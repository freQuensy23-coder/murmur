package app.murmur.data.models

import app.murmur.core.catalog.CatalogParser
import app.murmur.core.catalog.GgufTemplates
import app.murmur.core.catalog.ModelEntry
import app.murmur.core.catalog.Origin
import app.murmur.core.model.ArtifactId
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.ModelId
import app.murmur.core.model.UnitPreference
import app.murmur.core.store.InstallState
import app.murmur.core.store.Settings
import app.murmur.testing.FakeCatalogRepository
import app.murmur.testing.FakeDeviceProfileRepository
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class DefaultCatalogRepositoryTest {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val bundled = CatalogParser.bundled()

    private class MemStore : CustomModelStore {
        val rows = MutableStateFlow<Map<String, String>>(emptyMap())
        override val entries: Flow<List<String>> = rows.map { it.values.toList() }
        override suspend fun put(id: String, json: String, addedMs: Long) { rows.value = rows.value + (id to json) }
        override suspend fun delete(id: String) { rows.value = rows.value - id }
    }

    @After
    fun tearDown() = scope.cancel()

    private fun custom(): ModelEntry {
        val q = bundled.model(ModelId("qwen3.5-0.8b"))!!
        val a = q.artifacts.first()
        return q.copy(
            id = ModelId("custom-unsloth-qwen3.5-0.8b-gguf"), origin = Origin.CUSTOM,
            llm = q.llm!!.copy(prefixTokens = null, nCtx = GgufTemplates.DEFAULT_CUSTOM_NCTX),
            artifacts = listOf(a.copy(id = ArtifactId("custom-unsloth-qwen3.5-0.8b-gguf@q4_0"), memoryMb = emptyMap())),
        )
    }

    @Test
    fun customEntriesAreMergedUpsertedAndRemoved() = runBlocking {
        val store = MemStore()
        val repo = DefaultCatalogRepository(bundled, store, scope)
        val e = custom()
        repo.addCustom(e)
        val merged = withTimeout(5_000) { repo.catalog.first { it.model(e.id) != null } }
        assertEquals(bundled.models.size + 1, merged.models.size)
        assertEquals(emptyList<String>(), app.murmur.core.catalog.CatalogValidator.validate(merged))
        // After Preparing measured the prefix, the entry is stored again (upsert).
        repo.addCustom(GgufTemplates.withMeasuredPrefix(e, 3_129))
        assertEquals(9_216, withTimeout(5_000) { repo.catalog.first { it.model(e.id)?.llm?.prefixTokens == 3_129 } }.model(e.id)!!.llm!!.nCtx)
        repo.removeCustom(e.id)
        assertEquals(bundled.models.size, withTimeout(5_000) { repo.catalog.first { it.model(e.id) == null } }.models.size)
    }

    @Test
    fun brokenOrCollidingStoredRowsNeverBreakTheCatalog() = runBlocking {
        val store = MemStore()
        store.rows.value = mapOf(
            "junk" to "{not json",
            "collide" to CatalogParser.json.encodeToString(ModelEntry.serializer(), bundled.models.first().copy(origin = Origin.CUSTOM)),
            "ok" to CatalogParser.json.encodeToString(ModelEntry.serializer(), custom()),
        )
        val repo = DefaultCatalogRepository(bundled, store, scope)
        val c = withTimeout(5_000) { repo.catalog.first { it.models.size > bundled.models.size } }
        assertEquals(bundled.models.size + 1, c.models.size)
        try {
            repo.addCustom(bundled.models.first())
            fail("bundled entries are not stored as custom")
        } catch (e: IllegalArgumentException) { /* expected */ }
    }
}

class ModelDownloadRulesTest {
    private val catalog = FakeCatalogRepository(CatalogParser.bundled())
    private val installed = app.murmur.testing.FakeInstalledModelsRepository(
        CatalogParser.bundled().models.flatMap { m -> m.artifacts.map { it.id } }.toMutableSet(),
    )

    @Test
    fun packQueuesOnePerModelPlusVadAndOcrSkippingInstalled() = runBlocking {
        val rules = ModelDownloadRules(catalog, FakeDeviceProfileRepository(FakeDeviceProfileRepository.EMULATOR), installed)
        installed.download(ArtifactId("silero-vad-v5@onnx"), true)   // already installed
        val queued = rules.downloadPack("recommended", Settings())
        assertEquals(
            listOf("parakeet-tdt-0.6b-v3@onnx-int8", "pp-ocrv6-tiny-det@onnx", "pp-ocrv5-cyrillic-mobile-rec@onnx", "murmur-cleanup-qwen3.5-2b@q4_0"),
            queued.map { it.artifact.id.value },
        )
        assertTrue(queued.all { installed.state(it.artifact.id).first() is InstallState.Installed })
        assertTrue(rules.planPack("recommended", Settings(useScreen = false)).none { it.artifact.runtime.startsWith("ort/ppocr") })
    }

    @Test
    fun hubModelChoiceFollowsThePinAndUseScreenQueuesOcr() = runBlocking {
        val rules = ModelDownloadRules(catalog, FakeDeviceProfileRepository(FakeDeviceProfileRepository.FOLD7), installed)
        val pinCpu = Settings(unitPreference = mapOf("parakeet-tdt-0.6b-v3" to UnitPreference.Pinned(ComputeUnit.CPU)))
        assertEquals(listOf("parakeet-tdt-0.6b-v3@onnx-int8", "silero-vad-v5@onnx"), rules.downloadModel(ModelId("parakeet-tdt-0.6b-v3"), pinCpu).map { it.artifact.id.value })
        assertEquals(listOf("parakeet-tdt-0.6b-v3@qnn-sm8750-20s"), rules.downloadModel(ModelId("parakeet-tdt-0.6b-v3"), Settings()).map { it.artifact.id.value })
        assertEquals(listOf("pp-ocrv6-tiny-det@onnx", "pp-ocrv5-cyrillic-mobile-rec@onnx"), rules.onUseScreenEnabled(Settings()).map { it.artifact.id.value })
        assertEquals(emptyList<String>(), rules.onUseScreenEnabled(Settings()).map { it.artifact.id.value })
    }
}
