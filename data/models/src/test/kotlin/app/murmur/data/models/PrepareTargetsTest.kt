package app.murmur.data.models

import app.murmur.core.catalog.CatalogParser
import app.murmur.core.dictation.PrepareTargets
import app.murmur.core.model.ArtifactId
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.ModelKind
import app.murmur.core.model.MurmurError
import app.murmur.core.model.Outcome
import app.murmur.core.store.InstallState
import app.murmur.core.store.InstalledModelsRepository
import app.murmur.core.store.Settings
import app.murmur.core.store.StorageInfo
import app.murmur.testing.FakeCatalogRepository
import app.murmur.testing.FakeDeviceProfileRepository
import app.murmur.testing.FakeUnitHealth
import app.murmur.testing.InMemoryBenchmarkRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** DefaultModelResolver.prepareTargets: the EnginePreparer input of ARCHITECTURE 8.5 (ModelResolver KDoc). */
class PrepareTargetsTest {
    private val pNpu = ArtifactId("parakeet-tdt-0.6b-v3@qnn-sm8750-20s")
    private val pCpu = ArtifactId("parakeet-tdt-0.6b-v3@onnx-int8")
    private val vad = ArtifactId("silero-vad-v5@onnx")
    private val det = ArtifactId("pp-ocrv6-tiny-det@onnx")
    private val rec = ArtifactId("pp-ocrv5-ru-en-he-mobile-rec@onnx")
    private val lfm = ArtifactId("lfm3-1.7b@q4_0-qad")

    private class States(private val map: Map<ArtifactId, InstallState>) : InstalledModelsRepository {
        override val states: StateFlow<Map<ArtifactId, InstallState>> = MutableStateFlow(map)
        override fun state(id: ArtifactId): Flow<InstallState> = flowOf(map[id] ?: InstallState.NotInstalled)
        override suspend fun download(id: ArtifactId, wifiOnly: Boolean) = Unit
        override suspend fun pause(id: ArtifactId) = Unit
        override suspend fun delete(id: ArtifactId) = Unit
        override suspend fun storage() = StorageInfo(0, 0)
    }

    private fun installed(vararg ids: String) = ids.map(::ArtifactId).associateWith { InstallState.Installed("/data/models/${it.value}", 1) as InstallState }

    private val health = FakeUnitHealth()

    private fun resolver(states: Map<ArtifactId, InstallState>, dirOf: ((ArtifactId) -> String)? = null) = DefaultModelResolver(
        FakeCatalogRepository(CatalogParser.bundled()), States(states),
        FakeDeviceProfileRepository(FakeDeviceProfileRepository.FOLD7), health, InMemoryBenchmarkRepository(), dirOf,
    )

    private fun ok(r: Outcome<PrepareTargets>) = (r as? Outcome.Ok)?.value ?: error("expected Ok, got $r")

    @Test
    fun npuAsrArtifactComesWithItsVadAndTheCpuArtifactThatIsNotInstalled() = runBlocking {
        val t = ok(resolver(installed(pNpu.value, vad.value)).prepareTargets(pNpu, Settings()))
        assertEquals(ModelKind.ASR, t.kind)
        assertEquals(listOf(ComputeUnit.NPU), t.specs.map { it.unit })
        assertTrue(t.specs.all { it.artifactId == pNpu })
        assertTrue(t.needsVad)
        assertEquals(vad, t.vad?.artifactId)
        assertEquals(pCpu, t.cpuArtifact)
        assertFalse(t.cpuArtifactInstalled)
        val t2 = ok(resolver(installed(pNpu.value, pCpu.value, vad.value)).prepareTargets(pNpu, Settings()))
        assertTrue(t2.cpuArtifactInstalled)
    }

    @Test
    fun llmIsProvenOnEveryUsableUnitAndAFailedUnitIsSkipped() = runBlocking {
        val t = ok(resolver(installed(lfm.value)).prepareTargets(lfm, Settings()))
        assertEquals(ModelKind.LLM, t.kind)
        assertTrue(t.specs.size >= 2)
        assertEquals(ComputeUnit.NPU, t.specs.first().unit)
        val llm = t.llm!!
        assertEquals("lfm3", llm.template)
        assertEquals(7_168, llm.nCtx)
        assertEquals(64, llm.modelSha256.length)
        assertTrue(llm.memoryMb > 0)
        health.markFailed(lfm, ComputeUnit.NPU, "engine.load.llm")
        val after = ok(resolver(installed(lfm.value)).prepareTargets(lfm, Settings()))
        assertFalse(after.specs.any { it.unit == ComputeUnit.NPU })
    }

    @Test
    fun anArtifactBeingPreparedUsesTheStorageFolder() = runBlocking {
        val t = ok(resolver(mapOf(det to InstallState.Preparing(0f)) + installed(rec.value)) { "/store/${it.value}" }.prepareTargets(det, Settings()))
        assertEquals("/store/${det.value}", t.specs.single().dir)
        assertEquals(rec, t.ocrPartner?.artifactId)
    }

    @Test
    fun unknownOrMissingArtifactsAreErrors() = runBlocking {
        val r = resolver(installed(pNpu.value))
        assertEquals(Outcome.Err(MurmurError.CatalogInvalid("unknown artifact")), r.prepareTargets(ArtifactId("nope@x"), Settings()))
        assertTrue(r.prepareTargets(pCpu, Settings()) is Outcome.Err)
        // Preparing without a storage folder resolver cannot be located.
        assertTrue(resolver(mapOf(det to InstallState.Preparing(0f))).prepareTargets(det, Settings()) is Outcome.Err)
    }
}
