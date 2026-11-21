package app.murmur.data.models

import app.murmur.core.catalog.Candidate
import app.murmur.core.catalog.CatalogParser
import app.murmur.core.catalog.RuntimeKeys
import app.murmur.core.engine.HostState
import app.murmur.core.engine.InferenceHost
import app.murmur.core.engine.LoadOptions
import app.murmur.core.engine.ProcessMemory
import app.murmur.core.engine.ProcessProbe
import app.murmur.core.dictation.ActiveModels
import app.murmur.core.dictation.LlmSpeed
import app.murmur.core.model.ArtifactId
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.ComputeUnit.CPU
import app.murmur.core.model.ComputeUnit.GPU
import app.murmur.core.model.ComputeUnit.NPU
import app.murmur.core.model.InferProcess
import app.murmur.core.model.ModelId
import app.murmur.core.model.MurmurError
import app.murmur.core.model.Outcome
import app.murmur.core.model.UnitPreference
import app.murmur.core.benchmark.BenchScores
import app.murmur.core.store.BenchmarkRecord
import app.murmur.core.store.BenchmarkResultData
import app.murmur.core.store.CleanupChoice
import app.murmur.core.store.InstallState
import app.murmur.core.store.InstalledModelsRepository
import app.murmur.core.store.Settings
import app.murmur.core.store.StorageInfo
import app.murmur.testing.FakeCatalogRepository
import app.murmur.testing.FakeDeviceProfileRepository
import app.murmur.testing.FakeUnitHealth
import app.murmur.testing.InMemoryBenchmarkRepository
import java.io.File
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Resolver tables with fake install states (WP-13 acceptance). */
class DefaultModelResolverTest {
    @get:org.junit.Rule val timeout: org.junit.rules.Timeout = org.junit.rules.Timeout.seconds(60)

    private val catalog = FakeCatalogRepository(CatalogParser.bundled())
    private val health = FakeUnitHealth()
    private val bench = InMemoryBenchmarkRepository()

    private val pNpu = ArtifactId("parakeet-tdt-0.6b-v3@qnn-sm8750-20s")
    private val pCpu = ArtifactId("parakeet-tdt-0.6b-v3@onnx-int8")
    private val vad = ArtifactId("silero-vad-v5@onnx")
    private val det = ArtifactId("pp-ocrv6-tiny-det@onnx")
    private val rec = ArtifactId("pp-ocrv5-cyrillic-mobile-rec@onnx")
    private val lfm = ArtifactId("lfm3-1.7b@q4_0-qad")
    private val qwen = ArtifactId("qwen3.5-0.8b@q4_0")

    /** Explicit install states: only [ids] are installed, under /data/models/<id>. */
    private class Installed(ids: List<ArtifactId>) : InstalledModelsRepository {
        private val map = MutableStateFlow(ids.associateWith { InstallState.Installed("/data/models/${it.value}", 1) as InstallState })
        override val states: StateFlow<Map<ArtifactId, InstallState>> = map
        override fun state(id: ArtifactId): Flow<InstallState> = flowOf(map.value[id] ?: InstallState.NotInstalled)
        override suspend fun download(id: ArtifactId, wifiOnly: Boolean) = Unit
        override suspend fun pause(id: ArtifactId) = Unit
        override suspend fun delete(id: ArtifactId) = Unit
        override suspend fun storage() = StorageInfo(0, 0)
    }

    private fun resolver(installed: InstalledModelsRepository, device: FakeDeviceProfileRepository = FakeDeviceProfileRepository(FakeDeviceProfileRepository.FOLD7)) =
        DefaultModelResolver(catalog, installed, device, health, bench)

    private fun ok(r: Outcome<ActiveModels>): ActiveModels = (r as? Outcome.Ok)?.value ?: error("expected Ok, got $r")

    private val everything = listOf(pNpu, pCpu, vad, det, rec, lfm, qwen)
    private val withLfm = Settings(cleanup = CleanupChoice.Model(ModelId("lfm3-1.7b")))

    @Test
    fun emulatorProfileGetsCpuArtifactsAndNoAsrFallback() = runBlocking {
        val m = ok(resolver(Installed(everything), FakeDeviceProfileRepository(FakeDeviceProfileRepository.EMULATOR)).resolve(withLfm))
        assertEquals(pCpu, m.asr.artifactId)
        assertEquals(CPU, m.asr.unit)
        assertNull(m.asrFallback)
        assertEquals(listOf(CPU, CPU, CPU), listOf(m.vad!!.unit, m.ocrDet!!.unit, m.ocrRec!!.unit))
        assertEquals(lfm to CPU, m.llm!!.spec.artifactId to m.llm!!.spec.unit)
        assertNull(m.llm!!.fallback)
        assertEquals("0", m.llm!!.spec.options[LoadOptions.USE_MMAP])
    }

    @Test
    fun fold7WithBothParakeetArtifactsGetsNpuPrimaryAndCpuFallback() = runBlocking {
        val m = ok(resolver(Installed(everything)).resolve(Settings()))
        assertEquals(pNpu to NPU, m.asr.artifactId to m.asr.unit)
        assertEquals(pCpu to CPU, m.asrFallback!!.artifactId to m.asrFallback!!.unit)
        assertEquals(20.0, m.asr.maxWindowSec)
        // The tarball extracts into one folder: the spec points at it and drops the "dir" role.
        assertEquals("/data/models/${pNpu.value}/sherpa-onnx-qnn-SM8750-binary-parakeet-tdt-0.6b-v3-20s-transducer", m.asr.dir)
        assertEquals(setOf("encoder", "decoder", "joiner", "tokens"), m.asr.layout.keys)
        assertEquals("/data/models/${pCpu.value}", m.asrFallback!!.dir)
    }

    @Test
    fun llmFallbackIsTheSameArtifactOnGpu() = runBlocking {
        val m = ok(resolver(Installed(everything)).resolve(withLfm))
        val llm = m.llm!!
        assertEquals(lfm to NPU, llm.spec.artifactId to llm.spec.unit)
        assertEquals(lfm to GPU, llm.fallback!!.artifactId to llm.fallback!!.unit)
        assertEquals("lfm3", llm.template)
        assertEquals("default.v1", llm.profile)
        assertEquals(7_168, llm.nCtx)
        assertEquals(1_072, llm.prefixTokens)
        val gguf = catalog.catalog.value.artifact(lfm)!!.second.files.single()
        assertEquals(gguf.sha256, llm.modelSha256)
        assertEquals(catalog.catalog.value.artifact(lfm)!!.second.memoryMb[NPU], llm.memoryMb)
        assertEquals(mapOf(LoadOptions.N_CTX to "7168", LoadOptions.THREADS to "4"), llm.spec.options)
        // GPU fallback keeps mmap; only CPU specs set use_mmap=0 (ADR-031).
        assertNull(llm.fallback!!.options[LoadOptions.USE_MMAP])
        assertNull(llm.speed)
    }

    @Test
    fun failedNpuInUnitHealthGivesCpuPrimary() = runBlocking {
        health.markFailed(pNpu, NPU, "engine.load.asr")
        val m = ok(resolver(Installed(everything)).resolve(Settings()))
        assertEquals(pCpu to CPU, m.asr.artifactId to m.asr.unit)
        assertNull(m.asrFallback)
        health.markFailed(lfm, NPU, "engine.load.llm")
        val llm = ok(resolver(Installed(everything)).resolve(withLfm)).llm!!
        assertEquals(GPU, llm.spec.unit)
        assertEquals(CPU, llm.fallback!!.unit)
    }

    @Test
    fun previewOnlyModelSelectedFallsBackToTheDefaultAsr() = runBlocking {
        val nemotron = catalog.catalog.value.models.first { it.asr?.previewOnly == true }
        val installed = Installed(everything + nemotron.artifacts.map { it.id })
        val m = ok(resolver(installed).resolve(Settings(asrModel = nemotron.id)))
        assertEquals(ModelId("parakeet-tdt-0.6b-v3"), m.asr.modelId)
    }

    @Test
    fun aSelectedCustomModelIsUsedOnTheFirstResolveAfterAColdStart() = runBlocking {
        // The custom entry arrives with the store's first read, after the resolver was asked.
        val bundled = CatalogParser.bundled()
        val base = bundled.model(ModelId("parakeet-tdt-0.6b-v3"))!!
        val cpuArtifact = base.artifacts.first { it.id == pCpu }
        val custom = base.copy(
            id = ModelId("custom-parakeet"), origin = app.murmur.core.catalog.Origin.CUSTOM,
            artifacts = listOf(cpuArtifact.copy(id = ArtifactId("custom-parakeet@onnx-int8"))),
        )
        val late = object : app.murmur.core.catalog.CatalogRepository {
            private val flow = MutableStateFlow(bundled)
            override val catalog: StateFlow<app.murmur.core.catalog.Catalog> = flow
            override suspend fun loaded(): app.murmur.core.catalog.Catalog {
                kotlinx.coroutines.delay(200)
                flow.value = bundled.copy(models = bundled.models + custom)
                return flow.value
            }
            override suspend fun addCustom(entry: app.murmur.core.catalog.ModelEntry) = Unit
            override suspend fun removeCustom(id: ModelId) = Unit
        }
        val installed = Installed(everything + custom.artifacts.single().id)
        val m = ok(DefaultModelResolver(late, installed, FakeDeviceProfileRepository(FakeDeviceProfileRepository.FOLD7), health, bench)
            .resolve(Settings(asrModel = custom.id)))
        assertEquals(custom.id to custom.artifacts.single().id, m.asr.modelId to m.asr.artifactId)
    }

    @Test
    fun batchAsrWithoutVadIsNoVadModel() = runBlocking {
        val r = resolver(Installed(listOf(pNpu, pCpu, det, rec, lfm))).resolve(Settings())
        assertEquals(Outcome.Err(MurmurError.NoVadModel), r)
        // A streaming model needs no VAD.
        val tone = ArtifactId("t-one@onnx")
        val m = ok(resolver(Installed(listOf(tone))).resolve(Settings(asrModel = ModelId("t-one"), languages = listOf("ru"))))
        assertEquals(tone, m.asr.artifactId)
        assertTrue(m.asrTraits.streaming)
        assertNull(m.vad)
    }

    @Test
    fun noInstalledAsrIsNoAsrModel() = runBlocking {
        assertEquals(Outcome.Err(MurmurError.NoAsrModel), resolver(Installed(listOf(vad, lfm))).resolve(Settings()))
        // Only the NPU build installed on the emulator: unusable, so still no speech model.
        assertEquals(Outcome.Err(MurmurError.NoAsrModel),
            resolver(Installed(listOf(pNpu, vad)), FakeDeviceProfileRepository(FakeDeviceProfileRepository.EMULATOR)).resolve(Settings()))
    }

    @Test
    fun probeFailureGivesCpuOnlyProfileAndNoException() = runBlocking {
        val broken = object : InferenceHost {
            override val process = InferProcess.ASR
            override val state: StateFlow<HostState> = MutableStateFlow(HostState.Down)
            override suspend fun connect(): Boolean = throw IllegalStateException("binder died")
            override suspend fun probe(): ProcessProbe = error("unreachable")
            override suspend fun memory(): ProcessMemory = error("unreachable")
            override fun scheduleIdleExit(ms: Long) = Unit
            override suspend fun shutdown() = Unit
        }
        val cache = File(tempDir("probe"), "device-profile.json")
        val prober = DeviceProfileProber(cache, "1.0+1", broken, broken, { app.murmur.data.models.BaseDeviceInfo("SM8750", "arm64-v8a", 11_500) })
        val device = object : app.murmur.core.catalog.DeviceProfileRepository {
            override val profile = prober.profile
            override suspend fun ensure() = prober.ensure()
            override suspend fun reprobe() = prober.reprobe()
        }
        val r = DefaultModelResolver(catalog, Installed(everything), device, health, bench).resolve(withLfm)
        val m = ok(r)
        assertEquals(pCpu to CPU, m.asr.artifactId to m.asr.unit)
        assertNull(m.asrFallback)
        assertEquals(CPU, m.llm!!.spec.unit)
        assertEquals(setOf(CPU), prober.profile.value!!.availableUnits)
        assertEquals(null, prober.profile.value!!.soc)
        assertTrue("a failed probe is not cached on disk", !cache.exists())
    }

    @Test
    fun cleanupChoicesNoneDefaultMissingAndChosen() = runBlocking {
        val all = resolver(Installed(everything))
        ok(all.resolve(Settings(cleanup = CleanupChoice.None))).let { assertNull(it.llm); assertEquals("none", it.llmUnavailable) }
        ok(all.resolve(Settings())).let { assertNull(it.llm); assertEquals("none", it.llmUnavailable) }
        ok(resolver(Installed(listOf(pCpu, vad))).resolve(withLfm)).let { assertNull(it.llm); assertEquals("no_model", it.llmUnavailable) }
        ok(all.resolve(Settings(cleanup = CleanupChoice.Model(ModelId("lfm2.5-350m"))))).let { assertNull(it.llm); assertEquals("no_model", it.llmUnavailable) }
        ok(all.resolve(Settings(cleanup = CleanupChoice.Model(ModelId("murmur-cleanup-qwen3.5-0.8b"))))).let { assertEquals("no_model", it.llmUnavailable) }
        val q = ok(all.resolve(Settings(cleanup = CleanupChoice.Model(ModelId("qwen3.5-0.8b"))))).llm!!
        assertEquals(qwen to CPU, q.spec.artifactId to q.spec.unit)   // Qwen3.5 prefers CPU (llm spike D4)
        assertEquals("qwen3.5", q.template)
        assertEquals(1_047, q.prefixTokens)
    }

    @Test
    fun useScreenOffOrOcrPairIncompleteGivesNoOcr() = runBlocking {
        ok(resolver(Installed(everything)).resolve(Settings(useScreen = false))).let { assertNull(it.ocrDet); assertNull(it.ocrRec) }
        ok(resolver(Installed(listOf(pCpu, vad, det))).resolve(Settings())).let { assertNull(it.ocrDet); assertNull(it.ocrRec) }
    }

    @Test
    fun pinnedUnitIsHonouredAndAPinNothingInstalledServesFallsBackToAuto() = runBlocking {
        val pinCpu = Settings(unitPreference = mapOf("parakeet-tdt-0.6b-v3" to UnitPreference.Pinned(CPU)))
        ok(resolver(Installed(everything)).resolve(pinCpu)).let {
            assertEquals(pCpu, it.asr.artifactId)
            assertEquals(pNpu, it.asrFallback!!.artifactId)
        }
        val pinNpu = Settings(unitPreference = mapOf("parakeet-tdt-0.6b-v3" to UnitPreference.Pinned(NPU)))
        ok(resolver(Installed(listOf(pCpu, vad))).resolve(pinNpu)).let { assertEquals(pCpu, it.asr.artifactId) }
    }

    @Test
    fun validBenchmarkOverridesAndInvalidLatestIsIgnored() = runBlocking {
        fun asr(unit: ComputeUnit, rtfx: Double, valid: Boolean, at: Long, artifact: ArtifactId) = BenchmarkRecord(
            artifact, unit, "1", "fp", at,
            BenchmarkResultData.Asr(rtfx, 100, 1000, null, 900, unit.name, emptyList(), valid = valid, invalidReason = if (valid) null else "wer_high"),
        )
        // CPU measured faster than NPU: CPU wins.
        bench.save(asr(NPU, 5.0, true, 1, pNpu))
        bench.save(asr(CPU, 20.0, true, 1, pCpu))
        assertEquals(pCpu, ok(resolver(Installed(everything)).resolve(Settings())).asr.artifactId)
        // The latest CPU run is invalid (wrong text): it no longer scores, NPU's valid run wins.
        bench.save(asr(CPU, 40.0, false, 2, pCpu))
        assertEquals(pNpu, ok(resolver(Installed(everything)).resolve(Settings())).asr.artifactId)
        // LLM speed comes from the newest record of the chosen (artifact, unit) (ADR-071).
        bench.save(BenchmarkRecord(lfm, NPU, "1", "fp", 3, BenchmarkResultData.Llm(40.9, 569.0, 120.0, 1300.0, 900, 1200, "NPU", 1.0)))
        assertEquals(LlmSpeed(prefillTps = 569.0, decodeTps = 40.9), ok(resolver(Installed(everything)).resolve(withLfm)).llm!!.speed)
        val scores = BenchScores.lookup(listOf(asr(CPU, 20.0, true, 1, pCpu), asr(CPU, 40.0, false, 2, pCpu)))
        assertNull(scores.score(pCpu, CPU))
        assertEquals(50.0, BenchScores.lookup(listOf(asr(NPU, 20.0, true, 1, pNpu))).score(pNpu, NPU)!!, 1e-9)
    }

    /** The benchmark loads what a dictation loads, on the candidate's unit, even when UnitHealth marked it failed (ADR-080). */
    @Test
    fun benchSpecsFollowLoadSpecAndIgnoreUnitHealth() = runBlocking {
        val c = catalog.catalog.value
        fun entryOf(id: ArtifactId) = c.models.first { m -> m.artifacts.any { it.id == id } }
        fun cand(id: ArtifactId, unit: ComputeUnit) = Candidate(entryOf(id).artifacts.first { it.id == id }, unit, usable = true)
        health.markFailed(pNpu, NPU, "test")
        val r = resolver(Installed(everything))
        val asr = (r.benchSpecs.specs(entryOf(pNpu), cand(pNpu, NPU)) as Outcome.Ok).value
        assertEquals(pNpu, asr.main.artifactId)
        assertEquals(NPU, asr.main.unit)
        assertEquals(vad, asr.vad!!.artifactId)
        val llm = (r.benchSpecs.specs(entryOf(qwen), cand(qwen, CPU)) as Outcome.Ok).value
        assertEquals(ggufFile(entryOf(qwen).artifacts.first { it.id == qwen })!!.sha256, llm.modelSha256)
        assertEquals(rec, (r.benchSpecs.specs(entryOf(det), cand(det, CPU)) as Outcome.Ok).value.ocrPartner!!.artifactId)
        // Batch ASR without the VAD installed, or a candidate that is not installed: an error, never a throw.
        assertEquals(Outcome.Err(MurmurError.NoVadModel), resolver(Installed(listOf(pCpu))).benchSpecs.specs(entryOf(pCpu), cand(pCpu, CPU)))
        assertTrue(resolver(Installed(listOf(vad))).benchSpecs.specs(entryOf(pCpu), cand(pCpu, CPU)) is Outcome.Err)
    }

    @Test
    fun languageProfilePicksTheDefaultAsr() = runBlocking {
        assertEquals("ru", DefaultModelResolver.languageProfile(listOf("ru")))
        assertEquals("en", DefaultModelResolver.languageProfile(listOf("en")))
        assertEquals("ru+en", DefaultModelResolver.languageProfile(listOf("ru", "en")))
        assertEquals("ru+en", DefaultModelResolver.languageProfile(listOf("de")))
        val giga = ArtifactId("gigaam-v3-e2e-rnnt@onnx-int8")
        val m = ok(resolver(Installed(listOf(giga, vad))).resolve(Settings(languages = listOf("ru"))))
        assertEquals(giga, m.asr.artifactId)
    }

    @Test
    fun resolverNeverThrows() = runBlocking {
        val exploding = object : InstalledModelsRepository by Installed(emptyList()) {
            override fun state(id: ArtifactId): Flow<InstallState> = throw IllegalStateException("db closed")
        }
        val r = resolver(exploding).resolve(Settings())
        assertTrue(r is Outcome.Err && r.error is MurmurError.Internal)
        assertTrue(RuntimeKeys.isBatchAsr(RuntimeKeys.SHERPA_OFFLINE_TRANSDUCER))
    }
}
