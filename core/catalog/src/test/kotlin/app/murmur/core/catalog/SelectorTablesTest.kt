package app.murmur.core.catalog

import app.murmur.core.model.ArtifactId
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.ComputeUnit.CPU
import app.murmur.core.model.ComputeUnit.GPU
import app.murmur.core.model.ComputeUnit.NPU
import app.murmur.core.model.ModelId
import app.murmur.core.model.UnitPreference
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Decision tables for ArtifactSelector (D2, ARCHITECTURE 8.2) and the download rules (ADR-050), over the bundled
 * catalog. Each row names the situation the brief lists.
 */
class SelectorTablesTest {
    private val catalog = CatalogParser.bundled()
    private val fold = DeviceProfile("SM8750", "arm64-v8a", setOf(NPU, GPU, CPU), RuntimeKeys.known, 11_500)
    private val emu = DeviceProfile(null, "x86_64", setOf(CPU), RuntimeKeys.implemented, 3_072)
    private val parakeet = catalog.model(ModelId("parakeet-tdt-0.6b-v3"))!!
    private val lfm3 = catalog.model(ModelId("lfm3-1.7b"))!!
    private val qwen = catalog.model(ModelId("qwen3.5-0.8b"))!!
    private val pNpu = ArtifactId("parakeet-tdt-0.6b-v3@qnn-sm8750-20s")
    private val pCpu = ArtifactId("parakeet-tdt-0.6b-v3@onnx-int8")
    private val lfmQ40 = ArtifactId("lfm3-1.7b@q4_0-qad")
    private val lfmQ4km = ArtifactId("lfm3-1.7b@q4_k_m")

    private fun health(vararg failed: Pair<ArtifactId, ComputeUnit>) = object : UnitHealth {
        override fun failed(artifact: ArtifactId, unit: ComputeUnit) = (artifact to unit) in failed
        override fun proven(artifact: ArtifactId, unit: ComputeUnit) = false
    }

    private fun List<Candidate>.keys() = map { it.artifact.id.value + "/" + it.unit }

    @Test
    fun pinnedUnitUnavailableOnThisDeviceRanksNothingAndChooseFallsBackToAutoForDownload() {
        // Emulator has no NPU: a pinned NPU cannot be served at all.
        assertEquals(emptyList<Candidate>(), ArtifactSelector.ranked(parakeet, emu, UnitPreference.Pinned(NPU)))
        assertNull(ArtifactSelector.choose(parakeet, emu, UnitPreference.Pinned(NPU)))
        // What the hub would download instead: the CPU build (Auto), never an error.
        assertEquals(pCpu, DownloadPlanner.artifactFor(parakeet, emu, UnitPreference.Pinned(NPU))!!.id)
    }

    @Test
    fun pinnedUnitNotInstalledIsOfferedForDownload() {
        // Fold7, user pinned NPU, only the CPU build is installed: nothing usable for the pin among installed ...
        assertNull(ArtifactSelector.choose(parakeet, fold, UnitPreference.Pinned(NPU), installed = setOf(pCpu)))
        // ... and the artifact to offer is the NPU build.
        assertEquals(pNpu, ArtifactSelector.choose(parakeet, fold, UnitPreference.Pinned(NPU))!!.artifact.id)
        assertEquals(pNpu, DownloadPlanner.artifactFor(parakeet, fold, UnitPreference.Pinned(NPU))!!.id)
    }

    @Test
    fun benchmarkOverridesCatalogOrderAndPreferredUnit() {
        // LFM3 Q4_0 prefers NPU; a valid benchmark showing GPU faster wins.
        assertEquals("$lfmQ40/NPU", ArtifactSelector.choose(lfm3, fold, UnitPreference.Auto)!!.let { "${it.artifact.id}/${it.unit}" })
        val gpuFaster = BenchLookup { a, u -> if (a == lfmQ40) mapOf(NPU to 2_400.0, GPU to 1_800.0, CPU to 5_000.0)[u] else null }
        val c = ArtifactSelector.choose(lfm3, fold, UnitPreference.Auto, gpuFaster)!!
        assertEquals(lfmQ40 to GPU, c.artifact.id to c.unit)
        // A benchmarked second artifact beats an unbenchmarked first one.
        val kq = BenchLookup { a, u -> if (a == lfmQ4km && u == CPU) 900.0 else null }
        assertEquals(lfmQ4km to CPU, ArtifactSelector.choose(lfm3, fold, UnitPreference.Auto, kq)!!.let { it.artifact.id to it.unit })
    }

    @Test
    fun invalidBenchmarkIsIgnored() {
        // BenchLookup returns null for an invalid record (fast but wrong text): order falls back to the catalog's.
        val invalidNpuValidCpu = BenchLookup { a, u -> if (a == pCpu && u == CPU) 300.0 else null }
        val allInvalid = BenchLookup { _, _ -> null }
        assertEquals(listOf("$pNpu/NPU", "$pCpu/CPU"), ArtifactSelector.ranked(parakeet, fold, UnitPreference.Auto, allInvalid).keys())
        assertEquals(listOf("$pCpu/CPU", "$pNpu/NPU"), ArtifactSelector.ranked(parakeet, fold, UnitPreference.Auto, invalidNpuValidCpu).keys())
    }

    @Test
    fun wrongSocBlocksSocSpecificArtifacts() {
        val sm8650 = fold.copy(soc = "SM8650")
        val cands = ArtifactSelector.candidates(parakeet, sm8650)
        assertEquals(Blocker.WRONG_SOC, cands.single { it.artifact.id == pNpu }.blocker)
        assertEquals(listOf("$pCpu/CPU"), ArtifactSelector.ranked(parakeet, sm8650, UnitPreference.Auto).keys())
    }

    @Test
    fun runtimeNotShippedBlocksTheArtifact() {
        val qwenAsr = catalog.model(ModelId("qwen3-asr-0.6b"))!!
        val cands = ArtifactSelector.candidates(qwenAsr, fold.copy(supportedRuntimes = RuntimeKeys.implemented))
        assertEquals(Blocker.RUNTIME_NOT_SHIPPED, cands.single { it.artifact.runtime == RuntimeKeys.LITERT_NPU }.blocker)
        assertTrue(cands.single { it.artifact.runtime == RuntimeKeys.SHERPA_OFFLINE_QWEN3_ASR }.usable)
    }

    @Test
    fun unitNotAvailableAndUnitFailedAreDistinctBlockers() {
        val noGpu = fold.copy(availableUnits = setOf(NPU, CPU))
        val failedNpu = health(lfmQ40 to NPU)
        val cands = ArtifactSelector.candidates(lfm3, noGpu, failedNpu).filter { it.artifact.id == lfmQ40 }
        assertEquals(mapOf(NPU to Blocker.UNIT_FAILED, GPU to Blocker.UNIT_NOT_AVAILABLE, CPU to null), cands.associate { it.unit to it.blocker })
        // Failed NPU: the same GGUF on CPU is chosen (no download).
        assertEquals(lfmQ40 to CPU, ArtifactSelector.choose(lfm3, noGpu, UnitPreference.Auto, health = failedNpu)!!.let { it.artifact.id to it.unit })
    }

    @Test
    fun installedOnlySelection() {
        // Only the K-quant LFM3 is installed: it is chosen even though Q4_0 comes first in the catalog.
        val r = ArtifactSelector.ranked(lfm3, fold, UnitPreference.Auto, installed = setOf(lfmQ4km))
        assertTrue(r.all { it.artifact.id == lfmQ4km })
        assertEquals(listOf(NPU, GPU, CPU), r.map { it.unit })
        assertEquals(emptyList<Candidate>(), ArtifactSelector.ranked(lfm3, fold, UnitPreference.Auto, installed = emptySet()))
    }

    @Test
    fun fallbackOrderIsSameFileOnNextUnitThenOtherArtifacts() {
        val r = ArtifactSelector.ranked(lfm3, fold, UnitPreference.Auto)
        assertEquals(
            listOf("$lfmQ40/NPU", "$lfmQ40/GPU", "$lfmQ40/CPU", "$lfmQ4km/NPU", "$lfmQ4km/GPU", "$lfmQ4km/CPU"),
            r.keys(),
        )
        assertEquals(lfmQ40 to GPU, ArtifactSelector.fallbackAfter(r, r[0])!!.let { it.artifact.id to it.unit })
        assertNull(ArtifactSelector.fallbackAfter(r, r.last()))
        // Qwen3.5 prefers CPU (llm spike D4): CPU first, then NPU > GPU on the same file.
        assertEquals(listOf(CPU, NPU, GPU), ArtifactSelector.ranked(qwen, fold, UnitPreference.Auto).take(3).map { it.unit })
        // A pinned unit that fails still falls back to the Auto order.
        val pinned = ArtifactSelector.ranked(parakeet, fold, UnitPreference.Pinned(CPU))
        assertEquals(listOf("$pCpu/CPU", "$pNpu/NPU"), pinned.keys())
    }

    @Test
    fun packOnTheEmulatorQueuesCpuBuildsVadAndOcrPair() {
        val pack = catalog.packs.single { it.id == "recommended" }
        val plan = DownloadPlanner.forPack(catalog, pack, emu, useScreen = true)
        assertEquals(
            listOf("parakeet-tdt-0.6b-v3@onnx-int8", "silero-vad-v5@onnx", "pp-ocrv6-tiny-det@onnx", "pp-ocrv5-ru-en-he-mobile-rec@onnx", "murmur-cleanup-qwen3.5-2b@q4_0"),
            plan.map { it.artifact.id.value },
        )
        // One artifact per model.
        assertEquals(plan.size, plan.map { it.model }.toSet().size)
    }

    @Test
    fun packOnFold7QueuesTheNpuBuildAndOcrFollowsUseScreen() {
        val speechOnly = catalog.packs.single { it.id == "speech-only" }
        val on = DownloadPlanner.forPack(catalog, speechOnly, fold, useScreen = true)
        assertEquals(listOf(pNpu.value, "silero-vad-v5@onnx", "pp-ocrv6-tiny-det@onnx", "pp-ocrv5-ru-en-he-mobile-rec@onnx"), on.map { it.artifact.id.value })
        assertEquals(PlannedDownload.Reason.OCR_PAIR, on.last().reason)
        val recommendedOff = DownloadPlanner.forPack(catalog, catalog.packs.single { it.id == "recommended" }, fold, useScreen = false)
        assertTrue(recommendedOff.none { it.artifact.runtime.startsWith("ort/ppocr") })
        assertEquals(listOf("pp-ocrv6-tiny-det@onnx", "pp-ocrv5-ru-en-he-mobile-rec@onnx"), DownloadPlanner.ocrPair(catalog, fold).map { it.artifact.id.value })
    }

    @Test
    fun choosingABatchAsrModelInTheHubAddsVadAndAStreamingOneDoesNot() {
        val giga = DownloadPlanner.forModels(catalog, listOf(ModelId("gigaam-v3-e2e-rnnt")), emu, useScreen = false)
        assertEquals(listOf("gigaam-v3-e2e-rnnt@onnx-int8" to PlannedDownload.Reason.REQUESTED, "silero-vad-v5@onnx" to PlannedDownload.Reason.VAD_FOR_BATCH_ASR),
            giga.map { it.artifact.id.value to it.reason })
        val tone = DownloadPlanner.forModels(catalog, listOf(ModelId("t-one")), emu, useScreen = false)
        assertEquals(listOf("t-one@onnx"), tone.map { it.artifact.id.value })
        // NPU-only model on the emulator: nothing to download.
        assertEquals(emptyList<PlannedDownload>(), DownloadPlanner.forModels(catalog, listOf(ModelId("nemotron-speech-streaming-en-0.6b")), emu, useScreen = false))
    }

    @Test
    fun validateEntryCatchesBrokenCustomEntries() {
        val ok = qwen.copy(origin = Origin.CUSTOM)
        assertEquals(emptyList<String>(), CatalogValidator.validateEntry(ok))
        val a = ok.artifacts.first()
        val broken = ok.copy(llm = ok.llm!!.copy(template = "llama"), artifacts = listOf(a.copy(layout = mapOf("model" to "../x.gguf"))))
        val problems = CatalogValidator.validateEntry(broken)
        assertTrue(problems.toString(), problems.any { "unknown template" in it } && problems.any { "layout[model]" in it })
        assertTrue(CatalogValidator.isAllowedHost("cas-bridge.xethub.hf.co"))
        assertTrue(CatalogValidator.isAllowedHost("objects.githubusercontent.com"))
        assertTrue(!CatalogValidator.isAllowedHost("evilhuggingface.co") && !CatalogValidator.isAllowedHost("hf.co.evil.com"))
    }

    @Test
    fun measuredPrefixReDerivesTheContextWindow() {
        val custom = qwen.copy(origin = Origin.CUSTOM, llm = qwen.llm!!.copy(prefixTokens = null, nCtx = GgufTemplates.DEFAULT_CUSTOM_NCTX))
        val measured = GgufTemplates.withMeasuredPrefix(custom, 97)
        assertEquals(97, measured.llm!!.prefixTokens)
        assertEquals(6_144, measured.llm.nCtx)
        assertEquals(emptyList<String>(), CatalogValidator.validateEntry(measured))
    }
}
