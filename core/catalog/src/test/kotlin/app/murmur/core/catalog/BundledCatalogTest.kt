package app.murmur.core.catalog

import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.ModelId
import app.murmur.core.model.ModelKind
import app.murmur.core.model.UnitPreference
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class BundledCatalogTest {
    private val catalog = CatalogParser.bundled()

    @Test
    fun bundledCatalogIsValid() {
        assertEquals(emptyList<String>(), CatalogValidator.validate(catalog))
    }

    @Test
    fun npuContextFilesComeFromTheOwnersRepoAndStayPinned() {
        // OWNER_QUESTIONS 17 answered 2026-11-20: the compiled contexts live in the owner's public HF repo, at a tagged
        // revision, sha256-pinned per file. They stay on the ort-qnn runtimes only, which are never in
        // RuntimeKeys.implemented: RUNTIME_NOT_SHIPPED in every build without -Pmurmur.qnn, never picked by Auto.
        val npu = catalog.models.flatMap { m -> m.artifacts.filter { it.runtime in RuntimeKeys.qnnEp }.flatMap { a -> a.files.map { a to it } } }
        assertTrue("the catalog carries no NPU context files", npu.isNotEmpty())
        for ((a, f) in npu) {
            assertEquals("${a.id}/${f.name}", NPU_CTX_REPO, f.repo)
            assertTrue("${a.id}: ${a.runtime}", a.runtime !in RuntimeKeys.implemented)
            assertTrue("${a.id}/${f.name}", f.sha256.matches(Regex("[0-9a-f]{64}")) && f.bytes > 0)
        }
    }

    @Test
    fun ownerRequestedFamiliesAreListed() {
        // docs/BRIEF.md "Model hub": Whisper (sizes), Parakeet, GigaAM, T-one, Qwen3-ASR, LFM2.5, LFM3, Qwen3.5, fine-tunes, None.
        val families = catalog.models.map { it.family }.toSet()
        for (f in listOf("whisper", "parakeet", "gigaam", "t-one", "qwen3-asr", "lfm2.5", "lfm3", "qwen3.5", "murmur-cleanup"))
            assertTrue("missing family $f in $families", f in families)
        assertTrue(catalog.models.count { it.family == "whisper" } >= 2)
    }

    @Test
    fun everyKindHasADefault() {
        assertNotNull(catalog.model(catalog.defaults.vad))
        assertEquals(ModelKind.OCR_DET, catalog.model(catalog.defaults.ocrDet)!!.kind)
        assertEquals(ModelKind.OCR_REC, catalog.model(catalog.defaults.ocrRec)!!.kind)
        assertEquals(ModelKind.ASR, catalog.model(catalog.defaults.asr.getValue("ru+en"))!!.kind)
    }

    @Test
    fun emulatorGetsCpuArtifactOfDefaultAsr() {
        val emu = DeviceProfile(null, "x86_64", setOf(ComputeUnit.CPU), RuntimeKeys.known, 3072)
        val entry = catalog.model(catalog.defaults.asr.getValue("ru+en"))!!
        val c = ArtifactSelector.choose(entry, emu, UnitPreference.Auto)
        assertNotNull(c); assertEquals(ComputeUnit.CPU, c!!.unit)
    }

    @Test
    fun fold7PrefersNpuForDefaultAsr() {
        val fold = DeviceProfile("SM8750", "arm64-v8a", setOf(ComputeUnit.NPU, ComputeUnit.GPU, ComputeUnit.CPU), RuntimeKeys.known, 11_500)
        val entry = catalog.model(catalog.defaults.asr.getValue("ru+en"))!!
        assertEquals(ComputeUnit.NPU, ArtifactSelector.choose(entry, fold, UnitPreference.Auto)!!.unit)
        assertEquals(ComputeUnit.CPU, ArtifactSelector.choose(entry, fold, UnitPreference.Pinned(ComputeUnit.CPU))!!.unit)
    }

    @Test
    fun sherpaLayoutDetection() {
        val (rt, layout) = SherpaLayout.detect(listOf("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"))!!
        assertEquals(RuntimeKeys.SHERPA_OFFLINE_TRANSDUCER, rt)
        assertEquals("tokens.txt", layout["tokens"])
    }

    @Test
    fun noneIsNotACatalogModel() {
        // "None" (no cleanup) is a UI choice (CleanupChoice.None), not a catalog entry.
        assertTrue(catalog.model(ModelId("none")) == null)
    }

    private val fold = DeviceProfile("SM8750", "arm64-v8a", setOf(ComputeUnit.NPU, ComputeUnit.GPU, ComputeUnit.CPU), RuntimeKeys.known, 11_500)

    @Test
    fun everyLlmContextHoldsItsPrefixPlusAFiveMinuteDictation() {
        // ADR-045 (review blocker): a 5-minute dictation must fit after the prefix (default.v4 3,116-3,256 tokens could not
        // clean up 60 s in n_ctx 4,096). Stock default.v1 (ADR-085): 1,047-1,101 tokens, n_ctx 7,168.
        for (m in catalog.models.filter { it.kind == ModelKind.LLM }) {
            val l = m.llm!!
            val prefix = assertNotNull(m.id.value, l.prefixTokens).let { l.prefixTokens!! }
            assertTrue("${m.id}: nCtx ${l.nCtx} < $prefix + reserve", l.nCtx >= prefix + LlmTraits.CONTEXT_RESERVE_TOKENS)
            assertEquals(GgufTemplates.nCtxFor(prefix), l.nCtx)
        }
        // native/llama/tools/tokenizer_check.py (llama-tokenize on the real GGUFs); a prompt change must update the catalog.
        assertEquals(1_072, catalog.model(ModelId("lfm3-1.7b"))!!.llm!!.prefixTokens)
        assertEquals(1_047, catalog.model(ModelId("qwen3.5-0.8b"))!!.llm!!.prefixTokens)
        assertEquals(1_101, catalog.model(ModelId("lfm2.5-350m"))!!.llm!!.prefixTokens)
        assertEquals(1_062, catalog.model(ModelId("qwen3-0.6b"))!!.llm!!.prefixTokens)
        assertEquals(1_072, catalog.model(ModelId("lfm2.5-2.6b"))!!.llm!!.prefixTokens) // its own vocabulary
        // ADR-085: every stock (non-fine-tune) LLM ships the stock profile.
        val stock = catalog.models.filter { it.kind == ModelKind.LLM && it.family != "murmur-cleanup" }
        assertEquals(8, stock.size)
        stock.forEach {
            assertEquals(it.id.value, GgufTemplates.STOCK_PROFILE, it.llm!!.promptProfile)
            assertEquals(it.id.value, 7_168, it.llm!!.nCtx)
        }
        assertEquals("finetuned.v2", catalog.model(ModelId("murmur-cleanup-qwen3.5-2b"))!!.llm!!.promptProfile)
        assertTrue(catalog.model(ModelId("murmur-cleanup-qwen3-0.6b")) == null)
    }

    @Test
    fun previewOnlyModelsAreNeverTheSpeechModel() {
        val nemotron = catalog.models.first { it.asr?.previewOnly == true }
        assertTrue(!nemotron.selectableAsAsr)
        catalog.defaults.asr.values.forEach { assertTrue(catalog.model(it)!!.selectableAsAsr) }
    }

    @Test
    fun failedUnitIsSkippedAndTheNextCandidateIsTheFallback() {
        val lfm = catalog.model(ModelId("lfm3-1.7b"))!!
        val ranked = ArtifactSelector.ranked(lfm, fold, UnitPreference.Auto)
        assertEquals(ComputeUnit.NPU, ranked.first().unit)
        val next = ArtifactSelector.fallbackAfter(ranked, ranked.first())!!
        assertEquals(ranked.first().artifact.id, next.artifact.id) // same GGUF on the next unit: no download
        val npuFailed = object : UnitHealth {
            override fun failed(artifact: app.murmur.core.model.ArtifactId, unit: ComputeUnit) = unit == ComputeUnit.NPU
            override fun proven(artifact: app.murmur.core.model.ArtifactId, unit: ComputeUnit) = false
        }
        val c = ArtifactSelector.choose(lfm, fold, UnitPreference.Auto, health = npuFailed)!!
        assertTrue(c.unit != ComputeUnit.NPU)
        assertTrue(ArtifactSelector.candidates(lfm, fold, npuFailed).any { it.blocker == Blocker.UNIT_FAILED })
    }

    @Test
    fun invalidBenchmarksDoNotWinAndPinnedUnitsStillFallBack() {
        val asr = catalog.model(catalog.defaults.asr.getValue("ru+en"))!!
        // Only a valid record may score (BenchLookup contract); a pinned NPU keeps CPU as its fallback.
        val ranked = ArtifactSelector.ranked(asr, fold, UnitPreference.Pinned(ComputeUnit.NPU))
        assertEquals(listOf(ComputeUnit.NPU, ComputeUnit.CPU), ranked.map { it.unit })
        val cpuFaster = BenchLookup { _, u -> if (u == ComputeUnit.CPU) 50.0 else 400.0 }
        assertEquals(ComputeUnit.CPU, ArtifactSelector.choose(asr, fold, UnitPreference.Auto, cpuFaster)!!.unit)
    }

    @Test
    fun ggufArchitectureMapsToTheFourRenderTemplates() {
        assertEquals("qwen3.5", GgufTemplates.templateFor("qwen35"))
        assertEquals("lfm2.5", GgufTemplates.templateFor("lfm2"))
        assertNull(GgufTemplates.templateFor("llama"))
        assertEquals("finetuned.v1", GgufTemplates.profileFor("murmur-cleanup-qwen3.5-0.8b", null))
        assertEquals("default.v5", GgufTemplates.profileFor("murmur-cleanup-x", "default.v5"))
        assertEquals("default.v1", GgufTemplates.profileFor("Qwen3.5-0.8B", null))
    }

    @Test
    fun everyBundledArtifactHasAMemoryEstimate() {
        for (m in catalog.models) for (a in m.artifacts) for (u in a.units)
            assertTrue("${a.id}/$u", MemoryEstimate.mb(m, a, u) > 0 && a.memoryMb[u] != null)
    }

    @Test
    fun finetunedCleanupModelsAreAvailableFromOwnerHfRepo() {
        for (id in listOf("murmur-cleanup-lfm3-1.7b", "murmur-cleanup-qwen3.5-0.8b", "murmur-cleanup-qwen3.5-2b")) {
            val m = catalog.model(ModelId(id))!!
            assertEquals(id, Status.AVAILABLE, m.status)
            assertEquals(id, 1, m.artifacts.size)
            val f = m.artifacts.single().files.single()
            assertEquals(id, "satoshiNakomoroReal/murmur-cleanup", f.repo)
            assertTrue(id, f.url.startsWith("https://huggingface.co/satoshiNakomoroReal/murmur-cleanup/resolve/"))
            assertTrue(id, f.bytes > 100_000_000L)
            assertEquals(id, 64, f.sha256.length)
        }
    }

    @Test
    fun hebrewAsrAndGirgurCleanupAreRegisteredInCatalog() {
        assertEquals(ModelId("zipformer2-66m-streaming-he"), catalog.defaults.asr["he"])
        val heAsr = catalog.model(ModelId("zipformer2-66m-streaming-he"))!!
        assertEquals(ModelKind.ASR, heAsr.kind)
        assertEquals(Status.AVAILABLE, heAsr.status)
        assertEquals(listOf("he"), heAsr.languages.codes)
        assertTrue(heAsr.asr!!.streaming)
        assertTrue(heAsr.selectableAsAsr)
        val art = heAsr.artifacts.single()
        assertEquals(RuntimeKeys.SHERPA_ONLINE_TRANSDUCER, art.runtime)
        assertEquals("zipformer2", art.options["model_type"])
        assertEquals(setOf("encoder", "decoder", "joiner", "tokens"), art.layout.keys)
        assertTrue(art.files.all { it.bytes > 0 && it.sha256.matches(Regex("[0-9a-f]{64}")) })

        val girgur = catalog.model(ModelId("girgur-cleanup-qwen3.5-2b"))!!
        assertEquals(ModelKind.LLM, girgur.kind)
        assertEquals(Status.AVAILABLE, girgur.status)
        assertEquals(listOf("ru", "en", "he"), girgur.languages.codes)
        assertEquals(catalog.model(ModelId("murmur-cleanup-qwen3.5-2b"))!!.llm, girgur.llm)
        val girgurFile = girgur.artifacts.single().files.single()
        assertEquals("satoshiNakomoroReal/murmur-cleanup", girgurFile.repo)
        assertTrue(girgurFile.url.startsWith("https://huggingface.co/satoshiNakomoroReal/murmur-cleanup/resolve/"))
        assertTrue(girgurFile.bytes > 1_000_000_000L)
        assertEquals(64, girgurFile.sha256.length)

        val ocrRec = catalog.model(ModelId("pp-ocrv5-ru-en-he-mobile-rec"))!!
        assertEquals(catalog.defaults.ocrRec, ocrRec.id)
        assertEquals(ModelKind.OCR_REC, ocrRec.kind)
        assertEquals(Status.AVAILABLE, ocrRec.status)
        assertTrue("he" in ocrRec.languages.codes)
        assertEquals(setOf("model", "dict"), ocrRec.artifacts.single().layout.keys)
        assertTrue(ocrRec.artifacts.single().files.all { it.bytes > 0 && it.sha256.matches(Regex("[0-9a-f]{64}")) })
    }
}

/** build_catalog.py NPU_CTX_REPO (OWNER_QUESTIONS 17 answered 2026-11-20). */
private const val NPU_CTX_REPO = "satoshiNakomoroReal/npu-contexts"
