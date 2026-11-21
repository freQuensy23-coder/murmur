package app.murmur.data.models

import app.murmur.core.catalog.CatalogValidator
import app.murmur.core.catalog.CustomModelRequest
import app.murmur.core.catalog.GgufTemplates
import app.murmur.core.catalog.ModelEntry
import app.murmur.core.catalog.Origin
import app.murmur.core.catalog.RuntimeKeys
import app.murmur.core.model.ModelKind
import app.murmur.core.model.MurmurError
import app.murmur.core.model.Outcome
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * HF custom models against recorded API answers (src/test/resources/hf, captured 2026-11-19 from huggingface.co for
 * unsloth/Qwen3.5-0.8B-GGUF and csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8) and a synthetic GGUF header.
 * The same flow against the live hub runs in the instrumented CustomModelLiveTest.
 */
class HfCustomModelResolverTest {
    private fun res(name: String) = javaClass.getResource("/hf/$name")!!.readText()
    private val net = FakeTransport()
    private val resolver = HfCustomModelResolver(net)

    private val qwenRepo = "unsloth/Qwen3.5-0.8B-GGUF"
    private val qwenSha = "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0"
    private val parakeetRepo = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    private val parakeetSha = "ed5d745623472f97cfc2e6c2916ee248798fc5a8"
    private val ggufUrl = "https://huggingface.co/$qwenRepo/resolve/$qwenSha/Qwen3.5-0.8B-Q4_0.gguf"

    private fun serveQwen(header: ByteArray) {
        net.json["https://huggingface.co/api/models/$qwenRepo/revision/main"] = res("qwen-revision.json")
        net.posts["https://huggingface.co/api/models/$qwenRepo/paths-info/$qwenSha"] = { res("qwen-paths.json") }
        net.files[ggufUrl] = header
    }

    private fun ok(r: Outcome<ModelEntry>) = (r as? Outcome.Ok)?.value ?: error("expected Ok, got $r")

    @Test
    fun ggufRepoResolvesToQ40WithTemplateProfileAndPinnedSha() = runBlocking {
        // Header longer than the first 4 MB step (tokenizer arrays, about 6 MB here), so the reader asks for the next 4 MB.
        serveQwen(GgufWriter.qwen35(vocab = 400_000).build())
        val e = ok(resolver.resolve(CustomModelRequest(ModelKind.LLM, qwenRepo)))
        assertEquals(Origin.CUSTOM, e.origin)
        val a = e.artifacts.single()
        val f = a.files.single()
        assertEquals("Qwen3.5-0.8B-Q4_0.gguf", f.path)
        assertEquals(qwenSha, f.revision)
        assertEquals("444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c", f.sha256)
        assertEquals(507_154_688L, f.bytes)
        assertEquals("q4_0", a.quant)
        assertEquals(RuntimeKeys.LLAMA_GGUF, a.runtime)
        assertEquals("qwen3.5", e.llm!!.template)
        assertEquals(GgufTemplates.STOCK_PROFILE, e.llm!!.promptProfile)
        assertEquals(GgufTemplates.DEFAULT_CUSTOM_NCTX, e.llm!!.nCtx)
        assertEquals(12_288, e.llm!!.kvBytesPerToken)
        assertEquals(emptyList<String>(), CatalogValidator.validateEntry(e))
        val ranges = net.requests.filter { it.url == ggufUrl }.map { it.headers["Range"] }
        assertEquals(listOf("bytes=0-4194303", "bytes=4194304-8388607"), ranges)
        assertTrue(resolver.lastHeaderBytes > 4L shl 20)
    }

    @Test
    fun profileComesFromTheGgufKeyTheFineTuneNameOrTheRequest() = runBlocking {
        serveQwen(GgufWriter.qwen35(profile = "finetuned.v1").build())
        assertEquals("finetuned.v1", ok(resolver.resolve(CustomModelRequest(ModelKind.LLM, qwenRepo))).llm!!.promptProfile)
        serveQwen(GgufWriter.qwen35(name = "murmur-cleanup-qwen3.5-0.8b").build())
        assertEquals("finetuned.v1", ok(resolver.resolve(CustomModelRequest(ModelKind.LLM, qwenRepo))).llm!!.promptProfile)
        serveQwen(GgufWriter.qwen35(name = "murmur-cleanup-qwen3.5-0.8b", profile = "finetuned.v2").build())
        assertEquals("finetuned.v2", ok(resolver.resolve(CustomModelRequest(ModelKind.LLM, qwenRepo))).llm!!.promptProfile)
        assertEquals("default.v4s", ok(resolver.resolve(CustomModelRequest(ModelKind.LLM, qwenRepo, promptProfile = "default.v4s"))).llm!!.promptProfile)
        // A small header is read in one request.
        assertEquals(listOf("bytes=0-4194303"), net.requests.filter { it.url == ggufUrl }.map { it.headers["Range"] }.distinct())
    }

    @Test
    fun unsupportedArchitectureIsAClearError() = runBlocking {
        serveQwen(GgufWriter().string("general.architecture", "llama").stringArray("tokenizer.ggml.tokens", 400_000) { "t$it" }.string("general.name", "Llama-3.2-1B").build())
        val r = resolver.resolve(CustomModelRequest(ModelKind.LLM, qwenRepo))
        val detail = ((r as Outcome.Err).error as MurmurError.CatalogInvalid).detail!!
        assertTrue(detail, detail.startsWith("unsupported GGUF architecture 'llama'") && "qwen35" in detail)
        assertEquals("rejected after the first 4 MB", listOf("bytes=0-4194303"), net.requests.filter { it.url == ggufUrl }.map { it.headers["Range"] })
    }

    @Test
    fun sherpaRepoResolvesToTheTransducerLayoutWithHashedSmallFile() = runBlocking {
        net.json["https://huggingface.co/api/models/$parakeetRepo/revision/main"] = res("parakeet-revision.json")
        net.posts["https://huggingface.co/api/models/$parakeetRepo/paths-info/$parakeetSha"] = { res("parakeet-paths.json") }
        val tokens = bytesOf(93_939, 5)
        net.files["https://huggingface.co/$parakeetRepo/resolve/$parakeetSha/tokens.txt"] = tokens
        val e = ok(resolver.resolve(CustomModelRequest(ModelKind.ASR, parakeetRepo)))
        val a = e.artifacts.single()
        assertEquals(RuntimeKeys.SHERPA_OFFLINE_TRANSDUCER, a.runtime)
        assertEquals(mapOf("encoder" to "encoder.int8.onnx", "decoder" to "decoder.int8.onnx", "joiner" to "joiner.int8.onnx", "tokens" to "tokens.txt"), a.layout)
        assertTrue(a.files.all { it.revision == parakeetSha && "/resolve/$parakeetSha/" in it.url })
        assertEquals(sha256(tokens), a.files.single { it.name == "tokens.txt" }.sha256)
        assertEquals("1afbe69c1ffdabee94755f42e4daf1d93549131a3da11356bda84fcef0f2c479", a.files.single { it.name == "encoder.int8.onnx" }.sha256)
        assertTrue("ru" in e.languages.codes && "en" in e.languages.codes)
        assertEquals("cc-by-4.0", e.license.id)
        assertTrue(!e.asr!!.streaming)
        assertEquals(emptyList<String>(), CatalogValidator.validateEntry(e))
        // test_wavs/* are not part of the model.
        assertEquals(4, a.files.size)
    }

    @Test
    fun gatedMissingAndMalformedRequests() = runBlocking {
        net.json["https://huggingface.co/api/models/meta-llama/Llama-3.2-1B/revision/main"] =
            """{"sha":"${"1".repeat(40)}","gated":"manual","siblings":[]}"""
        assertEquals(Outcome.Err(MurmurError.DownloadFailed(401, "gated or private repo")),
            resolver.resolve(CustomModelRequest(ModelKind.LLM, "meta-llama/Llama-3.2-1B")))
        val missing = resolver.resolve(CustomModelRequest(ModelKind.LLM, "nobody/nothing"))
        assertEquals("repo or revision not found", ((missing as Outcome.Err).error as MurmurError.CatalogInvalid).detail)
        assertTrue(resolver.resolve(CustomModelRequest(ModelKind.LLM, "../etc/passwd")) is Outcome.Err)
        assertTrue(resolver.resolve(CustomModelRequest(ModelKind.OCR_DET, qwenRepo)) is Outcome.Err)
    }

    @Test
    fun ggufPickingAndQuantNames() {
        val siblings = listOf("Qwen3.5-0.8B-BF16.gguf", "Qwen3.5-0.8B-Q4_K_M.gguf", "Qwen3.5-0.8B-Q4_0.gguf", "mmproj-F16.gguf", "Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
        assertEquals("Qwen3.5-0.8B-Q4_0.gguf", HfCustomModelResolver.pickGguf(siblings))
        assertEquals("Qwen3.5-0.8B-Q4_K_M.gguf", HfCustomModelResolver.pickGguf(siblings - "Qwen3.5-0.8B-Q4_0.gguf"))
        assertEquals(null, HfCustomModelResolver.pickGguf(listOf("a-Q8_0.gguf", "b-Q6_K.gguf")))
        assertEquals(null, HfCustomModelResolver.pickGguf(listOf("big-Q4_0-00001-of-00002.gguf", "mmproj-F16.gguf")))
        assertEquals("ud-q4_k_xl", HfCustomModelResolver.quantOf("Qwen3.5-0.8B-UD-Q4_K_XL.gguf"))
        assertEquals("custom-unsloth-qwen3.5-0.8b-gguf", HfCustomModelResolver.customId(qwenRepo).value)
    }
}
