package app.murmur.infer.asr

import app.murmur.core.catalog.CatalogParser
import app.murmur.core.catalog.RuntimeKeys
import app.murmur.core.engine.LoadOptions
import app.murmur.core.engine.LoadSpec
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.ModelKind
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.file.Files

/** LoadSpec -> sherpa-onnx config for every runtime key and unit, built from the bundled catalog's own layouts. */
class SherpaConfigsTest {
    private val catalog = CatalogParser.bundled()
    private val libs = QnnLibs.inDir("/app/lib")

    private fun spec(artifactId: String, unit: ComputeUnit? = null, options: Map<String, String> = emptyMap()): LoadSpec {
        val model = catalog.models.first { m -> m.artifacts.any { it.id.value == artifactId } }
        val a = model.artifacts.first { it.id.value == artifactId }
        return LoadSpec(model.kind, model.id, a.id, a.runtime, unit ?: a.units.first(),
            "/m/$artifactId/", a.layout, a.maxWindowSec, a.options + options)
    }

    @Test
    fun parakeetOnTheCpuReadsItsModelTypeFromTheOnnxMetadata() {
        val c = SherpaConfigs.offline(spec("parakeet-tdt-0.6b-v3@onnx-int8", options = mapOf(LoadOptions.THREADS to "2")), null)
        val m = c.modelConfig
        assertEquals("cpu", m.provider)
        assertEquals("", m.modelType)
        assertEquals(2, m.numThreads)
        assertEquals("/m/parakeet-tdt-0.6b-v3@onnx-int8/encoder.int8.onnx", m.transducer.encoder)
        assertEquals("/m/parakeet-tdt-0.6b-v3@onnx-int8/joiner.int8.onnx", m.transducer.joiner)
        assertEquals("/m/parakeet-tdt-0.6b-v3@onnx-int8/tokens.txt", m.tokens)
        assertEquals("", m.transducer.qnnConfig.contextBinary)
        assertEquals(16_000, c.featConfig.sampleRate)
    }

    @Test
    fun parakeetOnTheNpuUsesTheContextBinariesOfTheExtractedTarball() {
        val s = spec("parakeet-tdt-0.6b-v3@qnn-sm8750-20s")
        assertEquals(ComputeUnit.NPU, s.unit)
        val m = SherpaConfigs.offline(s, libs).modelConfig
        val d = "/m/parakeet-tdt-0.6b-v3@qnn-sm8750-20s/sherpa-onnx-qnn-SM8750-binary-parakeet-tdt-0.6b-v3-20s-transducer"
        assertEquals("qnn", m.provider)
        assertEquals("nemo_transducer", m.modelType)
        assertEquals("$d/encoder.bin,$d/decoder.bin,$d/joiner.bin", m.transducer.qnnConfig.contextBinary)
        assertEquals("/app/lib/libQnnHtp.so", m.transducer.qnnConfig.backendLib)
        assertEquals("/app/lib/libQnnSystem.so", m.transducer.qnnConfig.systemLib)
        assertEquals("", m.transducer.encoder)
        assertEquals("$d/tokens.txt", m.tokens)
        // without the QNN libraries the NPU is not available
        assertEquals("unit.npu", assertThrows(SpecException::class.java) { SherpaConfigs.offline(s, null) }.reason)
    }

    @Test
    fun gigaAmCtcAndQwen3AndWhisper() {
        val ctc = SherpaConfigs.offline(spec("gigaam-v3-e2e-ctc@onnx-int8"), null).modelConfig
        assertEquals("/m/gigaam-v3-e2e-ctc@onnx-int8/model.int8.onnx", ctc.nemo.model)

        val q = SherpaConfigs.offline(spec("qwen3-asr-0.6b@onnx-int8"), null).modelConfig.qwen3Asr
        assertEquals("/m/qwen3-asr-0.6b@onnx-int8/conv_frontend.onnx", q.convFrontend)
        assertEquals("/m/qwen3-asr-0.6b@onnx-int8", q.tokenizer)
        assertEquals(1024, q.maxTotalLen)
        assertEquals("Russian", SherpaConfigs.qwen3Language("ru"))
        assertEquals("English", SherpaConfigs.qwen3Language("en"))

        val w = SherpaConfigs.offline(spec("whisper-small@onnx-int8"), null, language = "ru").modelConfig.whisper
        assertEquals("ru", w.language)
        assertEquals("/m/whisper-small@onnx-int8/small-encoder.int8.onnx", w.encoder)
        assertEquals("", SherpaConfigs.offline(spec("whisper-small@onnx-int8"), null).modelConfig.whisper.language) // auto

        val wq = SherpaConfigs.offline(spec("whisper-small@qnn-sm8750"), libs).modelConfig.whisper
        assertTrue(wq.qnnConfig.contextBinary.endsWith("sherpa-onnx-qnn-SM8750-binary-whisper-small/decoder.bin"))
        assertEquals("", wq.encoder)
    }

    @Test
    fun streamingModels() {
        val n = SherpaConfigs.online(spec("nemotron-3.5-asr-streaming-0.6b@onnx-int8-560ms"), null)
        assertFalse(n.enableEndpoint)
        assertEquals("/m/nemotron-3.5-asr-streaming-0.6b@onnx-int8-560ms/encoder.int8.onnx", n.modelConfig.transducer.encoder)
        val nq = SherpaConfigs.online(spec("nemotron-3.5-asr-streaming-0.6b@qnn-sm8750-560ms"), libs)
        assertEquals("qnn", nq.modelConfig.provider)
        assertEquals("nemo_transducer", nq.modelConfig.modelType)
        val t = SherpaConfigs.online(spec("t-one@onnx"), null)
        assertEquals(8_000, t.featConfig.sampleRate)
        assertEquals("/m/t-one@onnx/model.onnx", t.modelConfig.toneCtc.model)
        assertFalse(t.enableEndpoint)
        val he = SherpaConfigs.online(spec("zipformer2-66m-streaming-he@onnx-int8"), null)
        assertEquals("cpu", he.modelConfig.provider)
        assertEquals("zipformer2", he.modelConfig.modelType)
        assertEquals("/m/zipformer2-66m-streaming-he@onnx-int8/encoder.int8.onnx", he.modelConfig.transducer.encoder)
        assertEquals("/m/zipformer2-66m-streaming-he@onnx-int8/tokens.txt", he.modelConfig.tokens)
        assertFalse(he.enableEndpoint)
    }

    @Test
    fun unsupportedUnitsAndRuntimes() {
        assertEquals("unit.gpu", assertThrows(SpecException::class.java) {
            SherpaConfigs.offline(spec("parakeet-tdt-0.6b-v3@onnx-int8", ComputeUnit.GPU), null)
        }.reason)
        assertEquals("unit.npu", assertThrows(SpecException::class.java) {
            SherpaConfigs.offline(spec("gigaam-v3-e2e-ctc@onnx-int8", ComputeUnit.NPU), libs)
        }.reason)
        assertEquals("runtime", assertThrows(SpecException::class.java) {
            SherpaConfigs.offline(spec("qwen3-asr-0.6b@litert-npu-sm8750"), libs)
        }.reason)
        assertTrue(SherpaConfigs.takesHint(RuntimeKeys.SHERPA_OFFLINE_WHISPER))
        assertFalse(SherpaConfigs.takesHint(RuntimeKeys.SHERPA_OFFLINE_TRANSDUCER))
        assertEquals(
            setOf("sherpa/offline-transducer", "sherpa/offline-ctc", "sherpa/offline-whisper", "sherpa/offline-qwen3-asr",
                "sherpa/online-transducer", "sherpa/online-t-one"),
            SherpaConfigs.SHERPA,
        )
    }

    @Test
    fun missingFilesAreReportedByRoleNeverByPath() {
        val dir = Files.createTempDirectory("asr").toFile()
        try {
            val base = spec("gigaam-v3-e2e-rnnt@onnx-int8")
            val s = base.copy(dir = dir.absolutePath)
            listOf("encoder.int8.onnx", "decoder.onnx", "tokens.txt").forEach { File(dir, it).writeText("x") }
            assertEquals("joiner", SherpaConfigs.missingRole(s))
            File(dir, "joiner.onnx").writeText("x")
            assertNull(SherpaConfigs.missingRole(s))
            val vad = spec("silero-vad-v5@onnx").copy(dir = dir.absolutePath)
            assertEquals(ModelKind.VAD, vad.kind)
            assertEquals("model", SherpaConfigs.missingRole(vad))
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun theSharedWhisperLanguageFollowsTheSessionOfEachDecode() {
        val applied = ArrayList<String>()
        val language = ConfigValue("") { applied += it }
        // two sessions attached at once (ru and en hints), their segments interleaved on the engine thread
        listOf("ru", "en", "en", "ru", "").forEach { language.use(it) }
        assertEquals(listOf("ru", "en", "ru", ""), applied)
    }
}
