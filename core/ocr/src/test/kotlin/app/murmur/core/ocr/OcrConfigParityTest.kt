package app.murmur.core.ocr

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File

/** Drift guard: OcrConfig defaults and the homoglyph tables == ml/ocr/config.py (the porting spec, D11). */
class OcrConfigParityTest {
    private val root = File(System.getProperty("murmur.repoRoot") ?: ".")
    private val py = File(root, "ml/ocr/config.py").readText()

    /** `name: type = value  # comment` lines of the OcrConfig dataclass. */
    private val fields: Map<String, String> by lazy {
        val body = py.substringAfter("class OcrConfig:").substringBefore("\n    def ")
        Regex("""^    (\w+):\s*\w+\s*=\s*(.+?)\s*(#.*)?$""", RegexOption.MULTILINE).findAll(body)
            .associate { it.groupValues[1] to it.groupValues[2] }
    }

    private fun num(s: String) = s.trim().toDouble()
    private fun tuple(s: String) = s.trim().removePrefix("(").removeSuffix(")").split(',').filter { it.isNotBlank() }.map(::num)

    /** Kotlin value (as numbers) per Python field; model paths and `extra` are checked separately / not used. */
    private val c = OcrConfig()
    private val numeric: Map<String, List<Double>> = mapOf(
        "det_limit" to listOf(c.detLimit.toDouble()),
        "det_multiple" to listOf(c.detMultiple.toDouble()),
        "det_tile" to listOf(c.detTile.toDouble()),
        "det_tile_overlap" to listOf(c.detTileOverlap.toDouble()),
        "det_mean" to c.detMean,
        "det_std" to c.detStd,
        "db_thresh" to listOf(c.dbThresh),
        "db_box_thresh" to listOf(c.dbBoxThresh),
        "db_unclip_ratio" to listOf(c.dbUnclipRatio),
        "db_min_side" to listOf(c.dbMinSide),
        "db_min_side_unclipped" to listOf(c.dbMinSideUnclipped),
        "rec_height" to listOf(c.recHeight.toDouble()),
        "rec_buckets" to c.recBuckets.map { it.toDouble() },
        "rec_batch" to listOf(c.recBatch.toDouble()),
        "min_conf" to listOf(c.minConf),
        "min_chars" to listOf(c.minChars.toDouble()),
        "exclude_min_overlap" to listOf(c.excludeMinOverlap),
        "dedup_iou" to listOf(c.dedupIou),
        "row_overlap" to listOf(c.rowOverlap),
        "pane_min_aspect" to listOf(c.paneMinAspect),
        "pane_gutter_range" to listOf(c.paneGutterRange.first, c.paneGutterRange.second),
        "pane_min_lines" to listOf(c.paneMinLines.toDouble()),
        "pane_min_span" to listOf(c.paneMinSpan),
        "screen_script_ratio" to listOf(c.screenScriptRatio),
        "conf_digits" to listOf(c.confDigits.toDouble()),
    )
    private val models = setOf("det_model", "rec_model", "rec_dict")

    @Test
    fun everyPythonFieldIsMapped() {
        // A new field in config.py fails here until it is ported (or listed as not applicable).
        assertEquals(fields.keys.sorted(), (numeric.keys + models + "threads" + "extra").sorted())
    }

    @Test
    fun numericDefaultsMatchPythonReference() {
        for ((name, kotlin) in numeric) {
            val v = fields.getValue(name)
            val pyValues = if (v.startsWith("(")) tuple(v) else listOf(num(v))
            assertEquals(name, pyValues, kotlin)
        }
    }

    @Test
    fun threadsAndVersionMatchPythonReference() {
        val threads = Regex("""MURMUR_OCR_THREADS",\s*"(\d+)"""").find(fields.getValue("threads"))!!.groupValues[1]
        assertEquals(threads.toInt(), c.threads)
        assertEquals("\"${c.pipelineVersion}\"", Regex("""PIPELINE_VERSION = (".+?")""").find(py)!!.groupValues[1])
    }

    private fun pyString(s: String) = s.trim().removeSurrounding("\"")

    /** `NAME = dict(zip("abc", "xyz"))`, optionally `| {"k": "v", ...}`, as the effective code point map. */
    private fun pyZipDict(name: String): Map<Int, Int> {
        val line = Regex("""^$name = (.+)$""", RegexOption.MULTILINE).find(py)!!.groupValues[1]
        val z = Regex("""dict\(zip\(("[^"]*"),\s*("[^"]*")\)\)""").find(line)!!
        val k = pyString(z.groupValues[1]).codePoints().toArray()
        val v = pyString(z.groupValues[2]).codePoints().toArray()
        assertEquals(name, k.size, v.size)
        val m = LinkedHashMap<Int, Int>()
        for (i in k.indices) m[k[i]] = v[i]
        line.substringAfter("|", "").let { over ->
            Regex(""""(.)":\s*"(.)"""").findAll(over).forEach { m[it.groupValues[1].codePointAt(0)] = it.groupValues[2].codePointAt(0) }
        }
        return m
    }

    @Test
    fun homoglyphTablesMatchPythonReference() {
        assertEquals(pyZipDict("LAT2CYR_STRONG"), Homoglyphs.lat2cyrStrong)
        assertEquals(pyZipDict("LAT2CYR_WEAK"), Homoglyphs.lat2cyrWeak)
        assertEquals(pyZipDict("CYR2LAT_WEAK"), Homoglyphs.cyr2latWeak)
        assertEquals(pyZipDict("NON_RU_TO_LAT"), Homoglyphs.nonRuToLat)
        assertEquals(pyZipDict("LAT2CYR_STRONG").entries.associate { (a, b) -> b to a }, Homoglyphs.cyr2latStrong)
        assertEquals("CYR2LAT_STRONG = {v: k for k, v in LAT2CYR_STRONG.items()}",
            Regex("""^CYR2LAT_STRONG = .+$""", RegexOption.MULTILINE).find(py)!!.value)
        assertEquals(num(Regex("""^WEAK_WEIGHT = ([\d.]+)""", RegexOption.MULTILINE).find(py)!!.groupValues[1]), Homoglyphs.WEAK_WEIGHT, 0.0)
    }

    /**
     * The models the app downloads (bundled catalog) are the ones ml/ocr was measured with: same Hugging Face repo and
     * file for det/rec/dictionary, and the catalog's ocr.pipelineVersion == PIPELINE_VERSION.
     */
    @Test
    fun catalogShipsTheReferenceModels() {
        val catalog = Json.parseToJsonElement(File(root, "core/catalog/src/main/resources/catalog/catalog.json").readText())
        val entries = collect(catalog).filter { it["kind"]?.jsonPrimitive?.content in setOf("ocr-det", "ocr-rec") }
        assertEquals(3, entries.size)
        fun files(id: String) = entries.single { it["id"]!!.jsonPrimitive.content == id }.let { e ->
            assertEquals(c.pipelineVersion, e.getValue("ocr").jsonObject.getValue("pipelineVersion").jsonPrimitive.content)
            // the CPU artifact (ort/ppocr-*) ships the reference files; the NPU artifact (ort-qnn/*, WP-21) holds their contexts
            e.getValue("artifacts").jsonArray.map { it.jsonObject }.single { it.getValue("runtime").jsonPrimitive.content.startsWith("ort/") }
                .getValue("files").jsonArray.map { f ->
                val o = f.jsonObject
                o.getValue("repo").jsonPrimitive.content.substringAfter('/') + "/" + o.getValue("path").jsonPrimitive.content
            }.toSet()
        }
        assertEquals(setOf(pyString(fields.getValue("det_model"))), files("pp-ocrv6-tiny-det"))
        assertEquals(setOf(pyString(fields.getValue("rec_model")), pyString(fields.getValue("rec_dict"))), files("pp-ocrv5-cyrillic-mobile-rec"))
        assertEquals(
            setOf("murmur-cleanup/ocr/pp-ocrv5-ru-en-he-mobile-rec/inference.onnx", "murmur-cleanup/ocr/pp-ocrv5-ru-en-he-mobile-rec/inference.yml"),
            files("pp-ocrv5-ru-en-he-mobile-rec"),
        )
    }

    private fun collect(e: JsonElement): List<JsonObject> = when (e) {
        is JsonObject -> listOf(e) + e.values.flatMap(::collect)
        is JsonArray -> e.flatMap(::collect)
        else -> emptyList()
    }
}
