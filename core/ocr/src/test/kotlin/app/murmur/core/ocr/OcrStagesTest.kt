package app.murmur.core.ocr

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/** Pre-processing, DB post-processing, CTC and the dictionary reader on synthetic inputs (no models). */
class OcrStagesTest {

    @Test
    fun detectorSizeRule() {
        // README section 2: cover 1080x2520 -> 416x960, inner 1968x2184 -> 864x960; never upscale; min 32.
        assertEquals(416 to 960, OcrPipelineSizes.detTargetSize(1080, 2520))
        assertEquals(864 to 960, OcrPipelineSizes.detTargetSize(1968, 2184))
        assertEquals(640 to 480, OcrPipelineSizes.detTargetSize(640, 480))
        assertEquals(32 to 96, OcrPipelineSizes.detTargetSize(10, 100))
        assertEquals(64 to 32, OcrPipelineSizes.detTargetSize(48, 47)) // 48/32 + 0.5 = 2.0 -> 64 (round half up)
        val native = OcrPipeline(OcrConfig(detLimit = 0), OcrPipelineSizesModels)
        assertEquals(1088 to 2528, native.detTargetSize(1080, 2520))
        assertEquals(1984 to 2176, native.detTargetSize(1968, 2184))
    }

    @Test
    fun recogniserWidthBuckets() {
        val p = OcrPipeline(OcrConfig(), OcrPipelineSizesModels)
        assertEquals(30 to 320, p.recWidth(30, 48))
        assertEquals(321 to 480, p.recWidth(321, 48))
        assertEquals(98 to 320, p.recWidth(100, 49)) // ceil(48*100/49) = ceil(97.96)
        assertEquals(3200 to 3200, p.recWidth(3200, 48))
        assertEquals(3200 to 3200, p.recWidth(5000, 48)) // squeezed
    }

    private fun argb(w: Int, h: Int, f: (Int, Int) -> Int) = IntArray(w * h) { f(it % w, it / w) }

    @Test
    fun bilinearIdentityAndConstant() {
        val src = argb(7, 5) { x, y -> (0xFF shl 24) or ((x * 30) shl 16) or ((y * 40) shl 8) or (x + y) }
        val out = Bilinear.resizeBgr(src, 7, 0, 0, 7, 5, 7, 5)
        for (i in 0 until 35) {
            assertEquals((src[i] and 0xFF).toFloat(), out[i], 0f)
            assertEquals(((src[i] shr 8) and 0xFF).toFloat(), out[35 + i], 0f)
            assertEquals(((src[i] shr 16) and 0xFF).toFloat(), out[70 + i], 0f)
        }
        val flat = Bilinear.resizeBgr(IntArray(64 * 40) { 0xFF336699.toInt() }, 64, 0, 0, 64, 40, 23, 17)
        assertTrue(flat.copyOfRange(0, 23 * 17).all { it == 0x99.toFloat() })
        assertTrue(flat.copyOfRange(2 * 23 * 17, 3 * 23 * 17).all { it == 0x33.toFloat() })
    }

    @Test
    fun bilinearHalfPixelCentres() {
        // 4 -> 2 columns: dst x samples src 0.5 and 2.5 -> means of (0,1) and (2,3); 2 -> 4 upsamples with edge clamp.
        val row = intArrayOf(0, 10, 20, 30).map { 0xFF000000.toInt() or it }.toIntArray()
        assertArrayEquals(floatArrayOf(5f, 25f), Bilinear.resizeBgr(row, 4, 0, 0, 4, 1, 2, 1).copyOfRange(0, 2), 0f)
        val two = intArrayOf(0, 100).map { 0xFF000000.toInt() or it }.toIntArray()
        assertArrayEquals(floatArrayOf(0f, 25f, 75f, 100f), Bilinear.resizeBgr(two, 2, 0, 0, 2, 1, 4, 1).copyOfRange(0, 4), 0f)
        // A crop reads only its region of a wider image.
        val img = argb(10, 3) { x, y -> if (x in 4..5 && y == 1) 0xFF0000C8.toInt() else 0xFF000000.toInt() }
        assertArrayEquals(floatArrayOf(200f, 200f), Bilinear.resizeBgr(img, 10, 4, 1, 2, 1, 2, 1).copyOfRange(0, 2), 0f)
    }

    @Test
    fun dbFindsUnclipsAndScalesBoxes() {
        val w = 40
        val h = 20
        val prob = FloatArray(w * h)
        // A 20x4 text blob at (10, 8) with probability 0.9, and a diagonal neighbour joined by 8-connectivity.
        for (y in 8 until 12) for (x in 10 until 30) prob[y * w + x] = 0.9f
        prob[12 * w + 30] = 0.9f
        // A 2-px tall blob: dropped (min side < 3).
        for (y in 2 until 4) for (x in 2 until 8) prob[y * w + x] = 0.9f
        val boxes = DbPostProcess.boxes(prob, w, h, 80, 40, OcrConfig())
        assertEquals(1, boxes.size)
        // Component bbox x 10..30, y 8..12 -> w 21, h 5; d = 21*5*1.4/(2*26) = 2.8269; scale 2.
        val d = 21 * 5 * 1.4 / (2.0 * 26)
        assertEquals(Rect(Math.floor((10 - d) * 2 + 0.5).toInt(), Math.floor((8 - d) * 2 + 0.5).toInt(),
            Math.floor((31 + d) * 2 + 0.5).toInt(), Math.floor((13 + d) * 2 + 0.5).toInt()), boxes[0].rect)
        assertEquals((80 * 0.9 + 0.9) / (21 * 5), boxes[0].score, 1e-6)
    }

    @Test
    fun dbThresholdIsStrictInFloat32AndScoreGates() {
        val w = 10
        val h = 10
        val atThreshold = FloatArray(w * h) { 0.2f }
        assertTrue(DbPostProcess.components(atThreshold, w, h, 0.2f).isEmpty())
        // Mean over the bbox below 0.4: a hollow 8x8 ring of 0.5 (28 of 64 pixels) has score 0.22 -> dropped.
        val ring = FloatArray(w * h)
        for (y in 1 until 9) for (x in 1 until 9) if (x == 1 || x == 8 || y == 1 || y == 8) ring[y * w + x] = 0.5f
        assertTrue(DbPostProcess.boxes(ring, w, h, w, h, OcrConfig()).isEmpty())
        assertEquals(1, DbPostProcess.components(ring, w, h, 0.2f).size)
    }

    @Test
    fun ctcGreedyDecoding() {
        val chars = listOf("", "a", "b", " ")
        // steps: a a blank a b b space -> "aab " ; ties take the first index.
        val p = floatArrayOf(
            0.1f, 0.8f, 0.05f, 0.05f,
            0.1f, 0.7f, 0.1f, 0.1f,
            0.9f, 0.05f, 0.05f, 0f,
            0.2f, 0.6f, 0.1f, 0.1f,
            0.1f, 0.1f, 0.4f, 0.4f, // tie b/space -> b
            0.1f, 0.1f, 0.7f, 0.1f,
            0.1f, 0.1f, 0.1f, 0.7f,
        )
        val (text, conf) = Ctc.decode(p, 0, 7, 4, chars)
        assertEquals("aab ", text)
        assertEquals((0.8f.toDouble() + 0.6f.toDouble() + 0.4f.toDouble() + 0.7f.toDouble()) / 4, conf, 1e-12)
        assertEquals("" to 0.0, Ctc.decode(FloatArray(8) { if (it % 4 == 0) 1f else 0f }, 0, 2, 4, chars))
    }

    @Test
    fun dictionaryReaderHandlesYamlQuoting() {
        val yml = """
            |Global:
            |  model_name: x
            |  character_dict:
            |  - nope
            |PostProcess:
            |  name: CTCLabelDecode
            |  character_dict:
            |  - '!'
            |  - ''''
            |  - "Ж\\"
            |  - $
            |  - \
            |  - 'a b'
            |Other:
            |  - z
            """.trimMargin()
        assertEquals(listOf("!", "'", "Ж\\", "$", "\\", "a b"), OcrDictionary.fromInferenceYml(yml))
    }

    /** ml/ocr README: the dictionary read from inference.yml equals golden/rec_keys.txt byte for byte. */
    @Test
    fun dictionaryFromInferenceYmlIsByteIdenticalToRecKeys() {
        val keys = OcrDictionary.fromInferenceYml(TestModels.recYml.readText())
        val recKeys = File(Golden.dir, "rec_keys.txt").readBytes()
        assertEquals(850, keys.size)
        assertArrayEquals(recKeys, (keys.joinToString("\n") + "\n").toByteArray(Charsets.UTF_8))
    }

    @Test
    fun dictionaryReaderHandlesTrilingualHebrewEntries() {
        val yml = """
            |Global:
            |  model_name: ru_en_he_PP-OCRv5_mobile_rec
            |PostProcess:
            |  name: CTCLabelDecode
            |  character_dict:
            |  - '!'
            |  - ''''
            |  - 'א'
            |  - 'ת'
            |  - '׳'
            |  - '״'
            |  - '₪'
            |  - '־'
            """.trimMargin()
        assertEquals(listOf("!", "'", "א", "ת", "׳", "״", "₪", "־"), OcrDictionary.fromInferenceYml(yml))
    }
}

/** A model stub for the size rules (no inference). */
internal object OcrPipelineSizesModels : OcrModels {
    override fun detect(input: FloatArray, h: Int, w: Int) = error("no models")
    override fun recognize(input: FloatArray, n: Int, bucketWidth: Int) = error("no models")
    override val classes = 3
    override val dictionary = listOf("a")
}
