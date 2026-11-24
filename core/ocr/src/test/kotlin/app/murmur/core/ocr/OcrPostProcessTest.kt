package app.murmur.core.ocr

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit cases of the post-processing port. Expected values were produced by ml/ocr/postprocess.py (Python 3.13,
 * `repair_homoglyphs(text, default)`) when the port was written; the golden suite covers the rest end to end.
 */
class OcrPostProcessTest {
    private val cyr = Script.CYR
    private val lat = Script.LAT

    @Test
    fun homoglyphRepairMatchesReference() {
        val cases = listOf(
            Triple("Bce", null, "Bce"), Triple("Bce", cyr, "Все"), Triple("c", cyr, "с"),
            Triple("Kubernetes по Helm", null, "Kubernetes по Helm"), Triple("по", cyr, "по"),
            Triple("НАРгоху", null, "HAProxy"), Triple("А100", null, "A100"), Triple("promрt", null, "prompt"),
            Triple("RNN-Т", null, "RNN-T"), Triple("llama.срр", null, "llama.cpp"), Triple("SIM-кaрта", null, "SIM-карта"),
            Triple("Сапагу", cyr, "Сапагу"), Triple("НеМо", null, "HeMo"), Triple("х86", null, "x86"), Triple("С#", null, "C#"),
            Triple("Привет, wоrld", null, "Привет, world"), Triple("Mama", cyr, "Мата"), Triple("Mama", null, "Mama"),
            Triple("В", cyr, "В"), Triple("Вce вышли", null, "Bce вышли"), Triple("ресурс", lat, "ресурс"),
            Triple("pecypc", cyr, "ресурс"), Triple("Тhe", null, "The"), Triple("Ѕtart", null, "Start"),
            Triple("іnput", null, "input"), Triple("Мы  используем  Grafana", null, "Мы  используем  Grafana"),
            Triple("a/b_c.d e-f", null, "a/b_c.d e-f"), Triple("Кubernetes", null, "Kubernetes"), Triple("тема", null, "тема"),
            Triple("тeмa", cyr, "тема"), Triple("kube-prometheus", null, "kube-prometheus"),
            Triple("Привет-Hello", null, "Привет-Hello"),
        )
        for ((text, dominant, want) in cases) assertEquals("$text / $dominant", want, OcrPostProcess.repairHomoglyphs(text, dominant))
        assertEquals("HAProxy", OcrPostProcess.repairHomoglyphs("НАРгоху"))
    }

    @Test
    fun screenScriptNeedsTwiceTheEvidence() {
        assertEquals(cyr, OcrPostProcess.screenScript(listOf("Привет мир", "ok"), 2.0))
        assertEquals(lat, OcrPostProcess.screenScript(listOf("Hello world", "ок"), 2.0))
        assertEquals(cyr, OcrPostProcess.screenScript(listOf("abc", "абв"), 2.0))
        assertNull(OcrPostProcess.screenScript(listOf("Hello", "Привет"), 2.0))
        assertNull(OcrPostProcess.screenScript(emptyList(), 2.0))
    }

    @Test
    fun textNormalisationFollowsPythonWhitespace() {
        assertEquals("a b c", OcrPostProcess.normText(" a  b c "))
        assertEquals("ё", OcrPostProcess.normText("ё")) // NFC composes ё
        assertEquals("x​y", OcrPostProcess.normText("x​y")) // ZWSP is not whitespace in Python
        assertEquals("z", OcrPostProcess.normText("　z\t"))
        assertEquals("", OcrPostProcess.normText(" \n "))
    }

    @Test
    fun filterDropsLowConfidenceAndShortLines() {
        val r = Rect(0, 0, 100, 20)
        val out = OcrPostProcess.process(listOf(
            RecLine(r, "ok", 0.49), RecLine(Rect(0, 30, 100, 50), "a ", 0.9), RecLine(Rect(0, 60, 100, 80), "ab", 0.5),
        ), 1080, 2520)
        assertEquals(listOf("ab"), out.map { it.text })
    }

    @Test
    fun roundingIsPythonRound() {
        assertEquals(0.962, OcrPostProcess.roundHalfEven(0.96249, 3), 0.0)
        assertEquals(0.062, OcrPostProcess.roundHalfEven(0.0625, 3), 0.0) // exact tie -> even
        assertEquals(0.064, OcrPostProcess.roundHalfEven(0.0635, 3), 0.0) // binary 0.0635 lies above the tie
        assertEquals(1.0, OcrPostProcess.roundHalfEven(0.99951, 3), 0.0)
        assertEquals(0.619, OcrPostProcess.roundHalfEven(0.6185, 3), 0.0) // binary 0.6185 lies above the tie
    }

    @Test
    fun pythonSumIsCompensated() {
        // Python 3.12+: sum([0.1] * 10) == 1.0 (Neumaier), while naive left-to-right addition gives 0.9999999999999999.
        val v = DoubleArray(10) { 0.1 }
        assertEquals(1.0, Ctc.pySum(v), 0.0)
        assertEquals(0.9999999999999999, v.fold(0.0) { a, b -> a + b }, 0.0)
        assertEquals(0.0, Ctc.pySum(DoubleArray(0)), 0.0)
    }

    @Test
    fun geometricDedupKeepsTheMoreConfidentLine() {
        val lines = listOf(
            OcrPostProcess.L("Hello", 0.8, Rect(10, 10, 110, 40)),
            OcrPostProcess.L("Hel1o", 0.9, Rect(12, 11, 112, 41)),   // IoU > 0.5 with the first
            OcrPostProcess.L("lo", 0.99, Rect(80, 12, 105, 38)),     // inside the first, but visited first: kept
            OcrPostProcess.L("World", 0.7, Rect(10, 50, 110, 80)),
        )
        val kept = OcrPostProcess.dedupGeometric(lines, 0.5)
        assertEquals(listOf("Hel1o", "lo", "World"), kept.map { it.text })
    }

    @Test
    fun textualDedupIsCaseAndSpaceFolded() {
        val l = { t: String, y: Int -> OcrPostProcess.L(t, 0.9, Rect(0, y, 50, y + 10)) }
        val out = OcrPostProcess.dedupText(listOf(l("Привет  Мир", 0), l("привет мир", 20), l("STRASSE", 40), l("straße", 60)))
        assertEquals(listOf("Привет  Мир", "STRASSE"), out.map { it.text })
    }

    @Test
    fun rowsGroupByVerticalOverlapLeftToRight() {
        val l = { t: String, b: Rect -> OcrPostProcess.L(t, 0.9, b) }
        val lines = listOf(
            l("time", Rect(900, 102, 1000, 130)), l("name", Rect(100, 100, 400, 140)),
            l("second", Rect(100, 160, 400, 200)), l("over", Rect(350, 105, 600, 135)),
        )
        // "over" intersects "name" so it cannot join that row: it starts its own row.
        assertEquals(listOf("name", "time", "over", "second"), OcrPostProcess.rows(lines, 0.5).map { it.text })
    }

    @Test
    fun twoPaneLayoutReadsLeftPaneThenRightPane() {
        val config = OcrConfig()
        val l = { t: String, b: Rect -> OcrPostProcess.L(t, 0.9, b) }
        val left = (0 until 8).map { l("L$it", Rect(40, 300 + it * 200, 600, 340 + it * 200)) }
        val right = (0 until 8).map { l("R$it", Rect(1000, 310 + it * 200, 1900, 350 + it * 200)) }
        val title = l("Title", Rect(40, 100, 1900, 160))
        val footer = l("Footer", Rect(40, 2000, 1900, 2050))
        val all = right + left + footer + title
        // Reference (postprocess.find_gutter): x = (600 + 1000) // 2, band = [310, 1740].
        assertEquals(Triple(800, 310, 1740), OcrPostProcess.findGutter(all, 1968, 2184, config))
        assertEquals(listOf("Title") + left.map { it.text } + right.map { it.text } + "Footer",
            OcrPostProcess.readingOrder(all, 1968, 2184, config).map { it.text })
        // Cover display (aspect < 0.75): no pane detection, plain rows.
        assertNull(OcrPostProcess.findGutter(all, 1080, 2520, config))
    }

    @Test
    fun exclusionNeedsHalfOfTheBox() {
        val box = Rect(0, 0, 100, 10)
        assertTrue(OcrPostProcess.excluded(box, listOf(Rect(0, 0, 50, 10)), 0.5))
        assertFalse(OcrPostProcess.excluded(box, listOf(Rect(0, 0, 49, 10)), 0.5))
        assertTrue(OcrPostProcess.excluded(Rect(5, 5, 5, 9), emptyList(), 0.5)) // zero area
        assertFalse(OcrPostProcess.excluded(box, emptyList(), 0.5))
    }

    @Test
    fun visualToLogicalMatchesPythonReference() {
        val cases = listOf(
            "םלוע םולש" to "שלום עולם",
            "!םלוע םולש" to "שלום עולם!",
            "(תירבע) םולש" to "שלום (עברית)",
            "ל״הצ" to "צה״ל",
            "םולש ר״ד" to "ד״ר שלום",
            "GitHub-ל push יתישע" to "עשיתי push ל-GitHub",
            ".GitHub-ל push יתישע" to "עשיתי push ל-GitHub.",
            "kube-prometheus רובע Grafana קניל" to "לינק Grafana עבור kube-prometheus",
            "Hello World םולש" to "שלום Hello World",
            "15:30-ב השגפ" to "פגשה ב-15:30",
            "50₪ ריחמ" to "מחיר 50₪",
            "₪50 ריחמ" to "מחיר ₪50",
            "₪ 50 ריחמ" to "מחיר 50 ₪",
            "100% חוטב" to "בטוח 100%",
            "054-123-4567 רפסמ" to "מספר 054-123-4567",
            "באהטיגב PR #4182 יתחתפ" to "פתחתי PR #4182 בגיטהאב",
            "Hello םולש world" to "Hello שלום world",
            "Hello םלוע םולש world" to "Hello שלום עולם world",
            "Привет םולש мир" to "Привет שלום мир",
            "Привет, Grafana! 15:30" to "Привет, Grafana! 15:30",
            "PR #4182 (main) — 100%" to "PR #4182 (main) — 100%",
            "" to "",
        )
        for ((visual, logical) in cases) {
            assertEquals(visual, logical, OcrPostProcess.visualToLogical(visual))
        }
    }

    @Test
    fun rowsSortRtlDominantRightToLeftAndEndToEndProcessConvertsVisualHebrew() {
        val l = { t: String, b: Rect -> OcrPostProcess.L(t, 0.9, b) }
        val hebRow = listOf(
            l("12:30", Rect(50, 104, 150, 130)),
            l("שלום עולם", Rect(500, 100, 980, 140)),
            l("עוד שורה", Rect(500, 160, 980, 195)),
        )
        assertEquals(listOf("שלום עולם", "12:30", "עוד שורה"), OcrPostProcess.rows(hebRow, 0.5).map { it.text })
        val twoHeb = listOf(l("שמאל", Rect(100, 100, 400, 140)), l("ימין", Rect(600, 102, 950, 142)))
        assertEquals(listOf("ימין", "שמאל"), OcrPostProcess.rows(twoHeb, 0.5).map { it.text })
        val ltrRow = listOf(l("12:30", Rect(900, 104, 1000, 130)), l("Привет мир", Rect(50, 100, 600, 140)))
        assertEquals(listOf("Привет мир", "12:30"), OcrPostProcess.rows(ltrRow, 0.5).map { it.text })

        // End-to-end process() on visual CTC Hebrew lines
        val out = OcrPostProcess.process(
            listOf(
                RecLine(Rect(50, 104, 150, 130), "12:30", 0.95),
                RecLine(Rect(500, 100, 980, 140), "GitHub-ל push יתישע", 0.96),
                RecLine(Rect(500, 160, 980, 195), "םלוע םולש", 0.97),
            ),
            1080,
            2520,
        )
        assertEquals(listOf("עשיתי push ל-GitHub", "12:30", "שלום עולם"), out.map { it.text })
    }
}
