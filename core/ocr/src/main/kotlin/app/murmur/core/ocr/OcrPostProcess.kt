package app.murmur.core.ocr

import java.math.BigDecimal
import java.math.RoundingMode
import java.text.Normalizer
import java.util.Locale

/**
 * ml/ocr/postprocess.py, line by line (README section 6). Order: filter -> homoglyph repair -> geometric
 * de-duplication -> reading order -> textual de-duplication -> conf rounding. Python string semantics are kept where
 * they matter: code points (not UTF-16 units), `str.isspace` / `str.split()` whitespace, `str.casefold`, `round()`.
 */
object OcrPostProcess {

    /** Feed `stages.rec` of a golden file, get `ocr.lines` back (the strict half of the parity test). */
    fun process(lines: List<RecLine>, width: Int, height: Int, config: OcrConfig = OcrConfig()): List<OcrLine> {
        val kept = lines.map { L(normText(it.text), it.conf, it.rect) }.filter { keep(it, config) }
        val dominant = screenScript(kept.map { it.text }, config.screenScriptRatio)
        val repaired = kept.map { it.copy(text = visualToLogical(repairHomoglyphs(it.text, dominant))) }
        val deduped = dedupGeometric(repaired, config.dedupIou)
        val ordered = readingOrder(deduped, width, height, config)
        return dedupText(ordered).map {
            OcrLine(it.text, roundHalfEven(it.conf, config.confDigits), listOf(it.box.x0, it.box.y0, it.box.x1, it.box.y1))
        }
    }

    /** Homoglyph repair of one line with no screen-level script ([repairHomoglyphs] with `dominant = null`). */
    fun repairHomoglyphs(text: String): String = repairHomoglyphs(text, null)

    // ------------------------------------------------------------------------------------------- geometry

    internal data class L(val text: String, val conf: Double, val box: Rect)

    internal fun area(b: Rect): Long = maxOf(0, b.x1 - b.x0).toLong() * maxOf(0, b.y1 - b.y0)

    internal fun inter(a: Rect, b: Rect): Long =
        area(Rect(maxOf(a.x0, b.x0), maxOf(a.y0, b.y0), minOf(a.x1, b.x1), minOf(a.y1, b.y1)))

    /** README section 4: dropped when `intersection >= minOverlap * area(box)` for any rectangle (zero-area boxes too). */
    fun excluded(box: Rect, rects: List<Rect>, minOverlap: Double): Boolean {
        val a = area(box)
        if (a == 0L) return true
        return rects.any { inter(box, it) >= minOverlap * a }
    }

    internal fun iou(a: Rect, b: Rect): Double {
        val i = inter(a, b)
        val u = area(a) + area(b) - i
        return if (u != 0L) i.toDouble() / u else 0.0
    }

    // ------------------------------------------------------------------------------------------- filters

    /** Python `str.isspace()` for one code point (Unicode White_Space minus U+180E, plus the ASCII separators). */
    internal fun isPySpace(cp: Int): Boolean = when (cp) {
        in 0x09..0x0D, in 0x1C..0x20, 0x85, 0xA0, 0x1680, in 0x2000..0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 -> true
        else -> false
    }

    /** Python `" ".join(s.split())`. */
    internal fun collapseSpaces(s: String): String {
        val sb = StringBuilder(s.length)
        var pending = false
        var i = 0
        while (i < s.length) {
            val cp = s.codePointAt(i)
            i += Character.charCount(cp)
            if (isPySpace(cp)) {
                pending = sb.isNotEmpty()
            } else {
                if (pending) sb.append(' ')
                pending = false
                sb.appendCodePoint(cp)
            }
        }
        return sb.toString()
    }

    /** NFC, collapse whitespace runs to one space, strip. */
    fun normText(s: String): String = collapseSpaces(Normalizer.normalize(s, Normalizer.Form.NFC))

    internal fun keep(line: L, config: OcrConfig): Boolean {
        if (line.conf < config.minConf) return false
        return line.text.codePoints().filter { !isPySpace(it) }.count() >= config.minChars
    }

    // ------------------------------------------------------------------------------------------- homoglyph repair

    private val strongL2C = Homoglyphs.lat2cyrStrong
    private val weakL2C = Homoglyphs.lat2cyrWeak
    private val strongC2L = Homoglyphs.cyr2latStrong
    private val weakC2L = Homoglyphs.cyr2latWeak
    private val nonRu = Homoglyphs.nonRuToLat

    internal fun isCyr(c: Int): Boolean = c in 0x0400..0x04FF

    internal fun isLat(c: Int): Boolean = c in 'a'.code..'z'.code || c in 'A'.code..'Z'.code ||
        (c in 0x00C0..0x024F && Character.isLetter(c))

    internal fun isHeb(c: Int): Boolean = c in 0x0590..0x05FF

    internal fun isHebLetter(c: Int): Boolean = c in 0x0590..0x05FF && Character.isLetter(c)

    /** (cyrillic, latin) evidence: 1 per letter with no twin, WEAK_WEIGHT per weak twin, 0 per strong twin. */
    internal fun scriptScore(tok: IntArray): DoubleArray {
        var cy = 0.0
        var la = 0.0
        for (c in tok) {
            if (c in nonRu) {
                la += 1.0
            } else if (isCyr(c)) {
                cy += if (c in strongC2L) 0.0 else if (c in weakC2L) Homoglyphs.WEAK_WEIGHT else 1.0
            } else if (isLat(c)) {
                la += if (c in strongL2C) 0.0 else if (c in weakL2C) Homoglyphs.WEAK_WEIGHT else 1.0
            }
        }
        return doubleArrayOf(cy, la)
    }

    private fun toCyr(tok: IntArray): IntArray = IntArray(tok.size) { strongL2C[tok[it]] ?: weakL2C[tok[it]] ?: tok[it] }

    private fun toLat(tok: IntArray): IntArray =
        IntArray(tok.size) { strongC2L[tok[it]] ?: weakC2L[tok[it]] ?: nonRu[tok[it]] ?: tok[it] }

    private fun hasUnambiguous(tok: IntArray): Boolean = tok.any { c ->
        c in nonRu || (isCyr(c) && c !in strongC2L && c !in weakC2L) || (isLat(c) && c !in strongL2C && c !in weakL2C)
    }

    private fun hasBoth(tok: IntArray): Boolean = tok.any { isCyr(it) } && tok.any { isLat(it) }

    /** CYR / LAT when one script's evidence over all lines is >= [ratio] times the other's, else null. */
    fun screenScript(texts: List<String>, ratio: Double): Script? {
        var cy = 0.0
        var la = 0.0
        for (t in texts) {
            val s = scriptScore(t.codePoints().toArray())
            cy += s[0]
            la += s[1]
        }
        if (cy >= ratio * maxOf(la, 1e-9) && cy > 0) return Script.CYR
        if (la >= ratio * maxOf(cy, 1e-9) && la > 0) return Script.LAT
        return null
    }

    private val CODE = Regex("""[0-9#+@$=<>\\]""")
    private val CAMEL = Regex("""[a-zа-яё][A-ZА-ЯЁ]|[A-ZА-ЯЁ]{2}[a-zа-яё]""")

    private fun str(cps: IntArray): String = String(cps, 0, cps.size)

    private fun decisive(s: DoubleArray): Script? = if (s[0] == s[1]) null else if (s[0] > s[1]) Script.CYR else Script.LAT

    /** A token made only of twin letters (e.g. "Bce", "по", "НАРгоху", "А100", "promрt"): README section 6.3 (a)-(e). */
    private fun twinOnly(i: Int, toks: List<IntArray>, scores: List<DoubleArray>, default: Script?, seps: List<Char>): IntArray {
        val t = toks[i]
        val letters = t.filter { isCyr(it) || isLat(it) }.toIntArray()
        if (letters.isEmpty()) return t
        if (CODE.containsMatchIn(str(t))) return toLat(t)
        val ncy = letters.count { isCyr(it) }
        if (ncy > 0 && ncy < letters.size) {
            if (2 * ncy > letters.size) return toCyr(t)
            if (2 * ncy < letters.size) return toLat(t)
        }
        if (CAMEL.containsMatchIn(str(letters))) return toLat(t)
        val glued = HashSet<Script>()
        for ((j, sepI) in listOf(i - 1 to i - 1, i + 1 to i)) {
            if (j in toks.indices && sepI in seps.indices && seps[sepI] != ' ') decisive(scores[j])?.let { glued += it }
        }
        if (glued.size == 1) return if (Script.CYR in glued) toCyr(t) else toLat(t)
        val votes = HashSet<Script>()
        for (j in maxOf(0, i - 2) until minOf(toks.size, i + 3)) {
            if (j != i) decisive(scores[j])?.let { votes += it }
        }
        if (votes.isEmpty() && default != null && letters.any { Character.isLowerCase(it) }) votes += default
        // Never flip to Latin on context alone: "по", "с", "и" between English terms are Russian.
        return if (votes == setOf(Script.CYR)) toCyr(t) else t
    }

    private fun isSep(c: Char) = c == ' ' || c == '-' || c == '/' || c == '_' || c == '.'

    /**
     * Makes every token single-script where the evidence says so (README section 6.3). Tokens are separated by
     * space, "-", "/", "_", "." (kept). [dominant] = the screen's script ([screenScript]) for rule (d).
     */
    fun repairHomoglyphs(text: String, dominant: Script?): String {
        // re.split(r"([ \-/_.])", text): tokens and the separators between them (an empty token between two separators).
        val toks = ArrayList<IntArray>()
        val seps = ArrayList<Char>()
        var start = 0
        for (k in text.indices) {
            if (isSep(text[k])) {
                toks += text.substring(start, k).codePoints().toArray()
                seps += text[k]
                start = k + 1
            }
        }
        toks += text.substring(start).codePoints().toArray()
        val scores = toks.map { scriptScore(it) }
        val sb = StringBuilder(text.length)
        for (i in toks.indices) {
            val t = toks[i]
            val cy = scores[i][0]
            val la = scores[i][1]
            val r = when {
                !hasUnambiguous(t) -> twinOnly(i, toks, scores, dominant, seps)
                (!hasBoth(t) && t.none { it in nonRu }) || cy == la -> t
                cy > la -> toCyr(t)
                else -> toLat(t)
            }
            for (c in r) sb.appendCodePoint(c)
            if (i < seps.size) sb.append(seps[i])
        }
        return sb.toString()
    }

    // ------------------------------------------------------------------------------------------- BiDi visual-to-logical

    private fun mirrorBracket(cp: Int): Int = when (cp) {
        '('.code -> ')'.code
        ')'.code -> '('.code
        '['.code -> ']'.code
        ']'.code -> '['.code
        '{'.code -> '}'.code
        '}'.code -> '{'.code
        '<'.code -> '>'.code
        '>'.code -> '<'.code
        '«'.code -> '»'.code
        '»'.code -> '«'.code
        else -> cp
    }

    private fun isNumEt(cp: Int): Boolean =
        cp == '%'.code || cp == '₪'.code || cp == '$'.code || cp == '€'.code || cp == '£'.code || cp == '+'.code

    private fun reverseRtlRun(cps: IntArray, from: Int, to: Int): String {
        val clusters = ArrayList<String>(to - from)
        for (i in from until to) {
            val cp = cps[i]
            if (clusters.isNotEmpty() && Character.getType(cp) == Character.NON_SPACING_MARK.toInt()) {
                clusters[clusters.lastIndex] = clusters.last() + String(Character.toChars(cp))
            } else {
                clusters.add(String(Character.toChars(mirrorBracket(cp))))
            }
        }
        val sb = StringBuilder()
        for (i in clusters.lastIndex downTo 0) sb.append(clusters[i])
        return sb.toString()
    }

    /**
     * Convert visual left-to-right CTC output to logical Unicode order when a line contains Hebrew (U+0590..U+05FF).
     * 100% no-op on lines without Hebrew characters. Matches `ml/ocr/postprocess.py:visual_to_logical` identically.
     */
    fun visualToLogical(text: String): String {
        val cps = text.codePoints().toArray()
        if (cps.none { isHeb(it) }) return text
        val n = cps.size
        val cls = CharArray(n) { i ->
            val c = cps[i]
            when {
                isHeb(c) -> 'R'
                isLat(c) || isCyr(c) || Character.isLetter(c) -> 'L'
                c in '0'.code..'9'.code -> 'D'
                else -> 'N'
            }
        }
        var base = 'R'
        for (cl in cls) if (cl == 'R' || cl == 'L') base = cl
        val dirs = CharArray(n) { i -> if (cls[i] == 'L' || cls[i] == 'D') 'L' else cls[i] }
        for (i in 1 until n) {
            if (isNumEt(cps[i]) && (cls[i - 1] == 'D' || (isNumEt(cps[i - 1]) && dirs[i - 1] == 'L'))) dirs[i] = 'L'
        }
        for (i in n - 2 downTo 0) {
            if (isNumEt(cps[i]) && (cls[i + 1] == 'D' || (isNumEt(cps[i + 1]) && dirs[i + 1] == 'L'))) dirs[i] = 'L'
        }
        var i = 0
        while (i < n) {
            if (dirs[i] != 'N') { i++; continue }
            var j = i
            while (j < n && dirs[j] == 'N') j++
            val left = if (i > 0) dirs[i - 1] else base
            val right = if (j < n) dirs[j] else base
            val fill = if (left == right) left else base
            for (k in i until j) dirs[k] = fill
            i = j
        }
        val runs = ArrayList<String>()
        i = 0
        while (i < n) {
            var j = i + 1
            while (j < n && dirs[j] == dirs[i]) j++
            runs.add(if (dirs[i] == 'R') reverseRtlRun(cps, i, j) else String(cps, i, j - i))
            i = j
        }
        if (base == 'R') runs.reverse()
        return runs.joinToString("")
    }

    // ------------------------------------------------------------------------------------------- de-duplication

    /**
     * Drop a line that overlaps an already kept line with IoU >= [thr], or lies >= 80 % inside it. Lines are visited
     * by descending conf, then (y0, x0), so the result does not depend on input order; box order is restored.
     */
    internal fun dedupGeometric(lines: List<L>, thr: Double): List<L> {
        val order = lines.indices.sortedWith(compareBy({ -lines[it].conf }, { lines[it].box.y0 }, { lines[it].box.x0 }))
        val kept = ArrayList<Int>()
        for (i in order) {
            val b = lines[i].box
            if (kept.any { k -> iou(b, lines[k].box) >= thr || inter(b, lines[k].box) >= 0.8 * maxOf(1L, area(b)) }) continue
            kept += i
        }
        return kept.sorted().map { lines[it] }
    }

    /**
     * Python `" ".join(s.casefold().split())`. Full case folding as the per-code-point upper-then-lower mapping
     * (per code point, so that no context rule such as the Greek final sigma applies); equal to `str.casefold` on
     * every character of the recogniser dictionary (checked against Python when the port was written).
     */
    internal fun fold(s: String): String {
        val sb = StringBuilder(s.length)
        s.codePoints().forEach { cp ->
            sb.append(String(Character.toChars(cp)).uppercase(Locale.ROOT).lowercase(Locale.ROOT))
        }
        return collapseSpaces(sb.toString())
    }

    /** Keep the first occurrence (in reading order) of each case/space-folded text. */
    internal fun dedupText(lines: List<L>): List<L> {
        val seen = HashSet<String>()
        return lines.filter { seen.add(fold(it.text)) }
    }

    // ------------------------------------------------------------------------------------------- reading order

    private fun isRtlRow(row: List<L>): Boolean {
        var heb = 0
        var ltr = 0
        var rtlStarts = 0
        var ltrStarts = 0
        for (ln in row) {
            var firstStrong = 0
            ln.text.codePoints().forEach { c ->
                if (isHebLetter(c)) {
                    heb++
                    if (firstStrong == 0) firstStrong = 1
                } else if (isLat(c) || isCyr(c)) {
                    ltr++
                    if (firstStrong == 0) firstStrong = -1
                }
            }
            if (firstStrong == 1) rtlStarts++ else if (firstStrong == -1) ltrStarts++
        }
        return heb > 0 && (heb > ltr || rtlStarts > ltrStarts)
    }

    /** Rows: a line joins the first row whose first line overlaps it vertically enough and none of whose lines it intersects. */
    internal fun rows(lines: List<L>, overlap: Double): List<L> {
        val order = lines.sortedWith(compareBy({ it.box.y0 }, { it.box.x0 }))
        val rows = ArrayList<MutableList<L>>()
        for (ln in order) {
            val b = ln.box
            val row = rows.firstOrNull { row ->
                val r = row[0].box
                val ov = minOf(b.y1, r.y1) - maxOf(b.y0, r.y0)
                ov >= overlap * minOf(b.y1 - b.y0, r.y1 - r.y0) && row.none { inter(b, it.box) != 0L }
            }
            if (row != null) row += ln else rows += mutableListOf(ln)
        }
        return rows.flatMap { row ->
            if (isRtlRow(row)) row.sortedWith(compareBy({ -it.box.x1 }, { it.box.y0 }))
            else row.sortedWith(compareBy({ it.box.x0 }, { it.box.y0 }))
        }
    }

    /** Two-pane layout: (gutter x, band y0, band y1) or null. The widest free gap inside paneGutterRange wins. */
    internal fun findGutter(lines: List<L>, width: Int, height: Int, config: OcrConfig): Triple<Int, Int, Int>? {
        if (lines.isEmpty() || width.toDouble() / height < config.paneMinAspect) return null
        val lo = (config.paneGutterRange.first * width).toInt()
        val hi = (config.paneGutterRange.second * width).toInt()
        var bestGap = 0
        var best: Triple<Int, Int, Int>? = null
        var x = lo
        while (x <= hi) {
            val gx = x
            x += 4
            val left = lines.filter { it.box.x1 <= gx }
            val right = lines.filter { it.box.x0 >= gx }
            if (left.size < config.paneMinLines || right.size < config.paneMinLines) continue
            val y0 = maxOf(left.minOf { it.box.y0 }, right.minOf { it.box.y0 })
            val y1 = minOf(left.maxOf { it.box.y1 }, right.maxOf { it.box.y1 })
            if (y1 - y0 < config.paneMinSpan * height) continue
            if (lines.any { it.box.x0 < gx && gx < it.box.x1 && it.box.y1 > y0 && it.box.y0 < y1 }) continue
            val lspan = left.count { it.box.y0 >= y0 && it.box.y1 <= y1 }
            val rspan = right.count { it.box.y0 >= y0 && it.box.y1 <= y1 }
            if (lspan < config.paneMinLines || rspan < config.paneMinLines) continue
            val gl = left.maxOf { it.box.x1 }
            val gr = right.minOf { it.box.x0 }
            val gap = gr - gl
            if (best == null || gap > bestGap) {
                bestGap = gap
                best = Triple(Math.floorDiv(gl + gr, 2), y0, y1)
            }
        }
        return best
    }

    internal fun readingOrder(lines: List<L>, width: Int, height: Int, config: OcrConfig): List<L> {
        val g = findGutter(lines, width, height, config) ?: return rows(lines, config.rowOverlap)
        val (x, y0, y1) = g
        val above = lines.filter { it.box.y1 <= y0 || (it.box.y0 < y0 && it.box.x0 < x && x < it.box.x1) }
        val below = lines.filter { it !in above && it.box.y0 >= y1 }
        val rest = lines.filter { it !in above && it !in below }
        val left = rest.filter { (it.box.x0 + it.box.x1) / 2.0 < x }
        val right = rest.filter { (it.box.x0 + it.box.x1) / 2.0 >= x }
        val o = config.rowOverlap
        return rows(above, o) + rows(left, o) + rows(right, o) + rows(below, o)
    }

    // ------------------------------------------------------------------------------------------- numbers

    /** Python `round(v, digits)` for a float: the exact binary value rounded half-even. */
    fun roundHalfEven(v: Double, digits: Int): Double =
        if (!v.isFinite()) v else BigDecimal(v).setScale(digits, RoundingMode.HALF_EVEN).toDouble()
}
