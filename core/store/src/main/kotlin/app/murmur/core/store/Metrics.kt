package app.murmur.core.store

import kotlinx.coroutines.flow.Flow
import java.time.Instant
import java.text.Normalizer
import java.time.LocalDate
import java.util.Locale

/**
 * Numbers-only record of one dictation, written next to the history row and kept for the stats retention
 * period (90 days proposed, open question) even after the history text is gone (DESIGN 6.2). Every
 * Dashboard number is derivable from these rows (ARCHITECTURE.md section 12): no text, no context.
 */
data class DictationMetrics(
    val createdAt: Instant,
    val localDay: LocalDate,
    val pkg: String,
    val status: EntryStatus,
    val language: String?,
    /** True when >= 2 tokens of the minority language were present (Dashboard "Mixed"). */
    val mixedLanguage: Boolean,
    val wordsFinal: Int,
    val wordsRaw: Int,
    val recordMs: Long,
    val speechMs: Long,
    val stopToDoneMs: Long?,
    /** Word-level edit distance raw -> final, filler deletions, other substitutions (Dashboard "Changed by cleanup"). */
    val editDistanceWords: Int,
    val fillersRemoved: Int,
    val fixes: Int,
    val cleanupApplied: Boolean,
)

interface MetricsRepository {
    fun since(day: LocalDate): Flow<List<DictationMetrics>>
    suspend fun add(m: DictationMetrics)
    suspend fun deleteBefore(day: LocalDate): Int
}

/** Dashboard values (DESIGN 6.2) computed from metrics rows. Pure; WP-12 implements [StatsCalculator]. */
data class DashboardStats(
    val todayWords: Int,
    val todaySpeakingMin: Double,
    val timeSavedMin: Double?,
    val last7Days: List<Pair<LocalDate, Int>>,
    val paceWpmMedian: Double?,
    val readyMedianMs: Long?,
    val readyP90Ms: Long?,
    val typedShare: Double?,
    val copiedOnlyCount: Int,
    val copiedOnlyTopApp: String?,
    val changedShare: Double?,
    val fillersRemoved: Int,
    val fixes: Int,
    val languages: Map<String, Int>,
    val topApps: List<Pair<String, Int>>,
    val otherAppsCount: Int,
    val otherAppsWords: Int,
)

interface StatsCalculator {
    fun compute(rows: List<DictationMetrics>, today: LocalDate, typingWpm: Int): DashboardStats
}

/**
 * Numbers computed once at delivery by WP-11 (ARCHITECTURE 12), implemented and tested by WP-12. Pure. The
 * filler lexicon comes from OutputGuard.fillers (the prompts/guard.py FILLERS port, WP-04).
 */
interface DeliveryMetrics {
    fun wordCount(text: String): Int
    /** Word-level Levenshtein distance raw -> final (casefolded, punctuation stripped). */
    fun editDistanceWords(raw: String, final: String): Int
    /** Filler tokens of [raw] absent from [final]. */
    fun fillersRemoved(raw: String, final: String, fillers: Set<String>): Int
    /** Substitutions that are not filler deletions (= edit distance - fillers removed, floored at 0). */
    fun fixes(raw: String, final: String, fillers: Set<String>): Int
    /** True when >= 2 tokens of the minority script (Cyrillic vs Latin) are present. */
    fun mixedLanguage(text: String): Boolean
}

/**
 * The real [DeliveryMetrics] (ARCHITECTURE 12). Tokens are [Words.tokens], the prompts/guard.py `_words` of the NFC
 * text, so the filler lexicon (OutputGuard.fillers = guard.py FILLERS) matches token for token.
 */
class DefaultDeliveryMetrics : DeliveryMetrics {
    override fun wordCount(text: String): Int = Words.tokens(text).size

    override fun editDistanceWords(raw: String, final: String): Int = levenshtein(Words.tokens(raw), Words.tokens(final))

    /**
     * Per filler type, how many more times it occurs in [raw] than in [final] (a multiset difference), so a filler
     * that the cleanup kept is not counted and one filler replaced by another counts once, not twice.
     */
    override fun fillersRemoved(raw: String, final: String, fillers: Set<String>): Int {
        if (fillers.isEmpty()) return 0
        val lexicon = fillers.mapTo(HashSet()) { Words.fold(it) }
        val before = Words.tokens(raw).filter { it in lexicon }.groupingBy { it }.eachCount()
        val after = Words.tokens(final).filter { it in lexicon }.groupingBy { it }.eachCount()
        return before.entries.sumOf { (w, n) -> (n - (after[w] ?: 0)).coerceAtLeast(0) }
    }

    override fun fixes(raw: String, final: String, fillers: Set<String>): Int =
        (editDistanceWords(raw, final) - fillersRemoved(raw, final, fillers)).coerceAtLeast(0)

    /**
     * Each token is assigned the script most of its letters are written in (Cyrillic, Latin, or Hebrew; digits and
     * other scripts count for none); mixed = at least 2 scripts have at least 2 tokens each (DESIGN 6.2 "Languages").
     */
    override fun mixedLanguage(text: String): Boolean {
        var cyrillic = 0
        var latin = 0
        var hebrew = 0
        for (t in Words.tokens(text)) {
            var c = 0
            var l = 0
            var h = 0
            for (ch in t) {
                if (!ch.isLetter()) continue
                when (Character.UnicodeScript.of(ch.code)) {
                    Character.UnicodeScript.CYRILLIC -> c++
                    Character.UnicodeScript.LATIN -> l++
                    Character.UnicodeScript.HEBREW -> h++
                    else -> {}
                }
            }
            when {
                c > l && c > h -> cyrillic++
                l > c && l > h -> latin++
                h > c && h > l -> hebrew++
            }
        }
        var scriptsWithTwo = 0
        if (cyrillic >= 2) scriptsWithTwo++
        if (latin >= 2) scriptsWithTwo++
        if (hebrew >= 2) scriptsWithTwo++
        return scriptsWithTwo >= 2
    }

    private fun levenshtein(a: List<String>, b: List<String>): Int {
        if (a.isEmpty()) return b.size
        if (b.isEmpty()) return a.size
        val row = IntArray(b.size + 1) { it }
        for (i in 1..a.size) {
            var diagonal = row[0]
            row[0] = i
            for (j in 1..b.size) {
                val above = row[j]
                row[j] = minOf(above + 1, row[j - 1] + 1, diagonal + if (a[i - 1] == b[j - 1]) 0 else 1)
                diagonal = above
            }
        }
        return row[b.size]
    }
}

/**
 * Word tokens shared by the delivery metrics and the history search folding: prompts/guard.py `_WORD` + `_words`
 * applied to NFC text (guard.py `precheck` NFC-normalises the transcript before its words are taken).
 *
 * - `\w` in Python's `re` is `str.isalnum()` or "_": letters and numbers of every kind, never combining marks. A
 *   stress mark therefore splits a word ("за́мок" -> "за", "мок") exactly as in guard.py; after NFC, a mark that has a
 *   precomposed form ("е" + U+0308) is part of its letter ("ё").
 * - [fold] is Python's `str.casefold()` per code point ("ß" -> "ss", final "ς" -> "σ"), then "ё" -> "е".
 */
object Words {
    private val WORD = Regex("""[\p{L}\p{N}_]+(?:[-'’][\p{L}\p{N}_]+)*""")

    fun tokens(text: String): List<String> = WORD.findAll(nfc(text)).map { fold(it.value) }.toList()

    /** NFC, then Python's `str.casefold()`, then "ё" -> "е" (guard.py compares words this way). */
    fun fold(s: String): String {
        val n = nfc(s)
        val out = StringBuilder(n.length)
        var i = 0
        while (i < n.length) {
            val cp = n.codePointAt(i)
            caseFold(cp, out)
            i += Character.charCount(cp)
        }
        for (k in out.indices) if (out[k] == 'ё') out.setCharAt(k, 'е')
        return out.toString()
    }

    private fun nfc(s: String): String =
        if (Normalizer.isNormalized(s, Normalizer.Form.NFC)) s else Normalizer.normalize(s, Normalizer.Form.NFC)

    /**
     * Full Unicode case folding of one code point. Upper-casing the code point alone (no final-sigma context, which
     * only lower-casing a whole string applies) and lower-casing each result equals CaseFolding.txt (C + F) for every
     * code point except the four handled first; checked against Python over all code points the JVM defines.
     */
    private fun caseFold(cp: Int, out: StringBuilder) {
        when {
            cp == 0x130 -> out.append("i\u0307") // İ: F mapping, not the Turkic one
            cp == 0x131 -> out.appendCodePoint(cp) // ı has no folding
            cp == 0x1E9E -> out.append("ss") // ẞ
            cp in 0x13A0..0x13FD || cp in 0xAB70..0xABBF -> out.appendCodePoint(Character.toUpperCase(cp)) // Cherokee folds to upper
            else -> {
                val upper = String(Character.toChars(cp)).uppercase(Locale.ROOT)
                var j = 0
                while (j < upper.length) {
                    val u = upper.codePointAt(j)
                    out.appendCodePoint(Character.toLowerCase(u))
                    j += Character.charCount(u)
                }
            }
        }
    }
}
