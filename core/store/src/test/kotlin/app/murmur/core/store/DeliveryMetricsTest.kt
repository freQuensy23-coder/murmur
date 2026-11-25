package app.murmur.core.store

import app.murmur.testing.SimpleDeliveryMetrics
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class DeliveryMetricsTest {
    private val m = DefaultDeliveryMetrics()
    private val reference = SimpleDeliveryMetrics()

    /** The lexicon OutputGuard.fillers ports: `FILLERS = frozenset("""…""".split())` in prompts/guard.py. */
    private val guardFillers: Set<String> by lazy {
        val src = File(System.getProperty("murmur.repoRoot"), "prompts/guard.py").readText()
        val body = Regex("""FILLERS\s*=\s*frozenset\(\s*\"\"\"(.*?)\"\"\"\s*\.split\(\)\s*\)""", RegexOption.DOT_MATCHES_ALL)
            .find(src)?.groupValues?.get(1) ?: error("FILLERS not found in prompts/guard.py")
        body.split(Regex("\\s+")).filter { it.isNotEmpty() }.toSet()
    }

    /** Cases where both implementations must agree (no hyphenated or apostrophe tokens, no "ё"). */
    private val shared = listOf(
        Case("ну короче я буду через десять минут", "Короче, я буду через десять минут.", dist = 1, fillers = 1, fixes = 0),
        Case("эм привет как дела", "Привет! Как дела?", dist = 1, fillers = 1, fixes = 0),
        Case("um so I think we should uh ship it", "So I think we should ship it.", dist = 2, fillers = 2, fixes = 0),
        Case("встреча в пятницу нет в четверг", "Встреча в четверг.", dist = 3, fillers = 0, fixes = 3),
        Case("send the deck to dima", "Send the deck to Dima.", dist = 0, fillers = 0, fixes = 0),
        Case("мм open the терминал and run gradle", "Open the терминал and run Gradle.", dist = 1, fillers = 1, fixes = 0),
        Case("hmm the the meeting is at five", "The meeting is at 5.", dist = 3, fillers = 1, fixes = 2),
        Case("", "Hello there", dist = 2, fillers = 0, fixes = 2),
        Case("uh um", "", dist = 2, fillers = 2, fixes = 0),
    )

    data class Case(val raw: String, val final: String, val dist: Int, val fillers: Int, val fixes: Int)

    @Test
    fun lexiconIsTheGuardOne() {
        // Sanity of the parse: Russian and English hesitations, hyphenated forms included.
        assertTrue(guardFillers.containsAll(listOf("э", "ээ", "э-э", "мм", "ну", "um", "uh", "uh-huh", "hmm")))
    }

    @Test
    fun sharedCasesMatchExpectationsAndTheReference() {
        for (c in shared) {
            assertEquals(c.toString(), c.dist, m.editDistanceWords(c.raw, c.final))
            assertEquals(c.toString(), c.fillers, m.fillersRemoved(c.raw, c.final, guardFillers))
            assertEquals(c.toString(), c.fixes, m.fixes(c.raw, c.final, guardFillers))
            assertEquals(c.toString(), reference.editDistanceWords(c.raw, c.final), m.editDistanceWords(c.raw, c.final))
            assertEquals(c.toString(), reference.fillersRemoved(c.raw, c.final, guardFillers), m.fillersRemoved(c.raw, c.final, guardFillers))
            assertEquals(c.toString(), reference.fixes(c.raw, c.final, guardFillers), m.fixes(c.raw, c.final, guardFillers))
            assertEquals(c.toString(), reference.wordCount(c.raw), m.wordCount(c.raw))
            assertEquals(c.toString(), reference.wordCount(c.final), m.wordCount(c.final))
        }
    }

    @Test
    fun wordCountIgnoresPunctuationAndKeepsNumbers() {
        assertEquals(7, m.wordCount("Привет, Дима! Буду в 10:30 — ок?"))
        assertEquals(0, m.wordCount("  …  — !! "))
        assertEquals(3, m.wordCount("snake_case variable42 ok"))
    }

    @Test
    fun guardTokensKeepHyphenatedFillersAndContractionsWhole() {
        // guard.py _WORD: "э-э" and "uh-huh" are single filler tokens, "don't" one word (the reference splits them).
        assertEquals(1, m.fillersRemoved("э-э да", "Да.", guardFillers))
        assertEquals(0, m.fixes("э-э да", "Да.", guardFillers))
        assertEquals(1, m.fillersRemoved("uh-huh sounds good", "Sounds good.", guardFillers))
        assertEquals(3, m.wordCount("I don't know"))
        assertEquals(1, m.wordCount("кто-нибудь"))
    }

    @Test
    fun editDistanceIsCasefoldedAndFoldsYo() {
        assertEquals(0, m.editDistanceWords("ЁЛКА зелёная", "елка зеленая"))
        assertEquals(0, m.editDistanceWords("Telegram", "TELEGRAM"))
        assertEquals(1, m.editDistanceWords("a b c", "a c"))
        assertEquals(1, m.editDistanceWords("a c", "a b c"))
        assertEquals(1, m.editDistanceWords("a b c", "a x c"))
    }

    @Test
    fun keptFillerIsNotCountedAndFillerSwapCountsOnce() {
        // "ну" stays in the final text: nothing removed, the one deletion of "вот" is a fix.
        assertEquals(0, m.fillersRemoved("ну вот и всё", "Ну и всё.", guardFillers))
        assertEquals(1, m.fixes("ну вот и всё", "Ну и всё.", guardFillers))
        // Two of three "uh" removed.
        assertEquals(2, m.fillersRemoved("uh uh fine uh", "Uh, fine.", guardFillers))
        // One filler replaced by another: one removed (the multiset difference), not 0 (total counts are equal)
        // and not 2; the substitution is that filler, so no fix.
        assertEquals(1, m.fillersRemoved("um ok", "uh ok", guardFillers))
        assertEquals(0, m.fixes("um ok", "uh ok", guardFillers))
        assertEquals(1, m.fillersRemoved("ну ok", "э ok", guardFillers))
        assertEquals(2, m.fillersRemoved("ну ну ладно", "э ладно", guardFillers))
        // No lexicon: everything is a fix.
        assertEquals(0, m.fillersRemoved("um ok", "ok", emptySet()))
        assertEquals(1, m.fixes("um ok", "ok", emptySet()))
    }

    /**
     * Expected tokens are prompts/guard.py `_words(unicodedata.normalize("NFC", text))`, computed with python3 when this
     * table was written. Python's `\w` takes no combining marks, `casefold()` is full case folding.
     */
    @Test
    fun tokensAreGuardPyWordsOfTheNfcText() {
        val expected = listOf(
            "за\u0301мок" to listOf("за", "мок"), // combining stress mark splits, as in guard.py
            "e\u0301clair" to listOf("éclair"), // NFD: composes under NFC
            "\u0435\u0308лка" to listOf("елка"), // NFD "ё" composes, then folds to "е"
            "Ёлка" to listOf("елка"),
            "Straße" to listOf("strasse"),
            "ẞ" to listOf("ss"),
            "ΟΔΟΣ" to listOf("οδοσ"), // casefold has no final-sigma rule
            "ὈΔΌΣ" to listOf("ὀδόσ"),
            "İstanbul" to listOf("i\u0307stanbul"),
            "ılık" to listOf("ılık"),
            "ﬁle" to listOf("file"),
            "ǅemal" to listOf("ǆemal"),
            "µs" to listOf("μs"),
            "Ꭰᏸ" to listOf("ᎠᏰ"), // Cherokee folds to upper case
            "snake_case x²" to listOf("snake_case", "x²"),
            "Ⅻ legion" to listOf("ⅻ", "legion"),
            "don't кто-нибудь uh-huh э-э-э" to listOf("don't", "кто-нибудь", "uh-huh", "э-э-э"),
        )
        for ((text, tokens) in expected) assertEquals(text, tokens, Words.tokens(text))
        // The same word in NFC and NFD is the same token, so nothing counts as an edit.
        assertEquals(0, m.editDistanceWords("\u0435\u0308лка e\u0301clair", "Ёлка éclair"))
    }

    @Test
    fun mixedLanguageNeedsTwoMinorityTokens() {
        assertTrue(m.mixedLanguage("Скинь мне pull request и code review до вечера"))
        assertFalse(m.mixedLanguage("Открой Telegram и напиши Диме"))
        assertFalse(m.mixedLanguage("Just a plain English sentence"))
        assertFalse(m.mixedLanguage("Обычное русское предложение 2026 года"))
        assertTrue(m.mixedLanguage("Let's meet завтра утром at the office"))
        // Digits and punctuation belong to neither script.
        assertFalse(m.mixedLanguage("Встреча в 10:30, room 5"))
        // Latin with diacritics is Latin.
        assertTrue(m.mixedLanguage("Привет, café crème и круассан"))
        // Hebrew script counts as a third distinct script: mixed with English or Russian when >= 2 tokens each.
        assertFalse(m.mixedLanguage("שלום עולם מה שלומך היום"))
        assertFalse(m.mixedLanguage("שלח לי את זה ב Telegram בבקשה"))
        assertTrue(m.mixedLanguage("שלח לי pull request לפני code review"))
        assertTrue(m.mixedLanguage("שלום חברים привет мир сегодня"))
        for (text in listOf("Скинь мне pull request и code review", "Открой Telegram и напиши", "hello world", "")) {
            assertEquals(text, reference.mixedLanguage(text), m.mixedLanguage(text))
        }
    }
}
