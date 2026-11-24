package app.murmur.core.prompt

import java.util.Locale

/**
 * The Python `str` semantics that prompts/render.py and prompts/guard.py rely on, reproduced for the ports (D8).
 *
 * Where Kotlin and Python differ and how this object matches Python:
 * - Indexing and length: Python counts code points, Kotlin counts UTF-16 units. Every length, slice and budget in the
 *   ports goes through [len], [head], [tail] and [cps] (emoji and other non-BMP characters count as one).
 * - Whitespace (`str.isspace`, `strip()`, `split()`, regex `\s`): Kotlin's `isWhitespace` omits U+0085, U+00A0, U+2007
 *   and U+202F; [isSpace] is Python's exact set.
 * - Regex `\w` (str patterns): Python = `isalnum()` or `_`, without combining marks (Java's Unicode `\w` includes them
 *   and Android's ICU regex differs again); [isWord] = letter, Nd, Nl or No, or `_`. Regex `\d` = Nd ([isDecimal]).
 * - `str.isdigit()`: Nd plus the Numeric_Type=Digit characters (superscripts, circled digits, ...), table below.
 * - `str.casefold()`: full case folding; Java has none. [casefold] = full uppercase then full lowercase, which equals
 *   Python's folding except for U+0131, U+1E9E and Cherokee (handled explicitly).
 * - `str.lower()` / `str.upper()` of one character: Java's full (SpecialCasing) mappings with [Locale.ROOT].
 * - `str.isupper()` / `str.islower()` of one character: Lu / Ll plus Other_Uppercase / Other_Lowercase, spelled out
 *   (OpenJDK's Character includes them, Android's does not).
 * - Regexes that stay on java.util.regex use explicit classes for `\w`/`\s`/`\b` (Android's regex is ICU-based).
 * - re.I: sre equates a letter with the characters of the same simple lowercase plus a few extra equivalences
 *   (ſ = s, K = k, İ = ı = i, Cyrillic ᲀ = в ...), one character for one character. ICU's CASE_INSENSITIVE uses full
 *   case folding instead (the ligature "ﬆ" matches "st", "İ" stops matching "i"), so no pattern uses the flag:
 *   [ignoreCase] spells every letter out as its exact class ([ignoreCaseVariants]).
 * PyTextTest checks every one of these against goldens/unicode.json (CPython's tables and sre's re.I classes) for
 * every code point that both Unicode versions assign, on the host JVM and on ART (the API 36 emulator runs the same
 * suite through dalvikvm).
 */
internal object Py {
    fun len(s: String): Int = s.codePointCount(0, s.length)

    fun cps(s: String): IntArray = s.codePoints().toArray()

    fun str(cp: Int): String = String(Character.toChars(cp))

    fun str(cps: IntArray, from: Int, to: Int): String = if (to <= from) "" else String(cps, from, to - from)

    /** s[:n] for n >= 0. */
    fun head(s: String, n: Int): String {
        if (n <= 0) return ""
        val c = cps(s)
        return if (n >= c.size) s else str(c, 0, n)
    }

    /** s[-n:] for n >= 0 (Python: s[-0:] is the whole string). */
    fun tail(s: String, n: Int): String {
        val c = cps(s)
        return if (n <= 0 || n >= c.size) s else str(c, c.size - n, c.size)
    }

    /** End index of Python's s[:n] on a sequence of [size] items. */
    fun sliceEnd(size: Int, n: Int): Int = if (n >= 0) minOf(n, size) else maxOf(size + n, 0)

    /** Start index of Python's s[-n:] on a sequence of [size] items. */
    fun sliceStartFromEnd(size: Int, n: Int): Int = when {
        n > 0 -> maxOf(size - n, 0)
        n == 0 -> 0
        else -> minOf(-n, size)
    }

    // ------------------------------------------------------------ character classes
    fun isSpace(cp: Int): Boolean = when (cp) {
        in 0x09..0x0D, in 0x1C..0x20, 0x85, 0xA0, 0x1680, in 0x2000..0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 -> true
        else -> false
    }

    fun isAlpha(cp: Int): Boolean = Character.isLetter(cp)

    fun isDecimal(cp: Int): Boolean = Character.getType(cp) == Character.DECIMAL_DIGIT_NUMBER.toInt()

    fun isWord(cp: Int): Boolean {
        if (cp == '_'.code) return true
        if (Character.isLetter(cp)) return true
        val t = Character.getType(cp)
        return t == Character.DECIMAL_DIGIT_NUMBER.toInt() || t == Character.LETTER_NUMBER.toInt() ||
            t == Character.OTHER_NUMBER.toInt()
    }

    /** Numeric_Type=Digit characters that are not Nd (CPython 3.14 / Unicode 16; stable since Unicode 6). */
    private val DIGIT_NOT_DECIMAL = intArrayOf(
        0xB2, 0xB3, 0xB9, 0xB9, 0x1369, 0x1371, 0x19DA, 0x19DA, 0x2070, 0x2070, 0x2074, 0x2079, 0x2080, 0x2089,
        0x2460, 0x2468, 0x2474, 0x247C, 0x2488, 0x2490, 0x24EA, 0x24EA, 0x24F5, 0x24FD, 0x24FF, 0x24FF,
        0x2776, 0x277E, 0x2780, 0x2788, 0x278A, 0x2792, 0x10A40, 0x10A43, 0x10E60, 0x10E68, 0x11052, 0x1105A,
        0x1F100, 0x1F10A,
    )

    fun isDigit(cp: Int): Boolean = isDecimal(cp) || inRanges(DIGIT_NOT_DECIMAL, cp)

    /**
     * Other_Uppercase / Other_Lowercase (Unicode 16, as CPython 3.14 uses them). Spelled out because Android's
     * Character.isUpperCase/isLowerCase are Lu/Ll only, while OpenJDK's include these (seen on the API 36 emulator).
     */
    private val OTHER_UPPERCASE = intArrayOf(0x2160, 0x216F, 0x24B6, 0x24CF, 0x1F130, 0x1F149, 0x1F150, 0x1F169, 0x1F170, 0x1F189)
    private val OTHER_LOWERCASE = intArrayOf(
        0xAA, 0xAA, 0xBA, 0xBA, 0x2B0, 0x2B8, 0x2C0, 0x2C1, 0x2E0, 0x2E4, 0x345, 0x345, 0x37A, 0x37A, 0x10FC, 0x10FC,
        0x1D2C, 0x1D6A, 0x1D78, 0x1D78, 0x1D9B, 0x1DBF, 0x2071, 0x2071, 0x207F, 0x207F, 0x2090, 0x209C, 0x2170, 0x217F,
        0x24D0, 0x24E9, 0x2C7C, 0x2C7D, 0xA69C, 0xA69D, 0xA770, 0xA770, 0xA7F2, 0xA7F4, 0xA7F8, 0xA7F9, 0xAB5C, 0xAB5F,
        0xAB69, 0xAB69, 0x10780, 0x10780, 0x10783, 0x10785, 0x10787, 0x107B0, 0x107B2, 0x107BA, 0x1E030, 0x1E06D,
    )

    private fun inRanges(r: IntArray, cp: Int): Boolean {
        var i = 0
        while (i < r.size) {
            if (cp >= r[i] && cp <= r[i + 1]) return true
            i += 2
        }
        return false
    }

    /** One-character `str.isupper()`: Lu or Other_Uppercase. */
    fun isUpper(cp: Int): Boolean = Character.getType(cp) == Character.UPPERCASE_LETTER.toInt() || inRanges(OTHER_UPPERCASE, cp)

    /** One-character `str.islower()`: Ll or Other_Lowercase. */
    fun isLower(cp: Int): Boolean = Character.getType(cp) == Character.LOWERCASE_LETTER.toInt() || inRanges(OTHER_LOWERCASE, cp)

    /** `str.isdigit()` of a whole string. */
    fun isDigitString(s: String): Boolean = s.isNotEmpty() && s.codePoints().allMatch { isDigit(it) }

    // ------------------------------------------------------------ case mappings
    fun lower(cp: Int): String = if (cp < 0x80) str(Character.toLowerCase(cp)) else str(cp).lowercase(Locale.ROOT)

    fun upper(cp: Int): String = if (cp < 0x80) str(Character.toUpperCase(cp)) else str(cp).uppercase(Locale.ROOT)

    fun casefold(cp: Int): String = when {
        cp < 0x80 -> str(Character.toLowerCase(cp))
        cp == 0x131 -> "ı"
        cp == 0x1E9E -> "ss"
        cp in 0x13A0..0x13F5 -> str(cp)
        cp in 0x13F8..0x13FD -> str(cp - 8)
        cp in 0xAB70..0xABBF -> str(cp - 0xAB70 + 0x13A0)
        else -> str(cp).uppercase(Locale.ROOT).lowercase(Locale.ROOT)
    }

    fun casefold(s: String): String {
        var ascii = true
        for (ch in s) if (ch.code >= 0x80 || ch in 'A'..'Z') { ascii = false; break }
        if (ascii) return s
        val b = StringBuilder(s.length)
        s.codePoints().forEach { b.append(casefold(it)) }
        return b.toString()
    }

    fun upperFirst(s: String): String {
        if (s.isEmpty()) return s
        val cp = s.codePointAt(0)
        return upper(cp) + s.substring(Character.charCount(cp))
    }

    /** Python `"a" <= x <= "z"` (code point order) for x = one character's lower(). */
    fun lowerIsAsciiLetter(cp: Int): Boolean = between(lower(cp), "a", "z")

    private fun between(x: String, lo: String, hi: String): Boolean = cmp(lo, x) <= 0 && cmp(x, hi) <= 0

    /** Python string comparison: by code point. */
    fun cmp(a: String, b: String): Int {
        val x = cps(a)
        val y = cps(b)
        for (i in 0 until minOf(x.size, y.size)) if (x[i] != y[i]) return x[i].compareTo(y[i])
        return x.size.compareTo(y.size)
    }

    fun isCyrillic(cp: Int): Boolean = cp in 0x0400..0x04FF

    fun isHebrew(cp: Int): Boolean = cp in 0x0590..0x05FF && isAlpha(cp)

    fun isHebrewMark(cp: Int): Boolean =
        cp in 0x0591..0x05BD || cp == 0x05BF || cp == 0x05C1 || cp == 0x05C2 || cp == 0x05C4 || cp == 0x05C5 || cp == 0x05C7

    // ------------------------------------------------------------ strip / split
    fun strip(s: String): String = rstrip(lstrip(s))

    fun lstrip(s: String): String {
        var i = 0
        while (i < s.length) {
            val cp = s.codePointAt(i)
            if (!isSpace(cp)) break
            i += Character.charCount(cp)
        }
        return s.substring(i)
    }

    fun rstrip(s: String): String {
        var j = s.length
        while (j > 0) {
            val cp = s.codePointBefore(j)
            if (!isSpace(cp)) break
            j -= Character.charCount(cp)
        }
        return s.substring(0, j)
    }

    /** `s.strip(chars)`: [chars] holds BMP characters only. */
    fun strip(s: String, chars: String): String {
        var i = 0
        var j = s.length
        while (i < j && chars.indexOf(s[i]) >= 0) i++
        while (j > i && chars.indexOf(s[j - 1]) >= 0) j--
        return s.substring(i, j)
    }

    /** `s.split()` (runs of whitespace, no empty items). */
    fun split(s: String): List<String> {
        val out = ArrayList<String>()
        var i = 0
        var start = -1
        while (i < s.length) {
            val cp = s.codePointAt(i)
            if (isSpace(cp)) {
                if (start >= 0) { out.add(s.substring(start, i)); start = -1 }
            } else if (start < 0) {
                start = i
            }
            i += Character.charCount(cp)
        }
        if (start >= 0) out.add(s.substring(start))
        return out
    }

    /** `s.splitlines()` (no keepends). */
    fun splitlines(s: String): List<String> {
        val out = ArrayList<String>()
        var start = 0
        var i = 0
        while (i < s.length) {
            val c = s[i]
            val brk = c == '\n' || c == '\r' || c == '\u000B' || c == '\u000C' || c in '\u001C'..'\u001E' ||
                c == '\u0085' || c == ' ' || c == ' '
            if (brk) {
                out.add(s.substring(start, i))
                i += if (c == '\r' && i + 1 < s.length && s[i + 1] == '\n') 2 else 1
                start = i
            } else {
                i++
            }
        }
        if (start < s.length) out.add(s.substring(start))
        return out
    }

    /** Last code point of a non-empty string as a string (Python s[-1:]). */
    fun last(s: String): String = if (s.isEmpty()) "" else str(s.codePointBefore(s.length))

    // ------------------------------------------------------------ java.util.regex building blocks with Python meaning
    /** Python `\w` as a Java character class body. */
    const val W = "\\p{L}\\p{Nd}\\p{Nl}\\p{No}_"
    /** Python `\s` as a Java character class body. */
    const val S = "\\t\\n\\x0B\\f\\r\\x1C-\\x20\\x85\\xA0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000"

    // ------------------------------------------------------------ re.I without the regex engine's case folding
    /** sre's extra equivalences (re._casefix) for the supported letters, beyond the letter's own upper case. */
    private val IGNORECASE_EXTRA = mapOf(
        'i' to "İı", 'k' to "K", 's' to "ſ",
        'в' to "ᲀ", 'д' to "ᲁ", 'о' to "ᲂ", 'с' to "ᲃ", 'т' to "ᲄᲅ", 'ъ' to "ᲆ",
    )
    private val IGNORECASE: Map<Char, String> = buildMap {
        for (lo in ('a'..'z') + ('а'..'я') + 'ё') {
            val up = if (lo == 'ё') 'Ё' else lo - 32
            val v = "$lo$up" + IGNORECASE_EXTRA[lo].orEmpty()
            put(lo, v)
            put(up, v)
        }
    }

    /**
     * The characters Python's re.I (str pattern) matches the literal letter [c] with: those whose simple lowercase is
     * c's or one of sre's extra equivalences of it, e.g. "sSſ", "iIİı", "kKK", "тТᲄᲅ". Supported: Latin a-z and
     * Cyrillic а-я, ё in either case (every re.I pattern of guard.py and render.py; unicode.json `re_ignorecase`
     * holds CPython's answer for each of them).
     */
    fun ignoreCaseVariants(c: Char): String =
        IGNORECASE[c] ?: throw IllegalArgumentException("no re.I class for U+%04X".format(c.code))

    /** [cp] matches the literal letter [c] under Python's re.I. */
    fun ignoreCaseEquals(cp: Int, c: Char): Boolean = cp <= 0xFFFF && ignoreCaseVariants(c).indexOf(cp.toChar()) >= 0

    /**
     * A Python re.I pattern as a pattern to compile WITHOUT Pattern.CASE_INSENSITIVE: every letter outside a
     * character class becomes the class [ignoreCaseVariants]. Neither engine's own flag matches sre: Android's ICU
     * regex applies full case folding ("ﬆ" U+FB06 matches "st", "ﬁ" matches "fi", while "İ" U+0130 and "ı" U+0131 no
     * longer match "i"), and the JDK uses its own Unicode tables. Character classes are copied unchanged and must not
     * contain letters (only `\p{..}`, `\x..`, `\u....` and control escapes); an escaped letter outside a class (`\s`,
     * `\b`, ...) is rejected: spell it as an explicit class ([S], [W]).
     */
    fun ignoreCase(pattern: String): String {
        val b = StringBuilder(pattern.length * 3)
        var i = 0
        while (i < pattern.length) {
            val c = pattern[i]
            when {
                c == '\\' -> {
                    val n = pattern[i + 1]
                    require(!n.isLetterOrDigit()) { "escape \\$n outside a class in a re.I pattern" }
                    b.append(c).append(n)
                    i += 2
                }
                c == '[' -> {
                    val end = classEnd(pattern, i)
                    b.append(pattern, i, end)
                    i = end
                }
                c.isLetter() -> {
                    val v = IGNORECASE[c]
                    if (v != null) {
                        b.append('[').append(v).append(']')
                    } else if (c.code in 0x0590..0x05FF) {
                        b.append(c)
                    } else {
                        throw IllegalArgumentException("no re.I class for U+%04X".format(c.code))
                    }
                    i++
                }
                else -> {
                    b.append(c)
                    i++
                }
            }
        }
        return b.toString()
    }

    /** End (exclusive) of the character class that opens at [start]. */
    private fun classEnd(p: String, start: Int): Int {
        var j = start + 1
        if (p[j] == '^') j++
        while (p[j] != ']') {
            if (p[j] == '\\') {
                j += when (p[j + 1]) {
                    'p', 'P' -> p.indexOf('}', j) + 1 - j
                    'x' -> 4
                    'u' -> 6
                    't', 'n', 'f', 'r' -> 2
                    else -> {
                        require(!p[j + 1].isLetterOrDigit()) { "escape \\${p[j + 1]} in a class of a re.I pattern" }
                        2
                    }
                }
            } else {
                require(!p[j].isLetter()) { "letter ${p[j]} in a class of a re.I pattern" }
                j++
            }
        }
        return j + 1
    }
}
