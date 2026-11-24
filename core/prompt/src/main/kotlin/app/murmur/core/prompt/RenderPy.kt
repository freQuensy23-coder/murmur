package app.murmur.core.prompt

import app.murmur.core.context.CtxV2
import app.murmur.core.context.FieldInfo
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import java.text.Normalizer

/**
 * The functions of prompts/render.py, ported line by line (D8). Names follow the Python ones; Python `str` semantics
 * that Kotlin lacks come from [Py]. Only ctx v2 is ported (ADR-040). Further places where the languages differ:
 * - Python `//` floors, Kotlin `/` truncates: box centres use [Math.floorDiv] (boxes may be partly off-screen).
 * - unicodedata.normalize("NFC") is [Normalizer] (same Unicode algorithm; the JVM's Unicode version may be older).
 * - The Python code works on dicts, the port on the typed [CtxV2]: `not field` (an empty field object) is a
 *   [FieldInfo] whose members are all absent; a field object holding only explicit JSON nulls decodes the same way
 *   and renders "no text field focused" where Python says "text field" (the phone never sends one).
 * - Truthiness: `if hint`, `if lang`, `field.get("kind") or ...` treat "" like absent; ctx2 options follow Python
 *   truthiness of JSON values ([Ctx2]).
 * - `sorted(key=...)` is stable; the port sorts with the same key plus the block index Python's key already holds.
 * - `render_context(None)` renders the v1 layout of an empty context; the port returns that constant ([EMPTY_CONTEXT]).
 */
internal object RenderPy {
    const val MAX_BEFORE_CURSOR = 300

    val LANG_NAMES = mapOf("ru" to "Russian", "en" to "English", "he" to "Hebrew", "uk" to "Ukrainian", "de" to "German",
        "fr" to "French", "es" to "Spanish")

    val FIELD_PHRASES = mapOf(
        "message" to "message field", "email_body" to "email body", "email_subject" to "email subject",
        "search" to "search field", "address" to "address field", "note" to "note", "code" to "code or terminal input",
        "generic" to "text field", "none" to "no text field focused",
    )

    /** render.render_context(None) and of {}: the v1 layout of an empty context (no app, no field). */
    const val EMPTY_CONTEXT = "<context>\nApp: unknown app — text field\n</context>"

    // ------------------------------------------------------------ sanitising
    /** _CTRL: C0 except \t \n, DEL + C1, zero-width and bidi controls, BOM. All BMP, so a UTF-16 filter is exact. */
    private fun isCtrl(c: Char): Boolean {
        val x = c.code
        return x in 0x00..0x08 || x in 0x0B..0x1F || x in 0x7F..0x9F || x in 0x200B..0x200F ||
            x in 0x202A..0x202E || x in 0x2066..0x2069 || x == 0xFEFF
    }

    /** _WS.sub(" ", s): runs of space, tab, no-break space -> one space. */
    private fun collapseWs(s: String): String {
        val b = StringBuilder(s.length)
        var inRun = false
        for (c in s) {
            if (c == ' ' || c == '\t' || c == ' ') {
                if (!inRun) b.append(' ')
                inRun = true
            } else {
                b.append(c)
                inRun = false
            }
        }
        return b.toString()
    }

    /** re.sub(r"\n{3,}", "\n\n", s). */
    private fun collapseBlankLines(s: String): String {
        val b = StringBuilder(s.length)
        var run = 0
        for (c in s) {
            if (c == '\n') {
                run++
                if (run <= 2) b.append(c)
            } else {
                run = 0
                b.append(c)
            }
        }
        return b.toString()
    }

    fun clean(s: String?, keepNewlines: Boolean = false): String {
        if (s.isNullOrEmpty()) return ""
        var t = Normalizer.normalize(s, Normalizer.Form.NFC)
        if (t.any { isCtrl(it) }) t = buildString(t.length) { for (c in t) if (!isCtrl(c)) append(c) }
        t = t.replace('<', '‹').replace('>', '›')
        return if (keepNewlines) {
            Py.strip(collapseBlankLines(Py.splitlines(t).joinToString("\n") { Py.strip(collapseWs(it)) }))
        } else {
            Py.strip(collapseWs(t.replace('\r', ' ').replace('\n', ' ')))
        }
    }

    /** Keep the first n code points, cut at a word boundary, mark with an ellipsis. */
    fun clipEnd(s: String, n: Int): String {
        val c = Py.cps(s)
        if (c.size <= n) return s
        var end = Py.sliceEnd(c.size, n)
        var sp = -1
        for (k in end - 1 downTo 0) if (c[k] == ' '.code) { sp = k; break }
        if (sp > n * 0.6) end = sp
        return Py.rstrip(Py.str(c, 0, end)) + "…"
    }

    /** Keep the last n code points (text right before the cursor matters most). */
    fun clipStart(s: String, n: Int): String {
        val c = Py.cps(s)
        if (c.size <= n) return s
        var start = Py.sliceStartFromEnd(c.size, n)
        var sp = -1
        for (k in start until c.size) if (c[k] == ' '.code) { sp = k - start; break }
        if (sp >= 0 && sp < n * 0.4) start += sp + 1
        return "…" + Py.lstrip(Py.str(c, start, c.size))
    }

    // ------------------------------------------------------------ ctx v2
    /** render.CTX2_DEFAULT merged with a profile's ctx2 (`{**CTX2_DEFAULT, **ctx2}`; unknown keys are ignored). */
    class Ctx2(private val o: JsonObject?) {
        private fun get(k: String): JsonElement? = o?.get(k)
        val budget: Double = num(get("budget"), 1400.0)
        val order: String? = get("order").let { if (it == null) "near" else (it as? JsonPrimitive)?.takeIf { p -> p.isString }?.content }
        val minConf: Double = num(get("min_conf"), 0.5)
        val minChars: Double = num(get("min_chars"), 2.0)
        val dropClock: Boolean = truthy(get("drop_clock"), true)
        val dedup: Boolean = truthy(get("dedup"), true)
        val merge: Boolean = truthy(get("merge"), true)
        val side: Boolean = truthy(get("side"), false)
        val terms: Boolean = truthy(get("terms"), false)
        val lineMax: Int = get("line_max").let { e ->
            if (e == null) 240 else {
                val d = num(e, 240.0)
                require(e is JsonPrimitive && !e.isString && d == Math.floor(d)) { "ctx2.line_max must be an integer" }
                d.toInt()
            }
        }

        private companion object {
            fun num(e: JsonElement?, default: Double): Double = when (e) {
                null -> default
                is JsonPrimitive -> e.booleanOrNull?.let { if (it) 1.0 else 0.0 }
                    ?: if (e.isString || e == JsonNull) throw IllegalArgumentException("ctx2 option must be a number") else e.content.toDouble()
                else -> throw IllegalArgumentException("ctx2 option must be a number")
            }

            /** Python truthiness of a JSON value. */
            fun truthy(e: JsonElement?, default: Boolean): Boolean = when (e) {
                null -> default
                JsonNull -> false
                is JsonPrimitive -> e.booleanOrNull ?: if (e.isString) e.content.isNotEmpty() else e.content.toDouble() != 0.0
                is JsonArray -> e.isNotEmpty()
                is JsonObject -> e.isNotEmpty()
            }
        }
    }

    /** A block while lines are merged; [last] = box of its last line. */
    private class Block(var text: String, var box: IntArray, var last: IntArray)

    /** Python `not field` for a decoded field object: every member absent. */
    private fun FieldInfo.isEmptyObject() =
        kind == null && hint == null && beforeCursor == null && box == null && inputTypeVariation == null && password == null

    private fun size(ctx: CtxV2): IntArray {
        val s = ctx.screen?.size
        return if (s.isNullOrEmpty()) intArrayOf(1080, 2520) else intArrayOf(s[0], s[1])
    }

    private fun alnumCount(t: String): Int = t.codePoints().filter { it != '_'.code && Py.isWord(it) }.count().toInt()

    /**
     * Clock/status block (_CLOCK.match, re.I): `^A*(?:[ap]\.?m\.?|ч|мин|lte|5g|4g|g)?A*$`. Letters compare as sre's
     * re.I does ([Py.ignoreCaseEquals]), not by the platform's case tables.
     */
    fun isClock(text: String): Boolean {
        val c = Py.cps(text)
        val n = c.size
        fun inA(cp: Int) = Py.isDecimal(cp) || Py.isSpace(cp) || cp in CLOCK_PUNCT
        fun eq(cp: Int, letter: Char) = Py.ignoreCaseEquals(cp, letter)
        // suffixOk[k] = c[k:] consists of A only
        val suffixOk = BooleanArray(n + 1)
        suffixOk[n] = true
        for (k in n - 1 downTo 0) suffixOk[k] = suffixOk[k + 1] && inA(c[k])
        var k = 0
        while (true) {
            if (suffixOk[k]) return true  // empty optional group
            // the alternatives at position k ("5g"/"4g" add nothing: their digit is also in A)
            val ends = ArrayList<Int>(4)
            if (k < n && (eq(c[k], 'a') || eq(c[k], 'p'))) {
                var p = k + 1
                if (p < n && c[p] == '.'.code) p++
                if (p < n && eq(c[p], 'm')) {
                    p++
                    ends.add(p)
                    if (p < n && c[p] == '.'.code) ends.add(p + 1)
                }
            }
            if (k < n && eq(c[k], 'ч')) ends.add(k + 1)
            if (k + 2 < n && eq(c[k], 'м') && eq(c[k + 1], 'и') && eq(c[k + 2], 'н')) ends.add(k + 3)
            if (k + 2 < n && eq(c[k], 'l') && eq(c[k + 1], 't') && eq(c[k + 2], 'e')) ends.add(k + 3)
            if (k < n && eq(c[k], 'g')) ends.add(k + 1)
            if (ends.any { suffixOk[it] }) return true
            if (k < n && inA(c[k])) k++ else return false
        }
    }

    private val CLOCK_PUNCT = ":.,/%+-–—()|·•".codePoints().toArray().toSet()

    fun ocrBlocks(ctx: CtxV2, o: Ctx2): List<Block2> {
        val w = size(ctx)[0]
        val fbox = ctx.field?.box?.takeIf { it.isNotEmpty() }
        val hint = Py.casefold(clean(ctx.field?.hint))
        val kept = ArrayList<Pair<String, IntArray>>()
        val seen = HashSet<String>()
        for (ln in ctx.screen?.lines.orEmpty()) {
            val t = clean(ln.text)
            val box = ln.box.toIntArray()
            if (t.isEmpty() || ln.conf < o.minConf || alnumCount(t) < o.minChars) continue
            if (fbox != null) {
                val cx = Math.floorDiv(box[0] + box[2], 2)
                val cy = Math.floorDiv(box[1] + box[3], 2)
                if (fbox[0] <= cx && cx <= fbox[2] && fbox[1] <= cy && cy <= fbox[3]) continue
            }
            val tf = Py.casefold(t)
            if (hint.isNotEmpty() && tf == hint) continue
            if (o.dedup) {
                if (!seen.add(tf)) continue
            }
            kept.add(t to box)
        }
        var blocks = ArrayList<Block>()
        for ((text, box) in kept) {
            val b = blocks.lastOrNull()
            val lh = box[3] - box[1]
            val bh = if (b != null) b.last[3] - b.last[1] else 0
            val aligned = b != null && (Math.abs(box[0] - b.last[0]) * 50 <= w || Math.abs(box[2] - b.last[2]) * 50 <= w)
            if (o.merge && aligned && Math.abs(lh - bh) * 10 <= bh &&
                0 <= box[1] - b.last[3] && (box[1] - b.last[3]) * 2 < maxOf(1, bh)
            ) {
                b.text += " $text"
                b.box = intArrayOf(minOf(b.box[0], box[0]), b.box[1], maxOf(b.box[2], box[2]), box[3])
                b.last = box
            } else {
                blocks.add(Block(text, box.copyOf(), box))
            }
        }
        if (o.dropClock) blocks = blocks.filterTo(ArrayList()) { !(Py.len(it.text) <= 12 && isClock(it.text)) }
        return blocks.mapIndexed { i, b ->
            Block2(i, clipEnd(b.text, o.lineMax), b.box, b.box[0] * 100 >= 15 * w && (w - b.box[2]) * 10 < w)
        }
    }

    /** A finished block: reading-order index, clipped text, merged box, right-aligned flag (render._ocr_blocks). */
    class Block2(val i: Int, val text: String, val box: IntArray, val right: Boolean)

    private fun fieldAnchorY(ctx: CtxV2): Int {
        val fbox = ctx.field?.box?.takeIf { it.isNotEmpty() }
        return if (fbox != null) Math.floorDiv(fbox[1] + fbox[3], 2) else size(ctx)[1]
    }

    private const val TERM_STRIP = ".,;:!?()«»\"“”‹›'’-–—…"

    private fun isCamelLower(cp: Int) = cp in 'a'.code..'z'.code || cp in 0x430..0x44F || cp == 0x451
    private fun isCamelUpper(cp: Int) = cp in 'A'.code..'Z'.code || cp in 0x410..0x42F || cp == 0x401
    private fun isAlnumNotDecimal(cp: Int) = cp != '_'.code && Py.isWord(cp) && !Py.isDecimal(cp)

    /** re.search(r"\w[#@_./\-]\w|[^\W\d_]\d|\d[^\W\d_]", core). */
    private fun special(c: IntArray): Boolean {
        for (k in c.indices) {
            if (k + 2 < c.size && Py.isWord(c[k]) && c[k + 1] in SPECIAL_JOIN && Py.isWord(c[k + 2])) return true
            if (k + 1 < c.size && isAlnumNotDecimal(c[k]) && Py.isDecimal(c[k + 1])) return true
            if (k + 1 < c.size && Py.isDecimal(c[k]) && isAlnumNotDecimal(c[k + 1])) return true
        }
        return false
    }

    private val SPECIAL_JOIN = "#@_./-".codePoints().toArray().toSet()

    /** render.ocr_terms. */
    fun ocrTerms(texts: List<String>, limit: Int = 20): List<String> {
        val out = ArrayList<String>()
        val seen = HashSet<String>()
        fun add(x: String) {
            if (Py.len(x) >= 3 && seen.add(Py.casefold(x))) out.add(x)
        }
        for (t in texts) {
            val tc = Py.cps(t)
            val latN = tc.count { Py.lowerIsAsciiLetter(it) }
            val cyrLine = tc.count { Py.isCyrillic(it) } > latN
            val hebLine = tc.count { Py.isHebrew(it) } > latN
            val toks = Py.split(t)
            val run = ArrayList<String>()
            var runStart = false
            fun flush() {
                if (run.isNotEmpty() && (run.size >= 2 || !runStart)) add(run.joinToString(" "))
                run.clear()
            }
            for ((j, tok) in toks.withIndex()) {
                val core = Py.strip(tok, TERM_STRIP)
                val cc = Py.cps(core)
                val sentStart = j == 0 || ".!?…".contains(Py.last(toks[j - 1]))
                val letters = cc.filter { Py.isAlpha(it) }
                val special = special(cc)
                val camel = (0 until cc.size - 1).any { isCamelLower(cc[it]) && isCamelUpper(cc[it + 1]) }
                val caps = letters.size >= 2 && letters.all { Py.isUpper(it) }
                val latin = (cyrLine || hebLine) && letters.size >= 4 && letters.all { Py.lowerIsAsciiLetter(it) }
                if (cc.isEmpty()) {
                    flush()
                } else if (special || camel || caps) {
                    flush()
                    add(core)
                } else if (Py.isUpper(cc[0]) && cc.size >= 2 && Py.isLower(cc[1])) {
                    if (run.isEmpty()) runStart = sentStart
                    run.add(core)
                    if (",.!?…;:".contains(Py.last(tok))) flush()
                } else {
                    flush()
                    if (latin) add(core)
                }
            }
            flush()
        }
        return out.subList(0, Py.sliceEnd(out.size, limit)).toList()
    }

    fun renderContextV2(ctx: CtxV2, ctx2: JsonObject?): String {
        val o = Ctx2(ctx2)
        val privacy = ctx.privacy?.level ?: "full"
        val field = ctx.field
        var label = clean(ctx.app.label).ifEmpty { "unknown app" }
        if (privacy == "none") label = "unknown app"
        val kind = field?.kind?.takeIf { it.isNotEmpty() } ?: if (field == null || field.isEmptyObject()) "none" else "generic"
        val out = ArrayList<String>()
        out.add("App: $label — ${FIELD_PHRASES[kind] ?: "text field"}")
        val hint = clipEnd(clean(field?.hint), 60)
        if (hint.isNotEmpty() && privacy != "none") out.add("Field hint: «$hint»")
        val lang = ctx.language?.asr
        if (!lang.isNullOrEmpty()) out.add("Dictation language: ${LANG_NAMES[lang] ?: lang}")
        if (privacy == "app_only" || privacy == "none") return "<context>\n" + out.joinToString("\n") + "\n</context>"
        val before = clipStart(clean(field?.beforeCursor), MAX_BEFORE_CURSOR)
        if (before.isNotEmpty()) out.add("Text already in the field before the cursor: «$before»")
        val blocks = ocrBlocks(ctx, o)
        val ranked = if (o.order == "reading") blocks else {
            val ay = fieldAnchorY(ctx)
            blocks.sortedWith(compareBy<Block2>({ Math.abs(Math.floorDiv(it.box[1] + it.box[3], 2) - ay) }, { it.i }))
        }
        var shown = ArrayList<Pair<Int, String>>()
        var used = 0
        for (b in ranked) {
            val s = (if (o.side && b.right) "» " else "") + b.text
            val l = Py.len(s)
            if ((used + l + 1).toDouble() > o.budget) break
            shown.add(b.i to s)
            used += l + 1
        }
        if (o.order == "near_reading") shown = shown.sortedBy { it.first }.toCollection(ArrayList())
        if (shown.isNotEmpty()) {
            val head = when (o.order) {
                "near" -> "Screen text (OCR, nearest to the field first):"
                "near_reading", "reading" -> "Screen text (OCR, top to bottom):"
                else -> throw IllegalArgumentException("ctx2.order must be near, near_reading or reading")
            }
            out.add(head)
            for ((_, s) in shown) out.add(s)
            if (o.terms) {
                val terms = ocrTerms(shown.sortedBy { it.first }.map { it.second })
                if (terms.isNotEmpty()) out.add("Terms on screen: " + terms.joinToString("; "))
            }
        }
        return "<context>\n" + out.joinToString("\n") + "\n</context>"
    }

    /** render.render_context: ctx v2 only (ADR-040); null renders the empty v1 block exactly as Python does. */
    fun renderContext(ctx: CtxV2?, ctx2: JsonObject?): String {
        if (ctx == null) return EMPTY_CONTEXT
        require(ctx.v == 2) { "only ctx v2 is rendered (ADR-040)" }
        return renderContextV2(ctx, ctx2)
    }

    // ------------------------------------------------------------ prompt
    /** render.style_line(ctx, tones, override). */
    fun styleLine(ctx: CtxV2?, tones: JsonObject, override: String?): String {
        if (!override.isNullOrEmpty()) return "Style: " + clean(override)
        var cat = ctx?.app?.category?.takeIf { it.isNotEmpty() } ?: "default"
        val variation = ctx?.field?.inputTypeVariation
        if (!variation.isNullOrEmpty()) {
            val fo = tones["field_overrides"] as? JsonObject
            fo?.get(variation)?.let { cat = (it as JsonPrimitive).content }
        }
        val cats = tones.getValue("categories") as JsonObject
        val entry = (cats[cat] as? JsonObject)?.takeIf { it.isNotEmpty() } ?: cats.getValue("default") as JsonObject
        return "Style: " + (entry.getValue("text") as JsonPrimitive).content
    }

    /** render._versioned: 'default' -> 'default.v1'; 'default.v2' stays (re.search(r"\.v\d+$", name)). */
    fun versioned(name: String): String {
        val n = if (name.endsWith("\n")) name.dropLast(1) else name
        val c = Py.cps(n)
        var k = c.size
        while (k > 0 && Py.isDecimal(c[k - 1])) k--
        val ok = k < c.size && k >= 2 && c[k - 1] == 'v'.code && c[k - 2] == '.'.code
        return if (ok) name else "$name.v1"
    }

    fun userMessage(ctx: CtxV2?, transcript: String, styleOverride: String?, ctx2: JsonObject?, tones: JsonObject): String {
        val style = styleLine(ctx, tones, styleOverride)
        val place = (ctx2?.get("place") as? JsonPrimitive)?.takeIf { it.isString }?.content
        return if (ctx != null && ctx.v == 2 && place == "after") {
            style + "\n<transcript>\n" + clean(transcript) + "\n</transcript>\n" + renderContext(ctx, ctx2)
        } else {
            style + "\n" + renderContext(ctx, ctx2) + "\n" + "<transcript>\n" + clean(transcript) + "\n</transcript>"
        }
    }

    /** render.TEMPLATES: bos, prior assistant turn opener, final assistant opener (thinking disabled). */
    class Template(val bos: String, val prior: String, val final: String)

    val TEMPLATES = mapOf(
        "qwen3.5" to Template("", "<|im_start|>assistant\n", "<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "qwen3" to Template("", "<|im_start|>assistant\n", "<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "lfm3" to Template("<|startoftext|>", "<|im_start|>assistant\n<think></think>", "<|im_start|>assistant\n<think></think>"),
        "lfm2.5" to Template("<|startoftext|>", "<|im_start|>assistant\n", "<|im_start|>assistant\n"),
    )

    fun template(name: String): Template = TEMPLATES[name] ?: throw IllegalArgumentException("unknown chat template: $name")

    /**
     * render.chat_turns up to and including the real user turn's "<|im_start|>user\n" = render.cache_prefix (the last
     * marker of chat_turns(system, turns, "") is always the real user turn's: only "<|im_end|>\n" and the final
     * opener follow it).
     */
    fun prefix(system: String, turns: List<Pair<String, String>>, tpl: Template): String = buildString {
        append(tpl.bos).append("<|im_start|>system\n").append(system).append("<|im_end|>\n")
        for ((u, a) in turns) append("<|im_start|>user\n").append(u).append("<|im_end|>\n").append(tpl.prior).append(a).append("<|im_end|>\n")
        append("<|im_start|>user\n")
    }

    /** The rest of render.render_prompt after [prefix]. */
    fun suffix(user: String, tpl: Template, answerPrefix: String): String = user + "<|im_end|>\n" + tpl.final + answerPrefix
}
