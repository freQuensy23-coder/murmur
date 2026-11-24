package app.murmur.feature.history.ui

import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.style.TextDirection
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import app.murmur.feature.history.DiffSpan
import app.murmur.ui.design.MurmurTheme

/** A range of text drawn with a rounded fill behind it (search `mark`, diff additions) or a struck line over it. */
internal data class Mark(val range: IntRange, val kind: Kind) {
    enum class Kind { FILL, STRIKE }
}

/**
 * Text with the prototype's inline marks: `mark` / `.diff-ins` = accentSoft fill with a 3 dp radius and 1 dp side
 * padding behind accentInk text; `.diff-del` = ink3 text with a line-through in err (Compose's own line-through
 * takes the text colour, so the line is drawn here). [selectable] wraps it in a SelectionContainer (Final text).
 */
@Composable
internal fun MarkedText(
    text: AnnotatedString,
    marks: List<Mark>,
    style: TextStyle,
    color: Color,
    modifier: Modifier = Modifier,
    maxLines: Int = Int.MAX_VALUE,
    selectable: Boolean = false,
) {
    val c = MurmurTheme.colors
    var layout by remember { mutableStateOf<TextLayoutResult?>(null) }
    val fill = c.accentSoft
    val strike = c.err
    val decorated = modifier
        .drawBehind { layout?.let { l -> marks.filter { it.kind == Mark.Kind.FILL }.forEach { drawFill(l, it.range, fill) } } }
        .drawWithContent {
            drawContent()
            layout?.let { l -> marks.filter { it.kind == Mark.Kind.STRIKE }.forEach { drawStrike(l, it.range, strike) } }
        }
    val content = @Composable {
        Text(
            text, modifier = decorated, style = style.copy(textDirection = TextDirection.Content), color = color, maxLines = maxLines, overflow = TextOverflow.Ellipsis,
            onTextLayout = { layout = it },
        )
    }
    if (selectable) SelectionContainer { content() } else content()
}

/** Per-line boxes of [range] (end exclusive) in [l], skipping the part hidden by the line clamp. */
private inline fun forEachLineBox(l: TextLayoutResult, range: IntRange, block: (left: Float, right: Float, line: Int) -> Unit) {
    val len = l.layoutInput.text.length
    val start = range.first.coerceIn(0, len)
    val end = (range.last + 1).coerceIn(0, len)
    if (start >= end) return
    val firstLine = l.getLineForOffset(start)
    val lastLine = l.getLineForOffset(end - 1)
    for (line in firstLine..lastLine) {
        if (line >= l.lineCount) break
        val s = maxOf(start, l.getLineStart(line))
        val e = minOf(end, l.getLineEnd(line, visibleEnd = true))
        if (s >= e) continue
        val boxS = l.getBoundingBox(s)
        val boxE = l.getBoundingBox(e - 1)
        block(minOf(boxS.left, boxE.left), maxOf(boxS.right, boxE.right), line)
    }
}

private fun DrawScope.drawFill(l: TextLayoutResult, range: IntRange, color: Color) {
    val pad = 1.dp.toPx()
    val r = 3.dp.toPx()
    forEachLineBox(l, range) { left, right, line ->
        val top = l.getLineTop(line)
        val bottom = l.getLineBottom(line)
        // The prototype's inline background spans the font's ascent..descent, a little inside the line box.
        val inset = ((bottom - top) - l.layoutInput.style.fontSize.toPx() * FILL_HEIGHT) / 2
        drawRoundRect(color, Offset(left - pad, top + inset.coerceAtLeast(0f)), Size(right - left + 2 * pad, bottom - top - 2 * inset.coerceAtLeast(0f)), CornerRadius(r, r))
    }
}

private fun DrawScope.drawStrike(l: TextLayoutResult, range: IntRange, color: Color) {
    val w = 1.25.dp.toPx()
    forEachLineBox(l, range) { left, right, line ->
        val y = l.getLineBaseline(line) - l.layoutInput.style.fontSize.toPx() * STRIKE_RISE
        drawLine(color, Offset(left, y), Offset(right, y), strokeWidth = w)
    }
}

/** Height of the mark fill relative to the font size (CSS inline box ≈ ascent + descent of Onest). */
private const val FILL_HEIGHT = 1.22f

/** Strike line height above the baseline relative to the font size (about half the x-height). */
private const val STRIKE_RISE = 0.27f

/** [text] with [ranges] as accentInk fills (search highlight). */
@Composable
internal fun highlighted(text: String, ranges: List<IntRange>): Pair<AnnotatedString, List<Mark>> {
    val ink = MurmurTheme.colors.accentInk
    val annotated = buildAnnotatedString {
        append(text)
        for (r in ranges) addStyle(SpanStyle(color = ink), r.first, (r.last + 1).coerceAtMost(text.length))
    }
    return annotated to ranges.map { Mark(it, Mark.Kind.FILL) }
}

/** The Changes view: additions accentInk on accentSoft, removals ink3 with an err line (DESIGN 6.3). */
@Composable
internal fun diffText(spans: List<DiffSpan>): Pair<AnnotatedString, List<Mark>> {
    val c = MurmurTheme.colors
    val marks = ArrayList<Mark>()
    val annotated = buildAnnotatedString {
        for (s in spans) {
            val at = length
            when (s.kind) {
                DiffSpan.Kind.SAME -> append(s.text)
                DiffSpan.Kind.ADDED -> withStyle(SpanStyle(color = c.accentInk)) { append(s.text) }
                DiffSpan.Kind.REMOVED -> withStyle(SpanStyle(color = c.ink3)) { append(s.text) }
            }
            if (s.kind != DiffSpan.Kind.SAME) marks += Mark(at until length, if (s.kind == DiffSpan.Kind.ADDED) Mark.Kind.FILL else Mark.Kind.STRIKE)
        }
    }
    return annotated to marks
}
