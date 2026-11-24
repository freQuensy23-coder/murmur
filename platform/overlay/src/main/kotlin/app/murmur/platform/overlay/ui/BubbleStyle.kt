package app.murmur.platform.overlay.ui

import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithCache
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Outline
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.graphics.asAndroidPath
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.text.PlatformTextStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.LineHeightStyle
import androidx.compose.ui.unit.sp
import app.murmur.ui.design.MurmurFonts
import app.murmur.ui.design.TokenType
import app.murmur.ui.design.TypeSpec

/**
 * Bubble text styles from tokens.json (DESIGN 3.2: bubble 14/18 600, monoTimer 14/18 500 tnum) in the fonts that
 * :ui:design bundles (Onest, JetBrains Mono).
 */
internal object BubbleType {
    val sans: FontFamily = MurmurFonts.sans
    val mono: FontFamily = MurmurFonts.mono

    fun style(spec: TypeSpec, scale: Float = 1f): TextStyle = TextStyle(
        fontFamily = if (spec.mono) mono else sans,
        fontWeight = FontWeight(spec.weight),
        fontSize = (spec.size * scale).sp,
        lineHeight = (spec.line * scale).sp,
        letterSpacing = spec.tracking.sp,
        fontFeatureSettings = if (spec.tabular) "tnum" else null,
        platformStyle = PlatformTextStyle(includeFontPadding = false),
        lineHeightStyle = LineHeightStyle(LineHeightStyle.Alignment.Center, LineHeightStyle.Trim.None),
    )

    fun label(scale: Float) = style(TokenType.bubble, scale)
    fun timer(scale: Float) = style(TokenType.monoTimer, scale)
    /** Error action button: prototype `.bub .act` 600 13/16. */
    fun action(scale: Float) = style(TypeSpec(13, 16, 600, 0f, mono = false, uppercase = false, tabular = false), scale)
    /** Hide-target label: prototype `.target .lbl` 500 13/16. */
    val targetLabel = style(TypeSpec(13, 16, 500, 0f, mono = false, uppercase = false, tabular = false))
    val snackBody = style(TokenType.body)
    val snackButton = style(TokenType.button)
}

/**
 * The bubble's e2 shadow, drawn inside its window's transparent band: a window clips everything outside it, so an
 * elevation shadow would end in a hard edge. tokens.json e2 is (y 2, blur 6, 8%) + (y 8, blur 24, 10%); both layers
 * are compressed so that offset + blur stays within [reachPx] (the 6 dp band), keeping their alphas. Dark theme uses
 * the prototype's dark shadow alphas (40% / 45%).
 */
internal fun Modifier.bubbleShadow(shape: Shape, dark: Boolean, reachPx: Float, enabled: Boolean = true): Modifier =
    if (!enabled || reachPx <= 0f) this else drawWithCache {
        val outline = shape.createOutline(size, layoutDirection, this)
        val fillColor = Color.Black.copy(alpha = 0.004f).toArgb()
        val a1 = Color.Black.copy(alpha = if (dark) 0.40f else 0.08f).toArgb()
        val a2 = Color.Black.copy(alpha = if (dark) 0.45f else 0.10f).toArgb()
        val p1 = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            color = fillColor
            setShadowLayer(reachPx * 0.50f, 0f, reachPx * 0.15f, a1)
        }
        val p2 = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            color = fillColor
            setShadowLayer(reachPx * 0.65f, 0f, reachPx * 0.35f, a2)
        }
        onDrawBehind {
            drawIntoCanvas { canvas ->
                val nc = canvas.nativeCanvas
                drawOutline(nc, outline, size, p1)
                drawOutline(nc, outline, size, p2)
            }
        }
    }

private fun drawOutline(c: android.graphics.Canvas, outline: Outline, size: Size, paint: android.graphics.Paint) {
    when (outline) {
        is Outline.Rectangle -> c.drawRect(0f, 0f, size.width, size.height, paint)
        is Outline.Rounded -> {
            // One uniform round rect (HWUI blurs it on the GPU; a shadowed path stalled the emulator's renderer).
            // A side whose two corners are square is the docked screen edge (the Rest tab): the rect runs past it by the
            // radius, and the window edge clips it, so the visible shape is the same.
            val r = outline.roundRect
            val radius = maxOf(r.topLeftCornerRadius.x, r.topRightCornerRadius.x, r.bottomLeftCornerRadius.x, r.bottomRightCornerRadius.x)
            val left = if (r.topLeftCornerRadius.x == 0f && r.bottomLeftCornerRadius.x == 0f) r.left - radius else r.left
            val right = if (r.topRightCornerRadius.x == 0f && r.bottomRightCornerRadius.x == 0f) r.right + radius else r.right
            c.drawRoundRect(left, r.top, right, r.bottom, radius, radius, paint)
        }
        is Outline.Generic -> c.drawPath(outline.path.asAndroidPath(), paint)
    }
}
