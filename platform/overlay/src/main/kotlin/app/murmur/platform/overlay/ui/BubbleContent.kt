package app.murmur.platform.overlay.ui

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedContentTransitionScope
import androidx.compose.animation.ContentTransform
import androidx.compose.animation.SizeTransform
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.CubicBezierEasing
import androidx.compose.animation.core.StartOffset
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.keyframes
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.snap
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.State
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.AbsoluteAlignment
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.RoundRect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.TransformOrigin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.layout.boundsInWindow
import androidx.compose.ui.layout.layout
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.CustomAccessibilityAction
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.customActions
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.onClick
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import app.murmur.core.dictation.DictationBubble
import app.murmur.core.model.ProductName
import app.murmur.platform.overlay.BubbleCopy
import app.murmur.platform.overlay.BubbleMetrics
import app.murmur.platform.overlay.BubbleVisual
import app.murmur.platform.overlay.Live
import app.murmur.platform.overlay.R
import app.murmur.platform.overlay.TouchRegion
import app.murmur.ui.design.MurmurColors
import app.murmur.ui.design.MurmurTheme
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.ceil

/** What the bubble UI reports to its host (the overlay window or a test activity). */
interface BubbleActions {
    /** TalkBack activation of [region] (touches go through the host's gesture recognizer instead). */
    fun tap(region: TouchRegion)
    /** TalkBack custom action "Hide for 1 hour". */
    fun hide()
    /** Bounds of the visual in window pixels (slide-off distance, drag grab offset). */
    fun visualBounds(rect: Rect) {}
    /** Bounds of the Error action button in window pixels; null when the pill has none. */
    fun actionBounds(rect: Rect?) {}
}

/**
 * Everything the bubble window draws. [visual] is never [BubbleVisual.Hidden]: while hidden the host passes the last
 * visible visual with [shown] = false, so it fades out instead of vanishing (and the window keeps a drawable root).
 */
@Immutable
data class BubbleUi(
    val visual: BubbleVisual,
    val shown: Boolean,
    val rightEdge: Boolean,
    val metrics: BubbleMetrics,
    val reducedMotion: Boolean,
    val talkBack: Boolean,
)

internal val LocalDark = staticCompositionLocalOf { false }

/** DESIGN 3.5 easings. */
internal val Decelerate = CubicBezierEasing(0.05f, 0.7f, 0.1f, 1f)
internal val Accelerate = CubicBezierEasing(0.3f, 0f, 0.8f, 0.15f)

/** The bubble window's content: the visual inside its transparent shadow band, with the appear/disappear motion. */
@Composable
fun BubbleWindowContent(
    ui: BubbleUi,
    dark: Boolean,
    levels: State<FloatArray>,
    clock: () -> Long,
    actions: BubbleActions,
    modifier: Modifier = Modifier,
) {
    MurmurTheme(darkTheme = dark) {
        androidx.compose.runtime.CompositionLocalProvider(LocalDark provides dark) {
            val appear = remember { Animatable(if (ui.shown) 1f else 0f) }
            LaunchedEffect(ui.shown, ui.reducedMotion) {
                val target = if (ui.shown) 1f else 0f
                when {
                    ui.reducedMotion -> appear.animateTo(target, tween(150))
                    // DESIGN 3.5 "Bubble appears": scale 0.6 -> 1 and alpha 0 -> 1, spring(400, 0.75), from the docked edge.
                    ui.shown -> appear.animateTo(target, spring(dampingRatio = 0.75f, stiffness = 400f))
                    else -> appear.animateTo(target, tween(150, easing = Accelerate))
                }
            }
            val layer = Modifier.graphicsLayer {
                val a = appear.value
                alpha = a.coerceIn(0f, 1f)
                val s = if (ui.reducedMotion) 1f else 0.6f + 0.4f * a
                scaleX = s; scaleY = s
                transformOrigin = TransformOrigin(if (ui.rightEdge) 1f else 0f, 0.5f)
            }
            val hidden = if (ui.shown) Modifier else Modifier.clearAndSetSemantics { }
            Box(modifier.then(layer).then(hidden)) {
                if (ui.visual == BubbleVisual.Rest) RestTab(ui, actions) else Pill(ui, levels, clock, actions)
            }
        }
    }
}

/** DESIGN 5.3 Rest: inverse edge tab 22 x 60 with a 14 dp mark, hit 48 x 60, flush with the docked edge. */
@Composable
private fun RestTab(ui: BubbleUi, actions: BubbleActions) {
    val c = MurmurTheme.colors
    val inner = 12.dp
    val shape = if (ui.rightEdge) RoundedCornerShape(topStart = inner, bottomStart = inner)
    else RoundedCornerShape(topEnd = inner, bottomEnd = inner)
    val desc = stringResource(R.string.bubble_rest_cd, ProductName.NAME)
    val hide = stringResource(R.string.bubble_hide_action)
    Box(
        Modifier
            .padding(vertical = BubbleMetrics.TAB_PAD_Y.dp)
            .size(BubbleMetrics.TAB_HIT_WIDTH.dp, BubbleMetrics.TAB_HEIGHT.dp)
            .testTag(TAG_BUBBLE)
            .clearAndSetSemantics {
                contentDescription = desc
                role = Role.Button
                onClick { actions.tap(TouchRegion.BODY); true }
                if (ui.talkBack) customActions = listOf(CustomAccessibilityAction(hide) { actions.hide(); true })
            },
        contentAlignment = if (ui.rightEdge) Alignment.CenterEnd else Alignment.CenterStart,
    ) {
        Box(
            Modifier
                .size(BubbleMetrics.TAB_WIDTH.dp, BubbleMetrics.TAB_HEIGHT.dp)
                // e1 (1 dp): an elevation shadow fits the tab's 2 dp band.
                .shadow(1.dp, shape)
                // Prototype `.bub.tab` opacity .92 on the fill; the mark stays fully opaque for legibility.
                .background(c.inverse.copy(alpha = 0.92f), shape)
                .onGloballyPositioned { actions.visualBounds(it.boundsInWindow()) },
            contentAlignment = Alignment.Center,
        ) {
            Icon(BubbleIcons.mark, null, Modifier.size(BubbleMetrics.TAB_MARK.dp), tint = c.onInverse)
        }
    }
}

private enum class Face { CIRCLE, LISTEN, HOLD, TRANSCRIBING, POLISHING, DONE, COPIED, ERROR }

private fun faceOf(v: BubbleVisual): Face = when (v) {
    BubbleVisual.Ready, BubbleVisual.Rest, BubbleVisual.Hidden, is BubbleVisual.Dragging -> Face.CIRCLE
    is BubbleVisual.Active -> when (val s = v.state) {
        is DictationBubble.Listening -> if (s.hold) Face.HOLD else Face.LISTEN
        DictationBubble.Transcribing -> Face.TRANSCRIBING
        DictationBubble.Polishing -> Face.POLISHING
        is DictationBubble.Inserted -> Face.DONE
        is DictationBubble.Copied -> Face.COPIED
        is DictationBubble.Error -> Face.ERROR
        DictationBubble.Idle -> Face.CIRCLE
    }
}

private fun fill(face: Face, c: MurmurColors): Color = when (face) {
    Face.LISTEN, Face.HOLD -> c.accent
    Face.TRANSCRIBING, Face.POLISHING -> c.inverse
    Face.CIRCLE, Face.DONE, Face.COPIED, Face.ERROR -> c.surface
}

/** Surface visuals carry a 1 dp ring (prototype: line, line2 for the dark Ready circle). */
private fun ring(face: Face, c: MurmurColors, dark: Boolean): Color = when (face) {
    Face.CIRCLE -> if (dark) c.line2 else c.line
    Face.DONE, Face.COPIED, Face.ERROR -> c.line
    else -> Color.Transparent
}

private val PILL_SHAPE = RoundedCornerShape(percent = 50)

private fun AnimatedContentTransitionScope<BubbleVisual>.morph(reduced: Boolean): ContentTransform =
    if (reduced) {
        // DESIGN 3.5 reduced motion: 150 ms cross-fades, no morph.
        (fadeIn(tween(150)) togetherWith fadeOut(tween(150))) using SizeTransform(clip = true) { _, _ -> snap() }
    } else {
        // Layout size snaps inside a held-bounds container while the pill shape springs in the draw phase:
        // avoids per-frame WRAP_CONTENT WindowManager relayouts and SurfaceFlinger BLAST buffer reallocations.
        (fadeIn(tween(150, delayMillis = 60)) togetherWith fadeOut(tween(60))) using
            SizeTransform(clip = false) { _, _ -> snap() }
    }

/**
 * The circle or a pill inside its [BubbleMetrics.shadowPad] band. [lifted]: drawn by the drag layer (DESIGN 5.3
 * Dragging, e3 instead of e2; the caller scales it 1.08 and clears its semantics).
 */
@Composable
internal fun Pill(ui: BubbleUi, levels: State<FloatArray>, clock: () -> Long, actions: BubbleActions, lifted: Boolean = false) {
    val c = MurmurTheme.colors
    val dark = LocalDark.current
    val m = ui.metrics
    val face = faceOf(ui.visual)
    val colorSpec = if (ui.reducedMotion) tween<Color>(150) else tween(150, delayMillis = 60)
    // Keep State objects rather than delegating with `by` so color frames redraw without recomposing Pill().
    val bgState = animateColorAsState(fill(face, c), colorSpec, label = "bubbleFill")
    val ringState = animateColorAsState(ring(face, c, dark), colorSpec, label = "bubbleRing")
    val reach = with(LocalDensity.current) { m.shadowPad.dp.toPx() }
    val shadowPaints = remember(dark, reach, lifted) {
        if (lifted || reach <= 0f) null else BubbleShadowPaints(dark, reach)
    }
    val clipPath = remember { Path() }
    var targetSize by remember { mutableStateOf(IntSize.Zero) }
    var holdSize by remember { mutableStateOf(IntSize.Zero) }
    val animW = remember { Animatable(0f) }
    val animH = remember { Animatable(0f) }

    LaunchedEffect(targetSize, ui.reducedMotion, m) {
        if (targetSize.width <= 0 || targetSize.height <= 0) return@LaunchedEffect
        val tw = targetSize.width.toFloat()
        val th = targetSize.height.toFloat()
        if (ui.reducedMotion || animW.value <= 0f || animH.value <= 0f) {
            animW.snapTo(tw)
            animH.snapTo(th)
            holdSize = targetSize
            return@LaunchedEffect
        }
        // Hold the window at max(current, target) for the whole spring so WindowManager never relayouts mid-animation:
        // growing does 1 relayout at frame 0; shrinking does 0 at frame 0 and 1 invisible relayout after settling.
        holdSize = IntSize(
            maxOf(holdSize.width, targetSize.width, ceil(animW.value).toInt()),
            maxOf(holdSize.height, targetSize.height, ceil(animH.value).toInt()),
        )
        coroutineScope {
            val spec = spring<Float>(dampingRatio = 0.9f, stiffness = 1500f)
            if (animW.value != tw) launch { animW.animateTo(tw, spec) }
            if (animH.value != th) launch { animH.animateTo(th, spec) }
        }
        holdSize = targetSize
    }

    // Error carries two nodes (reason and action button) inside; every other face is one node, laid over the pill at
    // >= 48 dp inside the band (DESIGN 5.6: the hit area is never below 48 dp, also at Small).
    val ownSemantics = face != Face.ERROR
    Box {
        Box(
            Modifier
                .padding(m.shadowPad.dp)
                .then(if (lifted) Modifier.shadow(12.dp, PILL_SHAPE, clip = false) else Modifier)
                .drawWithContent {
                    val w = (if (animW.value > 0f) animW.value else size.width).coerceIn(0f, size.width)
                    val h = (if (animH.value > 0f) animH.value else size.height).coerceIn(0f, size.height)
                    val left = if (ui.rightEdge) size.width - w else 0f
                    val top = (size.height - h) / 2f
                    val radius = minOf(w, h) / 2f
                    if (shadowPaints != null && w > 0f && h > 0f) {
                        drawIntoCanvas { canvas ->
                            shadowPaints.drawRoundRect(canvas.nativeCanvas, left, top, left + w, top + h, radius)
                        }
                    }
                    if (w > 0f && h > 0f) {
                        drawRoundRect(
                            color = bgState.value,
                            topLeft = Offset(left, top),
                            size = Size(w, h),
                            cornerRadius = CornerRadius(radius, radius),
                        )
                        clipPath.rewind()
                        clipPath.addRoundRect(RoundRect(left, top, left + w, top + h, CornerRadius(radius, radius)))
                        clipPath(clipPath) {
                            this@drawWithContent.drawContent()
                        }
                        val ring = ringState.value
                        if (ring.alpha > 0f) {
                            val stroke = 1.dp.toPx()
                            val half = stroke / 2f
                            drawRoundRect(
                                color = ring,
                                topLeft = Offset(left + half, top + half),
                                size = Size((w - stroke).coerceAtLeast(0f), (h - stroke).coerceAtLeast(0f)),
                                cornerRadius = CornerRadius((radius - half).coerceAtLeast(0f), (radius - half).coerceAtLeast(0f)),
                                style = Stroke(width = stroke),
                            )
                        }
                    }
                }
                .layout { measurable, constraints ->
                    val p = measurable.measure(constraints)
                    val layoutW = maxOf(p.width, holdSize.width)
                    val layoutH = maxOf(p.height, holdSize.height)
                    layout(layoutW, layoutH) {
                        val x = if (ui.rightEdge) layoutW - p.width else 0
                        val y = (layoutH - p.height) / 2
                        p.place(x, y)
                    }
                },
        ) {
            AnimatedContent(
                targetState = ui.visual,
                contentKey = ::faceOf,
                modifier = Modifier
                    .testTag(TAG_VISUAL)
                    .onSizeChanged { targetSize = it }
                    .then(if (ownSemantics) Modifier.clearAndSetSemantics { } else Modifier)
                    .onGloballyPositioned { actions.visualBounds(it.boundsInWindow()) },
                contentAlignment = if (ui.rightEdge) AbsoluteAlignment.CenterRight else AbsoluteAlignment.CenterLeft,
                transitionSpec = { morph(ui.reducedMotion) },
                label = "bubble",
            ) { v ->
                when (faceOf(v)) {
                    Face.CIRCLE -> ReadyCircle(ui)
                    Face.LISTEN -> ListeningPill((v as BubbleVisual.Active).state as DictationBubble.Listening, ui, levels, clock)
                    Face.HOLD -> HoldPill(ui, levels)
                    Face.TRANSCRIBING -> WorkingPill(ui, R.string.bubble_transcribing) { TranscribingBars(m) }
                    Face.POLISHING -> WorkingPill(ui, R.string.bubble_polishing) { PolishingDots(m, ui.reducedMotion) }
                    Face.DONE -> {
                        val inserted = (v as BubbleVisual.Active).state as DictationBubble.Inserted
                        DonePill(ui, if (inserted.noCleanup) R.string.bubble_inserted_nocleanup else R.string.bubble_inserted, BubbleIcons.check, c.ok)
                    }
                    Face.COPIED -> DonePill(ui, R.string.bubble_copied, BubbleIcons.clipboard, c.accentInk)
                    Face.ERROR -> ErrorPill((v as BubbleVisual.Active).state as DictationBubble.Error, ui, actions)
                }
            }
        }
        val text = faceText(ui, face)
        if (text != null) {
            Box(
                Modifier.matchParentSize().touchTargetInside(m.shadowPad.dp, BubbleMetrics.TOUCH_MIN.dp).bubbleSemantics(
                    text.description, text.state, text.live, if (text.activates) ({ actions.tap(TouchRegion.BODY) }) else null,
                    text.hide, actions::hide,
                ),
            )
        }
    }
}

/**
 * The visual's own rectangle (the parent minus the [band] on every side), grown to at least [min] while it fits in the
 * parent, centred. It stays inside the window content, so the accessibility bounds (clipped to the ancestors) keep it.
 */
private fun Modifier.touchTargetInside(band: androidx.compose.ui.unit.Dp, min: androidx.compose.ui.unit.Dp): Modifier = layout { measurable, c ->
    val pad = band.roundToPx()
    val minPx = min.roundToPx()
    fun span(outer: Int) = (outer - 2 * pad).coerceAtLeast(0).let { maxOf(it, minPx) }.coerceAtMost(outer)
    val w = span(c.maxWidth)
    val h = span(c.maxHeight)
    val p = measurable.measure(Constraints.fixed(w, h))
    layout(c.maxWidth, c.maxHeight) { p.place((c.maxWidth - w) / 2, (c.maxHeight - h) / 2) }
}

/** DESIGN 5.3 TalkBack texts of one face. */
private class FaceText(val description: String, val state: String?, val live: Live, val activates: Boolean, val hide: String?)

/** The texts of every face but Error, whose two nodes (reason, action button) live inside its pill. */
@Composable
private fun faceText(ui: BubbleUi, face: Face): FaceText? {
    return when (face) {
        Face.CIRCLE -> FaceText(stringResource(R.string.bubble_ready_cd), null, Live.OFF, activates = true,
            hide = if (ui.talkBack) stringResource(R.string.bubble_hide_action) else null)
        Face.LISTEN -> FaceText(stringResource(R.string.bubble_finish_cd), stringResource(R.string.bubble_listening), Live.POLITE, true, null)
        Face.HOLD -> FaceText(stringResource(R.string.bubble_hold_live), null, Live.POLITE, false, null)
        Face.TRANSCRIBING -> FaceText(stringResource(R.string.bubble_transcribing), null, Live.POLITE, false, null)
        Face.POLISHING -> FaceText(stringResource(R.string.bubble_polishing), null, Live.POLITE, false, null)
        Face.DONE, Face.COPIED -> BubbleCopy.of(ui.visual)!!.let { FaceText(stringResource(it.description), null, it.live, true, null) }
        Face.ERROR -> null
    }
}

/** Semantics of a bubble body: DESIGN 5.3 TalkBack text, live region, activation. */
private fun Modifier.bubbleSemantics(
    description: String,
    state: String?,
    live: Live,
    onActivate: (() -> Unit)?,
    hideLabel: String?,
    onHide: () -> Unit,
): Modifier = testTag(TAG_BUBBLE).clearAndSetSemantics {
    contentDescription = description
    if (state != null) stateDescription = state
    when (live) {
        Live.POLITE -> liveRegion = LiveRegionMode.Polite
        Live.ASSERTIVE -> liveRegion = LiveRegionMode.Assertive
        Live.OFF -> Unit
    }
    if (onActivate != null) { role = Role.Button; onClick { onActivate(); true } }
    if (hideLabel != null) customActions = listOf(CustomAccessibilityAction(hideLabel) { onHide(); true })
}

/** DESIGN 5.3 Ready: surface circle, 26 dp ink mark, e2 + 1 dp line. */
@Composable
private fun ReadyCircle(ui: BubbleUi) {
    val m = ui.metrics
    Box(Modifier.size(m.diameter.dp), contentAlignment = Alignment.Center) {
        Icon(BubbleIcons.mark, null, Modifier.size(m.markReady.dp), tint = MurmurTheme.colors.ink)
    }
}

/**
 * DESIGN 5.3 Listening: accent pill 164 x 52, 7 live level bars, mono timer (warn from 4:30), 40 dp stop circle.
 * The whole pill finishes; the cancel circle above it is a separate window ([CancelButton]).
 */
@Composable
private fun ListeningPill(
    s: DictationBubble.Listening,
    ui: BubbleUi,
    levels: State<FloatArray>,
    clock: () -> Long,
) {
    val c = MurmurTheme.colors
    val m = ui.metrics
    Row(
        Modifier
            .size(m.listeningWidth.dp, m.pillHeight.dp)
            .padding(start = m.listenPadStart.dp, end = m.listenPadEnd.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        LevelBars(levels, m, c.onAccent, ui.reducedMotion)
        Spacer(Modifier.width(m.contentGap.dp))
        Box(Modifier.weight(1f)) { Timer(s, m, clock) }
        Spacer(Modifier.width(m.contentGap.dp))
        Box(
            Modifier.size(m.stopDiameter.dp).background(if (LocalDark.current) STOP_FILL_DARK else STOP_FILL_LIGHT, CircleShape),
            contentAlignment = Alignment.Center,
        ) {
            Icon(BubbleIcons.stop, null, Modifier.size((18f * m.scale).dp), tint = c.onAccent)
        }
    }
}

/** Prototype `.bub .stop`: white 18% on the light accent, rgba(11,16,51,.16) on the dark accent. */
private val STOP_FILL_LIGHT = Color(0x2EFFFFFF)
private val STOP_FILL_DARK = Color(0x290B1033)

/** m:ss since the press; frozen at the stop for the first 300 ms of Finishing (BubbleMapping). */
@Composable
private fun Timer(s: DictationBubble.Listening, m: BubbleMetrics, clock: () -> Long) {
    val c = MurmurTheme.colors
    var now by remember(s.startedAtMs) { mutableLongStateOf(clock()) }
    LaunchedEffect(s.startedAtMs, s.frozenAtMs) {
        if (s.frozenAtMs != null) return@LaunchedEffect
        while (true) {
            now = clock()
            val elapsed = (now - s.startedAtMs).coerceAtLeast(0L)
            delay(1_000L - elapsed % 1_000L)
        }
    }
    val text = formatElapsed((s.frozenAtMs ?: now) - s.startedAtMs)
    if (s.warn) {
        // DESIGN 5.5: the timer turns warn at 4:30. warn on warnSoft keeps >= 4.7:1 on the accent pill.
        Text(text, style = BubbleType.timer(m.scale), color = c.warn, maxLines = 1, softWrap = false,
            modifier = Modifier.background(c.warnSoft, RoundedCornerShape(6.dp)).padding(horizontal = 4.dp))
    } else {
        Text(text, style = BubbleType.timer(m.scale), color = c.onAccent, maxLines = 1, softWrap = false)
    }
}

internal fun formatElapsed(ms: Long): String {
    val total = (ms.coerceAtLeast(0L) / 1_000L)
    return "%d:%02d".format(total / 60, total % 60)
}

/** DESIGN 5.3 Hold to talk: accent pill, bars plus "Release to finish". */
@Composable
private fun HoldPill(ui: BubbleUi, levels: State<FloatArray>) {
    val c = MurmurTheme.colors
    val m = ui.metrics
    val label = stringResource(R.string.bubble_hold)
    Row(
        Modifier
            .heightIn(min = m.pillHeight.dp)
            .widthIn(max = m.maxWidth.dp)
            .padding(start = m.listenPadStart.dp, end = m.holdPadEnd.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        LevelBars(levels, m, c.onAccent, ui.reducedMotion)
        Spacer(Modifier.width(m.contentGap.dp))
        Label(label, c.onAccent, m)
    }
}

/** DESIGN 5.3 Transcribing / Polishing: inverse pill, a quiet indicator plus the word; taps ignored. */
@Composable
private fun WorkingPill(ui: BubbleUi, label: Int, indicator: @Composable () -> Unit) {
    val c = MurmurTheme.colors
    val m = ui.metrics
    val text = stringResource(label)
    Row(
        Modifier
            .heightIn(min = m.pillHeight.dp)
            .widthIn(max = m.maxWidth.dp)
            .padding(start = m.workPadStart.dp, end = m.workPadEnd.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        indicator()
        Spacer(Modifier.width(m.contentGap.dp))
        Label(text, c.onInverse, m)
    }
}

/** DESIGN 5.3 Inserted / Inserted, no cleanup / Copied only: surface pill, status icon plus the word. */
@Composable
private fun DonePill(
    ui: BubbleUi,
    label: Int,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    tint: Color,
) {
    val c = MurmurTheme.colors
    val m = ui.metrics
    val text = stringResource(label)
    Row(
        Modifier
            .heightIn(min = m.pillHeight.dp)
            .widthIn(max = m.maxWidth.dp)
            .padding(start = m.donePadStart.dp, end = m.donePadEnd.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(icon, null, Modifier.size(m.statusIcon.dp), tint = tint)
        Spacer(Modifier.width(m.contentGap.dp))
        Label(text, c.ink, m)
    }
}

/** DESIGN 5.3 / 5.5 Error: surface pill, err alert, short reason, optional 40 dp tonal action (48 dp target). */
@Composable
private fun ErrorPill(
    s: DictationBubble.Error,
    ui: BubbleUi,
    actions: BubbleActions,
) {
    val c = MurmurTheme.colors
    val m = ui.metrics
    val reason = stringResource(BubbleCopy.errorLabel(s.reason))
    Row(
        Modifier
            .heightIn(min = m.pillHeight.dp)
            .widthIn(max = m.maxWidth.dp)
            .padding(start = m.donePadStart.dp, end = if (s.action != null) m.errPadEnd.dp else m.donePadEnd.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Row(
            Modifier.weight(1f, fill = false).heightIn(min = BubbleMetrics.TOUCH_MIN.dp)
                .bubbleSemantics(reason, null, Live.ASSERTIVE, { actions.tap(TouchRegion.BODY) }, null) {},
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(BubbleIcons.alert, null, Modifier.size(m.statusIcon.dp), tint = c.err)
            Spacer(Modifier.width(m.contentGap.dp))
            Label(reason, c.ink, m)
        }
        val action = s.action
        if (action == null) {
            LaunchedEffect(Unit) { actions.actionBounds(null) }
        } else {
            val label = stringResource(BubbleCopy.actionLabel(action))
            Spacer(Modifier.width(m.contentGap.dp))
            Box(
                Modifier
                    .height(BubbleMetrics.TOUCH_MIN.dp.coerceAtLeast(m.actionHeight.dp))
                    .testTag(TAG_ACTION)
                    .onGloballyPositioned { actions.actionBounds(it.boundsInWindow()) }
                    .clearAndSetSemantics {
                        contentDescription = label
                        role = Role.Button
                        onClick { actions.tap(TouchRegion.ACTION); true }
                    },
                contentAlignment = Alignment.Center,
            ) {
                Box(
                    Modifier.height(m.actionHeight.dp).background(c.surface2, RoundedCornerShape(percent = 50))
                        .padding(horizontal = (14f * m.scale).dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(label, style = BubbleType.action(m.scale), color = c.ink, maxLines = 1)
                }
            }
        }
    }
}

/** Bubble labels wrap to 2 lines at large font scales; the pill then grows (DESIGN 8). */
@Composable
private fun Label(text: String, color: Color, m: BubbleMetrics) {
    Text(text, style = BubbleType.label(m.scale), color = color, maxLines = 2, overflow = TextOverflow.Ellipsis,
        modifier = Modifier.padding(vertical = 4.dp))
}

/**
 * DESIGN 3.5 level bars: 7 bars of 3 dp, 3 dp apart, 4-24 dp tall, heights straight from the LevelMeter (which
 * smooths with 40/120 ms attack/release). Silence is flat: no fake animation. Reduced motion: a static 3-step meter
 * (three fixed bars that light up with the level, nothing moves).
 */
@Composable
internal fun LevelBars(levels: State<FloatArray>, m: BubbleMetrics, color: Color, reducedMotion: Boolean) {
    val w = m.barCount * m.barWidth + (m.barCount - 1) * m.barGap
    Canvas(Modifier.size(w.dp, m.barMax.dp)) {
        val l = levels.value
        val bw = m.barWidth.dp.toPx()
        val gap = m.barGap.dp.toPx()
        val r = CornerRadius(2.dp.toPx())
        if (reducedMotion) {
            val level = l.maxOrNull() ?: 0f
            val lit = MeterSteps.lit(level)
            for (i in 0 until 3) {
                val h = (m.barMax * (i + 1) / 3f).dp.toPx()
                drawRoundRect(color.copy(alpha = if (i < lit) 1f else 0.35f), Offset(i * (bw + gap), (size.height - h) / 2f), Size(bw, h), r)
            }
        } else {
            for (i in 0 until m.barCount) {
                val h = m.barHeight(l.getOrElse(i) { 0f }).dp.toPx()
                drawRoundRect(color, Offset(i * (bw + gap), (size.height - h) / 2f), Size(bw, h), r)
            }
        }
    }
}

/** Static 3-step meter of reduced motion: how many of the three bars are lit for a 0..1 level. */
internal object MeterSteps {
    fun lit(level: Float): Int = when {
        level < 0.05f -> 0
        level < 0.35f -> 1
        level < 0.7f -> 2
        else -> 3
    }
}

/** Transcribing: five dimmed static bars (prototype heights 6/10/14/10/6 at 60%). */
@Composable
private fun TranscribingBars(m: BubbleMetrics) {
    val color = MurmurTheme.colors.onInverse.copy(alpha = 0.6f)
    val heights = floatArrayOf(6f, 10f, 14f, 10f, 6f)
    val w = heights.size * m.barWidth + (heights.size - 1) * m.barGap
    Canvas(Modifier.size(w.dp, m.barMax.dp)) {
        val bw = m.barWidth.dp.toPx()
        val gap = m.barGap.dp.toPx()
        heights.forEachIndexed { i, hDp ->
            val h = (hDp * m.scale).dp.toPx()
            drawRoundRect(color, Offset(i * (bw + gap), (size.height - h) / 2f), Size(bw, h), CornerRadius(2.dp.toPx()))
        }
    }
}

/** Polishing: three 5 dp dots pulsing 35% -> 100% -> 35% over 1.2 s, 150 ms apart; static under reduced motion. */
@Composable
private fun PolishingDots(m: BubbleMetrics, reducedMotion: Boolean) {
    val color = MurmurTheme.colors.onInverse
    val d = 5f * m.scale
    Row(horizontalArrangement = Arrangement.spacedBy((4f * m.scale).dp), verticalAlignment = Alignment.CenterVertically) {
        if (reducedMotion) {
            repeat(3) { Box(Modifier.size(d.dp).background(color.copy(alpha = 0.6f), CircleShape)) }
        } else {
            val t = rememberInfiniteTransition(label = "dots")
            repeat(3) { i ->
                val a by t.animateFloat(
                    0.35f, 0.35f,
                    infiniteRepeatable(
                        keyframes {
                            durationMillis = 1_200
                            0.35f at 0
                            1f at 360
                            0.35f at 720
                        },
                        initialStartOffset = StartOffset(150 * i),
                    ),
                    label = "dot$i",
                )
                Box(Modifier.size(d.dp).graphicsLayer { alpha = a }.background(color, CircleShape))
            }
        }
    }
}

const val TAG_BUBBLE = "bubble"
/** The pill / circle shape itself (its bounds are the visual, without the window's shadow band). */
const val TAG_VISUAL = "bubble.visual"
const val TAG_ACTION = "bubble.action"
const val TAG_CANCEL = "bubble.cancel"
