package app.murmur.platform.overlay.ui

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.VectorConverter
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.snap
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.State
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.TransformOrigin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.layout
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import app.murmur.core.dictation.DictationBubble
import app.murmur.platform.overlay.BubbleMetrics
import app.murmur.platform.overlay.BubblePlacement
import app.murmur.platform.overlay.BubbleVisual
import app.murmur.platform.overlay.R
import app.murmur.platform.overlay.TouchRegion
import app.murmur.ui.design.MurmurTheme
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

/**
 * Floating controls above the Listening pill: always the x cancel button, and when Hebrew is enabled with an installed
 * Hebrew ASR model, the clean "עב" toggle circle right next to it (docked side keeps x nearest the screen edge).
 */
@Composable
fun ListeningFloatingControls(
    dark: Boolean,
    shown: Boolean,
    rightEdge: Boolean,
    hebrewAvailable: Boolean,
    hebrewActive: Boolean,
    onCancel: () -> Unit,
    onToggleHebrew: () -> Unit,
) {
    if (!hebrewAvailable) {
        CancelButton(dark, shown, onCancel)
        return
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        if (rightEdge) {
            HebrewToggleButton(dark, shown, hebrewActive, onToggleHebrew)
            CancelButton(dark, shown, onCancel)
        } else {
            CancelButton(dark, shown, onCancel)
            HebrewToggleButton(dark, shown, hebrewActive, onToggleHebrew)
        }
    }
}

/**
 * Floating Hebrew ASR toggle button above the Listening pill: strictly "עב" (no localized label text). Active state
 * uses inverse fill + accent ring.
 */
@Composable
fun HebrewToggleButton(dark: Boolean, shown: Boolean, active: Boolean, onToggle: () -> Unit) {
    MurmurTheme(darkTheme = dark) {
        val desc = stringResource(if (active) R.string.bubble_hebrew_on_cd else R.string.bubble_hebrew_off_cd)
        val alpha by animateFloatAsState(if (shown) 1f else 0f, tween(150), label = "hebrew")
        val reach = with(LocalDensity.current) { ((BubbleMetrics.CANCEL_HIT - BubbleMetrics.CANCEL_DIAMETER) / 2f).dp.toPx() }
        Box(
            Modifier
                .size(BubbleMetrics.CANCEL_HIT.dp)
                .graphicsLayer { this.alpha = alpha }
                .testTag(TAG_HEBREW)
                .then(
                    if (shown) Modifier
                        .semantics { contentDescription = desc; selected = active }
                        .clickable(role = Role.Button, onClick = onToggle)
                    else Modifier.clearAndSetSemantics { },
                ),
            contentAlignment = Alignment.Center,
        ) {
            HebrewCircle(active, Modifier.bubbleShadow(CircleShape, dark, reach))
        }
    }
}

/** Clean "עב" circle (40 dp): surface when off, inverse fill + 2 dp accent ring when active. */
@Composable
private fun HebrewCircle(active: Boolean, shadow: Modifier) {
    val c = MurmurTheme.colors
    Box(
        Modifier
            .size(BubbleMetrics.CANCEL_DIAMETER.dp)
            .then(shadow)
            .background(if (active) c.inverse else c.surface, CircleShape)
            .border(if (active) 2.dp else 1.dp, if (active) c.accent else c.line, CircleShape),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            text = HEBREW_BADGE_TEXT,
            style = BubbleType.label(1f).copy(fontSize = 14.sp, lineHeight = 16.sp, fontWeight = FontWeight.SemiBold),
            color = if (active) c.onInverse else c.ink,
            maxLines = 1,
            softWrap = false,
        )
    }
}

/**
 * DESIGN 5.3 Listening: the cancel circle (40 dp visual, 48 dp hit, surface, x) that floats 50 dp above the pill. It
 * lives in its own small window so that the space beside it stays touchable for the app underneath.
 */
@Composable
fun CancelButton(dark: Boolean, shown: Boolean, onCancel: () -> Unit) {
    MurmurTheme(darkTheme = dark) {
        val c = MurmurTheme.colors
        val desc = stringResource(R.string.bubble_cancel_cd)
        val alpha by animateFloatAsState(if (shown) 1f else 0f, tween(150), label = "cancel")
        val reach = with(LocalDensity.current) { ((BubbleMetrics.CANCEL_HIT - BubbleMetrics.CANCEL_DIAMETER) / 2f).dp.toPx() }
        Box(
            Modifier
                .size(BubbleMetrics.CANCEL_HIT.dp)
                .graphicsLayer { this.alpha = alpha }
                .testTag(TAG_CANCEL)
                .then(
                    if (shown) Modifier.semantics { contentDescription = desc }.clickable(role = Role.Button, onClick = onCancel)
                    else Modifier.clearAndSetSemantics { },
                ),
            contentAlignment = Alignment.Center,
        ) {
            CancelCircle(Modifier.bubbleShadow(CircleShape, dark, reach))
        }
    }
}

/** The cancel circle's visual: 40 dp surface circle with a 1 dp line and the x. */
@Composable
private fun CancelCircle(shadow: Modifier) {
    val c = MurmurTheme.colors
    Box(
        Modifier
            .size(BubbleMetrics.CANCEL_DIAMETER.dp)
            .then(shadow)
            .background(c.surface, CircleShape)
            .border(1.dp, c.line, CircleShape),
        contentAlignment = Alignment.Center,
    ) {
        Icon(BubbleIcons.close, null, Modifier.size(20.dp), tint = c.ink2)
    }
}

/**
 * The full-screen, untouchable layer shown while dragging (DESIGN 5.3 Dragging, 3.5): 14% scrim, dashed accent guides
 * on both snap edges, the ghost where the bubble will land, the lifted bubble (1.08x, e3) and the hide target. All
 * positions are screen pixels of the bubble's display.
 *
 * Dragging the idle bubble lifts the Ready circle and arms the hide target. A working or result pill (DESIGN 5.3:
 * "can only be dragged") is lifted as itself, live bars, timer and cancel circle included, and the hide target stays
 * away ([hideArmed] false).
 */
@Immutable
data class DragScene(
    val circleX: Float,
    val circleY: Float,
    val diameterPx: Int,
    val screenW: Int,
    /** Usable range of the circle's top..bottom (guides span it). */
    val guideTop: Int,
    val guideBottom: Int,
    val ghostX: Int,
    val ghostY: Int,
    val overTarget: Boolean,
    val targetCenterX: Float,
    val targetCenterY: Float,
    /** The target appears after 300 ms of drag. */
    val targetShown: Boolean,
    /** Release: animate the circle to this top-left, then [DragLayer]'s onSettled. Null while the finger is down. */
    val settleTo: Offset?,
    val reducedMotion: Boolean,
    /** The hide target is in play (only for the idle bubble). */
    val hideArmed: Boolean = true,
    /** The working or result pill being moved (its visual, docked side at the drag start); null = the Ready circle. */
    val pill: BubbleUi? = null,
    /** The edge the release snaps to ([settleTo]); the pill re-anchors to it while it settles. */
    val settleRightEdge: Boolean? = null,
)

@Composable
fun DragLayer(scene: DragScene, dark: Boolean, levels: State<FloatArray>, clock: () -> Long, onSettled: () -> Unit) {
    MurmurTheme(darkTheme = dark) {
        CompositionLocalProvider(LocalDark provides dark) {
            val c = MurmurTheme.colors
            val density = LocalDensity.current
            val d = scene.diameterPx.toFloat()
            Box(Modifier.fillMaxSize().clearAndSetSemantics { }) {
                // Scrim: the token scrim (40% light / 56% dark) at 35% = 14% of ink over the app (prototype).
                Box(Modifier.fillMaxSize().background(c.scrim.copy(alpha = c.scrim.alpha * 0.35f)))
                val margin = with(density) { BubblePlacement.EDGE_MARGIN_DP.dp.toPx() }
                Canvas(Modifier.fillMaxSize()) {
                    val dash = PathEffect.dashPathEffect(floatArrayOf(6.dp.toPx(), 4.dp.toPx()))
                    val guide = c.accent.copy(alpha = 0.55f)
                    val sw = 2.dp.toPx()
                    for (x in floatArrayOf(margin, scene.screenW - margin)) {
                        drawLine(guide, Offset(x, scene.guideTop.toFloat()), Offset(x, scene.guideBottom.toFloat()), sw, pathEffect = dash)
                    }
                    if (!scene.overTarget && scene.settleTo == null) {
                        // Ghost: where the bubble lands; trail from the lifted circle to it.
                        drawCircle(c.accent.copy(alpha = 0.8f), d / 2f - sw / 2f, Offset(scene.ghostX + d / 2f, scene.ghostY + d / 2f),
                            style = Stroke(sw, pathEffect = dash))
                        val cy = scene.circleY + d / 2f
                        val fromX = if (scene.ghostX > scene.circleX) scene.circleX + d * 1.08f + 4.dp.toPx() else scene.ghostX + d + 4.dp.toPx()
                        val toX = if (scene.ghostX > scene.circleX) scene.ghostX - 4.dp.toPx() else scene.circleX - 4.dp.toPx()
                        if (toX - fromX > 8.dp.toPx()) {
                            drawLine(c.accent.copy(alpha = 0.6f), Offset(fromX, cy), Offset(toX, cy), sw,
                                pathEffect = PathEffect.dashPathEffect(floatArrayOf(6.dp.toPx(), 6.dp.toPx())))
                        }
                    }
                }
                HideTarget(scene)
                val pill = scene.pill
                if (pill == null) LiftedCircle(scene, onSettled) else LiftedPill(scene, pill, levels, clock, onSettled)
            }
        }
    }
}

@Composable
private fun HideTarget(scene: DragScene) {
    val c = MurmurTheme.colors
    val density = LocalDensity.current
    val shown by animateFloatAsState(if (scene.hideArmed && scene.targetShown && scene.settleTo == null) 1f else 0f,
        if (scene.reducedMotion) tween(150) else tween(240, easing = Decelerate), label = "target")
    val hot by animateFloatAsState(if (scene.overTarget) 1f else 0f, if (scene.reducedMotion) snap() else tween(150), label = "hot")
    val label = stringResource(R.string.bubble_hide_target)
    val size = BubbleMetrics.HIDE_TARGET.dp
    val rise = with(density) { BubbleMetrics.HIDE_TARGET_RISE.dp.toPx() }
    Column(
        Modifier
            .layout { measurable, constraints ->
                val p = measurable.measure(constraints.copy(minWidth = 0, minHeight = 0))
                layout(p.width, p.height) {
                    // The circle's centre is the magnet's centre.
                    val circleHalf = (size.toPx() / 2f)
                    p.place((scene.targetCenterX - p.width / 2f).roundToInt(), (scene.targetCenterY - circleHalf).roundToInt())
                }
            }
            .graphicsLayer { alpha = shown; translationY = (1f - shown) * rise },
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(BubbleMetrics.HIDE_LABEL_GAP.dp),
    ) {
        val scale = 1f + (BubbleMetrics.TARGET_HOT_SCALE - 1f) * hot
        Box(
            Modifier
                .size(size)
                .graphicsLayer { scaleX = scale; scaleY = scale }
                .shadow(6.dp, CircleShape)
                .background(androidx.compose.ui.graphics.lerp(c.surface, c.ink, hot), CircleShape)
                .border(1.dp, c.line2.copy(alpha = 1f - hot), CircleShape),
            contentAlignment = Alignment.Center,
        ) {
            Icon(BubbleIcons.close, null, Modifier.size(28.dp), tint = androidx.compose.ui.graphics.lerp(c.ink, c.bg, hot))
        }
        Text(
            label, style = BubbleType.targetLabel, color = c.onInverse,
            modifier = Modifier.background(c.inverse, RoundedCornerShape(percent = 50)).padding(horizontal = 12.dp, vertical = 6.dp),
        )
    }
}

/** The Ready circle lifted 1.08x with e3; pulled to the target centre inside the magnet; springs to the snap edge on release. */
@Composable
private fun LiftedCircle(scene: DragScene, onSettled: () -> Unit) {
    val c = MurmurTheme.colors
    val dark = LocalDark.current
    val d = scene.diameterPx.toFloat()
    val follow = Offset(scene.circleX, scene.circleY)
    val magnet = Offset(scene.targetCenterX - d / 2f, scene.targetCenterY - d / 2f)
    val pos = remember { Animatable(follow, Offset.VectorConverter) }
    LaunchedEffect(scene) {
        val settle = scene.settleTo
        when {
            settle != null -> {
                // DESIGN 3.5 snap: spring(400 = StiffnessMediumLow, 0.8).
                if (scene.reducedMotion) pos.snapTo(settle) else pos.animateTo(settle, spring(dampingRatio = 0.8f, stiffness = 400f))
                onSettled()
            }
            scene.overTarget -> if (scene.reducedMotion) pos.snapTo(magnet) else pos.animateTo(magnet, spring(dampingRatio = 0.8f, stiffness = 400f))
            else -> pos.snapTo(follow)
        }
    }
    val diameterDp = with(LocalDensity.current) { d.toDp() }
    Box(
        Modifier
            .offset { IntOffset(pos.value.x.roundToInt(), pos.value.y.roundToInt()) }
            .size(diameterDp)
            .graphicsLayer { scaleX = BubbleMetrics.DRAG_LIFT; scaleY = BubbleMetrics.DRAG_LIFT }
            .shadow(12.dp, CircleShape)
            .background(c.surface, CircleShape)
            .border(1.dp, if (dark) c.line2 else c.line, CircleShape),
        contentAlignment = Alignment.Center,
    ) {
        Icon(BubbleIcons.mark, null, Modifier.size(diameterDp * (26f / 52f)), tint = c.ink)
    }
}

/**
 * A working or result pill moved by the finger: drawn as itself, lifted 1.08x with e3, its docked end on the dragged
 * circle position (the pill keeps growing toward the side it was docked to, so it never flips under the finger). On
 * release it springs to the snap edge and re-anchors to it, then the bubble window takes over.
 */
@Composable
private fun LiftedPill(scene: DragScene, pill: BubbleUi, levels: State<FloatArray>, clock: () -> Long, onSettled: () -> Unit) {
    val density = LocalDensity.current
    val d = scene.diameterPx.toFloat()
    val m = pill.metrics
    val follow = Offset(scene.circleX, scene.circleY)
    val pos = remember { Animatable(follow, Offset.VectorConverter) }
    /** 1 = the pill's docked end is the circle's right side (it grows to the left), 0 = its left side. */
    val anchor = remember { Animatable(if (pill.rightEdge) 1f else 0f) }
    LaunchedEffect(scene) {
        val settle = scene.settleTo
        if (settle == null) { pos.snapTo(follow); return@LaunchedEffect }
        val toAnchor = if (scene.settleRightEdge ?: pill.rightEdge) 1f else 0f
        if (scene.reducedMotion) {
            pos.snapTo(settle); anchor.snapTo(toAnchor)
        } else coroutineScope {
            // DESIGN 3.5 snap: spring(400 = StiffnessMediumLow, 0.8).
            launch { anchor.animateTo(toAnchor, spring(dampingRatio = 0.8f, stiffness = 400f)) }
            pos.animateTo(settle, spring(dampingRatio = 0.8f, stiffness = 400f))
        }
        onSettled()
    }
    val band = with(density) { m.shadowPad.dp.toPx() }
    val inset = with(density) { m.cancelInset.dp.toPx() }
    val cancelD = with(density) { BubbleMetrics.CANCEL_DIAMETER.dp.toPx() }
    val cancelUp = with(density) { m.cancelOffsetY.dp.toPx() }
    val listeningState = (pill.visual as? BubbleVisual.Active)?.state as? DictationBubble.Listening
    val listening = listeningState?.hold == false
    val noActions = remember {
        object : BubbleActions {
            override fun tap(region: TouchRegion) = Unit
            override fun hide() = Unit
        }
    }
    val shadowReach = with(density) { ((BubbleMetrics.CANCEL_HIT - BubbleMetrics.CANCEL_DIAMETER) / 2f).dp.toPx() }
    Box(
        Modifier
            .fillMaxSize()
            .testTag(TAG_LIFTED)
            .layout { measurable, constraints ->
                val p = measurable.measure(constraints.copy(minWidth = 0, minHeight = 0))
                layout(constraints.maxWidth, constraints.maxHeight) {
                    val a = anchor.value
                    // Window-content box (visual + band) whose docked end sits on the circle, centred on its centre line.
                    val leftAnchored = pos.value.x - band
                    val rightAnchored = pos.value.x + d + band - p.width
                    p.place(
                        (leftAnchored + (rightAnchored - leftAnchored) * a).roundToInt(),
                        (pos.value.y + d / 2f - p.height / 2f).roundToInt(),
                    )
                }
            },
    ) {
        Box(Modifier.graphicsLayer {
            scaleX = BubbleMetrics.DRAG_LIFT; scaleY = BubbleMetrics.DRAG_LIFT
            transformOrigin = TransformOrigin(anchor.value, 0.5f)
        }) {
            Pill(pill, levels, clock, noActions, lifted = true)
        }
    }
    if (listening) {
        val hebrewAvailable = listeningState?.hebrewAvailable == true
        val hebrewActive = listeningState?.hebrewActive == true
        val gap = with(density) { (BubbleMetrics.CANCEL_HIT - BubbleMetrics.CANCEL_DIAMETER).dp.toPx() }
        val totalW = if (hebrewAvailable) cancelD * 2f + gap else cancelD
        // The cancel circle (and optional "עב" toggle beside it) rides 50 dp above the pill.
        Row(
            Modifier.offset {
                val a = anchor.value
                val leftAnchored = pos.value.x + inset
                val rightAnchored = pos.value.x + d - inset - totalW
                IntOffset((leftAnchored + (rightAnchored - leftAnchored) * a).roundToInt(), (pos.value.y - cancelUp).roundToInt())
            },
            horizontalArrangement = Arrangement.spacedBy((BubbleMetrics.CANCEL_HIT - BubbleMetrics.CANCEL_DIAMETER).dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            val rightAnchored = (scene.settleRightEdge ?: pill.rightEdge)
            if (hebrewAvailable && rightAnchored) HebrewCircle(hebrewActive, Modifier.bubbleShadow(CircleShape, LocalDark.current, shadowReach))
            CancelCircle(Modifier.bubbleShadow(CircleShape, LocalDark.current, shadowReach))
            if (hebrewAvailable && !rightAnchored) HebrewCircle(hebrewActive, Modifier.bubbleShadow(CircleShape, LocalDark.current, shadowReach))
        }
    }
}

/**
 * DESIGN 5.4 / b-hidden: "Bubble hidden until 15:32" with Undo. Inverse fill, md radius, >= 52 dp, Undo in
 * inverseAccent; announced politely.
 */
@Composable
fun HiddenSnackbar(dark: Boolean, text: String, onUndo: () -> Unit) {
    MurmurTheme(darkTheme = dark) {
        val c = MurmurTheme.colors
        val undo = stringResource(R.string.bubble_action_undo)
        val reach = with(LocalDensity.current) { BubbleMetrics.SHADOW_PAD.dp.toPx() }
        val shape = RoundedCornerShape(14.dp)
        Row(
            Modifier
                .padding(BubbleMetrics.SHADOW_PAD.dp)
                .bubbleShadow(shape, dark, reach)
                .background(c.inverse, shape)
                .heightIn(min = 52.dp)
                .padding(start = 16.dp, end = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(text, style = BubbleType.snackBody, color = c.onInverse, modifier = Modifier.weight(1f)
                .clearAndSetSemantics { contentDescription = text; liveRegion = LiveRegionMode.Polite })
            Box(
                Modifier
                    .heightIn(min = 48.dp)
                    .testTag(TAG_UNDO)
                    .clickable(role = Role.Button, onClick = onUndo)
                    .padding(horizontal = 12.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(undo, style = BubbleType.snackButton, color = c.inverseAccent)
            }
        }
    }
}

const val TAG_UNDO = "bubble.undo"
/** The lifted pill in the drag layer. */
const val TAG_LIFTED = "bubble.lifted"
const val TAG_HEBREW = "bubble.hebrew"
internal const val HEBREW_BADGE_TEXT = "עב"
