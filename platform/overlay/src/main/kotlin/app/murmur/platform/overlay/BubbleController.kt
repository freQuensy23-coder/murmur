package app.murmur.platform.overlay

import app.murmur.core.dictation.DictationBubble
import app.murmur.core.dictation.HapticKind
import app.murmur.core.dictation.UserInput
import kotlin.math.hypot

/**
 * Screen geometry a drag needs, captured by the host when the drag starts (pixels of the bubble's display).
 * [grabX]/[grabY] = finger position minus the circle's top-left at ACTION_DOWN.
 */
data class DragGeometry(
    val placement: PlacementInput,
    val grabX: Float,
    val grabY: Float,
    val targetCenterX: Float,
    val targetCenterY: Float,
    val magnetRadiusPx: Float,
)

/** The lifted circle during a drag: top-left of the (unscaled) circle in screen pixels, and the magnet state. */
data class DragState(
    val circleX: Float,
    val circleY: Float,
    val overHideTarget: Boolean,
    val startedAtMs: Long,
    /** Side and circle top it will snap to on release (the ghost). */
    val snapRightEdge: Boolean,
    val snapY: Int,
    /**
     * Whether the hide target is in play: only for a drag of the idle bubble (Ready or Rest, DESIGN 5.3 Dragging). A
     * working or result pill can only be moved, and a dictation that starts mid-drag disarms it for the rest of the drag.
     */
    val hideArmed: Boolean,
)

/**
 * The overlay's local brain, pure and JVM-tested: merges the orchestrator's bubble, the visibility input, settings
 * and the overlay-local state (drag, Rest tapped open) into one [BubbleVisual]; turns gestures and accessibility
 * actions into [UserInput]s for the orchestrator; runs the drag (magnet, snap, hide). The Android host feeds it and
 * performs [Output].
 */
class BubbleController(private val clock: () -> Long, private val out: Output) {

    interface Output {
        fun send(input: UserInput)
        /** Overlay-local haptic (entering the hide magnet); the host gates it by Settings.feedback. */
        fun haptic(kind: HapticKind)
        /** Release on the hide target or the TalkBack action "Hide for 1 hour": Settings.hiddenUntilEpochMs + snackbar. */
        fun hideForAnHour()
        /** A drag ended away from the target: store side + y for this display (DESIGN 5.4) and animate there. */
        fun dragSettled(rightEdge: Boolean, circleTop: Int, stored: Boolean)
    }

    var dictation: DictationBubble = DictationBubble.Idle
        set(value) {
            // Back from a dictation with no field focused: the "Always" tab shows Ready for another 4 s (DESIGN 5.3:
            // Copied "after 4 s -> Hidden or Ready"), then folds back.
            if (value is DictationBubble.Idle && field !is DictationBubble.Idle) restOpenedAtMs = clock()
            field = value
        }
    var visibility: VisibilityInput? = null
        set(value) {
            // A focused field takes over; once it is gone again the no-field form is the closed tab.
            if (value?.editableFocused == true) restOpenedAtMs = null
            field = value
        }
    /** Settings.hiddenUntilEpochMs: the overlay reads it directly so "hide" and "Show now" apply at once. */
    var hiddenUntilMs: Long? = null
    var holdToTalk: Boolean = true

    /** Last local interaction with an idle bubble (tap on Rest, end of a drag): restarts the 4 s tuck-in wait. */
    private var lastInteractionMs: Long? = null
    /** The "Always" tab with no field focused was opened (tap) or used at this instant; see [BubblePolicy.merge]. */
    private var restOpenedAtMs: Long? = null
    private var dragFromOpenTab = false
    private var geometry: DragGeometry? = null

    var drag: DragState? = null
        private set

    fun effectiveVisibility(now: Long = clock()): VisibilityInput {
        val v = visibility ?: VisibilityInput.nothingFocused(now)
        val local = lastInteractionMs
        val idle = v.idleSinceMs?.let { if (local != null && local > it) local else it }
        return v.copy(nowMs = now, hiddenUntilMs = hiddenUntilMs, idleSinceMs = idle)
    }

    fun visual(now: Long = clock()): BubbleVisual =
        BubblePolicy.merge(dictation, effectiveVisibility(now), dragging = drag != null, overHideTarget = drag?.overHideTarget == true,
            restOpenedAtMs = restOpenedAtMs)

    /** When [visual] can next change without an event (tuck-in, the opened "Always" tab, end of the hide hour). */
    fun nextDeadline(now: Long = clock()): Long? =
        if (drag != null) null else BubblePolicy.nextDeadline(dictation, effectiveVisibility(now), restOpenedAtMs)

    /** No field focused and "Show the bubble: Always": the bubble is the tab (closed) or the opened tab (Ready). */
    private fun noFieldAlways(now: Long): Boolean = effectiveVisibility(now).let { !it.editableFocused && it.showAlways }

    /** What a press may become on the visual shown at ACTION_DOWN (DESIGN 5.3, ARCHITECTURE 5.1). */
    fun pressPolicy(region: TouchRegion, now: Long = clock()): PressPolicy {
        val none = PressPolicy(region, canTap = false, canHold = false, canDrag = false)
        if (region == TouchRegion.CANCEL) return PressPolicy(region, canTap = isTapListening(), canHold = false, canDrag = false)
        if (region == TouchRegion.HEBREW) return PressPolicy(region, canTap = isHebrewToggleable(), canHold = false, canDrag = false)
        return when (val v = visual(now)) {
            BubbleVisual.Hidden, is BubbleVisual.Dragging -> none
            BubbleVisual.Rest -> PressPolicy(region, canTap = true, canHold = false, canDrag = true)
            BubbleVisual.Ready -> PressPolicy(region, canTap = true, canHold = holdToTalk, canDrag = true)
            is BubbleVisual.Active -> when (val s = v.state) {
                // The finger that holds is already down; a second press changes nothing.
                is DictationBubble.Listening -> if (s.hold) none else PressPolicy(region, canTap = true, canHold = false, canDrag = true)
                DictationBubble.Transcribing, DictationBubble.Polishing -> PressPolicy(region, canTap = false, canHold = false, canDrag = true)
                is DictationBubble.Inserted -> PressPolicy(region, canTap = true, canHold = holdToTalk, canDrag = true)
                is DictationBubble.Copied -> PressPolicy(region, canTap = true, canHold = false, canDrag = true)
                is DictationBubble.Error -> if (region == TouchRegion.ACTION) PressPolicy(region, canTap = s.action != null, canHold = false, canDrag = false)
                    else PressPolicy(region, canTap = true, canHold = false, canDrag = true)
                DictationBubble.Idle -> none
            }
        }
    }

    private fun isTapListening(): Boolean = (dictation as? DictationBubble.Listening)?.hold == false

    private fun isHebrewToggleable(): Boolean =
        (dictation as? DictationBubble.Listening)?.let { !it.hold && it.hebrewAvailable } == true

    /** Tap / TalkBack double-tap on [region] of the visual now shown. */
    fun onTap(region: TouchRegion, now: Long = clock()) {
        if (region == TouchRegion.CANCEL) { if (isTapListening()) out.send(UserInput.Cancel); return }
        if (region == TouchRegion.HEBREW) { if (isHebrewToggleable()) out.send(UserInput.ToggleHebrew); return }
        when (val v = visual(now)) {
            BubbleVisual.Rest -> { lastInteractionMs = now; if (noFieldAlways(now)) restOpenedAtMs = now }
            BubbleVisual.Ready -> { lastInteractionMs = now; if (noFieldAlways(now)) restOpenedAtMs = now; out.send(UserInput.Tap) }
            is BubbleVisual.Active -> when (val s = v.state) {
                is DictationBubble.Listening -> if (!s.hold) out.send(UserInput.Tap)
                is DictationBubble.Inserted -> out.send(UserInput.Tap)
                is DictationBubble.Copied -> out.send(UserInput.ResultTap)
                is DictationBubble.Error ->
                    if (region == TouchRegion.ACTION) { if (s.action != null) out.send(UserInput.ResultAction) } else out.send(UserInput.ResultTap)
                DictationBubble.Transcribing, DictationBubble.Polishing, DictationBubble.Idle -> Unit // ignored while working
            }
            BubbleVisual.Hidden, is BubbleVisual.Dragging -> Unit
        }
    }

    fun onGesture(g: Gesture, now: Long = clock()) {
        when (g) {
            is Gesture.Tap -> onTap(g.region, now)
            Gesture.HoldStart -> out.send(UserInput.HoldStart)
            Gesture.HoldRelease -> out.send(UserInput.HoldRelease)
            Gesture.HoldSlideOff -> out.send(UserInput.HoldSlideOff)
            is Gesture.DragStart -> Unit // the host calls startDrag with the geometry first
            is Gesture.DragMove -> moveDrag(g.x, g.y, now)
            is Gesture.DragEnd -> endDrag(g.x, g.y, g.vxDpPerSec, now, cancelled = false)
            Gesture.DragCancel -> drag?.let { endDrag(it.circleX + (geometry?.grabX ?: 0f), it.circleY + (geometry?.grabY ?: 0f), 0f, now, cancelled = true) }
        }
    }

    /** Begin a drag at finger ([x], [y]); [geometry] describes the display. */
    fun startDrag(x: Float, y: Float, geometry: DragGeometry, now: Long = clock()) {
        dragFromOpenTab = noFieldAlways(now) && visual(now) == BubbleVisual.Ready
        this.geometry = geometry
        drag = stateAt(x, y, geometry, now, armed = dictation is DictationBubble.Idle)
    }

    private fun moveDrag(x: Float, y: Float, now: Long) {
        val g = geometry ?: return
        val prev = drag ?: return
        val next = stateAt(x, y, g, prev.startedAtMs, armed = prev.hideArmed && dictation is DictationBubble.Idle)
        if (next.overHideTarget && !prev.overHideTarget) out.haptic(HapticKind.CLOCK_TICK)
        drag = next
    }

    private fun stateAt(x: Float, y: Float, g: DragGeometry, startedAt: Long, armed: Boolean): DragState {
        val d = g.placement.bubbleDiameterPx
        val cx = x - g.grabX
        val cy = y - g.grabY
        val centerX = cx + d / 2f
        val centerY = cy + d / 2f
        val over = armed && hypot(centerX - g.targetCenterX, centerY - g.targetCenterY) <= g.magnetRadiusPx
        val right = BubblePlacement.snap(centerX.toInt(), g.placement.screenW, 0f)
        return DragState(cx, cy, over, startedAt, right, BubblePlacement.clampY(cy.toInt(), g.placement), armed)
    }

    private fun endDrag(x: Float, y: Float, vxDpPerSec: Float, now: Long, cancelled: Boolean) {
        val g = geometry ?: return
        val prev = drag ?: return
        val end = stateAt(x, y, g, prev.startedAtMs, armed = prev.hideArmed && dictation is DictationBubble.Idle)
        drag = null
        geometry = null
        lastInteractionMs = now
        if (dragFromOpenTab && noFieldAlways(now)) restOpenedAtMs = now
        dragFromOpenTab = false
        if (end.overHideTarget && !cancelled) { out.hideForAnHour(); return }
        val d = g.placement.bubbleDiameterPx
        val right = BubblePlacement.snap((end.circleX + d / 2f).toInt(), g.placement.screenW, vxDpPerSec)
        out.dragSettled(right, BubblePlacement.clampY(end.circleY.toInt(), g.placement), stored = !cancelled)
    }

    /** TalkBack custom action "Hide for 1 hour" (DESIGN 8: replaces drag-to-hide). */
    fun hideAction() = out.hideForAnHour()
}
