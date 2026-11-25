package app.murmur.platform.overlay

import kotlin.math.hypot
import kotlin.math.max

/** Where a touch went down: the bubble body, the Error pill's action button, the cancel circle, or the עב toggle above Listening. */
enum class TouchRegion { BODY, ACTION, CANCEL, HEBREW }

/** What the bubble's touch stream means (DESIGN 5.3 "Enters when / Leaves when"). Coordinates are screen pixels. */
sealed interface Gesture {
    data class Tap(val region: TouchRegion) : Gesture
    /** Press held >= 250 ms without moving 8 dp (only when hold-to-talk applies to the visual). */
    data object HoldStart : Gesture
    data object HoldRelease : Gesture
    /** While holding, the finger went more than 64 dp away from the pill: discard. */
    data object HoldSlideOff : Gesture
    data class DragStart(val x: Float, val y: Float) : Gesture
    data class DragMove(val x: Float, val y: Float) : Gesture
    data class DragEnd(val x: Float, val y: Float, val vxDpPerSec: Float) : Gesture
    /** The system cancelled the stream mid-drag (window removed, display change): settle without hiding. */
    data object DragCancel : Gesture
}

/** What a press may turn into, decided by the visual under the finger at ACTION_DOWN. */
data class PressPolicy(val region: TouchRegion, val canTap: Boolean, val canHold: Boolean, val canDrag: Boolean)

/** Axis-aligned rectangle in screen pixels (no android.graphics on the JVM). */
data class PxRect(val left: Float, val top: Float, val right: Float, val bottom: Float) {
    /** Distance from a point to the rectangle (0 inside). */
    fun distanceTo(x: Float, y: Float): Float {
        val dx = max(max(left - x, 0f), x - right)
        val dy = max(max(top - y, 0f), y - bottom)
        return hypot(dx, dy)
    }
}

/**
 * Pure touch recognizer for the bubble (DESIGN 5.3 / 3.5): tap; hold >= 250 ms (hold-to-talk); slide-off > 64 dp from
 * the pill while holding; drag once the finger moved >= 8 dp; fling velocity at release. Works on raw screen
 * coordinates so that moving windows never distort the deltas. The host calls [tick] at [holdDeadline].
 */
class BubbleGestureRecognizer(
    private val density: Float,
    /** The pill's current rectangle on screen (it grows from circle to pill when the hold starts). */
    private val pillRect: () -> PxRect?,
) {
    private enum class Phase { IDLE, PRESSED, HOLDING, DRAGGING, DONE }

    private var phase = Phase.IDLE
    private var policy = PressPolicy(TouchRegion.BODY, canTap = false, canHold = false, canDrag = false)
    private var downX = 0f
    private var downY = 0f
    private var downT = 0L
    private var moved = false
    private val samples = ArrayDeque<Triple<Long, Float, Float>>()

    val dragging: Boolean get() = phase == Phase.DRAGGING

    /** When [tick] must run to start a hold, or null. */
    fun holdDeadline(): Long? = if (phase == Phase.PRESSED && policy.canHold && !moved) downT + HOLD_MS else null

    fun down(x: Float, y: Float, t: Long, p: PressPolicy): List<Gesture> {
        phase = Phase.PRESSED; policy = p
        downX = x; downY = y; downT = t; moved = false
        samples.clear(); samples.addLast(Triple(t, x, y))
        return emptyList()
    }

    fun move(x: Float, y: Float, t: Long): List<Gesture> {
        when (phase) {
            Phase.PRESSED -> {
                // A hold that is due fires first, so a slow press that then wanders becomes a slide-off, not a drag.
                tick(t).takeIf { it.isNotEmpty() }?.let { return it + move(x, y, t) }
                if (hypot(x - downX, y - downY) < dp(SLOP_DP)) return emptyList()
                moved = true
                if (!policy.canDrag) return emptyList()
                phase = Phase.DRAGGING
                record(t, x, y)
                return listOf(Gesture.DragStart(downX, downY), Gesture.DragMove(x, y))
            }
            Phase.HOLDING -> {
                val rect = pillRect() ?: PxRect(downX, downY, downX, downY)
                if (rect.distanceTo(x, y) > dp(SLIDE_OFF_DP)) {
                    phase = Phase.DONE
                    return listOf(Gesture.HoldSlideOff)
                }
                return emptyList()
            }
            Phase.DRAGGING -> {
                // Only real movement counts for the velocity; a resting finger leaves the last sample to age.
                val last = samples.last()
                if (x != last.second || y != last.third) record(t, x, y)
                return listOf(Gesture.DragMove(x, y))
            }
            else -> return emptyList()
        }
    }

    fun up(x: Float, y: Float, t: Long): List<Gesture> {
        val out = when (phase) {
            Phase.PRESSED -> {
                val due = tick(t)
                when {
                    due.isNotEmpty() -> due + Gesture.HoldRelease
                    !moved && policy.canTap -> listOf(Gesture.Tap(policy.region))
                    else -> emptyList()
                }
            }
            Phase.HOLDING -> listOf(Gesture.HoldRelease)
            Phase.DRAGGING -> {
                val last = samples.last()
                if (x != last.second || y != last.third) record(t, x, y)
                listOf(Gesture.DragEnd(x, y, velocityDpPerSec(t)))
            }
            else -> emptyList()
        }
        phase = Phase.IDLE
        return out
    }

    /** ACTION_CANCEL: a hold finishes (the audio so far is kept), a drag settles, a press does nothing. */
    fun cancel(): List<Gesture> {
        val out = when (phase) {
            Phase.HOLDING -> listOf(Gesture.HoldRelease)
            Phase.DRAGGING -> listOf(Gesture.DragCancel)
            else -> emptyList()
        }
        phase = Phase.IDLE
        return out
    }

    fun tick(t: Long): List<Gesture> {
        val due = holdDeadline() ?: return emptyList()
        if (t < due) return emptyList()
        phase = Phase.HOLDING
        return listOf(Gesture.HoldStart)
    }

    private fun record(t: Long, x: Float, y: Float) {
        samples.addLast(Triple(t, x, y))
        while (samples.size > 2 && t - samples.first().first > VELOCITY_WINDOW_MS) samples.removeFirst()
    }

    /**
     * Horizontal release velocity over the 100 ms before the last movement, in dp/s. A finger that rested for more
     * than 40 ms before lifting has no velocity (as Android's VelocityTracker assumes the pointer stopped).
     */
    private fun velocityDpPerSec(upT: Long): Float {
        if (samples.size < 2) return 0f
        val (t1, x1, _) = samples.last()
        if (upT - t1 > STOPPED_MS) return 0f
        val (t0, x0, _) = samples.firstOrNull { t1 - it.first <= VELOCITY_WINDOW_MS } ?: return 0f
        val dt = (t1 - t0).coerceAtLeast(1L)
        return (x1 - x0) / density / dt * 1000f
    }

    private fun dp(v: Float) = v * density

    companion object {
        const val SLOP_DP = 8f
        const val HOLD_MS = 250L
        const val SLIDE_OFF_DP = 64f
        const val VELOCITY_WINDOW_MS = 100L
        const val STOPPED_MS = 40L
    }
}
