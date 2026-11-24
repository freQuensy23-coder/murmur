package app.murmur.platform.overlay

import app.murmur.core.dictation.BubbleAction
import app.murmur.core.dictation.BubbleError
import app.murmur.core.dictation.DictationBubble
import app.murmur.core.dictation.HapticKind
import app.murmur.core.dictation.UserInput
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.roundToInt

class BubbleControllerTest {
    private var now = 100_000L
    private val sent = mutableListOf<UserInput>()
    private val haptics = mutableListOf<HapticKind>()
    private var hides = 0
    private val settled = mutableListOf<Triple<Boolean, Int, Boolean>>()

    private val c = BubbleController({ now }, object : BubbleController.Output {
        override fun send(input: UserInput) { sent += input }
        override fun haptic(kind: HapticKind) { haptics += kind }
        override fun hideForAnHour() { hides++ }
        override fun dragSettled(rightEdge: Boolean, circleTop: Int, stored: Boolean) { settled += Triple(rightEdge, circleTop, stored) }
    })

    private val focused = VisibilityInput(editableFocused = true, bubbleAllowedForField = true, appHidden = false, hiddenUntilMs = null,
        nowMs = now, showAlways = false, idleSinceMs = now, tuckIn = true, talkBackOn = false)

    private val density = 2.625f
    private fun px(dp: Float) = (dp * density).roundToInt()
    private val d = px(52f)
    private val screen = PlacementInput(1080, 2520, density, 63, 2457, fieldTop = null, imeTop = null, stored = null, bubbleDiameterPx = d)
    private val targetX = 540f
    private val targetY = 2200f
    private val geometry = DragGeometry(screen, grabX = d / 2f, grabY = d / 2f, targetCenterX = targetX, targetCenterY = targetY,
        magnetRadiusPx = 72f * density)

    private fun show(b: DictationBubble) { c.dictation = b }

    @Test
    fun tapOnReadyStartsDictation_holdTalks_slideOffDiscards() {
        c.visibility = focused
        assertEquals(BubbleVisual.Ready, c.visual())
        c.onGesture(Gesture.Tap(TouchRegion.BODY))
        c.onGesture(Gesture.HoldStart)
        c.onGesture(Gesture.HoldRelease)
        c.onGesture(Gesture.HoldSlideOff)
        assertEquals(listOf(UserInput.Tap, UserInput.HoldStart, UserInput.HoldRelease, UserInput.HoldSlideOff), sent)
    }

    @Test
    fun tapOnRestOpensReadyLocally_andItTucksAgainAfter4s() {
        c.visibility = focused.copy(idleSinceMs = now - 10_000)
        assertEquals(BubbleVisual.Rest, c.visual())
        c.onTap(TouchRegion.BODY)
        assertEquals(emptyList<UserInput>(), sent) // expanding is overlay-local
        assertEquals(BubbleVisual.Ready, c.visual())
        assertEquals(now + 4_000, c.nextDeadline())
        now += 4_000
        assertEquals(BubbleVisual.Rest, c.visual())
    }

    private val alwaysTab = focused.copy(editableFocused = false, showAlways = true, idleSinceMs = null)

    @Test
    fun tapOnTheAlwaysTabWithoutAFieldOpensReady_andItFoldsBackAfter4s() {
        // Review F1: the tab of "Show the bubble: Always" ignored taps although TalkBack offers "show the dictation button".
        c.visibility = alwaysTab
        assertEquals(BubbleVisual.Rest, c.visual())
        assertTrue(c.pressPolicy(TouchRegion.BODY).canTap)
        c.onTap(TouchRegion.BODY)
        assertEquals(emptyList<UserInput>(), sent) // opening is overlay-local
        assertEquals(BubbleVisual.Ready, c.visual())
        assertEquals(now + 4_000, c.nextDeadline())
        now += 3_999
        assertEquals(BubbleVisual.Ready, c.visual())
        now += 1
        assertEquals(BubbleVisual.Rest, c.visual())
        assertNull(c.nextDeadline())
    }

    @Test
    fun theOpenedAlwaysTabStartsDictation_andIsReadyAgainAfterTheResult() {
        c.visibility = alwaysTab
        c.onTap(TouchRegion.BODY)
        now += 1_000
        c.onTap(TouchRegion.BODY)
        assertEquals(listOf(UserInput.Tap), sent)
        show(DictationBubble.Listening(startedAtMs = now, hold = false, warn = false))
        now += 30_000
        show(DictationBubble.Copied(null))
        now += 4_000
        show(DictationBubble.Idle)
        // Back from the result: Ready for another 4 s, then the tab.
        assertEquals(BubbleVisual.Ready, c.visual())
        now += 4_000
        assertEquals(BubbleVisual.Rest, c.visual())
    }

    @Test
    fun theOpenedAlwaysTabStaysReadyUnderTalkBack_untilAFieldTakesOver() {
        c.visibility = alwaysTab.copy(talkBackOn = true)
        c.onTap(TouchRegion.BODY) // TalkBack double-tap
        now += 60_000
        assertEquals(BubbleVisual.Ready, c.visual())
        assertNull(c.nextDeadline())
        // A field gets focus, then loses it: the no-field form is the tab again.
        c.visibility = focused.copy(talkBackOn = true, idleSinceMs = now)
        assertEquals(BubbleVisual.Ready, c.visual())
        c.visibility = alwaysTab.copy(talkBackOn = true)
        assertEquals(BubbleVisual.Rest, c.visual())
    }

    @Test
    fun listeningPillTapFinishes_cancelDiscards_holdPillIgnoresTaps() {
        c.visibility = focused
        show(DictationBubble.Listening(startedAtMs = now, hold = false, warn = false, hebrewAvailable = true))
        c.onTap(TouchRegion.BODY)
        c.onTap(TouchRegion.CANCEL)
        c.onTap(TouchRegion.HEBREW)
        assertEquals(listOf(UserInput.Tap, UserInput.Cancel, UserInput.ToggleHebrew), sent)
        sent.clear()
        // When Hebrew ASR is unavailable, taps on HEBREW do nothing.
        show(DictationBubble.Listening(startedAtMs = now, hold = false, warn = false, hebrewAvailable = false))
        assertFalse(c.pressPolicy(TouchRegion.HEBREW).canTap)
        c.onTap(TouchRegion.HEBREW)
        assertEquals(emptyList<UserInput>(), sent)
        show(DictationBubble.Listening(startedAtMs = now, hold = true, warn = false, hebrewAvailable = true))
        assertFalse(c.pressPolicy(TouchRegion.HEBREW).canTap)
        c.onTap(TouchRegion.BODY)
        c.onTap(TouchRegion.CANCEL)
        c.onTap(TouchRegion.HEBREW)
        assertEquals(emptyList<UserInput>(), sent)
    }

    @Test
    fun workingStatesIgnoreTaps() {
        for (s in listOf(DictationBubble.Transcribing, DictationBubble.Polishing)) {
            show(s)
            c.onTap(TouchRegion.BODY)
            c.onTap(TouchRegion.CANCEL)
        }
        assertEquals(emptyList<UserInput>(), sent)
    }

    @Test
    fun resultStates() {
        show(DictationBubble.Inserted(noCleanup = false))
        c.onTap(TouchRegion.BODY) // a new dictation from the result pill (ARCHITECTURE 5.2 row 46)
        show(DictationBubble.Copied(entry = null))
        c.onTap(TouchRegion.BODY)
        show(DictationBubble.Error(BubbleError.NO_SPEECH, BubbleAction.RETRY))
        c.onTap(TouchRegion.BODY)
        c.onTap(TouchRegion.ACTION)
        show(DictationBubble.Error(BubbleError.MIC_BUSY, null))
        c.onTap(TouchRegion.ACTION) // no action button: nothing
        assertEquals(listOf(UserInput.Tap, UserInput.ResultTap, UserInput.ResultTap, UserInput.ResultAction), sent)
    }

    @Test
    fun pressPolicyPerVisual() {
        c.visibility = focused
        c.holdToTalk = true
        assertEquals(PressPolicy(TouchRegion.BODY, canTap = true, canHold = true, canDrag = true), c.pressPolicy(TouchRegion.BODY))
        c.holdToTalk = false
        assertFalse(c.pressPolicy(TouchRegion.BODY).canHold)
        c.visibility = focused.copy(idleSinceMs = now - 10_000) // Rest
        assertEquals(PressPolicy(TouchRegion.BODY, canTap = true, canHold = false, canDrag = true), c.pressPolicy(TouchRegion.BODY))
        show(DictationBubble.Listening(now, hold = false, warn = false))
        assertEquals(PressPolicy(TouchRegion.BODY, canTap = true, canHold = false, canDrag = true), c.pressPolicy(TouchRegion.BODY))
        assertTrue(c.pressPolicy(TouchRegion.CANCEL).canTap)
        show(DictationBubble.Listening(now, hold = true, warn = false))
        assertEquals(PressPolicy(TouchRegion.BODY, canTap = false, canHold = false, canDrag = false), c.pressPolicy(TouchRegion.BODY))
        assertFalse(c.pressPolicy(TouchRegion.CANCEL).canTap)
        show(DictationBubble.Polishing)
        assertEquals(PressPolicy(TouchRegion.BODY, canTap = false, canHold = false, canDrag = true), c.pressPolicy(TouchRegion.BODY))
        show(DictationBubble.Error(BubbleError.LOAD_FAILED, BubbleAction.OPEN_MODELS))
        assertEquals(PressPolicy(TouchRegion.ACTION, canTap = true, canHold = false, canDrag = false), c.pressPolicy(TouchRegion.ACTION))
        show(DictationBubble.Idle)
        c.visibility = focused.copy(bubbleAllowedForField = false) // Hidden
        assertEquals(PressPolicy(TouchRegion.BODY, canTap = false, canHold = false, canDrag = false), c.pressPolicy(TouchRegion.BODY))
    }

    @Test
    fun dragShowsDraggingAndReleaseSnapsToTheNearerEdgeAndStoresIt() {
        c.visibility = focused
        c.startDrag(900f, 1200f, geometry)
        assertEquals(BubbleVisual.Dragging(false), c.visual())
        c.onGesture(Gesture.DragMove(300f, 1000f))
        val s = c.drag!!
        assertEquals(300f - d / 2f, s.circleX, 0f)
        assertFalse(s.snapRightEdge) // ghost on the left
        c.onGesture(Gesture.DragEnd(300f, 1000f, vxDpPerSec = 0f))
        assertNull(c.drag)
        assertEquals(listOf(Triple(false, (1000f - d / 2f).toInt(), true)), settled)
        assertEquals(BubbleVisual.Ready, c.visual())
        assertEquals(0, hides)
    }

    @Test
    fun flingDecidesTheSide() {
        c.startDrag(900f, 1200f, geometry)
        c.onGesture(Gesture.DragMove(300f, 1200f))
        c.onGesture(Gesture.DragEnd(300f, 1200f, vxDpPerSec = 1_500f))
        assertTrue(settled.single().first)
    }

    @Test
    fun releaseYIsClampedToTheUsableArea() {
        c.startDrag(900f, 1200f, geometry)
        c.onGesture(Gesture.DragMove(900f, 10f))
        c.onGesture(Gesture.DragEnd(900f, 10f, 0f))
        assertEquals(BubblePlacement.topLimit(screen), settled.single().second)
    }

    @Test
    fun enteringTheMagnetTicksOnce_releaseThereHidesForAnHour() {
        c.visibility = focused
        c.startDrag(900f, 1200f, geometry)
        c.onGesture(Gesture.DragMove(targetX, targetY - 72f * density - 5f)) // just outside
        assertFalse(c.drag!!.overHideTarget)
        c.onGesture(Gesture.DragMove(targetX, targetY - 60f * density))
        assertEquals(BubbleVisual.Dragging(true), c.visual())
        c.onGesture(Gesture.DragMove(targetX + 10f, targetY))
        assertEquals(listOf(HapticKind.CLOCK_TICK), haptics)
        c.onGesture(Gesture.DragEnd(targetX, targetY, 0f))
        assertEquals(1, hides)
        assertEquals(emptyList<Triple<Boolean, Int, Boolean>>(), settled)
    }

    @Test
    fun aWorkingOrResultPillCanOnlyBeMoved_theHideTargetIsNeverArmed() {
        // Review F2: dropping a Listening pill on the target wrote "hidden until" while the pill stayed on screen.
        c.visibility = focused
        val states = listOf(DictationBubble.Listening(startedAtMs = now, hold = false, warn = false), DictationBubble.Transcribing,
            DictationBubble.Polishing, DictationBubble.Inserted(false), DictationBubble.Copied(null), DictationBubble.Error(BubbleError.NO_SPEECH, null))
        for (s in states) {
            show(s)
            settled.clear()
            c.startDrag(900f, 1200f, geometry)
            assertFalse(c.drag!!.hideArmed)
            c.onGesture(Gesture.DragMove(targetX, targetY - 60f * density))
            c.onGesture(Gesture.DragMove(targetX, targetY))
            assertFalse("$s", c.drag!!.overHideTarget)
            assertEquals(BubbleVisual.Dragging(false), c.visual())
            c.onGesture(Gesture.DragEnd(targetX, targetY, 0f))
            assertEquals("$s", 0, hides)
            assertEquals(emptyList<HapticKind>(), haptics)
            // Released at the bottom centre: it snaps to an edge like any other drop and keeps its state.
            assertEquals(1, settled.size)
            assertTrue(settled.single().third)
            assertEquals(BubbleVisual.Active(s), c.visual())
        }
        assertEquals(emptyList<UserInput>(), sent)
    }

    @Test
    fun aDictationThatStartsMidDragDisarmsTheHideTarget() {
        // e.g. "Start dictation" from the notification while the Ready circle is being moved.
        c.visibility = focused
        c.startDrag(900f, 1200f, geometry)
        assertTrue(c.drag!!.hideArmed)
        show(DictationBubble.Listening(startedAtMs = now, hold = false, warn = false))
        c.onGesture(Gesture.DragMove(targetX, targetY))
        assertFalse(c.drag!!.hideArmed)
        assertFalse(c.drag!!.overHideTarget)
        c.onGesture(Gesture.DragEnd(targetX, targetY, 0f))
        assertEquals(0, hides)
        assertEquals(1, settled.size)
    }

    @Test
    fun leavingAndReenteringTheMagnetTicksAgain() {
        c.startDrag(900f, 1200f, geometry)
        c.onGesture(Gesture.DragMove(targetX, targetY))
        c.onGesture(Gesture.DragMove(targetX, 1000f))
        c.onGesture(Gesture.DragMove(targetX, targetY))
        assertEquals(listOf(HapticKind.CLOCK_TICK, HapticKind.CLOCK_TICK), haptics)
    }

    @Test
    fun cancelledDragSettlesWithoutStoringOrHiding() {
        c.startDrag(900f, 1200f, geometry)
        c.onGesture(Gesture.DragMove(targetX, targetY))
        c.onGesture(Gesture.DragCancel)
        assertEquals(0, hides)
        assertFalse(settled.single().third)
    }

    @Test
    fun hiddenUntilFromSettingsHidesAnIdleBubble_butNotARunningDictation() {
        c.visibility = focused
        c.hiddenUntilMs = now + 3_600_000
        assertEquals(BubbleVisual.Hidden, c.visual())
        assertEquals(now + 3_600_000, c.nextDeadline())
        show(DictationBubble.Polishing)
        assertEquals(BubbleVisual.Active(DictationBubble.Polishing), c.visual())
        show(DictationBubble.Idle)
        c.hiddenUntilMs = null // Undo / Show now
        assertEquals(BubbleVisual.Ready, c.visual())
    }

    @Test
    fun talkBackHideActionHides() {
        c.hideAction()
        assertEquals(1, hides)
    }

    @Test
    fun withoutAnyVisibilityInputTheBubbleIsHidden() {
        assertEquals(BubbleVisual.Hidden, c.visual())
        c.visibility = focused.copy(editableFocused = false, showAlways = true)
        assertEquals(BubbleVisual.Rest, c.visual())
    }
}
