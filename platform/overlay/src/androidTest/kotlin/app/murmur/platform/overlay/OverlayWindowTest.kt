package app.murmur.platform.overlay

import android.app.UiAutomation
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import android.view.Display
import android.view.InputDevice
import android.view.MotionEvent
import android.view.accessibility.AccessibilityNodeInfo
import android.view.accessibility.AccessibilityWindowInfo
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import app.murmur.core.dictation.DictationBubble
import app.murmur.core.dictation.UserInput
import app.murmur.testing.InMemorySettingsRepository
import kotlinx.coroutines.flow.MutableStateFlow
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * WP-03 acceptance "Overlay on the emulator": a test accessibility service hosts the real [OverlayHost] over
 * :testapps:target. Window type, drag/snap/hide, size (wm size cover <-> inner), posture (hinge, driven from the host by
 * tools/run_emulator_tests.sh) and rotation 0-3, with the same edge and proportional y. Positions are logged with the
 * tag WP03 (and MurmurBubble by the host).
 *
 * Needs murmur_fold7_inner (1968 x 2184) and app.murmur.testtarget installed.
 */
@RunWith(AndroidJUnit4::class)
class OverlayWindowTest {
    private val inst = InstrumentationRegistry.getInstrumentation()
    // A default UiAutomation suppresses every accessibility service, including the one that hosts the bubble.
    private val ua = inst.getUiAutomation(UiAutomation.FLAG_DONT_SUPPRESS_ACCESSIBILITY_SERVICES)
    private val ctx = inst.targetContext
    private val settings = InMemorySettingsRepository()
    private val dictation = MutableStateFlow<DictationBubble>(DictationBubble.Idle)
    private val levels = MutableStateFlow(FloatArray(7))
    private val inputs = CopyOnWriteArrayList<UserInput>()
    private lateinit var service: OverlayTestService
    private lateinit var host: OverlayHost
    private var previousServices = ""
    private var previousAutoRotate = ""
    private var previousHideErrorDialogs = ""

    private fun sh(cmd: String): String =
        ua.executeShellCommand(cmd).use { pfd -> android.os.ParcelFileDescriptor.AutoCloseInputStream(pfd).bufferedReader().readText() }

    private fun log(step: String, s: OverlayHost.Snapshot) =
        Log.i(TAG, "step=$step w=${s.screenW} h=${s.screenH} rot=${s.rotation} right=${s.rightEdge} y=${s.circleTop} " +
            "frac=${s.yFraction?.let { "%.3f".format(it) }} visual=${s.visual} alpha=${s.rootAlpha}")

    private fun <T> onMain(block: () -> T): T {
        var out: T? = null
        inst.runOnMainSync { out = block() }
        @Suppress("UNCHECKED_CAST")
        return out as T
    }

    private fun snap(): OverlayHost.Snapshot = onMain { host.snapshot() }

    /** Ready and drawn by its own window again: the drag layer's settle animation is over (the window is touchable). */
    private fun settled(s: OverlayHost.Snapshot) = s.visual == BubbleVisual.Ready && s.rootAlpha == 1f

    private fun waitUntil(what: String, timeoutMs: Long = 10_000, cond: () -> Boolean) {
        val end = SystemClock.uptimeMillis() + timeoutMs
        while (SystemClock.uptimeMillis() < end) {
            if (cond()) return
            SystemClock.sleep(50)
        }
        fail("timed out waiting for $what; inputs=$inputs")
    }

    private fun focusedInput(fieldTop: Int? = null, imeTop: Int? = null) = VisibilityInput(
        editableFocused = true, bubbleAllowedForField = true, appHidden = false, hiddenUntilMs = null,
        nowMs = System.currentTimeMillis(), showAlways = false, idleSinceMs = null, tuckIn = true, talkBackOn = false,
        fieldTop = fieldTop, imeTop = imeTop,
    )

    @Before
    fun setUp() {
        assertTrue("install :testapps:target first (tools/run_emulator_tests.sh does)", sh("pm path $TARGET_PKG").contains("package:"))
        previousServices = sh("settings get secure enabled_accessibility_services").trim()
        previousAutoRotate = sh("settings get system accelerometer_rotation").trim()
        previousHideErrorDialogs = sh("settings get global hide_error_dialogs").trim()
        // A loaded shared server makes system UI ANR now and then; its dialog would take the focus from the target app.
        sh("settings put global hide_error_dialogs 1")
        clearSystemDialogs()
        OverlayTestService.reset()
        sh("settings put secure enabled_accessibility_services ${ctx.packageName}/${OverlayTestService::class.java.name}")
        sh("settings put secure accessibility_enabled 1")
        service = OverlayTestService.await(20) ?: error("test accessibility service did not connect")
        // Up to three launches: a system dialog that appears meanwhile is closed and the target brought back.
        var lastError: AssertionError? = null
        for (attempt in 1..3) {
            sh("am start -W -n $TARGET_PKG/.TargetActivity")
            try {
                waitUntil("target window", 10_000) { service.windows.any { it.root?.packageName == TARGET_PKG } }
                lastError = null
                break
            } catch (e: AssertionError) {
                lastError = AssertionError("${e.message} (attempt $attempt); windows: ${describeWindows()}")
                Log.w(TAG, lastError.message.orEmpty())
                clearSystemDialogs()
            }
        }
        lastError?.let { throw it }
        onMain {
            host = OverlayHost(service, Display.DEFAULT_DISPLAY, dictation, levels, { inputs += it }, settings)
            check(host.attach()) { "bubble window not attached" }
        }
    }

    @After
    fun tearDown() {
        if (::host.isInitialized) onMain { host.detach() }
        sh("wm size reset")
        sh("settings put system user_rotation 0")
        if (previousAutoRotate.isNotEmpty() && previousAutoRotate != "null") sh("settings put system accelerometer_rotation $previousAutoRotate")
        if (previousServices.isEmpty() || previousServices == "null") sh("settings delete secure enabled_accessibility_services")
        else sh("settings put secure enabled_accessibility_services $previousServices")
        if (previousHideErrorDialogs.isEmpty() || previousHideErrorDialogs == "null") sh("settings delete global hide_error_dialogs")
        else sh("settings put global hide_error_dialogs $previousHideErrorDialogs")
    }

    /** Wake, unlock and close any system dialog (ANR, crash, notification shade) before the target is launched. */
    private fun clearSystemDialogs() {
        sh("input keyevent KEYCODE_WAKEUP")
        sh("wm dismiss-keyguard")
        sh("am broadcast -a android.intent.action.CLOSE_SYSTEM_DIALOGS")
    }

    /**
     * The keyboard for the target's field: TargetActivity asks for it (stateVisible), but after earlier tests or a
     * dialog the window may come back without it, so the field is tapped as a user would.
     */
    private fun awaitKeyboard() {
        for (attempt in 1..3) {
            try {
                waitUntil("keyboard", 5_000) { service.fieldAndIme().second != null }
                return
            } catch (e: AssertionError) {
                val field = service.editableBounds(TARGET_PKG)
                Log.w(TAG, "no keyboard yet (attempt $attempt); tapping the field at $field")
                if (field != null) tap(Pair(field.exactCenterX(), field.exactCenterY()), onOverlay = false)
            }
        }
        waitUntil("keyboard after tapping the field") { service.fieldAndIme().second != null }
    }

    @Test
    fun bubbleIsAnAccessibilityOverlayOverTheTargetApp_placedAboveTheKeyboard() {
        // Wait until the IME window is in the window list (tapping the field if it does not come up on its own).
        awaitKeyboard()
        val (fieldTop, imeTop) = service.fieldAndIme()
        Log.i(TAG, "target fieldTop=$fieldTop imeTop=$imeTop")
        assertNotNull("the target's EditText has input focus", fieldTop)
        assertNotNull(imeTop)
        onMain { host.updateVisibility(focusedInput(fieldTop, imeTop)) }
        waitUntil("Ready") { snap().visual == BubbleVisual.Ready }
        SystemClock.sleep(500)
        val s = snap()
        log("over-target", s)

        // 1. Window type, from WindowManager's own dump.
        val dump = sh("dumpsys window windows")
        val block = checkNotNull(dump.split("Window #").firstOrNull { it.contains("MurmurBubble}") }) { "MurmurBubble not in dumpsys window" }
        val typeLine = block.lines().firstOrNull { it.contains(" ty=") }.orEmpty().trim()
        Log.i(TAG, "dumpsys: ${block.lineSequence().first().trim()} | $typeLine")
        assertTrue(typeLine, typeLine.contains("ty=ACCESSIBILITY_OVERLAY"))

        // 2. No SYSTEM_ALERT_WINDOW anywhere in this package.
        val requested = ctx.packageManager.getPackageInfo(ctx.packageName, PackageManager.GET_PERMISSIONS).requestedPermissions.orEmpty()
        assertFalse(requested.contains("android.permission.SYSTEM_ALERT_WINDOW"))

        // 3. Over the target app: the overlay window is above the app's window.
        val windows = service.windows
        val overlay = checkNotNull(windows.firstOrNull { it.type == AccessibilityWindowInfo.TYPE_ACCESSIBILITY_OVERLAY }) { "no overlay window" }
        val app = windows.first { it.root?.packageName == TARGET_PKG }
        assertTrue("overlay layer ${overlay.layer} > app layer ${app.layer}", overlay.layer > app.layer)

        // 4. DESIGN 5.4: the target's field sits near the top, so the bubble goes 12 dp above the keyboard (or the bottom
        //    of the usable area when no keyboard is up), and never below imeTop - 12 dp.
        val d = ctx.resources.displayMetrics.density
        val top = s.circleTop!!
        assertEquals(imeTop!! - (12 * d).roundToInt() - s.diameterPx, top)
        assertTrue(top >= (16 * d).roundToInt())

        // 5. Pixels: the mark is drawn where the bubble is (polled: the first frame of a window can be slow here).
        val cx = s.screenW - (8 * d).roundToInt() - s.diameterPx / 2
        val cy = top + s.diameterPx / 2
        var ink = 0
        var shot: Bitmap? = null
        val end = SystemClock.uptimeMillis() + 10_000
        while (ink <= 20 && SystemClock.uptimeMillis() < end) {
            shot = checkNotNull(ua.takeScreenshot()) { "no screenshot" }
            ink = inkPixels(shot, cx, cy, s.diameterPx / 3)
            if (ink <= 20) SystemClock.sleep(250)
        }
        Shots.save("overlay-over-target", shot!!)
        assertTrue("ink pixels of the mark: $ink", ink > 20)
    }

    /** Dark (ink) pixels in the square of half-size [r] around ([cx], [cy]). */
    private fun inkPixels(b: Bitmap, cx: Int, cy: Int, r: Int): Int {
        var ink = 0
        for (dx in -r..r) for (dy in -r..r) {
            val p = b.getPixel(cx + dx, cy + dy)
            if (((p shr 16) and 0xFF) < 80 && ((p shr 8) and 0xFF) < 80 && (p and 0xFF) < 80) ink++
        }
        return ink
    }

    /**
     * Drag tests run without hold-to-talk: on a loaded emulator a synchronous injection can take longer than the
     * 250 ms hold threshold, which would turn the press into a hold. Hold-to-talk has its own test.
     */
    private fun withoutHoldToTalk() = kotlinx.coroutines.runBlocking { settings.update { it.copy(holdToTalk = false) } }

    @Test
    fun dragSnapsToTheNearerEdge_flingDecides_hideAndUndo() {
        withoutHoldToTalk()
        onMain { host.updateVisibility(focusedInput()) }
        waitUntil("Ready") { snap().visual == BubbleVisual.Ready }
        val d = ctx.resources.displayMetrics.density
        var s = snap(); log("start", s)
        assertTrue(s.rightEdge)

        // Slow drag to 30% of the width: nearer edge = left.
        val start = centre(s)
        swipe(start, Pair(s.screenW * 0.3f, start.second + 200f), steps = 20, durationMs = 600, holdAtEndMs = 200)
        waitUntil("snapped left") { snap().let { !it.rightEdge && settled(it) } }
        s = snap(); log("snap-left", s)
        val expectedTop = (start.second + 200f - s.diameterPx / 2f).roundToInt()
        assertTrue("y ${s.circleTop} vs $expectedTop", abs(s.circleTop!! - expectedTop) <= (2 * d).roundToInt())
        val key = BubblePlacement.displayKey(Display.DEFAULT_DISPLAY, s.screenW, s.screenH, d)
        val stored = settings.settings.value.bubblePositions[key]
        assertNotNull("stored for $key", stored)
        assertFalse(stored!!.rightEdge)
        assertEquals(s.yFraction!!, stored.yFraction, 0.01f)

        // Slow drag to 70%: right.
        swipe(centre(s), Pair(s.screenW * 0.7f, centre(s).second), steps = 20, durationMs = 600, holdAtEndMs = 200)
        waitUntil("snapped right") { snap().let { it.rightEdge && settled(it) } }
        s = snap(); log("snap-right", s)

        // Fast flick toward the left that ends right of the centre: the fling (> 1000 dp/s) wins.
        val c = centre(s)
        swipe(c, Pair(s.screenW * 0.6f, c.second), steps = 4, durationMs = 60, holdAtEndMs = 0)
        waitUntil("flung left") { snap().let { !it.rightEdge && settled(it) } }
        s = snap(); log("fling-left", s)

        // Drag onto the hide target (bottom centre): hidden for an hour, snackbar with Undo.
        val before = System.currentTimeMillis()
        val targetY = s.screenH - ((56 + 28 + 10 + 32) * d)
        swipe(centre(s), Pair(s.screenW / 2f, targetY), steps = 25, durationMs = 900, holdAtEndMs = 400)
        waitUntil("hidden") { snap().visual == BubbleVisual.Hidden }
        s = snap(); log("hidden", s)
        val until = settings.settings.value.hiddenUntilEpochMs
        assertNotNull(until)
        assertTrue(until!! - before in 3_590_000L..3_610_000L)
        assertTrue(s.snackbarShown)
        Log.i(TAG, "snackbar window: " + sh("dumpsys window windows").split("Window #").firstOrNull { it.contains("MurmurBubbleSnack}") }
            ?.lines()?.filter { it.contains("mAttrs") || it.contains("mFrame") || it.contains("mHasSurface") }?.joinToString(" | ") { it.trim().take(200) })
        Log.i(TAG, "a11y windows at snackbar: ${describeWindows()}")
        assertTrue("no input was sent to the orchestrator by dragging", inputs.isEmpty())

        // Undo through the accessibility tree (TalkBack's path): the bubble comes back, the setting is cleared.
        val undo = ctx.getString(R.string.bubble_action_undo)
        var node: AccessibilityNodeInfo? = null
        try {
            waitUntil("Undo in the overlay windows") {
                node = service.windows.mapNotNull { it.root }.firstNotNullOfOrNull { clickableWithText(it, undo) }
                node != null
            }
        } catch (e: AssertionError) {
            val dump = service.windows.joinToString(" || ") { w -> "${w.type}/${w.title}/${w.root?.let { dumpNode(it) }}" }
            throw AssertionError("${e.message}; snackbar=${snap().snackbarShown}; $dump")
        }
        Log.i(TAG, "undo node: clickable=${node!!.isClickable} windows=${describeWindows()}")
        assertTrue(node!!.performAction(AccessibilityNodeInfo.ACTION_CLICK))
        waitUntil("unhidden") { snap().visual == BubbleVisual.Ready }
        waitUntil("setting cleared") { settings.settings.value.hiddenUntilEpochMs == null }
        log("undo", snap())
    }

    @Test
    fun tapsReachTheOrchestrator_listeningShowsTheCancelCircle() {
        withoutHoldToTalk() // a tap must not become a hold when the loaded emulator delays the release
        onMain { host.updateVisibility(focusedInput()) }
        waitUntil("Ready") { snap().visual == BubbleVisual.Ready }
        tap(centre(snap()))
        waitUntil("Tap sent") { inputs == listOf(UserInput.Tap) }
        dictation.value = DictationBubble.Listening(startedAtMs = SystemClock.elapsedRealtime(), hold = false, warn = false)
        waitUntil("cancel shown") { snap().cancelShown }
        val s = snap()
        Shots.save("overlay-listening", checkNotNull(ua.takeScreenshot()))
        // The cancel circle: 40 dp, its centre 50 dp - 20 dp above the pill top, 12 dp + 20 dp from the docked edge.
        val d = ctx.resources.displayMetrics.density
        val cancel = Pair(s.screenW - (32 * d), s.circleTop!! - 30 * d)
        tap(cancel)
        waitUntil("Cancel sent") { inputs.lastOrNull() == UserInput.Cancel }
        // Hold to talk on Ready: press >= 250 ms, release.
        kotlinx.coroutines.runBlocking { settings.update { it.copy(holdToTalk = true) } }
        dictation.value = DictationBubble.Idle
        // Ready, drawn by its own (touchable) window, the cancel circle gone; then a beat for the pill -> circle relayout.
        waitUntil("Ready again") { snap().let { it.visual == BubbleVisual.Ready && !it.cancelShown && it.rootAlpha == 1f } }
        SystemClock.sleep(300)
        val c = centre(snap())
        swipe(c, c, steps = 1, durationMs = 0, holdAtEndMs = 450)
        waitUntil("hold start/release") { inputs.takeLast(2) == listOf(UserInput.HoldStart, UserInput.HoldRelease) }
        Log.i(TAG, "inputs=$inputs")
    }

    @Test
    fun survivesSizeAndRotationChangesWithTheSameEdgeAndProportionalY() {
        withoutHoldToTalk()
        onMain { host.updateVisibility(focusedInput()) }
        waitUntil("Ready") { snap().visual == BubbleVisual.Ready }
        val s0 = snap(); log("inner-0", s0)
        assertNull("no stored position yet", settings.settings.value.bubblePositions["inner"])

        fun check(step: String, expectRight: Boolean, frac: Float) {
            val s = snap(); log(step, s)
            assertEquals("$step edge", expectRight, s.rightEdge)
            assertEquals("$step fraction", frac, s.yFraction!!, 0.02f)
            assertEquals("$step visual", BubbleVisual.Ready, s.visual)
        }

        // Cover size and back (fold / unfold metrics, TOOLCHAIN 4.5).
        sh("wm size 1080x2520")
        waitUntil("cover size") { snap().screenW == 1080 }
        check("cover", true, s0.yFraction!!)
        sh("wm size 1968x2184")
        waitUntil("inner size") { snap().screenW == 1968 }
        check("inner", true, s0.yFraction)

        // Rotation 0-3 (a 180° flip arrives only through DisplayListener.onDisplayChanged).
        sh("settings put system accelerometer_rotation 0")
        for (r in listOf(1, 2, 3, 0)) {
            sh("settings put system user_rotation $r")
            waitUntil("rotation $r") { snap().rotation == r }
            SystemClock.sleep(300)
            check("rot$r", true, s0.yFraction)
        }

        // Left edge, with a stored position on the inner display only: the cover display reprojects it.
        val c = centre(snap())
        Log.i(TAG, "drag from $c; ${bubbleWindowDump()}")
        swipe(c, Pair(snap().screenW * 0.2f, c.second), steps = 20, durationMs = 600, holdAtEndMs = 200)
        waitUntil("left") { snap().let { !it.rightEdge && settled(it) } }
        val left = snap(); log("left", left)
        sh("wm size 1080x2520")
        waitUntil("cover size") { snap().screenW == 1080 }
        check("cover-left", false, left.yFraction!!)
        for (r in listOf(1, 2, 3, 0)) {
            sh("settings put system user_rotation $r")
            waitUntil("rotation $r") { snap().rotation == r }
            SystemClock.sleep(300)
            check("cover-left-rot$r", false, left.yFraction)
        }
        sh("wm size reset")
        waitUntil("inner size") { snap().screenW == 1968 }
        check("inner-left", false, left.yFraction)
    }

    /**
     * Review F2 + F3: a Listening pill dragged onto the bottom centre is moved as itself (the drag layer draws the
     * accent pill, not the Ready circle), the hide target never appears, and the release snaps it to an edge: nothing
     * is hidden, no snackbar, no input reaches the orchestrator.
     */
    @Test
    fun aWorkingPillIsMovedAsItself_andNeverHides() {
        withoutHoldToTalk()
        onMain { host.updateVisibility(focusedInput()) }
        waitUntil("Ready") { snap().visual == BubbleVisual.Ready }
        dictation.value = DictationBubble.Listening(startedAtMs = SystemClock.elapsedRealtime(), hold = false, warn = false)
        waitUntil("Listening with cancel") { snap().let { it.cancelShown && it.visual is BubbleVisual.Active } }
        SystemClock.sleep(500)
        val s = snap(); log("listen-start", s)
        val d = ctx.resources.displayMetrics.density
        val targetY = s.screenH - ((56 + 28 + 10 + 32) * d)
        val to = Pair(s.screenW / 2f, targetY)
        // Hold over the target for longer than its 300 ms delay.
        val (down, last) = press(centre(s), to, steps = 25, durationMs = 900, holdAtEndMs = 600)
        val mid = snap(); log("listen-drag", mid)
        val shot = checkNotNull(ua.takeScreenshot())
        Shots.save("overlay-drag-listening", shot)
        inject(MotionEvent.ACTION_UP, down, to.first, to.second, last + 8)
        assertEquals(BubbleVisual.Dragging(false), mid.visual)
        assertTrue("the drag layer draws the pill", mid.dragDrawsPill)
        assertFalse("no hide target for a working pill", mid.dragTargetShown)
        // The accent pill is under the finger: its docked (right) end, the stop circle, is where the finger holds it, and
        // its body (bars, timer on accent) reaches 138 dp to the left. The Ready circle would be surface-coloured there.
        val accent = app.murmur.ui.design.TokenColors.light.accent
        val body = accentPixels(shot, (to.first - 60 * d).roundToInt(), to.second.roundToInt(), (20 * d).roundToInt(), accent)
        assertTrue("accent pixels of the lifted pill's body: $body", body > (40 * d) * (40 * d) * 0.4f)

        waitUntil("settled as Listening") { snap().let { it.visual is BubbleVisual.Active && it.rootAlpha == 1f && it.cancelShown } }
        val after = snap(); log("listen-settled", after)
        assertNull("hiddenUntil", settings.settings.value.hiddenUntilEpochMs)
        assertFalse(after.snackbarShown)
        assertTrue("no input was sent by moving the pill", inputs.isEmpty())
    }

    /**
     * Review F1: with "Show the bubble: Always" and no field, a tap on the Rest tab opens it to Ready (no input to the
     * orchestrator), and it folds back into the tab 4 s later.
     */
    @Test
    fun theAlwaysTabOpensOnTap_andFoldsBackAfter4s() {
        withoutHoldToTalk()
        onMain { host.updateVisibility(focusedInput().copy(editableFocused = false, showAlways = true)) }
        waitUntil("Rest tab") { snap().visual == BubbleVisual.Rest && snap().rootAlpha == 1f }
        SystemClock.sleep(500)
        val s = snap(); log("always-tab", s)
        val d = ctx.resources.displayMetrics.density
        // The tab's 48 dp hit area is flush with the right edge, centred on the circle's centre line.
        tap(Pair(s.screenW - 24 * d, s.circleTop!! + s.diameterPx / 2f))
        waitUntil("opened to Ready") { snap().visual == BubbleVisual.Ready }
        val opened = SystemClock.uptimeMillis()
        log("always-open", snap())
        assertTrue("opening the tab is overlay-local", inputs.isEmpty())
        waitUntil("folded back", 10_000) { snap().visual == BubbleVisual.Rest }
        val after = SystemClock.uptimeMillis() - opened
        log("always-folded", snap())
        assertTrue("folded back after $after ms", after in 3_000L..6_000L)
    }

    /** Pixels within 16 per channel of [color] in the square of half-size [r] around ([cx], [cy]). */
    private fun accentPixels(b: Bitmap, cx: Int, cy: Int, r: Int, color: androidx.compose.ui.graphics.Color): Int {
        val cr = (color.red * 255).roundToInt(); val cg = (color.green * 255).roundToInt(); val cb = (color.blue * 255).roundToInt()
        var n = 0
        for (dx in -r..r) for (dy in -r..r) {
            val x = cx + dx; val y = cy + dy
            if (x !in 0 until b.width || y !in 0 until b.height) continue
            val p = b.getPixel(x, y)
            if (abs(((p shr 16) and 0xFF) - cr) <= 16 && abs(((p shr 8) and 0xFF) - cg) <= 16 && abs((p and 0xFF) - cb) <= 16) n++
        }
        return n
    }

    /** Review F9: destroy() is final: lifecycle DESTROYED, view models cleared, windows gone, no second attach. */
    @Test
    fun destroyEndsTheLifecycle_clearsViewModels_andRemovesTheWindows() {
        onMain { host.updateVisibility(focusedInput()) }
        waitUntil("Ready") { snap().visual == BubbleVisual.Ready }
        val probe = onMain { androidx.lifecycle.ViewModelProvider(host.owners)[Probe::class.java] }
        onMain { host.detach() }
        assertEquals(androidx.lifecycle.Lifecycle.State.CREATED, onMain { host.owners.lifecycle.currentState })
        assertFalse("detach keeps the view models", probe.cleared)
        assertTrue("attach after detach", onMain { host.attach() }) // detach is reversible
        waitUntil("Ready again") { snap().visual == BubbleVisual.Ready }
        onMain { host.destroy() }
        assertEquals(androidx.lifecycle.Lifecycle.State.DESTROYED, onMain { host.owners.lifecycle.currentState })
        assertTrue("view models cleared", probe.cleared)
        waitUntil("bubble window removed") { !sh("dumpsys window windows").contains("MurmurBubble}") }
        val again = onMain { runCatching { host.attach() }.exceptionOrNull() }
        assertTrue("attach after destroy: $again", again is IllegalStateException)
    }

    class Probe : androidx.lifecycle.ViewModel() {
        @Volatile var cleared = false
        override fun onCleared() { cleared = true }
    }

    /**
     * Posture CLOSED / OPENED through the emulator's hinge sensor (TOOLCHAIN 4.5). The console command runs on the
     * host: tools/run_emulator_tests.sh passes `-e hinge host` and answers the WP03_HINGE requests in logcat.
     *
     * A survival check only: on the headless emulator the hinge changes device_state but not the display, so nothing
     * reaches the host and the edge and fraction cannot change here. Re-placement on a real fold is DEVICE_CHECKLIST
     * DC10; the display-change path itself is covered by the wm size and rotation test.
     */
    @Test
    fun survivesPostureChanges() {
        assumeTrue("needs the host-side hinge driver (-e hinge host)", InstrumentationRegistry.getArguments().getString("hinge") == "host")
        onMain { host.updateVisibility(focusedInput()) }
        waitUntil("Ready") { snap().visual == BubbleVisual.Ready }
        val s0 = snap(); log("posture-open", s0)
        for ((angle, state) in listOf(0 to "CLOSED", 180 to "OPENED")) {
            Log.i(TAG, "WP03_HINGE $angle")
            waitUntil("device_state $state", 30_000) { committedState().contains(state) }
            SystemClock.sleep(500)
            val s = snap(); log("posture-$state", s)
            assertEquals(s0.rightEdge, s.rightEdge)
            assertEquals(s0.yFraction!!, s.yFraction!!, 0.02f)
            assertEquals(BubbleVisual.Ready, s.visual)
        }
    }

    private fun bubbleWindowDump(): String =
        sh("dumpsys window windows").split("Window #").firstOrNull { it.contains("MurmurBubble}") }.orEmpty()
            .lines().filter { it.contains("mFrame") || it.contains("ty=") || it.contains("touchable", ignoreCase = true) }
            .joinToString(" | ") { it.trim().take(160) }

    /** The clickable node that carries [text] (itself, or the nearest clickable ancestor), as TalkBack activates it. */
    private fun clickableWithText(n: AccessibilityNodeInfo, text: String): AccessibilityNodeInfo? {
        if (n.text?.toString() == text || n.contentDescription?.toString() == text) {
            var c: AccessibilityNodeInfo? = n
            while (c != null && !c.isClickable) c = c.parent
            return c
        }
        for (i in 0 until n.childCount) n.getChild(i)?.let { child -> clickableWithText(child, text)?.let { return it } }
        return null
    }

    private fun dumpNode(n: AccessibilityNodeInfo, depth: Int = 0): String {
        if (depth > 6) return ""
        val kids = (0 until n.childCount).mapNotNull { n.getChild(it) }.joinToString(",") { dumpNode(it, depth + 1) }
        return "[${n.className?.toString()?.substringAfterLast('.')} t=${n.text} d=${n.contentDescription}${if (kids.isEmpty()) "" else " $kids"}]"
    }

    private fun describeWindows(): String =
        service.windows.joinToString { "${it.type}/${it.root?.packageName}/${it.title}/${it.root?.let { r -> dumpNode(r) }}" }

    private fun committedState(): String =
        sh("dumpsys device_state").lines().firstOrNull { it.contains("mCommittedState") }.orEmpty()

    // --- input injection (display coordinates)

    private fun centre(s: OverlayHost.Snapshot): Pair<Float, Float> {
        val d = ctx.resources.displayMetrics.density
        val x = if (s.rightEdge) s.screenW - 8 * d - s.diameterPx / 2f else 8 * d + s.diameterPx / 2f
        return Pair(x, s.circleTop!! + s.diameterPx / 2f)
    }

    private fun inject(action: Int, down: Long, x: Float, y: Float, at: Long = SystemClock.uptimeMillis()) {
        val ev = MotionEvent.obtain(down, at, action, x, y, 0)
        ev.source = InputDevice.SOURCE_TOUCHSCREEN
        assertTrue(ua.injectInputEvent(ev, true))
        ev.recycle()
    }

    /**
     * The controller moves the bubble at once, WindowManager a little later: on a loaded emulator a press injected right
     * after a snap can land on the app underneath. Wait until an overlay window really covers [p].
     */
    private fun awaitOverlayAt(p: Pair<Float, Float>) {
        val r = android.graphics.Rect()
        // The bubble (<= 280 dp + band), cancel and snackbar windows; never the full-screen drag layer on its way out.
        val maxW = (400 * ctx.resources.displayMetrics.density).roundToInt()
        waitUntil("an overlay window at $p", 5_000) {
            service.windows.any { w ->
                w.getBoundsInScreen(r)
                w.type == AccessibilityWindowInfo.TYPE_ACCESSIBILITY_OVERLAY && r.width() < maxW &&
                    r.contains(p.first.roundToInt(), p.second.roundToInt())
            }
        }
        SystemClock.sleep(100)
    }

    private fun tap(p: Pair<Float, Float>, onOverlay: Boolean = true) {
        if (onOverlay) awaitOverlayAt(p)
        val down = SystemClock.uptimeMillis()
        inject(MotionEvent.ACTION_DOWN, down, p.first, p.second, down)
        SystemClock.sleep(60)
        inject(MotionEvent.ACTION_UP, down, p.first, p.second, down + 60)
    }

    private fun swipe(from: Pair<Float, Float>, to: Pair<Float, Float>, steps: Int, durationMs: Long, holdAtEndMs: Long) {
        val (down, last) = press(from, to, steps, durationMs, holdAtEndMs)
        inject(MotionEvent.ACTION_UP, down, to.first, to.second, last + 8)
    }

    /** DOWN at [from], MOVEs to [to], a still finger for [holdAtEndMs]; the finger stays down. Returns (down, last). */
    private fun press(from: Pair<Float, Float>, to: Pair<Float, Float>, steps: Int, durationMs: Long, holdAtEndMs: Long): Pair<Long, Long> {
        awaitOverlayAt(from)
        // Event times are the finger's own clock (as a touchscreen stamps them), independent of how long each
        // synchronous injection takes on a loaded emulator.
        val down = SystemClock.uptimeMillis()
        inject(MotionEvent.ACTION_DOWN, down, from.first, from.second, down)
        val dt = durationMs / steps
        for (i in 1..steps) {
            SystemClock.sleep(dt)
            val t = i / steps.toFloat()
            inject(MotionEvent.ACTION_MOVE, down, from.first + (to.first - from.first) * t, from.second + (to.second - from.second) * t, down + i * dt)
        }
        var last = down + steps * dt
        if (holdAtEndMs > 0) {
            // Still finger: the release velocity falls to zero.
            val end = SystemClock.uptimeMillis() + holdAtEndMs
            while (SystemClock.uptimeMillis() < end) {
                SystemClock.sleep(50)
                last = maxOf(last + 50, SystemClock.uptimeMillis())
                inject(MotionEvent.ACTION_MOVE, down, to.first, to.second, last)
            }
        }
        return down to last
    }

    private companion object {
        const val TAG = "WP03"
        const val TARGET_PKG = "app.murmur.testtarget"
    }
}
