package app.murmur.platform.overlay

import android.annotation.SuppressLint
import android.content.ComponentCallbacks
import android.content.Context
import android.content.res.Configuration
import android.database.ContentObserver
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.provider.Settings as SystemSettings
import android.text.format.DateFormat
import android.util.Log
import android.view.Display
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.FrameLayout
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.platform.ComposeView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import androidx.lifecycle.ViewModelStore
import androidx.lifecycle.ViewModelStoreOwner
import androidx.lifecycle.setViewTreeLifecycleOwner
import androidx.lifecycle.setViewTreeViewModelStoreOwner
import androidx.savedstate.SavedStateRegistry
import androidx.savedstate.SavedStateRegistryController
import androidx.savedstate.SavedStateRegistryOwner
import androidx.savedstate.setViewTreeSavedStateRegistryOwner
import app.murmur.core.dictation.DictationBubble
import app.murmur.core.dictation.HapticKind
import app.murmur.core.dictation.UserInput
import app.murmur.core.store.BubbleSize
import app.murmur.core.store.Settings
import app.murmur.core.store.SettingsRepository
import app.murmur.platform.overlay.ui.BubbleActions
import app.murmur.platform.overlay.ui.BubbleUi
import app.murmur.platform.overlay.ui.BubbleWindowContent
import app.murmur.platform.overlay.ui.CancelButton
import app.murmur.platform.overlay.ui.DragLayer
import app.murmur.platform.overlay.ui.DragScene
import app.murmur.platform.overlay.ui.HiddenSnackbar
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.util.Date
import kotlin.math.roundToInt

/**
 * Compose bubble in a TYPE_ACCESSIBILITY_OVERLAY window (no SYSTEM_ALERT_WINDOW; platform spike a), one per
 * display. Manual Lifecycle/SavedState/ViewModelStore owners, WRAP_CONTENT + FLAG_NOT_FOCUSABLE windows, hide via
 * alpha + FLAG_NOT_TOUCHABLE (never View.GONE: fold BLAST stall), re-place on DisplayListener.onDisplayChanged (180°
 * rotations arrive only there).
 *
 * Windows: the bubble (docked with gravity on its side so pills grow toward the screen centre), the cancel circle
 * while Listening, a full-screen untouchable drag layer while dragging, and the "hidden until" snackbar.
 * The logic lives in [BubbleController], [BubbleGestureRecognizer] and [BubblePlacement] (JVM-tested); this class
 * is the Android shell around them. Main thread only.
 *
 * [context] must be the AccessibilityService (window token of the accessibility overlay type).
 */
class OverlayHost(
    private val context: Context,
    private val displayId: Int,
    private val dictation: StateFlow<DictationBubble>,
    private val levels: StateFlow<FloatArray>,
    private val onInput: (UserInput) -> Unit,
    private val settings: SettingsRepository,
    private val clock: () -> Long = System::currentTimeMillis,
    /** Monotonic clock matching MurmurClock.elapsedMs() (used for the listening timer). */
    private val monotonicClock: () -> Long = SystemClock::elapsedRealtime,
) {

    /**
     * The view-tree owners of the bubble windows. Kept off OverlayHost's own type so that its users (the accessibility
     * service in :platform:a11y) need no lifecycle classes on their compile classpath. Internal for the window tests.
     */
    internal val owners = Owners()

    internal class Owners : LifecycleOwner, SavedStateRegistryOwner, ViewModelStoreOwner {
        val registry = LifecycleRegistry(this)
        val savedState = SavedStateRegistryController.create(this)
        override val lifecycle: Lifecycle get() = registry
        override val savedStateRegistry: SavedStateRegistry get() = savedState.savedStateRegistry
        override val viewModelStore = ViewModelStore()
    }

    private val lifecycleRegistry get() = owners.registry
    private val savedState get() = owners.savedState
    private val viewModelStore get() = owners.viewModelStore

    private val main = Handler(Looper.getMainLooper())
    private val display: Display = requireNotNull(context.getSystemService(DisplayManager::class.java).getDisplay(displayId)) {
        "display $displayId is gone"
    }
    /** Window context bound to the target display and the accessibility overlay type. */
    private val ctx: Context = context.createWindowContext(display, WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY, null)
    private val wm: WindowManager = ctx.getSystemService(WindowManager::class.java)

    private var scope: CoroutineScope? = null
    private var attached = false
    private var destroyed = false

    private val controller = BubbleController(clock, ControllerOutput())
    private val recognizer = BubbleGestureRecognizer(ctx.resources.displayMetrics.density) { pillRectOnScreen() }

    // --- state that the Compose windows read
    private var ui by mutableStateOf(BubbleUi(BubbleVisual.Ready, shown = false, rightEdge = true,
        metrics = BubbleMetrics.of(BubbleSize.STANDARD), reducedMotion = false, talkBack = false))
    private var dark by mutableStateOf(isDark(ctx.resources.configuration))
    private var cancelShown by mutableStateOf(false)
    private var dragScene by mutableStateOf<DragScene?>(null)
    private var snackText by mutableStateOf<String?>(null)

    // --- placement state (pixels of this display)
    private var currentSettings = Settings()
    private var metrics = BubbleMetrics.of(BubbleSize.STANDARD)
    private var rightEdge = true
    private var circleTop: Int? = null
    private var lastInput: PlacementInput? = null
    /** Fraction of the usable range and display size at the last explicit placement (reprojection keeps the fraction). */
    private var placedFraction = BubblePlacement.DEFAULT_FRACTION
    private var placedSize: Pair<Int, Int>? = null
    private var lastShownVisual: BubbleVisual = BubbleVisual.Ready
    private var lastVisual: BubbleVisual = BubbleVisual.Hidden
    private var visualRectInWindow: Rect? = null
    private var actionRectInWindow: Rect? = null
    private var dragGrab: Offset? = null
    private var dragInput: PlacementInput? = null
    private var settling = false
    private var reducedMotion = readReducedMotion()

    private val root = TouchRoot(ctx)
    private val bubbleActions = Actions()
    private val bubbleLp = overlayParams(WindowManager.LayoutParams.WRAP_CONTENT, WindowManager.LayoutParams.WRAP_CONTENT, title = "MurmurBubble")
    private var lastAppliedBubbleLp: AppliedLp? = null
    private var cancelView: View? = null
    private val cancelLp = overlayParams(WindowManager.LayoutParams.WRAP_CONTENT, WindowManager.LayoutParams.WRAP_CONTENT, title = "MurmurBubbleCancel")
    private var lastAppliedCancelLp: AppliedLp? = null
    private var dragView: View? = null
    private var snackView: View? = null
    private val dismissSnack = Runnable { removeSnack() }
    private val deadlineRunnable = Runnable { refresh("deadline") }
    private val holdTick = Runnable { handle(recognizer.tick(SystemClock.uptimeMillis())) }
    private val showTarget = Runnable { dragScene = dragScene?.let { if (it.hideArmed) it.copy(targetShown = true) else it } }
    private val fadedOut = Runnable { if (lastVisual == BubbleVisual.Hidden) root.alpha = 0f }
    private val warmCancel = Runnable { if (attached && cancelView == null) circleTop?.let { ensureCancelView(it, touchable = false) } }

    private data class AppliedLp(val gravity: Int, val x: Int, val y: Int, val flags: Int)

    init {
        root.setViewTreeLifecycleOwner(owners)
        root.setViewTreeSavedStateRegistryOwner(owners)
        root.setViewTreeViewModelStoreOwner(owners)
        root.addOnAttachStateChangeListener(object : View.OnAttachStateChangeListener {
            override fun onViewAttachedToWindow(v: View) {
                if (attached && lastAppliedBubbleLp == null) positionBubble(lastShownVisual)
            }
            override fun onViewDetachedFromWindow(v: View) {
                lastAppliedBubbleLp = null
            }
        })
        root.addView(composeView {
            val lv = levels.collectAsState()
            BubbleWindowContent(ui, dark, lv, monotonicClock, bubbleActions)
        })
    }

    /**
     * Shows the bubble on this display. Idempotent; allowed again after [detach], never after [destroy]. Returns
     * whether the bubble window is attached: false when WindowManager refused it (bad token, display gone).
     */
    fun attach(): Boolean {
        check(!destroyed) { "OverlayHost for display $displayId was destroyed" }
        if (attached) return true
        if (lifecycleRegistry.currentState == Lifecycle.State.INITIALIZED) savedState.performRestore(null)
        lifecycleRegistry.currentState = Lifecycle.State.RESUMED
        val s = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
        scope = s
        setTouchable(bubbleLp, false)
        root.alpha = 0f
        try {
            wm.addView(root, bubbleLp)
            attached = true
        } catch (e: RuntimeException) {
            // BadTokenException (service not connected yet) or a display that went away: no bubble on this display.
            Log.w(TAG, "bubble_attach_failed display=$displayId ${e.javaClass.simpleName}")
            s.cancel(); scope = null
            lifecycleRegistry.currentState = Lifecycle.State.CREATED
            return false
        }
        BubbleViews.register(root) { lastVisual != BubbleVisual.Hidden }
        ctx.registerComponentCallbacks(configCallbacks)
        context.getSystemService(DisplayManager::class.java).registerDisplayListener(displayListener, main)
        ctx.contentResolver.registerContentObserver(
            SystemSettings.Global.getUriFor(SystemSettings.Global.ANIMATOR_DURATION_SCALE), false, animationObserver)
        s.launch { settings.settings.collect { onSettings(it) } }
        s.launch { dictation.collect { controller.dictation = it; refresh("dictation") } }
        Log.i(TAG, "bubble_attached display=$displayId type=${bubbleLp.type} bounds=${wm.currentWindowMetrics.bounds.toShortString()}")
        refresh("attach")
        main.post(warmCancel)
        return true
    }

    fun updateVisibility(input: VisibilityInput) {
        if (Looper.myLooper() != Looper.getMainLooper()) { main.post { updateVisibility(input) }; return }
        if (input == controller.visibility && lastInput != null) return
        controller.visibility = input
        refresh("visibility")
    }

    /**
     * Removes every window of this display and stops listening; the host can [attach] again later (the service was
     * interrupted, the display turned off). The owners stay CREATED. For the end of the host use [destroy].
     */
    fun detach() {
        if (!attached) return
        attached = false
        main.removeCallbacks(deadlineRunnable); main.removeCallbacks(holdTick); main.removeCallbacks(showTarget)
        main.removeCallbacks(fadedOut); main.removeCallbacks(warmCancel)
        dragInput = null
        recognizer.cancel()
        removeDragLayer(); removeSnack()
        cancelView?.let { runCatching { wm.removeViewImmediate(it) } }; cancelView = null
        lastAppliedCancelLp = null
        runCatching { wm.removeViewImmediate(root) }
        lastAppliedBubbleLp = null
        BubbleViews.unregister(root)
        runCatching { ctx.unregisterComponentCallbacks(configCallbacks) }
        runCatching { context.getSystemService(DisplayManager::class.java).unregisterDisplayListener(displayListener) }
        runCatching { ctx.contentResolver.unregisterContentObserver(animationObserver) }
        scope?.cancel(); scope = null
        lifecycleRegistry.currentState = Lifecycle.State.CREATED
        Log.i(TAG, "bubble_detached display=$displayId")
    }

    /**
     * Final: [detach], then the lifecycle goes to DESTROYED (the Compose compositions dispose) and the view models are
     * cleared. Call it when the display is removed (DisplayListener.onDisplayRemoved) and when the service unbinds.
     */
    fun destroy() {
        if (destroyed) return
        detach()
        destroyed = true
        if (lifecycleRegistry.currentState == Lifecycle.State.INITIALIZED) lifecycleRegistry.currentState = Lifecycle.State.CREATED
        lifecycleRegistry.currentState = Lifecycle.State.DESTROYED
        viewModelStore.clear()
        Log.i(TAG, "bubble_destroyed display=$displayId")
    }

    // ------------------------------------------------------------------------------------------------------------
    // State -> windows

    private fun onSettings(s: Settings) {
        val sizeChanged = s.bubbleSize.dp != metrics.sizeDp
        currentSettings = s
        metrics = BubbleMetrics.of(s.bubbleSize)
        controller.hiddenUntilMs = s.hiddenUntilEpochMs
        controller.holdToTalk = s.holdToTalk
        refresh(if (sizeChanged) "size" else "settings", replace = sizeChanged)
    }

    /**
     * Recompute the visual, re-place when the context changed and the bubble is idle (the bubble never jumps under
     * the finger while a dictation runs), update the windows, and schedule the next policy deadline.
     */
    private fun refresh(reason: String, replace: Boolean = false) {
        if (!attached) return
        val now = clock()
        val visual = controller.visual(now)
        val input = placementInput()
        val idleVisual = visual == BubbleVisual.Ready || visual == BubbleVisual.Rest || visual == BubbleVisual.Hidden
        val becameVisible = lastVisual == BubbleVisual.Hidden && visual != BubbleVisual.Hidden
        if (controller.drag == null && !settling) {
            val top = circleTop
            val sizeChanged = placedSize != null && placedSize != input.screenW to input.screenH
            val changed = input.withoutCurrent() != lastInput?.withoutCurrent()
            // Without a field and without a stored position the bubble keeps its proportional y (DESIGN 5.4, platform
            // spike a): insets that change mid-transition, the keyboard, a fold or a rotation never shift that fraction.
            val proportional = input.stored == null && input.fieldTop == null
            when {
                top == null -> place(BubblePlacement.place(input), input, reason)
                // Fold / unfold / rotation / Size setting: same edge and proportional y, unless this display (cover vs
                // inner) has a stored position. A focused field re-places with the next visibility input.
                sizeChanged || replace || (proportional && (becameVisible || (idleVisual && changed))) ->
                    if (input.stored != null) place(BubblePlacement.place(input), input, reason)
                    else place(proportionalPlacement(input), input, reason, keepFraction = true)
                becameVisible || (idleVisual && changed) -> place(BubblePlacement.place(input), input, reason)
                else -> circleTop = BubblePlacement.clampY(top, input)
            }
        }
        lastInput = input
        lastVisual = visual
        if (visual != BubbleVisual.Hidden && visual !is BubbleVisual.Dragging) lastShownVisual = visual
        if (visual !is BubbleVisual.Active || (visual.state !is DictationBubble.Error)) actionRectInWindow = null

        val shown = visual != BubbleVisual.Hidden
        ui = BubbleUi(
            visual = lastShownVisual, shown = shown, rightEdge = rightEdge, metrics = metrics,
            reducedMotion = reducedMotion, talkBack = controller.effectiveVisibility(now).talkBackOn,
        )
        val dragging = controller.drag != null || settling
        main.removeCallbacks(fadedOut)
        when {
            dragging -> root.alpha = 0f // the drag layer draws the lifted bubble
            shown -> root.alpha = 1f
            // Hidden: Compose fades the content out (150 ms), then the root goes to alpha 0 (never View.GONE).
            root.alpha > 0f -> main.postDelayed(fadedOut, FADE_OUT_MS)
        }
        root.importantForAccessibility = if (shown) View.IMPORTANT_FOR_ACCESSIBILITY_AUTO else View.IMPORTANT_FOR_ACCESSIBILITY_NO_HIDE_DESCENDANTS
        // A running gesture keeps its touch stream even when the window turns untouchable.
        setTouchable(bubbleLp, shown && !settling)
        positionBubble(lastShownVisual)
        updateCancel(visual, dragging)
        if (controller.drag != null) updateDragScene()

        main.removeCallbacks(deadlineRunnable)
        controller.nextDeadline(now)?.let { main.postDelayed(deadlineRunnable, (it - now).coerceAtLeast(0L) + 1L) }
    }

    /** [keepFraction]: the placement was derived from [placedFraction] (and maybe clamped); keep the original. */
    private fun place(p: Placement, input: PlacementInput, reason: String, keepFraction: Boolean = false) {
        rightEdge = p.rightEdge
        circleTop = p.y
        if (!keepFraction) placedFraction = BubblePlacement.fractionOf(p.y, input)
        placedSize = input.screenW to input.screenH
        Log.i(TAG, "bubble_placed display=$displayId reason=$reason w=${input.screenW} h=${input.screenH} " +
            "rot=${display.rotation} right=${p.rightEdge} y=${p.y} frac=${"%.3f".format(placedFraction)}")
    }

    private fun proportionalPlacement(input: PlacementInput) = Placement(
        rightEdge, BubblePlacement.xFor(rightEdge, input.screenW, input.bubbleDiameterPx, input.density),
        BubblePlacement.clampY(BubblePlacement.yForFraction(placedFraction, input), input),
    )

    private fun positionBubble(visual: BubbleVisual) {
        val top = circleTop ?: return
        val d = density()
        bubbleLp.gravity = Gravity.TOP or if (rightEdge) Gravity.RIGHT else Gravity.LEFT
        bubbleLp.x = ((metrics.edgeMargin(visual) - metrics.windowPadEdge(visual)) * d).roundToInt()
        val visualTop = top + ((metrics.diameter - metrics.visualHeight(visual)) / 2f * d).roundToInt()
        bubbleLp.y = visualTop - (metrics.windowPadTop(visual) * d).roundToInt()
        val next = AppliedLp(bubbleLp.gravity, bubbleLp.x, bubbleLp.y, bubbleLp.flags)
        if (next != lastAppliedBubbleLp && update(root, bubbleLp)) {
            lastAppliedBubbleLp = next
        }
    }

    private fun updateCancel(visual: BubbleVisual, dragging: Boolean) {
        val listening = visual is BubbleVisual.Active && (visual.state as? DictationBubble.Listening)?.hold == false && !dragging
        val top = circleTop
        if (listening && cancelView == null && top != null) {
            ensureCancelView(top, touchable = true)
        }
        cancelShown = listening
        val view = cancelView ?: return
        setTouchable(cancelLp, listening)
        if (top != null) positionCancel(top)
        applyCancelLayout(view)
    }

    private fun ensureCancelView(top: Int, touchable: Boolean) {
        if (cancelView != null) return
        setTouchable(cancelLp, touchable)
        positionCancel(top)
        cancelView = composeView { CancelButton(dark, cancelShown) { controller.onTap(TouchRegion.CANCEL) } }.also { v ->
            v.addOnAttachStateChangeListener(object : View.OnAttachStateChangeListener {
                override fun onViewAttachedToWindow(view: View) {
                    if (attached) applyCancelLayout(view)
                }
                override fun onViewDetachedFromWindow(view: View) {
                    lastAppliedCancelLp = null
                }
            })
            runCatching { wm.addView(v, cancelLp) }
                .onSuccess { lastAppliedCancelLp = AppliedLp(cancelLp.gravity, cancelLp.x, cancelLp.y, cancelLp.flags) }
                .onFailure { e -> Log.w(TAG, "cancel_attach_failed ${e.javaClass.simpleName}"); cancelView = null }
        }
    }

    private fun applyCancelLayout(view: View) {
        val next = AppliedLp(cancelLp.gravity, cancelLp.x, cancelLp.y, cancelLp.flags)
        if (next != lastAppliedCancelLp && update(view, cancelLp)) {
            lastAppliedCancelLp = next
        }
    }

    private fun positionCancel(circleTop: Int) {
        val d = density()
        val pad = (BubbleMetrics.CANCEL_HIT - BubbleMetrics.CANCEL_DIAMETER) / 2f
        val padPx = (pad * d).roundToInt()
        val hitPx = (BubbleMetrics.CANCEL_HIT * d).roundToInt()
        val input = lastInput ?: placementInput()
        val aboveY = circleTop - ((metrics.cancelOffsetY + pad) * d).roundToInt()
        val gap = (metrics.cancelOffsetY - BubbleMetrics.CANCEL_DIAMETER).coerceAtLeast(8f)
        val belowY = circleTop + ((metrics.diameter + gap - pad) * d).roundToInt()
        // Flip below the pill when docked near the top edge so the cancel button never clips off-screen.
        val rawY = if (aboveY + padPx < input.safeTop) belowY else aboveY
        val minY = (input.safeTop - padPx).coerceAtLeast(0)
        val maxY = (input.safeBottom - hitPx + padPx).coerceAtLeast(minY)
        cancelLp.gravity = Gravity.TOP or if (rightEdge) Gravity.RIGHT else Gravity.LEFT
        cancelLp.x = ((BubblePlacement.EDGE_MARGIN_DP + metrics.cancelInset - pad) * d).roundToInt()
        cancelLp.y = rawY.coerceIn(minY, maxY)
    }

    // ------------------------------------------------------------------------------------------------------------
    // Touch

    private fun onTouch(ev: MotionEvent) {
        val t = ev.eventTime
        when (ev.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                BubbleViews.touched(root, clock())
                val action = actionRectInWindow
                val region = if (action != null && action.contains(Offset(ev.x, ev.y))) TouchRegion.ACTION else TouchRegion.BODY
                val p = controller.pressPolicy(region)
                dragGrab = grabOffset(ev.rawX, ev.rawY)
                handle(recognizer.down(ev.rawX, ev.rawY, t, p))
                recognizer.holdDeadline()?.let { main.postAtTime(holdTick, it) }
            }
            MotionEvent.ACTION_MOVE -> handle(recognizer.move(ev.rawX, ev.rawY, t))
            MotionEvent.ACTION_UP -> { main.removeCallbacks(holdTick); handle(recognizer.up(ev.rawX, ev.rawY, t)) }
            MotionEvent.ACTION_CANCEL -> { main.removeCallbacks(holdTick); handle(recognizer.cancel()) }
        }
    }

    private fun handle(gestures: List<Gesture>) {
        if (gestures.isEmpty()) return
        var onlyDragMove = true
        for (g in gestures) {
            if (g !is Gesture.DragMove) {
                onlyDragMove = false
                if (g is Gesture.DragEnd) Log.i(TAG, "bubble_drag_end display=$displayId vx_dp_s=${g.vxDpPerSec.roundToInt()}")
                else Log.i(TAG, "bubble_gesture display=$displayId $g")
            }
            if (g is Gesture.DragStart) startDrag(g.x, g.y) else controller.onGesture(g)
        }
        if (onlyDragMove) updateDragScene() else refresh("gesture")
    }

    /**
     * Finger offset inside the circle drawn while dragging, from the placement (not from view measurements, which lag
     * a moving window): every visual is centred on the docked circle, and a wide pill is grabbed at its docked end.
     */
    private fun grabOffset(rawX: Float, rawY: Float): Offset {
        val input = lastInput ?: placementInput()
        val d = input.bubbleDiameterPx.toFloat()
        val top = circleTop ?: return Offset(d / 2f, d / 2f)
        val left = BubblePlacement.xFor(rightEdge, input.screenW, input.bubbleDiameterPx, input.density)
        // A pill is moved as itself (the drag layer draws it), so it keeps the grab point anywhere on the pill.
        val pill = if (lastShownVisual is BubbleVisual.Active) pillRectOnScreen() else null
        if (pill != null) return Offset(rawX.coerceIn(pill.left, pill.right) - left, rawY.coerceIn(pill.top, pill.bottom) - top)
        return Offset((rawX - left).coerceIn(0f, d), (rawY - top).coerceIn(0f, d))
    }

    private fun startDrag(x: Float, y: Float) {
        val input = placementInput()
        dragInput = input
        val d = density()
        val grab = dragGrab ?: Offset(input.bubbleDiameterPx / 2f, input.bubbleDiameterPx / 2f)
        controller.startDrag(x, y, DragGeometry(input, grab.x, grab.y, input.screenW / 2f, targetCenter(input), BubbleMetrics.MAGNET_RADIUS * d))
        addDragLayer()
        updateDragScene()
        main.postDelayed(showTarget, TARGET_DELAY_MS)
        Log.i(TAG, "bubble_drag_start display=$displayId")
    }

    private fun updateDragScene() {
        val s = controller.drag ?: return
        val input = dragInput ?: placementInput()
        val prev = dragScene
        // The pill being moved, live (a dictation can change state mid-drag); null = the idle bubble: the Ready circle.
        val pill = controller.dictation.takeIf { it !is DictationBubble.Idle }
            ?.let { ui.copy(visual = BubbleVisual.Active(it), shown = true) }
        dragScene = DragScene(
            circleX = s.circleX, circleY = s.circleY, diameterPx = input.bubbleDiameterPx, screenW = input.screenW,
            guideTop = BubblePlacement.topLimit(input), guideBottom = BubblePlacement.bottomLimit(input),
            ghostX = BubblePlacement.xFor(s.snapRightEdge, input.screenW, input.bubbleDiameterPx, input.density), ghostY = s.snapY,
            overTarget = s.overHideTarget,
            targetCenterX = input.screenW / 2f,
            targetCenterY = targetCenter(input),
            targetShown = (prev?.targetShown ?: false) && s.hideArmed, settleTo = null, reducedMotion = reducedMotion,
            hideArmed = s.hideArmed, pill = pill,
        )
    }

    /** Centre of the hide target: the target group (64 dp circle, 10 dp gap, label) ends 56 dp above the screen bottom. */
    private fun targetCenter(input: PlacementInput): Float {
        val d = density()
        val labelH = 16f * ctx.resources.configuration.fontScale * d + 12f * d
        return input.screenH - BubbleMetrics.HIDE_TARGET_BOTTOM * d - labelH - BubbleMetrics.HIDE_LABEL_GAP * d - BubbleMetrics.HIDE_TARGET / 2f * d
    }

    private fun addDragLayer() {
        if (dragView != null) return
        val lp = overlayParams(WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.MATCH_PARENT, title = "MurmurBubbleDrag").apply {
            flags = flags or WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
        }
        dragView = composeView {
            val lv = levels.collectAsState()
            dragScene?.let { DragLayer(it, dark, lv, monotonicClock) { finishSettle() } }
        }.also { v -> runCatching { wm.addView(v, lp) }.onFailure { Log.w(TAG, "drag_layer_failed ${it.javaClass.simpleName}"); dragView = null } }
    }

    private fun finishSettle() {
        settling = false
        dragInput = null
        main.removeCallbacks(showTarget)
        removeDragLayer()
        refresh("settled")
    }

    private fun removeDragLayer() {
        dragView?.let { runCatching { wm.removeViewImmediate(it) } }
        dragView = null
        dragScene = null
    }

    // ------------------------------------------------------------------------------------------------------------
    // Hide 1 h + snackbar

    private fun hideForAnHour() {
        val until = clock() + HIDE_MS
        controller.hiddenUntilMs = until
        scope?.launch { settings.update { it.copy(hiddenUntilEpochMs = until) } }
        settling = false
        dragInput = null
        main.removeCallbacks(showTarget)
        removeDragLayer()
        showSnack(until)
        Log.i(TAG, "bubble_hidden display=$displayId for_ms=$HIDE_MS")
        refresh("hidden")
    }

    private fun showSnack(until: Long) {
        val time = DateFormat.getTimeFormat(ctx).format(Date(until))
        snackText = ctx.getString(R.string.bubble_hidden_snack, time)
        main.removeCallbacks(dismissSnack)
        if (snackView == null) {
            val d = density()
            val input = placementInput()
            val wide = input.screenW / d >= WIDE_DP
            val inset = ((SNACK_INSET_DP - BubbleMetrics.SHADOW_PAD) * d).roundToInt()
            val width = if (wide) ((SNACK_WIDE_DP + 2 * BubbleMetrics.SHADOW_PAD) * d).roundToInt()
                else input.screenW - 2 * inset
            val anchor = listOfNotNull(input.imeTop, input.fieldTop.takeIf { controller.visibility?.editableFocused == true })
                .minOrNull()?.let { it - (SNACK_GAP_DP * d).roundToInt() }
                ?: (input.screenH - (SNACK_BOTTOM_DP * d).roundToInt())
            val lp = overlayParams(width, WindowManager.LayoutParams.WRAP_CONTENT, title = "MurmurBubbleSnack").apply {
                gravity = Gravity.BOTTOM or if (wide) Gravity.RIGHT else Gravity.LEFT
                x = inset
                y = input.screenH - anchor - (BubbleMetrics.SHADOW_PAD * d).roundToInt()
            }
            snackView = composeView {
                snackText?.let { HiddenSnackbar(dark, it) { undoHide() } }
            }.also { v -> runCatching { wm.addView(v, lp) }.onFailure { snackView = null } }
        }
        main.postDelayed(dismissSnack, SNACK_MS)
    }

    private fun undoHide() {
        controller.hiddenUntilMs = null
        scope?.launch { settings.update { it.copy(hiddenUntilEpochMs = null) } }
        removeSnack()
        Log.i(TAG, "bubble_unhidden display=$displayId")
        refresh("undo")
    }

    private fun removeSnack() {
        main.removeCallbacks(dismissSnack)
        snackView?.let { runCatching { wm.removeViewImmediate(it) } }
        snackView = null
        snackText = null
    }

    // ------------------------------------------------------------------------------------------------------------
    // Geometry

    private fun density(): Float = ctx.resources.displayMetrics.density

    private fun placementInput(): PlacementInput {
        val m = wm.currentWindowMetrics
        val b = m.bounds
        val ins = m.windowInsets
        val top = ins.getInsets(WindowInsets.Type.statusBars() or WindowInsets.Type.displayCutout()).top
        val bottom = ins.getInsets(WindowInsets.Type.navigationBars()).bottom
        val d = density()
        val v = controller.visibility?.takeIf { it.editableFocused }
        val key = BubblePlacement.displayKey(displayId, b.width(), b.height(), d)
        return PlacementInput(
            screenW = b.width(), screenH = b.height(), density = d,
            safeTop = top, safeBottom = b.height() - bottom,
            fieldTop = v?.fieldTop, imeTop = controller.visibility?.imeTop,
            stored = currentSettings.bubblePositions[key],
            bubbleDiameterPx = (metrics.diameter * d).roundToInt(),
            currentY = circleTop, currentRightEdge = rightEdge,
        )
    }

    private fun PlacementInput.withoutCurrent() = copy(currentY = null, currentRightEdge = null)

    private fun pillRectOnScreen(): PxRect? {
        val r = visualRectInWindow ?: return null
        val loc = IntArray(2)
        root.getLocationOnScreen(loc)
        return PxRect(loc[0] + r.left, loc[1] + r.top, loc[0] + r.right, loc[1] + r.bottom)
    }

    private fun onDisplayChanged() {
        dragInput = null
        refresh("display")
    }

    // ------------------------------------------------------------------------------------------------------------
    // Plumbing

    private fun overlayParams(w: Int, h: Int, title: String) = WindowManager.LayoutParams(
        w, h,
        WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
            WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
        PixelFormat.TRANSLUCENT,
    ).apply {
        gravity = Gravity.TOP or Gravity.LEFT
        this.title = title
        fitInsetsTypes = 0
        layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS
        setCanPlayMoveAnimation(false)
    }

    private fun setTouchable(lp: WindowManager.LayoutParams, touchable: Boolean) {
        lp.flags = if (touchable) lp.flags and WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE.inv()
        else lp.flags or WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
    }

    private fun update(view: View, lp: WindowManager.LayoutParams): Boolean =
        attached && view.isAttachedToWindow && runCatching { wm.updateViewLayout(view, lp) }.isSuccess

    private fun composeView(content: @androidx.compose.runtime.Composable () -> Unit) = ComposeView(ctx).apply {
        setViewTreeLifecycleOwner(owners)
        setViewTreeSavedStateRegistryOwner(owners)
        setViewTreeViewModelStoreOwner(owners)
        setContent(content)
    }

    private fun isDark(c: Configuration) = (c.uiMode and Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES

    private fun readReducedMotion(): Boolean =
        SystemSettings.Global.getFloat(ctx.contentResolver, SystemSettings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f

    private fun haptic(kind: HapticKind) {
        if (OverlayFeedback.vibrates(currentSettings.feedback)) root.performHapticFeedback(OverlayFeedback.constantFor(kind))
    }

    /** The bubble window's root: every touch goes to the recognizer (raw coordinates); hover (TalkBack) passes through. */
    @SuppressLint("ViewConstructor", "ClickableViewAccessibility")
    private inner class TouchRoot(c: Context) : FrameLayout(c) {
        override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
            onTouch(ev)
            return true
        }
    }

    private inner class Actions : BubbleActions {
        override fun tap(region: TouchRegion) { controller.onTap(region); refresh("a11y") }
        override fun hide() = controller.hideAction()
        override fun visualBounds(rect: Rect) { visualRectInWindow = rect }
        override fun actionBounds(rect: Rect?) { actionRectInWindow = rect }
    }

    private inner class ControllerOutput : BubbleController.Output {
        override fun send(input: UserInput) = onInput(input)
        override fun haptic(kind: HapticKind) = this@OverlayHost.haptic(kind)
        override fun hideForAnHour() = this@OverlayHost.hideForAnHour()
        override fun dragSettled(rightEdge: Boolean, circleTop: Int, stored: Boolean) {
            val input = placementInput()
            val x = BubblePlacement.xFor(rightEdge, input.screenW, input.bubbleDiameterPx, input.density)
            place(Placement(rightEdge, x, circleTop), input, if (stored) "drag" else "drag-cancel")
            if (stored) {
                val key = BubblePlacement.displayKey(displayId, input.screenW, input.screenH, input.density)
                val pos = BubblePlacement.stored(rightEdge, circleTop, input)
                scope?.launch { settings.update { it.copy(bubblePositions = it.bubblePositions + (key to pos)) } }
                Log.i(TAG, "bubble_snapped display=$displayId key=$key right=$rightEdge y=$circleTop frac=${"%.3f".format(pos.yFraction)}")
            }
            settling = true
            main.removeCallbacks(showTarget)
            dragScene = dragScene?.copy(settleTo = Offset(x.toFloat(), circleTop.toFloat()), settleRightEdge = rightEdge)
            if (dragScene == null) finishSettle()
        }
    }

    private val configCallbacks = object : ComponentCallbacks {
        override fun onConfigurationChanged(newConfig: Configuration) {
            dark = isDark(newConfig)
            onDisplayChanged()
        }
        @Deprecated("Deprecated in Java")
        override fun onLowMemory() = Unit
    }

    private val displayListener = object : DisplayManager.DisplayListener {
        override fun onDisplayAdded(id: Int) = Unit
        override fun onDisplayRemoved(id: Int) = Unit
        override fun onDisplayChanged(id: Int) { if (id == displayId) this@OverlayHost.onDisplayChanged() }
    }

    private val animationObserver = object : ContentObserver(main) {
        override fun onChange(selfChange: Boolean) {
            reducedMotion = readReducedMotion()
            refresh("motion")
        }
    }

    /** What an instrumented test reads back (positions are logged as well: bubble_placed / bubble_snapped). Main thread. */
    data class Snapshot(
        val visual: BubbleVisual,
        val rightEdge: Boolean,
        val circleTop: Int?,
        val yFraction: Float?,
        val screenW: Int,
        val screenH: Int,
        val rotation: Int,
        val diameterPx: Int,
        val hiddenUntilMs: Long?,
        val snackbarShown: Boolean,
        val cancelShown: Boolean,
        /** Alpha of the bubble window's root view (0 while hidden or while the drag layer draws the bubble). */
        val rootAlpha: Float,
        /** While dragging: the drag layer draws the working/result pill itself (not the Ready circle). */
        val dragDrawsPill: Boolean,
        /** While dragging: the hide target is armed and shown. */
        val dragTargetShown: Boolean,
        val lifecycle: Lifecycle.State,
    )

    fun snapshot(): Snapshot {
        val input = placementInput()
        val top = circleTop
        return Snapshot(
            visual = controller.visual(), rightEdge = rightEdge, circleTop = top,
            yFraction = top?.let { BubblePlacement.fractionOf(it, input) },
            screenW = input.screenW, screenH = input.screenH, rotation = display.rotation, diameterPx = input.bubbleDiameterPx,
            hiddenUntilMs = controller.hiddenUntilMs, snackbarShown = snackView != null, cancelShown = cancelShown,
            rootAlpha = root.alpha,
            dragDrawsPill = dragScene?.pill != null,
            dragTargetShown = dragScene?.let { it.hideArmed && it.targetShown } == true,
            lifecycle = lifecycleRegistry.currentState,
        )
    }

    private companion object {
        const val TAG = "MurmurBubble"
        const val HIDE_MS = 60L * 60L * 1_000L
        const val TARGET_DELAY_MS = 300L
        const val FADE_OUT_MS = 200L
        const val SNACK_MS = 6_000L
        const val SNACK_INSET_DP = 12f
        const val SNACK_GAP_DP = 12f
        const val SNACK_BOTTOM_DP = 84f
        const val SNACK_WIDE_DP = 380f
        const val WIDE_DP = 600f
    }
}
