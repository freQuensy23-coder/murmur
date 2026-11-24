package app.murmur.platform.a11y

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.InputMethod
import android.content.Intent
import android.hardware.display.DisplayManager
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.Log
import android.view.Display
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityManager
import app.murmur.core.dictation.DictationBubble
import app.murmur.core.dictation.DictationOrchestrator
import app.murmur.core.dictation.UserInput
import app.murmur.core.store.SettingsRepository
import app.murmur.platform.overlay.OverlayHost
import app.murmur.platform.overlay.VisibilityInput
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.android.asCoroutineDispatcher
import kotlinx.coroutines.asExecutor
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Implemented by the Application (:app). Gives the service its collaborators without a DI framework
 * (ADR-010). Only the :main process has it.
 */
interface A11yServiceHost {
    val dictation: DictationOrchestrator
    val settings: SettingsRepository
    /**
     * Called when the service connects/disconnects (Dashboard/Settings "The bubble is off" banner). On disconnect the
     * host also calls DictationOrchestrator.onAccessibilityLost() (ExternalStop(A11Y_OFF) for a running dictation).
     */
    fun onServiceState(connected: Boolean)

    /** Bubble gestures and the notification's "Start dictation" go to the orchestrator. */
    fun onUserInput(input: UserInput) = dictation.onInput(input)

    /**
     * Packages whose UI counts as "Murmur's own UI" (DESIGN 5.2: no bubble there; the practice field has its own).
     * Default: the app itself.
     */
    fun ownUiPackages(service: AccessibilityService): Set<String> = setOf(service.packageName)

    /** The bubble on [displayId] (one per display). Default: WP-03's Compose bubble in an accessibility overlay. */
    fun bubbleFor(service: AccessibilityService, displayId: Int): BubbleSurface =
        OverlayBubbleSurface(
            OverlayHost(service, displayId, dictation.bubble, dictation.levels, ::onUserInput, settings, monotonicClock = dictation::elapsedMs),
        )
}

/** What the service needs from a per-display bubble window. */
interface BubbleSurface {
    /**
     * Shows the bubble. Returns false when no window could be added (bad token, display gone): the service then
     * destroys this surface and tries a fresh one on the display's next change.
     */
    fun attach(): Boolean
    fun updateVisibility(input: VisibilityInput)
    /** Removes the windows; the surface may be attached again (a temporary loss of the window token). */
    fun detach()
    /** Final: the display was removed or the service ends. The service never reuses the surface afterwards. */
    fun destroy() = detach()
}

private class OverlayBubbleSurface(private val host: OverlayHost) : BubbleSurface {
    override fun attach(): Boolean = host.attach()
    override fun updateVisibility(input: VisibilityInput) = host.updateVisibility(input)
    override fun detach() = host.detach()
    override fun destroy() = host.destroy()
}

/**
 * The accessibility service (WP-02). Owns: focus tracking -> VisibilityInput, one bubble per display (DisplayListener:
 * added / removed / changed, fold and unfold), the target snapshot, screenshots, the insertion ladder and the
 * accessibility IME (API 33) for commitText / getSurroundingText. Node reads run on one HandlerThread ("a11y-io"),
 * never on main (ARCHITECTURE 14). Logs carry codes and numbers only (D9).
 */
class MurmurAccessibilityService : AccessibilityService() {
    private val ioThread = HandlerThread("a11y-io")
    private lateinit var io: Handler
    /** Node reads, screenshots and their callbacks; survives onDestroy without cancelling or stranding callers. */
    internal val lane = IoLane({ io.post(it) }, Dispatchers.Default.asExecutor())
    internal val ioDispatcher: CoroutineDispatcher get() = lane.dispatcher
    private val main = Handler(Looper.getMainLooper())
    private var scope: CoroutineScope? = null

    internal val nodes = NodeReader(this)
    /** Bumped on every TYPE_WINDOWS_CHANGED (snapshotAfterShade waits on it). */
    internal val windowsVersion = MutableStateFlow(0L)
    @Volatile internal var ime: MurmurInputMethod? = null
        private set

    val targetReader = A11yTargetReader(this)
    val inserter = A11yInserter(this)

    private var host: A11yServiceHost? = null
    private var tracker: FocusTracker? = null
    private val surfaces = BubbleSurfaces { displayId -> checkNotNull(host).bubbleFor(this, displayId) }
    private var lastVisibility: Pair<Int, VisibilityInput>? = null
    private val focusReadQueued = AtomicBoolean(false)
    @Volatile private var connected = false

    override fun onCreate() {
        super.onCreate()
        ioThread.start()
        io = Handler(ioThread.looper)
    }

    override fun onCreateInputMethod(): InputMethod = MurmurInputMethod(this, ::requestFocusRead).also { ime = it }

    override fun onServiceConnected() {
        super.onServiceConnected()
        if (connected) return
        connected = true
        val h = application as? A11yServiceHost
        host = h
        val s = CoroutineScope(SupervisorJob() + main.asCoroutineDispatcher("a11y-main"))
        scope = s
        A11yStatus.get(this).onServiceConnected()
        val t = FocusTracker(s, h?.ownUiPackages(this) ?: setOf(packageName), System::currentTimeMillis, ::pushVisibility)
        tracker = t

        val dm = getSystemService(DisplayManager::class.java)
        dm.displays.filter { it.isPublic() }.forEach { attachBubble(it.displayId) }
        dm.registerDisplayListener(displayListener, main)

        val am = getSystemService(AccessibilityManager::class.java)
        t.onTalkBack(am.isTouchExplorationEnabled)
        am.addTouchExplorationStateChangeListener(talkBackListener)

        if (h != null) {
            s.launch { h.settings.settings.map(VisibilitySettings::of).distinctUntilChanged().collect { t.onSettings(it) } }
            s.launch {
                var previous: DictationBubble? = null
                h.dictation.bubble.collect { b ->
                    if (b is DictationBubble.Idle && previous != null && previous !is DictationBubble.Idle) t.onDictationIdle()
                    previous = b
                }
            }
        }
        A11yBridge.service.value = this
        Log.i(TAG, "service_state connected=true displays=${surfaces.size} host=${h != null}")
        h?.onServiceState(true)
        requestFocusRead()
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        when (event?.eventType) {
            AccessibilityEvent.TYPE_WINDOWS_CHANGED -> { windowsVersion.update { it + 1 }; requestFocusRead() }
            AccessibilityEvent.TYPE_VIEW_FOCUSED, AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED -> requestFocusRead()
            else -> Unit // text selection changes: nothing for the bubble
        }
    }

    override fun onInterrupt() {}

    override fun onUnbind(intent: Intent?): Boolean {
        teardown("unbind")
        return super.onUnbind(intent)
    }

    override fun onDestroy() {
        teardown("destroy")
        lane.close()
        windowsVersion.update { it + 1 } // wakes a snapshotAfterShade waiting for a window change
        ioThread.quitSafely()
        super.onDestroy()
    }

    /** Notification start: the shade does not collapse for a broadcast action (API 31 global action). */
    internal fun dismissNotificationShade() {
        performGlobalAction(GLOBAL_ACTION_DISMISS_NOTIFICATION_SHADE)
    }

    /** Conflated: one focus read in flight on a11y-io, re-queued when an event arrives meanwhile. */
    private fun requestFocusRead() {
        if (!connected || !focusReadQueued.compareAndSet(false, true)) return
        io.post {
            focusReadQueued.set(false)
            val f = runCatching { nodes.focus() }.getOrElse {
                Log.w(TAG, "focus_read_failed ${it.javaClass.simpleName}")
                return@post
            }
            main.post { if (connected) tracker?.onFocus(f) }
        }
    }

    private fun pushVisibility(focusDisplayId: Int, input: VisibilityInput) {
        lastVisibility = focusDisplayId to input
        surfaces.forEach { displayId, surface -> surface.updateVisibility(VisibilityRules.forDisplay(input, focusDisplayId, displayId)) }
    }

    private fun attachBubble(displayId: Int) {
        if (host == null) return
        when (val r = surfaces.attach(displayId)) {
            BubbleSurfaces.Result.Present -> Unit
            BubbleSurfaces.Result.Attached -> lastVisibility?.let { (focusDisplay, input) ->
                surfaces[displayId]?.updateVisibility(VisibilityRules.forDisplay(input, focusDisplay, displayId))
            }
            // e.g. a display that went away between the event and the attach (WindowManager.BadTokenException). The
            // surface is not kept: the display's next change tries again with a fresh one.
            is BubbleSurfaces.Result.Failed -> Log.w(TAG, "bubble_retry_later display=$displayId error=${r.error ?: "not_attached"}")
        }
    }

    /** Display removed or service ending: the surface is final ([BubbleSurface.destroy]); a re-added display gets a new one. */
    private fun destroyBubble(displayId: Int) = surfaces.destroy(displayId)

    private fun Display.isPublic() = (flags and Display.FLAG_PRIVATE) == 0

    private val displayListener = object : DisplayManager.DisplayListener {
        override fun onDisplayAdded(displayId: Int) {
            val d = getSystemService(DisplayManager::class.java).getDisplay(displayId) ?: return
            if (d.isPublic()) attachBubble(displayId)
            Log.i(TAG, "display_added id=$displayId bubbles=${surfaces.size}")
        }

        override fun onDisplayRemoved(displayId: Int) {
            destroyBubble(displayId)
            Log.i(TAG, "display_removed id=$displayId bubbles=${surfaces.size}")
        }

        /** Fold/unfold and rotation (a 180° flip arrives only here, platform spike a): focus and bounds may have moved. */
        override fun onDisplayChanged(displayId: Int) {
            // A display whose bubble failed to attach gets another try (F2: the failed surface is not kept).
            if (connected && displayId !in surfaces) {
                getSystemService(DisplayManager::class.java).getDisplay(displayId)?.takeIf { it.isPublic() }?.let { attachBubble(displayId) }
            }
            requestFocusRead()
            tracker?.republish()
        }
    }

    private val talkBackListener = AccessibilityManager.TouchExplorationStateChangeListener { on -> tracker?.onTalkBack(on) }

    private fun teardown(why: String) {
        if (!connected) return
        connected = false
        A11yBridge.service.compareAndSet(this, null)
        runCatching { getSystemService(DisplayManager::class.java).unregisterDisplayListener(displayListener) }
        runCatching { getSystemService(AccessibilityManager::class.java).removeTouchExplorationStateChangeListener(talkBackListener) }
        surfaces.destroyAll()
        scope?.cancel()
        scope = null
        tracker = null
        lastVisibility = null
        Log.i(TAG, "service_state connected=false reason=$why")
        host?.onServiceState(false)
        host = null
    }

    private companion object {
        const val TAG = "MurmurA11y"
    }
}
