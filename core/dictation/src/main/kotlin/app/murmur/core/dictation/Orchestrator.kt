package app.murmur.core.dictation

import app.murmur.core.audio.PcmChunk
import app.murmur.core.audio.PcmFormat
import app.murmur.core.catalog.UnitHealthRepository
import app.murmur.core.context.BuiltInDenylist
import app.murmur.core.context.ContextAssembler
import app.murmur.core.context.ContextSettings
import app.murmur.core.context.FieldKinds
import app.murmur.core.context.PrivacyDecision
import app.murmur.core.context.PrivacyLevel
import app.murmur.core.context.PrivacyReason
import app.murmur.core.context.ScreenOutcome
import app.murmur.core.engine.AsrSession
import app.murmur.core.engine.AsrSessionOptions
import app.murmur.core.engine.Engines
import app.murmur.core.engine.HostState
import app.murmur.core.engine.InferenceHost
import app.murmur.core.engine.LoadSpec
import app.murmur.core.engine.ProcessExits
import app.murmur.core.model.ArtifactId
import app.murmur.core.model.AudioKey
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.EntryId
import app.murmur.core.model.InferProcess
import app.murmur.core.model.ModelKind
import app.murmur.core.model.MurmurClock
import app.murmur.core.model.MurmurError
import app.murmur.core.model.Outcome
import app.murmur.core.model.SessionId
import app.murmur.core.model.Stage
import app.murmur.core.model.getOrNull
import app.murmur.core.ocr.ScreenText
import app.murmur.core.prompt.OutputGuard
import app.murmur.core.prompt.PromptAssets
import app.murmur.core.prompt.PromptRenderer
import app.murmur.core.store.AudioStore
import app.murmur.core.store.CleanupUse
import app.murmur.core.store.DeliveryMetrics
import app.murmur.core.store.EntryStatus
import app.murmur.core.store.HistoryRepository
import app.murmur.core.store.InsertMethod
import app.murmur.core.store.MetricsRepository
import app.murmur.core.store.ModelUse
import app.murmur.core.store.RetentionPolicy
import app.murmur.core.store.Settings
import app.murmur.core.store.SettingsRepository
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.drop
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.isActive
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.coroutines.ContinuationInterceptor
import app.murmur.core.store.Feedback as FeedbackSetting

/** Everything the orchestrator needs; wired in :app (real) or from :testing (fakes, FakeWorld). */
data class DictationDeps(
    val clock: MurmurClock,
    val settings: SettingsRepository,
    val models: ModelResolver,
    val engines: Engines,
    val audio: AudioCapture,
    val target: TargetReader,
    val inserter: Inserter,
    val context: ContextAssembler,
    val renderer: PromptRenderer,
    val guard: OutputGuard,
    val prompts: PromptAssets,
    val history: HistoryRepository,
    val metrics: MetricsRepository,
    val audioStore: AudioStore,
    val feedback: Feedback,
    val memory: MemoryGate,
    val unitHealth: UnitHealthRepository,
    val navigator: AppNavigator,
    val permissions: PermissionMonitor,
    val deliveryMetrics: DeliveryMetrics,
    val system: SystemConditions,
    /** ADR-083: exit records of the inference processes (a death during a load for memory is OutOfMemory). */
    val exits: ProcessExits = ProcessExits.NONE,
)

/**
 * Runs [DictationMachine] and executes its [Effect]s (ARCHITECTURE.md 5, threading 14). Lives in :main, one instance
 * per process (AppContainer). Single-threaded: every event, effect, timer and port callback runs on one serial view
 * (`limitedParallelism(1)`) of [scope]'s dispatcher, so the reducer state and the session memory have one writer.
 * Timers use coroutine time (virtual in tests). Also the [EnginePreparer]: preparation, Retry, Re-polish and Try it
 * share the inference processes; they wait for no dictation, and a press cancels them (preparation resumes after).
 */
class DictationOrchestrator(
    private val deps: DictationDeps,
    scope: CoroutineScope,
    private val config: DictationConfig = DictationConfig(),
    private val joinPolicy: JoinPolicy = DefaultJoinPolicy,
    /** Unexpected exceptions inside the event loop or an effect (the loop never stops); tests fail on them. */
    private val onInternalError: (Throwable) -> Unit = {},
) : EnginePreparer {
    private val serial: CoroutineDispatcher =
        ((scope.coroutineContext[ContinuationInterceptor] as? CoroutineDispatcher) ?: Dispatchers.Default).limitedParallelism(1)
    private val work = CoroutineScope(scope.coroutineContext + SupervisorJob(scope.coroutineContext[Job]) + serial)
    private val inbox = Channel<() -> Unit>(Channel.UNLIMITED)

    private val keeper = EngineKeeper(deps.engines, deps.unitHealth, deps.exits)
    private val cleaner = Cleaner(deps.engines.llm, deps.renderer, deps.guard, deps.prompts, deps.clock)
    private val recorder = Recorder(deps.history, deps.metrics, deps.audioStore, deps.deliveryMetrics, deps.guard, deps.clock)

    private val _state = MutableStateFlow<DictationState>(DictationState.Idle)
    val state: StateFlow<DictationState> = _state.asStateFlow()

    private val _bubble = MutableStateFlow<DictationBubble>(DictationBubble.Idle)
    val bubble: StateFlow<DictationBubble> = _bubble.asStateFlow()

    private val _levels = MutableStateFlow(FloatArray(LEVEL_BARS))
    val levels: StateFlow<FloatArray> = _levels.asStateFlow()

    private var run: SessionRun? = null

    /** The executor's memory of the last session; null once it ended and its writers finished (I4). Tests only. */
    internal val sessionMemory: SessionRun? get() = run
    private var lastSession = 0L
    private var pendingInApp: InAppTarget? = null
    private val timers = HashMap<Timer, Job>()

    private val backgroundLock = Mutex()
    private var backgroundJob: Job? = null
    private var rebuildJob: Job? = null

    init {
        work.launch {
            for (message in inbox) {
                try { message() } catch (e: CancellationException) { throw e } catch (t: Throwable) { onInternalError(t) }
            }
        }
        watchDeaths(deps.engines.asrHost)
        watchDeaths(deps.engines.llmHost)
        // The settings observer of ARCHITECTURE 13: any change of the prompt inputs re-prepares the active LLM's prefix.
        work.launch {
            deps.settings.settings
                .map { PrefixInputs(it.promptOverride, it.preferences, it.unitPreference, it.cleanup) }
                .distinctUntilChanged().drop(1)
                .collect { invalidatePrefix() }
        }
    }

    private data class PrefixInputs(val override: Any?, val preferences: String?, val units: Any?, val cleanup: Any?)

    // ================================================================ public API

    /** Monotonic milliseconds sharing the exact origin of [DictationBubble.Listening.startedAtMs]. */
    fun elapsedMs(): Long = deps.clock.elapsedMs()

    fun onInput(input: UserInput) {
        val at = deps.clock.elapsedMs()
        inbox.trySend { dispatch(DictationEvent.Input(input, at)) }
    }

    /** Accessibility service disconnected (WP-02 via A11yServiceHost): ExternalStop(A11Y_OFF) for the running session. */
    fun onAccessibilityLost() {
        val at = deps.clock.elapsedMs()
        inbox.trySend {
            val r = activeRun() ?: return@trySend
            r.a11yLost = true
            dispatch(DictationEvent.ExternalStop(r.session, StopReason.A11Y_OFF, at))
        }
    }

    /** Onboarding practice (ob-ready): one dictation into [target], StartMode.IN_APP, nothing stored. */
    fun startInApp(target: InAppTarget) {
        val at = deps.clock.elapsedMs()
        inbox.trySend {
            pendingInApp = target
            dispatch(DictationEvent.Input(UserInput.InAppStart, at))
            pendingInApp = null
        }
    }

    /**
     * History "Try again" (WP-11 Deliver 4): the stored audio of a Failed entry runs through the normal flow from
     * Finishing as a new session (the error pill's Retry does the same, row 49): ASR, cleanup with the current settings
     * (no screen context: it is never stored), delivery into the focused field when there is one outside this app,
     * else the clipboard. The same entry is updated. Waits for a running dictation to end first. Ok(entry) when the text
     * was inserted or copied, else the session's error.
     */
    suspend fun retry(entry: EntryId): Outcome<EntryId> {
        var started: SessionId? = null
        while (started == null) {
            state.first { it is DictationState.Idle || it is DictationState.Done }
            started = withContext(serial) {
                if (dictationRunning()) null
                else {
                    dispatch(DictationEvent.RetryRequested(entry, deps.clock.elapsedMs()))
                    (_state.value as? DictationState.Arming)?.takeIf { it.retryOf == entry }?.session
                }
            }
        }
        var last: DictationState? = null
        state.first { st ->
            if (st.session == started) last = st
            st.session != started || (st is DictationState.Done && !st.recording)
        }
        return when (val o = (last as? DictationState.Done)?.outcome) {
            is DictationOutcome.Inserted, is DictationOutcome.Copied -> Outcome.Ok(entry)
            is DictationOutcome.Failed -> Outcome.Err(o.error)
            is DictationOutcome.Discarded, null -> Outcome.Err(MurmurError.Internal("retry_not_delivered"))
        }
    }

    /** History "Re-polish": re-run cleanup on rawText with the current prompt/tone/model; Original stays, revision + 1. */
    suspend fun repolish(entry: EntryId): Outcome<EntryId> =
        background(resume = false) { guarded { doRepolish(entry) } } ?: Outcome.Err(MurmurError.Internal("interrupted"))

    /**
     * Settings "Try it" card (DESIGN 6.5): render [sample] with the given (unsaved) override and preferences through
     * the active LLM, guard, return the text. No context, nothing stored. Err when no LLM is installed.
     */
    suspend fun tryCleanup(sample: String, systemOverride: String?, preferences: String?): Outcome<String> =
        background(resume = false) { guarded { doTryCleanup(sample, systemOverride, preferences) } } ?: Outcome.Err(MurmurError.Internal("interrupted"))

    override suspend fun prepareArtifact(artifact: ArtifactId, onProgress: (Float) -> Unit): Outcome<PrepareReport> =
        background(resume = true) { guarded { doPrepare(artifact, onProgress) } } ?: Outcome.Err(MurmurError.Internal("interrupted"))

    override fun invalidatePrefix() {
        inbox.trySend {
            rebuildJob?.cancel()
            rebuildJob = work.launch {
                delay(config.prefixRebuildDebounceMs)
                background(resume = true) { guarded { rebuildPrefix(); Outcome.Ok(Unit) } }
            }
        }
    }

    /**
     * Work on the inference processes that is not a dictation (ADR-093: the device-profile retry probe). It runs in the
     * background slot like preparation (ARCHITECTURE 5.4): never during a dictation, one at a time; a press cancels it
     * and it runs again after the dictation. It loads nothing, so a cancelled run keeps the keeper's loaded models.
     */
    suspend fun whenNoDictation(block: suspend () -> Unit) {
        background(resume = true, forgetOnCancel = false) { block() }
    }

    /**
     * The idle exit after a device probe of a process the probe started (D5: a probe does not keep a process alive).
     * The orchestrator owns both idle timers (ADR-086 session hold, the Done re-arm), so the probe's short exit goes
     * through the event loop and applies only while no dictation runs; during one, the session hold stands (ADR-093).
     */
    fun releaseAfterProbe(host: InferenceHost, ms: Long) {
        inbox.trySend { if (!dictationRunning()) host.scheduleIdleExit(ms) }
    }

    // ================================================================ event loop

    private fun post(e: DictationEvent) {
        inbox.trySend { dispatch(e) }
    }

    private fun dispatch(event: DictationEvent) {
        val settings = deps.settings.settings.value
        val cfg = effectiveConfig(settings)
        val next = SessionId(lastSession + 1)
        val e = when {
            event is DictationEvent.Input && event.session == null -> event.copy(session = next)
            event is DictationEvent.RetryRequested && event.session == null -> event.copy(session = next)
            else -> event
        }
        val old = _state.value
        val t = DictationMachine.reduce(old, e, cfg)
        val new = t.state
        if (new is DictationState.Arming && new.session != old.session) beginSession(new, settings, cfg)
        if (new != old) {
            _state.value = new
            _bubble.value = BubbleMapping.of(new)
        }
        for (effect in t.effects) {
            try { execute(effect) } catch (x: CancellationException) { throw x } catch (x: Throwable) { onInternalError(x) }
        }
        if (new != old) afterTransition(new)
    }

    private fun effectiveConfig(s: Settings): DictationConfig = config.copy(
        timeoutScale = config.timeoutScale * (s.debugTimeoutScale.toDouble().takeIf { it > 0 } ?: 1.0),
        idleUnloadMs = s.idleUnloadSec * 1_000L,
        holdToTalk = s.holdToTalk,
    )

    private fun beginSession(arming: DictationState.Arming, settings: Settings, cfg: DictationConfig) {
        run?.let { endSession(it) }
        timers.values.forEach { it.cancel() }
        timers.clear()
        backgroundJob?.cancel()     // ARCHITECTURE 5.4: a press cancels preparation; the dictation never waits for it.
        // D5 / ADR-086: the idle exit armed by the previous Done must not fire during this dictation (a warm :llm died
        // 60 s after the last Done while the next dictation was still Listening). Hold both processes for the longest
        // possible session; every Done re-arms the normal timeout. The hold is finite so a missed Done cannot pin them.
        scheduleIdle(sessionHoldMs(cfg))
        lastSession = arming.session.value
        run = SessionRun(arming.session, arming.mode, deps.clock.now(), settings, cfg,
            inApp = if (arming.mode == StartMode.IN_APP) pendingInApp else null, retryOf = arming.retryOf)
    }

    /** Leaving the working states: timers do not outlive them, and the screen context is dropped (I4). */
    private fun afterTransition(new: DictationState) {
        val r = run ?: return
        when (new) {
            is DictationState.Done -> if (new.session == r.session) {
                timers.filterKeys { it != Timer.RESULT_DISMISS }.forEach { (k, j) -> j.cancel(); timers.remove(k) }
                endSession(r)
            }
            DictationState.Idle -> {
                timers.values.forEach { it.cancel() }
                timers.clear()
                endSession(r)
            }
            else -> Unit
        }
    }

    private fun endSession(r: SessionRun) {
        if (r.ended) return
        r.ended = true
        r.contextJob?.cancel()
        r.screen = null
        r.screenOutcome = null
        r.captureJobs.forEach { it.cancel() }
        r.pumpJob?.cancel()
        // The session PCM (up to 9.6 MB for 5 min) is only for replay inside the session; Retry reads the stored file.
        r.buffer.clear()
        r.buffer.trimToSize()
        _levels.value = FloatArray(LEVEL_BARS)
        // I4: the target snapshot (field text, window title), the privacy decision and the transcript go as soon as the
        // delivery and RecordFailure coroutines that write them out are done (row 44 delivers after Done).
        val writers = r.writers.toList()
        work.launch {
            writers.joinAll()
            r.target = null
            r.decision = null
            r.transcript = null
            r.retryEntry = null
            if (run === r) run = null
        }
    }

    private fun runFor(s: SessionId): SessionRun? = run?.takeIf { it.session == s }

    private fun activeRun(): SessionRun? {
        val r = run ?: return null
        val st = _state.value
        return r.takeIf { !it.ended && st.session == it.session && st !is DictationState.Done }
    }

    private fun dictationRunning(): Boolean = _state.value.let { it !is DictationState.Idle && it !is DictationState.Done }

    private fun launchFor(r: SessionRun, onError: (Throwable) -> DictationEvent?, block: suspend () -> Unit): Job = work.launch {
        try { block() } catch (e: CancellationException) { throw e } catch (t: Throwable) {
            onInternalError(t)
            onError(t)?.let { post(it) }
        }
    }

    // ================================================================ effects

    private fun execute(effect: Effect) {
        when (effect) {
            is Effect.Preflight -> runFor(effect.session)?.let { r ->
                launchFor(r, { DictationEvent.PreflightFailed(r.session, internal("preflight", it)) }) { post(preflight(r, effect.mode)) }
            }
            is Effect.StartCapture -> runFor(effect.session)?.let { r ->
                launchFor(r, { DictationEvent.MicFailed(r.session, MurmurError.MicStartFailed(it::class.simpleName)) }) { startCapture(r) }
            }
            is Effect.StopCapture -> runFor(effect.session)?.let { stopCapture(it) }
            is Effect.DiscardCapture -> runFor(effect.session)?.let { discardCapture(it) }
            is Effect.PrepareEngines -> runFor(effect.session)?.let { prepareEngines(it, effect.plan) }
            is Effect.CaptureContext -> runFor(effect.session)?.let { r ->
                r.contextJob = launchFor(r, { DictationEvent.ContextReady(r.session, false, 0) }) { captureContext(r, effect.delayMs) }
            }
            is Effect.FinishAsr -> runFor(effect.session)?.let { r ->
                r.finishJob = launchFor(r, { DictationEvent.AsrFailed(r.session, internal("asr", it)) }) { finishAsr(r) }
            }
            is Effect.RestartAsr -> runFor(effect.session)?.let { restartAsr(it, effect.thenFinish) }
            is Effect.ReloadLlm -> runFor(effect.session)?.let { r ->
                r.models?.llm?.let { llm -> r.llmJob = launchFor(r, { DictationEvent.LlmLoadFailed(r.session, internal("llm", it)) }) { loadLlm(r, llm, reload = true) } }
            }
            is Effect.RunCleanup -> runFor(effect.session)?.let { r ->
                r.cleanupJob = launchFor(r, { DictationEvent.CleanupDone(r.session, effect.transcript.text, false, "skipped:llm_failed") }) {
                    runCleanup(r, effect)
                }
            }
            is Effect.Deliver -> runFor(effect.session)?.let { r ->
                r.writers += launchFor(r, { DictationEvent.Delivered(r.session, DeliveryResult.CopiedOnly(null, internal("deliver", it))) }) { deliver(r, effect) }
            }
            is Effect.RecordFailure -> runFor(effect.session)?.let { r ->
                r.writers += launchFor(r, { DictationEvent.FailureRecorded(r.session, null) }) {
                    val recorded = recorder.failure(r, effect.error, effect.keepAudio)
                    post(DictationEvent.FailureRecorded(r.session, recorded?.id, audioKept = recorded?.audioKept == true))
                }
            }
            is Effect.StartTimer -> startTimer(effect)
            is Effect.CancelTimer -> timers.remove(effect.timer)?.cancel()
            is Effect.Haptic -> haptic(effect.kind)
            is Effect.ScheduleIdleUnload -> scheduleIdle(effect.ms)
            is Effect.CancelEngines -> runFor(effect.session)?.let { cancelEngines(it) }
            is Effect.Open -> deps.navigator.open(effect.target)
        }
    }

    private fun internal(stage: String, t: Throwable) = MurmurError.Internal("$stage:${t::class.simpleName}")

    private fun startTimer(e: Effect.StartTimer) {
        timers.remove(e.timer)?.cancel()
        lateinit var job: Job
        job = work.launch {
            delay(e.ms)
            if (timers[e.timer] === job) timers.remove(e.timer)
            post(DictationEvent.TimerFired(e.session, e.timer))
        }
        timers[e.timer] = job
    }

    private fun haptic(kind: HapticKind) {
        when (deps.settings.settings.value.feedback) {
            FeedbackSetting.VIBRATION -> deps.feedback.haptic(kind)
            FeedbackSetting.SOUND -> deps.feedback.sound(kind)
            FeedbackSetting.BOTH -> { deps.feedback.haptic(kind); deps.feedback.sound(kind) }
            FeedbackSetting.OFF -> Unit
        }
    }

    /**
     * Idle-exit delay while a dictation runs (ADR-086): the longest session (5 min recording, ASR tail, LLM ready wait,
     * LLM hard cap, delivery; scaled like the timers) plus the normal idle timeout.
     */
    private fun sessionHoldMs(cfg: DictationConfig): Long =
        (cfg.scaled(cfg.maxDurationMs + cfg.asrMaxTailMs + cfg.llmReadyWaitMs + cfg.llmHardCapMs) + cfg.deliveryTimerMs())
            .coerceAtLeast(cfg.maxDurationMs) + cfg.idleUnloadMs

    /** D5: keep both processes warm for the idle timeout, then unload + exit. */
    private fun scheduleIdle(ms: Long) {
        deps.engines.asrHost.scheduleIdleExit(ms)
        deps.engines.llmHost.scheduleIdleExit(ms)
    }

    // ---------------------------------------------------------------- preflight (row 6, in this order)

    private suspend fun preflight(r: SessionRun, mode: StartMode): DictationEvent {
        val s = r.session
        val settings = r.settings
        val c = r.config
        fun fail(e: MurmurError) = DictationEvent.PreflightFailed(s, e)
        // A retry records nothing (its audio is stored) and may come from History with the service off; it is delivered
        // into the focused field only when that is a usable one, else copied (NotDelivered without the service).
        val retry = r.retryOf != null
        fun refused(p: PermissionState) = (mode != StartMode.IN_APP && !retry && !p.a11yConnected) || (!retry && !p.micGranted)
        // The monitor's state can be stale after a grant made outside the app's screens (system Settings): re-read
        // before refusing, so a later grant is seen on the next start. The passing path costs nothing.
        var perms = deps.permissions.state.value
        if (refused(perms)) { deps.permissions.refresh(); perms = deps.permissions.state.value }
        if (mode != StartMode.IN_APP && !retry && !perms.a11yConnected) return fail(MurmurError.AccessibilityOff)
        if (!retry && !perms.micGranted) return fail(MurmurError.MicPermissionMissing)
        val snap = when (mode) {
            StartMode.NOTIFICATION -> deps.target.snapshotAfterShade(c.scaled(c.shadeCollapseWaitMs))
            StartMode.IN_APP -> r.inApp?.snapshot
            StartMode.TAP, StartMode.HOLD -> deps.target.snapshot()
        }
        if (snap == null && !retry) return fail(MurmurError.NoEditableField)
        val field = snap?.field
        if (field != null && (field.isPassword || FieldKinds.isPassword(field.inputType))) return fail(MurmurError.PasswordField)
        val excluded = BuiltInDenylist.packages + settings.hiddenInPackages
        val excludedApp = mode != StartMode.IN_APP && snap != null && (snap.pkg in excluded || snap.pkg == c.ownPackage)
        if (excludedApp && !retry) return fail(MurmurError.AppExcluded)
        // A retry from this app (History) or an excluded one has no target: no field context, clipboard only.
        r.target = snap?.takeIf { !excludedApp }
        val decision = when {
            // No target (a retry from this app or an excluded one): no ctx is rendered, the decision only gates the LLM.
            snap == null || excludedApp -> PrivacyDecision(PrivacyLevel.NONE, takeScreenshot = false, reason = null, skipLlm = false)
            // IN_APP: the practice field in this app (own package) -> app_only, no screenshot (ARCHITECTURE 9.2).
            else -> deps.context.decide(snap, ContextSettings(
                useScreen = settings.useScreen && mode != StartMode.IN_APP,
                excludedPackages = settings.hiddenInPackages,
                inAppPractice = mode == StartMode.IN_APP,
            ))
        }
        // The assembler's own refusals (number/phone fields; the checks above already caught password and excluded).
        if (!decision.dictationAllowed && !retry) return fail(
            when (decision.reason) {
                PrivacyReason.PASSWORD_FIELD -> MurmurError.PasswordField
                PrivacyReason.DENYLISTED_APP -> MurmurError.AppExcluded
                else -> MurmurError.NoEditableField
            },
        )
        // A retry into a field dictation may not use (ADR-064, ADR-070) has no target either: copied only, no ctx.
        r.decision = if (decision.dictationAllowed) decision else {
            r.target = null
            PrivacyDecision(PrivacyLevel.NONE, takeScreenshot = false, reason = null, skipLlm = false)
        }
        val models = when (val m = deps.models.resolve(settings)) {
            is Outcome.Err -> return fail(m.error)
            is Outcome.Ok -> m.value
        }
        r.models = models
        if (deps.system.freeStorageBytes() < c.minFreeStorageBytes) return fail(MurmurError.LowStorage)
        if (deps.system.thermalCritical()) return fail(MurmurError.ThermalCritical)
        val llm = models.llm
        val skip = when {
            llm == null -> models.llmUnavailable ?: "no_model"
            decision.skipLlm -> "password"
            // Memory refused (ARCHITECTURE 7): with :asr not running, shutting it down after the tail frees nothing, so
            // a load that does not fit now will not fit later either.
            !memoryAllows(r, llm) && deps.engines.asrHost.state.value !is HostState.Up -> "low_memory"
            else -> null
        }
        val plan = SessionPlan(
            cleanup = skip == null,
            cleanupSkipReason = skip,
            // A retry re-runs stored data only: the screen now is not the screen it was dictated against.
            screenshot = mode != StartMode.IN_APP && !retry && decision.takeScreenshot && settings.useScreen && models.ocrDet != null && models.ocrRec != null,
            asrUnit = models.asr.unit,
            speechMeasured = !models.asrTraits.streaming,
        )
        r.plan = plan
        return DictationEvent.PreflightPassed(s, plan)
    }

    private fun memoryAllows(r: SessionRun, llm: ActiveLlm): Boolean = memoryAllows(llm, r.settings, r.config)

    private fun memoryAllows(llm: ActiveLlm, settings: Settings, cfg: DictationConfig): Boolean =
        settings.debugIgnoreMemoryGate || keeper.llmWarm(llm) || deps.memory.availableMb() - llm.memoryMb >= cfg.llmMemoryMarginMb

    /**
     * ARCHITECTURE 7 outside a dictation (Re-polish, Try it, Preparing): the LLM loads only with the margin; when a warm
     * :asr is what stands in the way it is shut down first (nothing needs it until the next press).
     */
    private suspend fun roomForLlm(llm: ActiveLlm, settings: Settings, cfg: DictationConfig): Boolean {
        if (memoryAllows(llm, settings, cfg)) return true
        if (deps.engines.asrHost.state.value !is HostState.Up) return false
        keeper.shutdown(InferProcess.ASR)
        return memoryAllows(llm, settings, cfg)
    }

    // ---------------------------------------------------------------- capture

    private suspend fun startCapture(r: SessionRun) {
        if (r.retryOf != null) return loadStoredAudio(r, r.retryOf)
        val s = r.session
        val key = AudioKey.of(r.createdAt.toEpochMilli(), s).value
        val keep = r.mode != StartMode.IN_APP && r.settings.keepRecordings && RetentionPolicy.keepsHistory(r.settings.retentionDays)
        when (val res = deps.audio.start(s, keep, key)) {
            is Outcome.Err -> {
                r.captureStopped.complete(null)
                r.drained.complete(Unit)
                post(DictationEvent.MicFailed(s, res.error))
            }
            is Outcome.Ok -> {
                val h = res.value
                if (r.captureCancelled || r.ended) {
                    h.cancel()
                    r.captureStopped.complete(null)
                    r.drained.complete(Unit)
                    return
                }
                r.capture = h
                r.pumpJob = work.launch {
                    h.chunks.collect { chunk ->
                        r.buffer += chunk
                        r.asrSession?.accept(chunk)
                    }
                }
                r.captureJobs += work.launch { h.levels.collect { _levels.value = it } }
                r.captureJobs += work.launch {
                    h.signals.collect { sig -> post(DictationEvent.ExternalStop(s, stopReasonOf(sig), deps.clock.elapsedMs())) }
                }
                post(DictationEvent.MicStarted(s, deps.clock.elapsedMs()))
            }
        }
    }

    /** A retry's capture: the entry's stored recording becomes the session buffer, then MicStarted(recordedMs). */
    private suspend fun loadStoredAudio(r: SessionRun, id: EntryId) {
        val entry = deps.history.entry(id).first()
        val ref = entry?.audio
        val pcm = ref?.let { deps.audioStore.readPcm(it) }?.takeIf { it.isNotEmpty() }
        if (r.captureCancelled || r.ended) return
        if (entry == null || ref == null || pcm == null) {
            r.captureStopped.complete(null)
            r.drained.complete(Unit)
            post(DictationEvent.MicFailed(r.session, MurmurError.StorageFailed("audio")))
            return
        }
        r.retryEntry = entry
        // Like the live pump: buffered for a replay, and fed to the ASR session when it attached while this was read.
        var i = 0
        while (i < pcm.size) {
            val end = minOf(i + PcmFormat.FRAME_SAMPLES, pcm.size)
            val chunk = PcmChunk(pcm.copyOfRange(i, end), i.toLong())
            r.buffer += chunk
            r.asrSession?.accept(chunk)
            i = end
        }
        val ms = PcmFormat.samplesToMs(pcm.size.toLong())
        r.captureStopped.complete(CaptureResult(ms, ref, pcm.maxOf { kotlin.math.abs(it.toInt()) }, silenced = false))
        r.drained.complete(Unit)
        post(DictationEvent.MicStarted(r.session, deps.clock.elapsedMs(), recordedMs = ms))
    }

    private fun stopReasonOf(sig: CaptureSignal): StopReason = when (sig) {
        CaptureSignal.CallStarted -> StopReason.CALL
        CaptureSignal.Silenced -> StopReason.MIC_SILENCED
        is CaptureSignal.Lost -> StopReason.MIC_LOST
    }

    private fun stopCapture(r: SessionRun) {
        r.stopAtElapsedMs = deps.clock.elapsedMs()
        val h = r.capture
        if (h == null) {
            r.captureCancelled = true
            r.captureStopped.complete(null)
            r.drained.complete(Unit)
            return
        }
        work.launch {
            val result = try { h.stop() } catch (e: CancellationException) { throw e } catch (t: Throwable) { onInternalError(t); null }
            r.captureJobs.forEach { it.cancel() }
            _levels.value = FloatArray(LEVEL_BARS)
            r.captureStopped.complete(result)
            val pump = r.pumpJob
            if (pump != null && withTimeoutOrNull(r.config.captureDrainMs) { pump.join() } == null) pump.cancel()
            r.drained.complete(Unit)
        }
    }

    private fun discardCapture(r: SessionRun) {
        r.captureCancelled = true
        r.capture?.cancel()
        r.pumpJob?.cancel()
        r.captureJobs.forEach { it.cancel() }
        _levels.value = FloatArray(LEVEL_BARS)
        r.captureStopped.complete(null)
        r.drained.complete(Unit)
    }

    // ---------------------------------------------------------------- engines

    private fun prepareEngines(r: SessionRun, plan: SessionPlan) {
        val models = r.models ?: return
        r.asrJob = launchFor(r, { DictationEvent.AsrLoadFailed(r.session, internal("asr", it)) }) { prepareAsr(r, models, plan) }
        if (!plan.cleanup) return
        val llm = models.llm ?: return
        if (memoryAllows(r, llm)) {
            r.llmJob = launchFor(r, { DictationEvent.LlmLoadFailed(r.session, internal("llm", it)) }) { loadLlm(r, llm, reload = false) }
        } else {
            r.llmDeferred = true
        }
    }

    private suspend fun prepareAsr(r: SessionRun, models: ActiveModels, plan: SessionPlan) {
        val s = r.session
        when (val res = keeper.loadAsr(models.asr, models.asrFallback, models.vad)) {
            is Outcome.Err -> {
                r.asrSlot.value = AsrSlot.Failed(res.error)
                r.ocrLoaded.complete(false)
                post(DictationEvent.AsrLoadFailed(s, res.error))
            }
            is Outcome.Ok -> {
                r.asrSpec = res.value.spec
                r.asrFellBack = res.value.fellBack
                if (!attachNewSession(r)) { r.ocrLoaded.complete(false); return }
                post(DictationEvent.AsrReady(s, res.value.info, res.value.fellBack))
                val det = models.ocrDet
                val rec = models.ocrRec
                r.ocrLoaded.complete(plan.screenshot && det != null && rec != null && keeper.loadOcr(det, rec))
            }
        }
    }

    /** New ASR session fed with the whole session buffer from sample 0, then attached to the live stream (no gap). */
    private suspend fun attachNewSession(r: SessionRun): Boolean {
        val session = deps.engines.asr.startSession(sessionOptions(r.settings))
        if (r.ended) { session.cancel(); return false }
        for (chunk in r.buffer) session.accept(chunk)
        r.asrPid = (deps.engines.asrHost.state.value as? HostState.Up)?.pid
        r.asrSession = session
        r.asrSlot.value = AsrSlot.Ready(session)
        return true
    }

    /** A language hint only when exactly one dictation language is selected; otherwise the model detects it. */
    private fun sessionOptions(s: Settings) = AsrSessionOptions(language = s.languages.singleOrNull())

    private suspend fun loadLlm(r: SessionRun, llm: ActiveLlm, reload: Boolean) {
        val s = r.session
        val first = if (reload) r.llmSpec ?: llm.spec else llm.spec
        val override = cleaner.systemOverride(cleaner.profile(llm), r.settings)
        val prefix = cleaner.prefix(llm, r.settings.preferences, override)
        when (val res = keeper.loadLlm(llm, first, prefix)) {
            is Outcome.Err -> post(DictationEvent.LlmLoadFailed(s, res.error))
            is Outcome.Ok -> {
                val l = res.value
                r.llmSpec = l.spec
                val tokens = l.prefixTokens.takeIf { it > 0 } ?: llm.prefixTokens ?: 0
                // The benchmark speed belongs to the primary (artifact, unit); after a fallback use the unit default.
                val speed = llm.speed.takeIf { l.spec.unit == llm.spec.unit }
                post(DictationEvent.LlmReady(s, l.info, tokens, llm.nCtx, speed, fellBack = l.fellBack || l.spec != llm.spec))
            }
        }
    }

    private suspend fun finishAsr(r: SessionRun) {
        r.drained.await()
        when (val slot = r.asrSlot.first { it !is AsrSlot.Loading }) {
            is AsrSlot.Failed -> post(DictationEvent.AsrFailed(r.session, slot.error))
            is AsrSlot.Ready -> finishWith(r, slot.session)
            AsrSlot.Loading -> Unit
        }
    }

    private suspend fun finishWith(r: SessionRun, session: AsrSession) {
        val s = r.session
        val t0 = deps.clock.elapsedMs()
        val res = session.finish()
        r.asrTailMs = deps.clock.elapsedMs() - t0
        when (res) {
            is Outcome.Err -> {
                // The host watcher may already have reported this death (its notice came first).
                if (res.error is MurmurError.ProcessDied && !r.asrDeaths.add(r.asrPid)) return
                post(DictationEvent.AsrFailed(s, res.error))
            }
            is Outcome.Ok -> {
                val pre = deps.guard.precheck(res.value.text)
                val t = if (pre.skip) res.value else res.value.copy(text = pre.transcript)
                r.transcript = t
                // Only a transcript that goes on to Polishing is worth shutting :asr down for (row 26 ends the others).
                val plan = r.plan
                if (plan != null && !DictationMachine.isNoSpeech(t, pre.skip, plan, r.config)) loadDeferredLlm(r)
                post(DictationEvent.TranscriptReady(s, t, pre.skip))
            }
        }
    }

    /** ARCHITECTURE 7: the LLM that did not fit next to the ASR loads now that :asr is shut down, or cleanup is skipped. */
    private suspend fun loadDeferredLlm(r: SessionRun) {
        if (!r.llmDeferred) return
        r.llmDeferred = false
        val llm = r.models?.llm ?: return
        keeper.shutdown(InferProcess.ASR)
        if (memoryAllows(r, llm)) {
            r.llmJob = launchFor(r, { DictationEvent.LlmLoadFailed(r.session, internal("llm", it)) }) { loadLlm(r, llm, reload = false) }
        } else {
            post(DictationEvent.LlmLoadFailed(r.session, MurmurError.OutOfMemory(ModelKind.LLM)))
        }
    }

    /** Rows 21 and 29 (ARCHITECTURE 4.4): re-bind, new session, replay from sample 0; then finish when [thenFinish]. */
    private fun restartAsr(r: SessionRun, thenFinish: Boolean) {
        r.finishJob?.cancel()
        r.asrJob?.cancel()
        r.asrSession?.cancel()
        r.asrSession = null
        r.asrSlot.value = AsrSlot.Loading
        val job = launchFor(r, { DictationEvent.AsrFailed(r.session, internal("asr", it)) }) {
            val models = r.models ?: return@launchFor
            val s = r.session
            keeper.shutdown(InferProcess.ASR)
            val primary = r.asrSpec ?: models.asr
            val fallback = models.asrFallback.takeIf { primary == models.asr }
            when (val res = keeper.loadAsr(primary, fallback, models.vad)) {
                is Outcome.Err -> {
                    r.asrSlot.value = AsrSlot.Failed(res.error)
                    post(if (thenFinish) DictationEvent.AsrFailed(s, res.error) else DictationEvent.AsrLoadFailed(s, res.error))
                }
                is Outcome.Ok -> {
                    r.asrSpec = res.value.spec
                    r.asrFellBack = r.asrFellBack || res.value.fellBack
                    if (thenFinish) r.drained.await()
                    if (!attachNewSession(r)) return@launchFor
                    post(DictationEvent.AsrReady(s, res.value.info, r.asrFellBack))
                    if (thenFinish) (r.asrSlot.value as? AsrSlot.Ready)?.let { finishWith(r, it.session) }
                }
            }
        }
        r.asrJob = job
        if (thenFinish) r.finishJob = job
    }

    private fun cancelEngines(r: SessionRun) {
        r.asrJob?.cancel()
        r.finishJob?.cancel()
        r.asrSession?.cancel()
        r.asrSession = null
        r.contextJob?.cancel()
        r.cleanupJob?.let { if (it.isActive) { deps.engines.llm.cancel(); it.cancel() } }
        // An LLM load in flight continues: the processes stay warm for the next dictation (row 12, row 34).
    }

    /** A Crashed notice is the death of the instance last seen Up (null = none seen). */
    private fun watchDeaths(host: InferenceHost) {
        work.launch {
            var lastUp: Int? = null
            host.state.collect { st ->
                when (st) {
                    is HostState.Up -> lastUp = st.pid
                    is HostState.Crashed -> onDied(host.process, lastUp)
                    else -> Unit
                }
            }
        }
    }

    private fun onDied(process: InferProcess, pid: Int?) {
        keeper.forget(process)
        val r = activeRun() ?: return
        when (process) {
            InferProcess.ASR -> {
                // A death during a load is that load's error (it falls back). A death of another instance than the one
                // the session runs in is not this session's; one already reported by the failing finish() is not new.
                if (keeper.asrLoading > 0) return
                if (pid != null && r.asrPid != null && pid != r.asrPid) return
                if (!r.asrDeaths.add(pid ?: r.asrPid)) return
                r.asrSession = null
                r.asrSlot.value = AsrSlot.Loading
                post(DictationEvent.ProcessDied(r.session, InferProcess.ASR))
            }
            InferProcess.LLM -> {
                if (keeper.llmLoading > 0) return
                post(DictationEvent.ProcessDied(r.session, InferProcess.LLM))
            }
        }
    }

    // ---------------------------------------------------------------- context (D11, I4)

    private suspend fun captureContext(r: SessionRun, delayMs: Long) {
        val s = r.session
        delay(delayMs)
        val target = r.target
        val decision = r.decision
        if (target == null || decision == null) { post(DictationEvent.ContextReady(s, false, 0)); return }
        val t0 = deps.clock.elapsedMs()
        val shot = deps.target.screenshot(target)
        val outcome: ScreenOutcome = when (shot) {
            is Outcome.Err -> if (shot.error == MurmurError.SecureWindow) ScreenOutcome.SecureWindow else ScreenOutcome.Failed
            is Outcome.Ok -> withTimeoutOrNull(r.config.scaled(r.config.ocrBudgetMs)) {
                if (r.ocrLoaded.await()) deps.engines.ocr.recognize(shot.value).getOrNull() else null
            }?.let { ScreenOutcome.Text(it, shot.value.kind) } ?: ScreenOutcome.Failed
        }
        if (shot is Outcome.Ok) r.ocrMs = deps.clock.elapsedMs() - t0
        val ctx = deps.context.assemble(target, decision, outcome, null)
        r.screenOutcome = outcome
        r.screen = ctx.screen?.let { ScreenText(it.size, it.lines) }
        val words = ctx.screen?.lines?.sumOf { line -> line.text.split(WHITESPACE).count { it.isNotEmpty() } } ?: 0
        r.contextWords = words
        post(DictationEvent.ContextReady(s, hasScreen = r.screen != null, contextWords = words))
    }

    // ---------------------------------------------------------------- cleanup and delivery

    private suspend fun runCleanup(r: SessionRun, e: Effect.RunCleanup) {
        val s = r.session
        val t = e.transcript
        val llm = r.models?.llm
        if (llm == null) { post(DictationEvent.CleanupDone(s, t.text, false, "skipped:no_model")); return }
        val profile = cleaner.profile(llm)
        val override = cleaner.systemOverride(profile, r.settings)
        r.promptVersion = cleaner.promptVersion(profile, override)
        // No outcome yet = the screenshot was planned but OCR missed CONTEXT_WAIT (Failed), or none was planned.
        val outcome = r.screenOutcome ?: if (r.decision?.takeScreenshot == true) ScreenOutcome.Failed else ScreenOutcome.NotTaken
        val ctx = r.target?.let { tg -> r.decision?.let { d -> deps.context.assemble(tg, d, outcome, t.language) } }
        r.cleanupUsedScreen = ctx?.screen != null
        r.tone = cleaner.tone(r.settings, r.target?.pkg, ctx)
        val res = cleaner.run(llm, t.text, ctx, r.settings.preferences, override, e.budget, r.tone)
        r.llmMs = res.llmMs
        post(DictationEvent.CleanupDone(s, res.text, res.cleaned, res.outcome))
    }

    /** Row 37 / ARCHITECTURE 4.4: cancel(); no end within 1 s -> shutdown(:llm). Never delays the delivery. */
    private fun stopCleanup(r: SessionRun) {
        val job = r.cleanupJob ?: return
        if (!job.isActive) return
        deps.engines.llm.cancel()
        work.launch {
            if (withTimeoutOrNull(r.config.scaled(r.config.llmCancelGraceMs)) { job.join() } == null) {
                job.cancel()
                keeper.shutdown(InferProcess.LLM)
            }
        }
    }

    private suspend fun deliver(r: SessionRun, e: Effect.Deliver) {
        stopCleanup(r)
        post(DictationEvent.Delivered(r.session, deliverText(r, e)))
    }

    /** ARCHITECTURE 11. */
    private suspend fun deliverText(r: SessionRun, e: Effect.Deliver): DeliveryResult {
        val text = e.text
        val raw = e.raw ?: text
        val outcome = e.cleanupOutcome
        // Step 7: the in-app practice field; no clipboard, no history.
        if (r.mode == StartMode.IN_APP) {
            val inApp = r.inApp
            if (inApp == null || text.isBlank()) return DeliveryResult.NothingToInsert(null)
            inApp.insert(joinPolicy.join(inApp.snapshot.field?.beforeCursor, text, e.cleaned))
            return DeliveryResult.Inserted(null, InsertMethod.COMMIT_TEXT)
        }
        // Guard EMPTY: only fillers were spoken.
        if (text.isBlank()) {
            return DeliveryResult.NothingToInsert(recorder.text(r, EntryStatus.FAILED, null, raw, false, outcome, null, MurmurError.NoSpeech))
        }
        // Step 1: current focus; a password field discards everything (I5 exception).
        val current = deps.target.current()
        val field = current?.field
        if (field != null && (field.isPassword || FieldKinds.isPassword(field.inputType))) {
            recorder.discardAudio(r)
            return DeliveryResult.DiscardedPassword
        }
        if (r.a11yLost || !deps.permissions.state.value.a11yConnected) {
            val entry = recorder.text(r, EntryStatus.FAILED, text, raw, e.cleaned, outcome, null, MurmurError.AccessibilityOff)
            return DeliveryResult.NotDelivered(entry, MurmurError.AccessibilityOff)
        }
        // Step 2 (I5): the clipboard first.
        deps.inserter.copyToClipboard(text)
        // Step 3. A retry without a target (started in this app, an excluded one or a refused field) is copied only.
        val snap = r.target
        var why: MurmurError? = when {
            r.retryOf != null && snap == null -> MurmurError.NoEditableField
            current == null || field == null -> MurmurError.NoEditableField
            snap != null && current.pkg != snap.pkg -> MurmurError.TargetChanged
            else -> null
        }
        if (why == null && current != null && field != null) {
            // Steps 4-5: join against the current before_cursor, then the ladder (WP-02).
            when (val ins = deps.inserter.insert(current, joinPolicy.join(field.beforeCursor, text, e.cleaned))) {
                is InsertResult.Inserted -> {
                    val entry = recorder.text(r, EntryStatus.INSERTED, text, raw, e.cleaned, outcome, ins.method, null)
                    return DeliveryResult.Inserted(entry, ins.method)
                }
                is InsertResult.Failed -> why = ins.error
            }
        }
        // Step 6.
        val entry = recorder.text(r, EntryStatus.COPIED, text, raw, e.cleaned, outcome, InsertMethod.CLIPBOARD_ONLY, null)
        return DeliveryResult.CopiedOnly(entry, why ?: MurmurError.InsertRejected)
    }

    // ================================================================ background work (Preparing, Retry, Re-polish, Try it)

    private sealed interface Bg<out T> {
        data class Value<T>(val v: T) : Bg<T>
        data object Preempted : Bg<Nothing>
        data object Busy : Bg<Nothing>
    }

    /**
     * Runs [block] while no dictation runs, one at a time. A press cancels it (ARCHITECTURE 5.4); with [resume] it runs
     * again after the dictation, otherwise null is returned. A cancelled block may have been half-way through a load, so
     * the keeper forgets what it knew loaded, unless [forgetOnCancel] is false (a block that loads nothing).
     */
    private suspend fun <T> background(resume: Boolean, forgetOnCancel: Boolean = true, block: suspend () -> T): T? {
        while (true) {
            state.first { it is DictationState.Idle || it is DictationState.Done }
            val result: Bg<T> = backgroundLock.withLock {
                val job: Deferred<T> = withContext(serial) {
                    if (dictationRunning()) null else work.async { block() }.also { backgroundJob = it }
                } ?: return@withLock Bg.Busy
                try {
                    Bg.Value(job.await())
                } catch (e: CancellationException) {
                    if (!currentCoroutineContext().isActive) { job.cancel(); throw e }
                    Bg.Preempted
                } finally {
                    withContext(NonCancellable + serial) {
                        if (backgroundJob === job) backgroundJob = null
                        if (job.isCancelled && forgetOnCancel) { keeper.forget(InferProcess.ASR); keeper.forget(InferProcess.LLM) }
                    }
                }
            }
            when (result) {
                is Bg.Value -> return result.v
                Bg.Preempted -> if (!resume) return null
                Bg.Busy -> Unit
            }
        }
    }

    private suspend fun <T> guarded(block: suspend () -> Outcome<T>): Outcome<T> =
        try { block() } catch (e: CancellationException) { throw e } catch (t: Throwable) { onInternalError(t); Outcome.Err(internal("background", t)) }

    private data class OfflineCleanup(
        val text: String, val cleaned: Boolean, val outcome: String, val llmMs: Long?,
        val spec: LoadSpec?, val promptVersion: String, val error: MurmurError?,
    )

    /** One cleanup outside a dictation (no screen context): load + prefix, budget, generate with the adaptive timeout, guard. */
    private suspend fun cleanupOffline(llm: ActiveLlm, transcript: String, settings: Settings, cfg: DictationConfig, override: String?, preferences: String?): OfflineCleanup {
        val profile = cleaner.profile(llm)
        val version = cleaner.promptVersion(profile, override)
        val prefix = cleaner.prefix(llm, preferences, override)
        if (!roomForLlm(llm, settings, cfg)) {
            return OfflineCleanup(transcript, false, "skipped:low_memory", null, null, version, MurmurError.OutOfMemory(ModelKind.LLM))
        }
        val loaded = when (val l = keeper.loadLlm(llm, llm.spec, prefix)) {
            is Outcome.Err -> return OfflineCleanup(transcript, false, "skipped:${DictationMachine.skipReasonOf(l.error)}", null, null, version, l.error)
            is Outcome.Ok -> l.value
        }
        val speed = llm.speed.takeIf { loaded.spec.unit == llm.spec.unit } ?: cfg.defaultSpeed[loaded.spec.unit] ?: cfg.defaultSpeed.getValue(ComputeUnit.CPU)
        val plan = CleanupBudget.plan(transcript.length, loaded.prefixTokens.takeIf { it > 0 } ?: llm.prefixTokens ?: 0, llm.nCtx, speed, cfg)
        if (!plan.fits) return OfflineCleanup(transcript, false, "skipped:too_long", null, loaded.spec, version, MurmurError.LlmFailed("too_long"))
        val res = withTimeoutOrNull(plan.timeoutMs) { cleaner.run(llm, transcript, null, preferences, override, plan) }
        scheduleIdle(cfg.idleUnloadMs)
        if (res == null) {
            deps.engines.llm.cancel()
            return OfflineCleanup(transcript, false, "skipped:timeout", null, loaded.spec, version, MurmurError.Timeout(Stage.LLM))
        }
        val error = when {
            res.cleaned -> null
            res.outcome.startsWith("fallback:") -> MurmurError.GuardRejected(res.outcome.removePrefix("fallback:"))
            res.outcome.startsWith("empty:") -> MurmurError.NoSpeech
            res.outcome == "skipped:too_long" -> MurmurError.LlmFailed("too_long")
            res.outcome == "skipped:llm_died" -> MurmurError.ProcessDied(InferProcess.LLM)
            else -> MurmurError.LlmFailed(null)
        }
        return OfflineCleanup(res.text, res.cleaned, res.outcome, res.llmMs, loaded.spec, version, error)
    }

    private suspend fun doRepolish(id: EntryId): Outcome<EntryId> {
        val entry = deps.history.entry(id).first() ?: return Outcome.Err(MurmurError.Internal("entry_missing"))
        val raw = entry.rawText?.takeIf { it.isNotBlank() } ?: return Outcome.Err(MurmurError.Internal("no_text"))
        val settings = deps.settings.settings.value
        val cfg = effectiveConfig(settings)
        val models = when (val m = deps.models.resolve(settings)) { is Outcome.Err -> return m; is Outcome.Ok -> m.value }
        val llm = models.llm ?: return Outcome.Err(MurmurError.LlmFailed(models.llmUnavailable ?: "no_model"))
        val res = cleanupOffline(llm, raw, settings, cfg, cleaner.systemOverride(cleaner.profile(llm), settings), settings.preferences)
        if (!res.cleaned) return Outcome.Err(res.error ?: MurmurError.LlmFailed(res.outcome))
        deps.history.update(entry.copy(
            finalText = res.text,
            revision = entry.revision + 1,
            cleanup = CleanupUse(res.spec?.let { ModelUse(it.modelId, it.artifactId, it.unit) }, res.promptVersion, res.outcome, entry.cleanup?.tone),
            failureCode = null,
            timings = entry.timings.copy(llmMs = res.llmMs),
        ))
        return Outcome.Ok(entry.id)
    }

    private suspend fun doTryCleanup(sample: String, systemOverride: String?, preferences: String?): Outcome<String> {
        val settings = deps.settings.settings.value
        val cfg = effectiveConfig(settings)
        val models = when (val m = deps.models.resolve(settings)) { is Outcome.Err -> return m; is Outcome.Ok -> m.value }
        val llm = models.llm ?: return Outcome.Err(MurmurError.LlmFailed(models.llmUnavailable ?: "no_model"))
        val profile = cleaner.profile(llm)
        // Fine-tuned profiles were trained on their own prompt: an edit never applies to them (ADR-046).
        val override = systemOverride?.takeIf { !profile.system.startsWith("finetuned") }
        val res = cleanupOffline(llm, sample, settings, cfg, override, preferences)
        return if (res.cleaned) Outcome.Ok(res.text) else Outcome.Err(res.error ?: MurmurError.LlmFailed(res.outcome))
    }

    /**
     * invalidatePrefix: the active LLM's snapshot for the current prompt inputs (loads the model if needed). An
     * optimisation only: skipped when the model does not fit in memory now (the next dictation decides again).
     */
    private suspend fun rebuildPrefix() {
        val settings = deps.settings.settings.value
        val models = deps.models.resolve(settings).getOrNull() ?: return
        val llm = models.llm ?: return
        val cfg = effectiveConfig(settings)
        if (!memoryAllows(llm, settings, cfg)) return
        val prefix = cleaner.prefix(llm, settings.preferences, cleaner.systemOverride(cleaner.profile(llm), settings))
        keeper.loadLlm(llm, llm.spec, prefix)
        scheduleIdle(cfg.idleUnloadMs)
    }

    /**
     * ARCHITECTURE 8.5: prove [artifact] on every usable unit. An LLM unit that cannot get the memory margin now (even
     * with :asr shut down) is neither proven nor failed: its first dictation decides.
     */
    private suspend fun doPrepare(artifact: ArtifactId, onProgress: (Float) -> Unit): Outcome<PrepareReport> {
        val settings = deps.settings.settings.value
        val cfg = effectiveConfig(settings)
        val targets = when (val t = deps.models.prepareTargets(artifact, settings)) { is Outcome.Err -> return t; is Outcome.Ok -> t.value }
        val proven = linkedSetOf<ComputeUnit>()
        val failed = linkedMapOf<ComputeUnit, MurmurError>()
        val untested = linkedMapOf<ComputeUnit, MurmurError>()
        // ADR-066: only an error that says the unit cannot run the artifact marks it failed; OutOfMemory, a process
        // that could not be bound, a prefix error and a proof-decode timeout leave UnitHealth unchanged.
        suspend fun mark(unit: ComputeUnit, trial: EngineKeeper.Trial) {
            val error = trial.error
            when {
                error == null -> { deps.unitHealth.markProven(artifact, unit); proven += unit }
                trial.unitFailed -> { deps.unitHealth.markFailed(artifact, unit, error.code); failed[unit] = error }
                else -> untested[unit] = error
            }
        }
        onProgress(0f)
        when (targets.kind) {
            ModelKind.ASR -> {
                if (targets.needsVad && targets.vad == null) return Outcome.Err(MurmurError.NoVadModel)
                targets.specs.forEachIndexed { i, spec ->
                    mark(spec.unit, proveAsr(spec, targets.vad, cfg))
                    onProgress((i + 1f) / targets.specs.size)
                }
                deps.engines.asr.unload()
                keeper.forget(InferProcess.ASR)
            }
            ModelKind.LLM -> {
                val info = targets.llm ?: return Outcome.Err(MurmurError.CatalogInvalid("llm traits"))
                targets.specs.forEachIndexed { i, spec ->
                    val active = ActiveLlm(spec, null, info.template, info.profile, memoryMb = info.memoryMb, nCtx = info.nCtx, prefixTokens = null,
                        modelSha256 = info.modelSha256, speed = null)
                    if (!roomForLlm(active, settings, cfg)) { onProgress((i + 1f) / targets.specs.size); return@forEachIndexed }
                    val prefix = cleaner.prefix(active, settings.preferences, cleaner.systemOverride(cleaner.profile(active), settings))
                    mark(spec.unit, keeper.trialLlm(active, spec, prefix))
                    onProgress((i + 1f) / targets.specs.size)
                }
            }
            ModelKind.OCR_DET, ModelKind.OCR_REC -> {
                val self = targets.specs.firstOrNull()
                val partner = targets.ocrPartner
                if (self != null && partner != null) {
                    val (det, rec) = if (targets.kind == ModelKind.OCR_DET) self to partner else partner to self
                    mark(self.unit, keeper.trialOcr(det, rec))
                }
            }
            // A VAD is proven with the ASR artifact that uses it (load(asr, vad)).
            ModelKind.VAD -> Unit
        }
        onProgress(1f)
        scheduleIdle(cfg.idleUnloadMs)
        val needsCpu = targets.cpuArtifact.takeIf {
            targets.kind == ModelKind.ASR && ComputeUnit.NPU in failed && !targets.cpuArtifactInstalled
        }
        return Outcome.Ok(PrepareReport(artifact, proven, failed, needsCpu, untested))
    }

    /**
     * Load + a 1 s decode of the bundled clip. A NoSpeech verdict still proves the decode ran. A decode error is the
     * unit's (a runtime/op rejection the load did not catch) unless it is OutOfMemory; a decode timeout is not (a busy
     * device says nothing about the unit, and the dictation path does not mark decode timeouts either).
     */
    private suspend fun proveAsr(spec: LoadSpec, vad: LoadSpec?, cfg: DictationConfig): EngineKeeper.Trial {
        keeper.trialAsr(spec, vad).let { if (it.error != null) return it }
        val session = deps.engines.asr.startSession(AsrSessionOptions())
        feed(session, PrepareClip.samples)
        val r = withTimeoutOrNull(cfg.asrFinalTimeoutMs(PcmFormat.samplesToMs(PrepareClip.samples.size.toLong()))) { session.finish() }
            ?: return EngineKeeper.Trial.untested(MurmurError.Timeout(Stage.ASR_FINAL)).also { session.cancel() }
        return when (r) {
            is Outcome.Ok -> EngineKeeper.Trial.OK
            is Outcome.Err -> when (r.error) {
                MurmurError.NoSpeech -> EngineKeeper.Trial.OK
                is MurmurError.Timeout -> EngineKeeper.Trial.untested(r.error)
                else -> EngineKeeper.Trial.load(r.error)
            }
        }
    }

    private fun feed(session: AsrSession, pcm: ShortArray) {
        var i = 0
        while (i < pcm.size) {
            val end = minOf(i + PcmFormat.FRAME_SAMPLES, pcm.size)
            session.accept(PcmChunk(pcm.copyOfRange(i, end), i.toLong()))
            i = end
        }
    }

    private companion object {
        const val LEVEL_BARS = 7
        val WHITESPACE = Regex("\\s+")
    }
}

/** The bundled 1 s speech clip decoded while Preparing an ASR artifact (resources/prepare/, see its NOTICE). */
internal object PrepareClip {
    val samples: ShortArray by lazy {
        val bytes = PrepareClip::class.java.getResourceAsStream("/prepare/clip_1s.pcm")?.use { it.readBytes() }
            ?: error("prepare clip missing")
        ShortArray(bytes.size / 2) { i -> ((bytes[2 * i].toInt() and 0xff) or (bytes[2 * i + 1].toInt() shl 8)).toShort() }
    }
}
