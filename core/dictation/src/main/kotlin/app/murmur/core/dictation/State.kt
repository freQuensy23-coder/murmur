package app.murmur.core.dictation

import app.murmur.core.engine.LoadInfo
import app.murmur.core.engine.Transcript
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.EntryId
import app.murmur.core.model.InferProcess
import app.murmur.core.model.MurmurError
import app.murmur.core.model.SessionId

/**
 * Dictation state machine (ARCHITECTURE.md section 5). Pure types; [DictationMachine.reduce] is the only
 * place that decides transitions. One session at a time (I1); every completion event carries its [SessionId] and
 * is dropped when stale (I2).
 */
sealed interface DictationState {
    val session: SessionId?

    data object Idle : DictationState { override val session: SessionId? = null }

    /**
     * Bubble pressed; preflight, mic FGS + AudioRecord and engine binding run in parallel. Bubble shows Listening at
     * once. Arming completes (-> Listening) when both [plan] (PreflightPassed) and [micStartedAtMs] (MicStarted) are
     * known, in either order. [stopRequested] = a tap / hold release arrived >= 300 ms after the press but before
     * Arming completed: the completing transition continues straight to Finishing.
     */
    data class Arming(
        override val session: SessionId,
        val mode: StartMode,
        val pressedAtMs: Long,
        val plan: SessionPlan? = null,
        val micStartedAtMs: Long? = null,
        val engines: EngineStatus = EngineStatus(),
        val stopRequested: Boolean = false,
        /** When the early stop arrived (row 10); Finishing freezes the timer at max(micStartedAt, this). */
        val stopAtMs: Long? = null,
        /**
         * Retry of a Failed entry (row 49, DictationOrchestrator.retry): the entry's stored audio is the capture, so
         * Arming is born [stopRequested] and continues straight to Finishing; the bubble shows Transcribing.
         */
        val retryOf: EntryId? = null,
    ) : DictationState

    /** Audio flows to the ASR session (buffered until the model is loaded). */
    data class Listening(
        override val session: SessionId,
        val mode: StartMode,
        val startedAtMs: Long,
        val plan: SessionPlan,
        val engines: EngineStatus,
        val warn: Boolean = false,
        val hebrewActive: Boolean = false,
    ) : DictationState

    /** Stop received; waiting for the ASR tail. Bubble keeps Listening visuals (timer frozen) for 300 ms, then Transcribing. */
    data class Finishing(
        override val session: SessionId,
        val mode: StartMode,
        val startedAtMs: Long,
        val stoppedAtMs: Long,
        val plan: SessionPlan,
        val engines: EngineStatus,
        val showTranscribing: Boolean = false,
        /** True for stored-audio retries and crash replays where ASR must decode the full recording, not just the tail. */
        val wholeRecording: Boolean = false,
        val hebrewActive: Boolean = false,
    ) : DictationState

    /**
     * Transcript ready and cleanup planned. [cleanupStarted] = false while waiting for the screen context (at most
     * CONTEXT_WAIT) and/or a still-loading LLM (at most LLM_READY); the LLM timer starts only with RunCleanup
     * (review 2026-11-19: load time never eats the cleanup budget). Bubble: Polishing throughout.
     */
    data class Polishing(
        override val session: SessionId,
        val stoppedAtMs: Long,
        val transcript: Transcript,
        val plan: SessionPlan,
        val engines: EngineStatus,
        val cleanupStarted: Boolean = false,
        val contextWaitOver: Boolean = false,
    ) : DictationState

    /** Clipboard write + insertion ladder + history/metrics write. [shownAs] = the bubble of the previous working state. */
    data class Delivering(
        override val session: SessionId,
        val stoppedAtMs: Long,
        val text: String,
        val cleaned: Boolean,
        val shownAs: DictationBubble,
        /** cleanup_outcome of this delivery; becomes [DictationOutcome.Inserted.skipReason] when not cleaned. */
        val cleanupOutcome: String? = null,
    ) : DictationState

    /**
     * Result pill on screen; dismiss timer running. [recording] = the RecordFailure of this pill is still being written
     * (its entry arrives with FailureRecorded); [retryAtMs] = the pill's Retry was tapped meanwhile and runs once the
     * entry is known.
     */
    data class Done(
        override val session: SessionId,
        val outcome: DictationOutcome,
        val recording: Boolean = false,
        val retryAtMs: Long? = null,
    ) : DictationState
}

enum class StartMode {
    TAP,
    HOLD,
    /** Notification "Start dictation": the target snapshot waits until the shade has collapsed (ARCHITECTURE 5.2). */
    NOTIFICATION,
    /** In-app practice field (onboarding ob-ready); see [InAppTarget]. */
    IN_APP,
}

/** Result of the asynchronous preflight (executor), fixed for the rest of the session. */
data class SessionPlan(
    /** Cleanup will be attempted (choice != None, model resolved, not a password field, LLM allowed by memory). */
    val cleanup: Boolean,
    /** Why not, as cleanup_outcome "skipped:<code>" ("none", "no_model", "password", "low_memory"). */
    val cleanupSkipReason: String?,
    /** A screenshot will be taken (privacy decision FULL and "Use what's on screen" on). */
    val screenshot: Boolean,
    /** Batch ASR models need VAD; the resolver already guaranteed it is installed. */
    val asrUnit: ComputeUnit,
    /**
     * The ASR reports VAD speech time (batch models, fed VAD segments). false for streaming models, whose
     * Transcript.speechMs is 0 ("not measured"): row 26 then judges the text and the precheck only.
     */
    val speechMeasured: Boolean = true,
    /** Mid-recording עב toggle available (Hebrew enabled in settings and a distinct Hebrew ASR model is installed). */
    val hebrewAvailable: Boolean = false,
    /** Whether the Hebrew ASR model reports VAD speech time (false for streaming Zipformer2). */
    val hebrewSpeechMeasured: Boolean = false,
) {
    fun speechMeasuredFor(hebrewActive: Boolean): Boolean = if (hebrewActive) hebrewSpeechMeasured else speechMeasured
}

/** Readiness of the per-session engines, tracked by the reducer from completion events. */
data class EngineStatus(
    val asr: AsrReadiness = AsrReadiness.Loading,
    val llm: LlmReadiness = LlmReadiness.Loading,
    val context: ContextReadiness = ContextReadiness.Pending,
    /** :asr deaths in this session; the first is retried (re-bind + replay), the second fails (ARCHITECTURE 4.4). */
    val asrCrashes: Int = 0,
    val llmCrashes: Int = 0,
)

sealed interface AsrReadiness {
    data object Loading : AsrReadiness
    /** [fellBack] = the executor fell back to another unit/artifact after a load failure (UnitHealth marked). */
    data class Ready(val info: LoadInfo, val fellBack: Boolean) : AsrReadiness
    /** Remembered until Finishing, where it becomes Done(Failed) with the audio kept. */
    data class Failed(val error: MurmurError) : AsrReadiness
}

sealed interface LlmReadiness {
    /** Cleanup not planned for this session (SessionPlan.cleanup = false). */
    data class NotRequested(val reason: String) : LlmReadiness
    data object Loading : LlmReadiness
    data class Ready(val info: LoadInfo, val prefixTokens: Int, val nCtx: Int, val speed: LlmSpeed?, val fellBack: Boolean) : LlmReadiness
    data class Failed(val error: MurmurError) : LlmReadiness
}

sealed interface ContextReadiness {
    data object Pending : ContextReadiness
    data object NotRequested : ContextReadiness
    /** [contextWords] = words of OCR text the cleanup will see (history stores the number only, ADR-043). */
    data class Ready(val hasScreen: Boolean, val contextWords: Int) : ContextReadiness
}

sealed interface DictationOutcome {
    /** [noCleanup] = "Inserted, no cleanup" (cleanup failed / timed out / low memory / too long). */
    data class Inserted(val entry: EntryId?, val noCleanup: Boolean, val skipReason: String? = null) : DictationOutcome
    data class Copied(val entry: EntryId?) : DictationOutcome
    /** [audioKept] = [entry] holds the recording, so Retry can re-transcribe it (else Retry is a new dictation). */
    data class Failed(val error: MurmurError, val entry: EntryId?, val audioKept: Boolean = false) : DictationOutcome
    /** User cancelled (x, slide-off) or a password field took focus before delivery: nothing kept. */
    data class Discarded(val reason: String) : DictationOutcome
}

/** Bubble and notification inputs (DESIGN 5.3). */
sealed interface UserInput {
    /** Tap on Ready starts; tap anywhere on the Listening pill finishes; ignored while working. */
    data object Tap : UserInput
    /** Press held >= 250 ms on Ready (hold-to-talk, if enabled). */
    data object HoldStart : UserInput
    data object HoldRelease : UserInput
    /** Finger slid > 64 dp off the pill while holding: discard. */
    data object HoldSlideOff : UserInput
    /** The x button above the Listening pill. */
    data object Cancel : UserInput
    /** The עב button above the Listening pill: swap ASR between primary and Hebrew mid-recording. */
    data object ToggleHebrew : UserInput
    /** Notification action "Start dictation" (and the Quick Settings tile later). */
    data object NotificationStart : UserInput
    /** Onboarding practice field (row 4): posted by DictationOrchestrator.startInApp, never by the overlay. */
    data object InAppStart : UserInput
    /** Tap on the body of the Copied / Error pill (opens the entry / dismisses). */
    data object ResultTap : UserInput
    /** Tap on the 40 dp action button of an Error pill (Retry / Open / Open entry, DESIGN 5.5). */
    data object ResultAction : UserInput
}

/** Why recording stopped without the user (ARCHITECTURE 4.4, 5.2). Producers: WP-09 (capture), WP-02 (service). */
enum class StopReason {
    /** AudioManager mode IN_CALL / IN_COMMUNICATION, or the recording was silenced by a call. */
    CALL,
    /** AudioRecord read error / device lost mid-recording. */
    MIC_LOST,
    /** isClientSilenced became true mid-recording (another app took the mic). */
    MIC_SILENCED,
    /** Accessibility service disconnected (user disabled it, or the system unbound it). */
    A11Y_OFF,
    SCREEN_OFF,
}

enum class Timer { MIC_START, WARN, MAX_DURATION, TRANSCRIBING_LABEL, ASR_FINAL, CONTEXT_WAIT, LLM_READY, LLM, DELIVERY, RESULT_DISMISS }

/** Everything the reducer reacts to. Completion events come from [Effect] execution in the orchestrator. */
sealed interface DictationEvent {
    /**
     * [atMs] = MurmurClock.elapsedMs() of the input. [session] = the id a start input creates (rows 1-4, 46, 49): the
     * orchestrator stamps the next SessionId so the reducer stays pure; null = previous session + 1 (tests).
     */
    data class Input(val input: UserInput, val atMs: Long, val session: SessionId? = null) : DictationEvent
    data class PreflightPassed(val session: SessionId, val plan: SessionPlan) : DictationEvent
    /** Mic permission, service, ASR/VAD model, password field, excluded app, storage, thermal (ARCHITECTURE 5.2). */
    data class PreflightFailed(val session: SessionId, val error: MurmurError) : DictationEvent
    /**
     * The first buffer arrived at [atMs]. A Retry has its audio already: [recordedMs] of stored audio ending at
     * [atMs] (Finishing then spans exactly the stored recording).
     */
    data class MicStarted(val session: SessionId, val atMs: Long, val recordedMs: Long = 0) : DictationEvent
    data class MicFailed(val session: SessionId, val error: MurmurError) : DictationEvent
    /** ASR (+VAD) loaded, possibly after the executor's unit fallback. */
    data class AsrReady(val session: SessionId, val info: LoadInfo, val fellBack: Boolean) : DictationEvent
    /** Final: the primary and every installed fallback failed (UnitUnavailable(NPU) when no CPU artifact is installed, unless it ran out of memory: OutOfMemory(ASR), ADR-083). */
    data class AsrLoadFailed(val session: SessionId, val error: MurmurError) : DictationEvent
    /** LLM loaded and its prefix snapshot restored or built (PrefixInfo.tokens = [prefixTokens]). */
    data class LlmReady(val session: SessionId, val info: LoadInfo, val prefixTokens: Int, val nCtx: Int, val speed: LlmSpeed?, val fellBack: Boolean) : DictationEvent
    /** Final, after the executor's NPU -> GPU -> CPU fallback; OutOfMemory(LLM) when the memory gate refused. */
    data class LlmLoadFailed(val session: SessionId, val error: MurmurError) : DictationEvent
    /** ctx v2 assembled (screen may be absent); the ctx itself stays in the executor's session memory (I4). */
    data class ContextReady(val session: SessionId, val hasScreen: Boolean, val contextWords: Int) : DictationEvent
    /** [precheckSkip] = OutputGuard.precheck says noise/hallucination (no LLM, nothing inserted). */
    data class TranscriptReady(val session: SessionId, val transcript: Transcript, val precheckSkip: Boolean) : DictationEvent
    data class AsrFailed(val session: SessionId, val error: MurmurError) : DictationEvent
    /** [outcome] = history cleanup_outcome: guard "accept", "fallback:<rule>", "fallback:truncated", "skipped:<code>". */
    data class CleanupDone(val session: SessionId, val text: String, val cleaned: Boolean, val outcome: String) : DictationEvent
    data class Delivered(val session: SessionId, val result: DeliveryResult) : DictationEvent
    data class TimerFired(val session: SessionId, val timer: Timer) : DictationEvent
    data class ProcessDied(val session: SessionId, val process: InferProcess) : DictationEvent
    /** [atMs] = elapsedMs when the stop happened (freezes the Finishing timer); null = the recording start. */
    data class ExternalStop(val session: SessionId, val reason: StopReason, val atMs: Long? = null) : DictationEvent
    /**
     * Result of [Effect.RecordFailure]: the history entry (null when nothing is stored), for Retry / Open entry;
     * [audioKept] = the entry holds the recording.
     */
    data class FailureRecorded(val session: SessionId, val entry: EntryId?, val audioKept: Boolean = false) : DictationEvent
    /**
     * Re-run a Failed entry with stored audio through the normal flow (History "Try again", posted by
     * DictationOrchestrator.retry; the error pill's Retry does the same inside row 49). [session] as in [Input].
     */
    data class RetryRequested(val entry: EntryId, val atMs: Long, val session: SessionId? = null) : DictationEvent
    /** Mid-recording toggle of Hebrew ASR (dispatched via עב button or directly). */
    data class ToggleHebrew(val session: SessionId) : DictationEvent
}

/** Result of [Effect.Deliver]. */
sealed interface DeliveryResult {
    data class Inserted(val entry: EntryId?, val method: app.murmur.core.store.InsertMethod) : DeliveryResult
    data class CopiedOnly(val entry: EntryId?, val why: MurmurError) : DeliveryResult
    /** Focus moved to a password field before delivery: nothing inserted, copied or stored. */
    data object DiscardedPassword : DeliveryResult
    /** Guard EMPTY (only fillers were spoken): nothing inserted; stored as Failed(NoSpeech) with audio. */
    data class NothingToInsert(val entry: EntryId?) : DeliveryResult
    /** Accessibility service gone at delivery: text kept in the history entry, not inserted or copied. */
    data class NotDelivered(val entry: EntryId?, val error: MurmurError) : DeliveryResult
}

/** Where a result tap or action leads (implemented by WP-20's AppNavigator, deep links from WP-15). */
sealed interface OpenTarget {
    data class Entry(val id: EntryId) : OpenTarget
    /** Models tab; [kind] null = Speech segment. */
    data class Models(val kind: app.murmur.core.model.ModelKind? = null) : OpenTarget
}

/** Side effects the orchestrator executes; their completions come back as [DictationEvent]s. */
sealed interface Effect {
    /** Permission/service/storage/thermal checks, target snapshot (after the shade collapses for NOTIFICATION), privacy decision, model resolution -> PreflightPassed / PreflightFailed. */
    data class Preflight(val session: SessionId, val mode: StartMode) : Effect
    data class StartCapture(val session: SessionId) : Effect
    data class StopCapture(val session: SessionId) : Effect
    data class DiscardCapture(val session: SessionId) : Effect
    /**
     * D5: bind :asr and load ASR (+VAD), then OCR; bind :llm and load the LLM + prefix in parallel when the plan says
     * so and the memory gate allows (else after the ASR tail). Unit fallback on load failure happens inside the
     * executor (UnitHealth.markFailed, then ActiveModels.asrFallback / ActiveLlm.fallback).
     */
    data class PrepareEngines(val session: SessionId, val plan: SessionPlan) : Effect
    /** D11: screenshot ~0.6 s after start -> OCR -> ctx v2 -> ContextReady. */
    data class CaptureContext(val session: SessionId, val delayMs: Long) : Effect
    data class FinishAsr(val session: SessionId) : Effect
    /** After a :asr death: re-bind, new session, replay the session buffer from sample 0; [thenFinish] in Finishing. */
    data class RestartAsr(val session: SessionId, val thenFinish: Boolean) : Effect
    /** Mid-recording עב toggle: swap :asr to Hebrew (or back to primary) and replay the session buffer from sample 0. */
    data class SwapAsr(val session: SessionId, val hebrew: Boolean) : Effect
    /** After a :llm death before cleanup: re-bind, load, restore the prefix snapshot from disk. */
    data class ReloadLlm(val session: SessionId) : Effect
    /** render (ctx v2 + transcript) -> generate (budget) -> guard -> CleanupDone. */
    data class RunCleanup(val session: SessionId, val transcript: Transcript, val budget: CleanupBudget.Plan) : Effect
    /** Clipboard + insertion ladder + history + metrics -> Delivered. [cleanupOutcome] as in [DictationEvent.CleanupDone]. */
    data class Deliver(val session: SessionId, val text: String, val raw: String?, val cleaned: Boolean, val cleanupOutcome: String) : Effect
    data class RecordFailure(val session: SessionId, val error: MurmurError, val keepAudio: Boolean) : Effect
    data class StartTimer(val session: SessionId, val timer: Timer, val ms: Long) : Effect
    data class CancelTimer(val timer: Timer) : Effect
    data class Haptic(val kind: HapticKind) : Effect
    /** Keep processes warm for the idle timeout, then unload + exit (D5). */
    data class ScheduleIdleUnload(val ms: Long) : Effect
    data class CancelEngines(val session: SessionId) : Effect
    data class Open(val target: OpenTarget) : Effect
}

enum class HapticKind { CONFIRM, CLOCK_TICK, REJECT }

data class Transition(val state: DictationState, val effects: List<Effect> = emptyList())
