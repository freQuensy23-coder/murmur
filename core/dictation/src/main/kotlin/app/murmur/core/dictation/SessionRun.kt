package app.murmur.core.dictation

import app.murmur.core.audio.PcmChunk
import app.murmur.core.context.A11ySnapshot
import app.murmur.core.context.PrivacyDecision
import app.murmur.core.engine.AsrSession
import app.murmur.core.engine.LoadSpec
import app.murmur.core.engine.Transcript
import app.murmur.core.model.EntryId
import app.murmur.core.model.MurmurError
import app.murmur.core.model.SessionId
import app.murmur.core.ocr.ScreenText
import app.murmur.core.store.HistoryEntry
import app.murmur.core.store.Settings
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import java.time.Instant

/**
 * Executor-side memory of one dictation (never part of the reducer state). Touched only on the orchestrator's serial
 * dispatcher. The screenshot text and anything derived from it live here only and are dropped when the session ends
 * (invariant I4).
 */
internal class SessionRun(
    val session: SessionId,
    val mode: StartMode,
    val createdAt: Instant,
    val settings: Settings,
    val config: DictationConfig,
    val inApp: InAppTarget?,
    /** Retry of this Failed entry: its stored audio is the capture, and the recorder updates it instead of adding one. */
    val retryOf: EntryId? = null,
) {
    /** The entry being retried, as read when its audio was loaded. */
    var retryEntry: HistoryEntry? = null

    // Preflight
    var target: A11ySnapshot? = null
    var decision: PrivacyDecision? = null
    var models: ActiveModels? = null
    var plan: SessionPlan? = null

    // Capture: the session buffer holds every chunk from sample 0 (replay after a crash or a unit fallback).
    var capture: CaptureHandle? = null
    var captureCancelled = false
    val buffer = ArrayList<PcmChunk>()
    val captureJobs = mutableListOf<Job>()
    var pumpJob: Job? = null
    /** Completed by StopCapture / DiscardCapture / a failed start: the capture result (null = nothing recorded). */
    val captureStopped = CompletableDeferred<CaptureResult?>()
    /** Completed once the chunk stream delivered its last sample after the stop (or gave up after captureDrainMs). */
    val drained = CompletableDeferred<Unit>()
    var stopAtElapsedMs: Long? = null

    // ASR
    val asrSlot = MutableStateFlow<AsrSlot>(AsrSlot.Loading)
    var asrSession: AsrSession? = null
    var asrSpec: LoadSpec? = null
    var asrFellBack = false
    var hebrewActive = false
    var asrJob: Job? = null
    var finishJob: Job? = null
    /** Pid of the :asr instance the current ASR session runs in (HostState.Up at attach time). */
    var asrPid: Int? = null
    /**
     * :asr instances whose death this session already reported: the failing call (EPIPE / DeadObject) and the host's
     * Crashed notice report the same death in either order, possibly while the restart is already under way.
     */
    val asrDeaths = HashSet<Int?>()
    val ocrLoaded = CompletableDeferred<Boolean>()
    var transcript: Transcript? = null
    var asrTailMs: Long? = null

    // Context (I4: in memory only)
    var contextJob: Job? = null
    var screen: ScreenText? = null
    /** How the screen capture ended (cause of a missing screen for ctx v2 privacy, WP-08 ScreenOutcome). */
    var screenOutcome: app.murmur.core.context.ScreenOutcome? = null
    var ocrMs: Long? = null
    var contextWords: Int? = null

    // LLM
    var llmJob: Job? = null
    var llmSpec: LoadSpec? = null
    /** Memory gate refused the parallel load: load after the ASR tail, once :asr is shut down (ARCHITECTURE 7). */
    var llmDeferred = false
    var cleanupJob: Job? = null
    var cleanupUsedScreen = false
    var promptVersion: String? = null
    var llmMs: Long? = null
    /** The tone the cleanup ran with (ADR-088: tonePerApp, else defaultTone for an app without a category). */
    var tone: app.murmur.core.store.Tone? = null

    // Delivery
    var a11yLost = false
    /** Delivery and RecordFailure coroutines: they read the target and transcript, which are dropped after them (I4). */
    val writers = mutableListOf<Job>()
    var ended = false
}

internal sealed interface AsrSlot {
    data object Loading : AsrSlot
    data class Ready(val session: AsrSession) : AsrSlot
    data class Failed(val error: MurmurError) : AsrSlot
}
