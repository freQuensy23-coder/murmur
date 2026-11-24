package app.murmur.core.dictation

import app.murmur.core.audio.PcmChunk
import app.murmur.core.catalog.AsrTraits
import app.murmur.core.context.A11ySnapshot
import app.murmur.core.engine.LoadSpec
import app.murmur.core.model.ArtifactId
import app.murmur.core.model.MurmurError
import app.murmur.core.model.Outcome
import app.murmur.core.model.SessionId
import app.murmur.core.ocr.ScreenImage
import app.murmur.core.store.AudioRef
import app.murmur.core.store.InsertMethod
import app.murmur.core.store.Settings
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.StateFlow

/** :platform:audio. Mic FGS + AudioRecord in :main (platform spike), PCM also encoded to the history file. */
interface AudioCapture {
    /** [audioKey] = AudioKey.of(createdAt, session).value: the history audio file name (never the bare session). */
    suspend fun start(session: SessionId, keepRecording: Boolean, audioKey: String): Outcome<CaptureHandle>
}

interface CaptureHandle {
    /**
     * Every sample from the first buffer on, replayed to late collectors (the ASR session may attach after load).
     * Completes after [stop] once the last captured sample was emitted (the orchestrator waits at most
     * DictationConfig.captureDrainMs for that, then finishes the ASR with what it has).
     */
    val chunks: Flow<PcmChunk>
    /** 7 bar heights 0..1 at ~30 Hz for the bubble (LevelMeter). */
    val levels: StateFlow<FloatArray>
    /** Mid-recording problems (WP-09); WP-11 maps them to DictationEvent.ExternalStop. */
    val signals: Flow<CaptureSignal>
    suspend fun stop(): CaptureResult
    fun cancel()
}

sealed interface CaptureSignal {
    /** AudioRecordingCallback / isClientSilenced became true (another app took the mic). */
    data object Silenced : CaptureSignal
    /** AudioRecord.read error or the input device disappeared. */
    data class Lost(val error: MurmurError) : CaptureSignal
    /**
     * AudioManager mode changed during the recording to a call mode (IN_CALL, IN_COMMUNICATION, RINGTONE,
     * CALL_SCREENING, CALL_REDIRECT, COMMUNICATION_REDIRECT): incoming (ringing) or outgoing call. A call already up at
     * the start raises nothing; at most once per session (WP-09 CallWatch).
     */
    data object CallStarted : CaptureSignal
}

data class CaptureResult(val durationMs: Long, val audio: AudioRef?, val peakAbs: Int, val silenced: Boolean)

/** :platform:a11y. Reads the target at press time, captures the window screenshot, re-checks before delivery. */
interface TargetReader {
    suspend fun snapshot(): A11ySnapshot?
    /**
     * NOTIFICATION start (ARCHITECTURE 5.2): wait until the notification shade has collapsed (first WINDOWS_CHANGED with
     * a focused TYPE_APPLICATION window that is not SystemUI), at most [timeoutMs], then [snapshot]. null = no app window.
     */
    suspend fun snapshotAfterShade(timeoutMs: Long): A11ySnapshot?
    /** takeScreenshotOfWindow(target) with the fallbacks of ARCHITECTURE.md 9.1; Err(SecureWindow) for FLAG_SECURE. */
    suspend fun screenshot(target: A11ySnapshot): Outcome<ScreenImage>
    /** Current focus right before insertion (focus may have moved or become a password field). */
    suspend fun current(): A11ySnapshot?
}

/** :platform:a11y. Insertion ladder commitText -> ACTION_PASTE -> ACTION_SET_TEXT, verified by getSurroundingText. */
interface Inserter {
    suspend fun insert(target: A11ySnapshot, text: String): InsertResult
    /** ClipData with EXTRA_IS_SENSITIVE=false (D: always copied as well, BRIEF flow 7). */
    fun copyToClipboard(text: String): Boolean
}

sealed interface InsertResult {
    data class Inserted(val method: InsertMethod) : InsertResult
    data class Failed(val error: MurmurError) : InsertResult
}

/**
 * Bubble feedback (WP-03, from the overlay view): haptics per Settings.feedback (VIBRATION/BOTH) and earcons
 * (SOUND/BOTH). The start earcon plays before the microphone opens; earcon sounds are an open design question
 * (DESIGN 6.5 lists the setting, no asset exists): until decided, [sound] is a no-op and Settings shows the option.
 */
interface Feedback {
    fun haptic(kind: HapticKind)
    fun sound(kind: HapticKind)
}

/**
 * Resolves the selected models into load specs (catalog + install state + device profile + unit choice + unit
 * health + benchmark). Implemented in :data:models (WP-13). Never throws: a missing device profile resolves with
 * DeviceProfile.cpuOnly; errors are Err(NoAsrModel) (no installed artifact of a selectable ASR model; previewOnly
 * models are never selected, ADR-047) and Err(NoVadModel) (batch ASR without an installed VAD). A missing LLM is not
 * an error: ActiveModels.llm = null with [ActiveModels.llmUnavailable] = "none" / "no_model".
 */
interface ModelResolver {
    suspend fun resolve(settings: Settings): Outcome<ActiveModels>

    /**
     * EnginePreparer input (ARCHITECTURE 8.5): [artifact] (installed, or being installed: InstallState.Preparing) as
     * load specs on every usable unit of this phone (ArtifactSelector.candidates with UnitHealth, in ranked order),
     * built exactly like [resolve] builds them (absolute dir, layout, LoadOptions). Err(CatalogInvalid) for an
     * unknown artifact.
     */
    suspend fun prepareTargets(artifact: ArtifactId, settings: Settings): Outcome<PrepareTargets>
}

/** What [EnginePreparer.prepareArtifact] proves for one artifact. */
data class PrepareTargets(
    val kind: app.murmur.core.model.ModelKind,
    /** One spec per usable unit, ranked (NPU first on the Fold7). Empty = nothing usable on this phone. */
    val specs: List<LoadSpec>,
    /** ASR: the installed VAD for batch runtimes (null for streaming ones, or when it is not installed yet). */
    val vad: LoadSpec? = null,
    /** ASR: batch runtime (VAD-segmented); preparing it without [vad] is refused (Err(NoVadModel), nothing marked). */
    val needsVad: Boolean = false,
    /** LLM: what the prefix snapshot is built from (template, profile, nCtx, model sha). */
    val llm: LlmPrepare? = null,
    /** OCR det/rec: the other stage of the pair when installed (loadOcr needs both); null = proven when it arrives. */
    val ocrPartner: LoadSpec? = null,
    /** ASR: the model's CPU artifact (catalog), and whether it is installed (PrepareReport.needsCpuArtifact). */
    val cpuArtifact: ArtifactId? = null,
    val cpuArtifactInstalled: Boolean = false,
)

/** [memoryMb] = the same estimate as ActiveLlm.memoryMb: preparing obeys the memory gate of ARCHITECTURE 7 too. */
data class LlmPrepare(val template: String, val profile: String, val nCtx: Int, val modelSha256: String, val memoryMb: Int)

/**
 * Preflight's device checks (row 6): free storage and thermal state. :app implements it (StatFs on filesDir,
 * PowerManager.currentThermalStatus >= THERMAL_STATUS_CRITICAL).
 */
interface SystemConditions {
    fun freeStorageBytes(): Long
    fun thermalCritical(): Boolean
}

data class ActiveModels(
    val asr: LoadSpec,
    /**
     * Next usable installed candidate after [asr] (ArtifactSelector.ranked), loaded by the executor when [asr] fails
     * (ARCHITECTURE 4.4); null = none installed -> UnitUnavailable / "Download the CPU version".
     */
    val asrFallback: LoadSpec?,
    val asrTraits: AsrTraits,
    val vad: LoadSpec?,
    val ocrDet: LoadSpec?,
    val ocrRec: LoadSpec?,
    val llm: ActiveLlm?,
    /** cleanup_outcome code when [llm] is null: "none" (user chose None) or "no_model" (selected model not installed). */
    val llmUnavailable: String? = null,
    /** Optional Hebrew ASR model resolved when "he" is enabled in settings (mid-recording עב toggle). */
    val hebrewAsr: LoadSpec? = null,
    val hebrewAsrFallback: LoadSpec? = null,
    val hebrewAsrTraits: AsrTraits? = null,
)

data class ActiveLlm(
    val spec: LoadSpec,
    /** Same artifact on the next usable unit (NPU -> GPU -> CPU), then other installed artifacts; no download needed. */
    val fallback: LoadSpec?,
    /** render.py TEMPLATES key and prompt profile from the catalog (custom: from GgufTemplates). */
    val template: String,
    val profile: String,
    val memoryMb: Int,
    /** Catalog LlmTraits.nCtx (passed as LoadSpec option n_ctx). */
    val nCtx: Int,
    /** Catalog LlmTraits.prefixTokens (custom: from the Preparing PrefixInfo); PrefixInfo.tokens wins at runtime. */
    val prefixTokens: Int?,
    /** sha256 of the GGUF (catalog FileSpec), part of the prefix cache key. */
    val modelSha256: String,
    /** Latest VALID benchmark of (artifact, unit); null -> DictationConfig.defaultSpeed[unit]. */
    val speed: LlmSpeed?,
)

/** ActivityManager.MemoryInfo in :main; decides whether the LLM can load next to the ASR (ARCHITECTURE.md 7). */
fun interface MemoryGate {
    fun availableMb(): Long
}

/** Opens app screens from the bubble (WP-20: Intent to MainActivity with WP-15's deep-link extras). */
fun interface AppNavigator {
    fun open(target: OpenTarget)
}

/** Permission and service status for preflight, onboarding and Settings > Permissions (WP-20 implements in :app). */
data class PermissionState(
    val micGranted: Boolean,
    val notificationsGranted: Boolean,
    val a11yEnabled: Boolean,
    val a11yConnected: Boolean,
    /** onServiceConnected ran at least once since install (RestrictedSettingsProbe input). */
    val a11yEverConnected: Boolean,
    val batteryUnrestricted: Boolean,
    /** RestrictedSettingsProbe.likelyBlocked (DESIGN 6.1 ob-restricted). */
    val restrictedLikely: Boolean,
)

interface PermissionMonitor {
    val state: StateFlow<PermissionState>
    /** Re-read everything (on resume, on return from a system screen). */
    fun refresh()
}

/**
 * The persistent idle notification of DESIGN 5.1 ("Murmur is ready", models as subtitle, "Hide 1 h", "Start
 * dictation", "Show now" while hidden). Separate from the microphone FGS notification (I3). WP-09 implements; :app
 * (WP-20) shows it while the service is connected and notifications are allowed, and updates the subtitle from
 * the resolver and the hidden state from Settings.hiddenUntilEpochMs.
 */
interface StatusNotifier {
    fun show(modelsLine: String, hiddenUntilEpochMs: Long?)
    fun hide()
}

/**
 * Install-time and change-time preparation (ARCHITECTURE 8.5, review 2026-11-19). Implemented by the
 * orchestrator (it shares the inference processes and must never run during a dictation: a press cancels it).
 */
interface EnginePreparer {
    /**
     * After an artifact is installed (WP-13 calls it while InstallState = Preparing): load it on every usable unit;
     * ASR: decode a bundled 1 s clip; LLM: build and persist the prefix snapshot. Marks UnitHealth proven / failed
     * (never for OutOfMemory, ADR-066; see [PrepareReport]).
     * An NPU failure with no CPU artifact installed reports [PrepareReport.needsCpuArtifact] (WP-13 queues it).
     */
    suspend fun prepareArtifact(artifact: ArtifactId, onProgress: (Float) -> Unit): Outcome<PrepareReport>
    /**
     * The prefix of the active LLM changed (prompt edit, preferences edit, unit or model change): rebuild the snapshot
     * in the background (debounced 5 s, only while no dictation runs). Called by the settings observer in WP-11.
     */
    fun invalidatePrefix()
}

/**
 * [proven]: marked proven in UnitHealth. [failed]: marked failed (the unit cannot run the artifact; an NPU entry for
 * an ASR artifact gives [needsCpuArtifact]). [untested]: tried, but the error says nothing about the unit (OutOfMemory,
 * no process, a prefix error, a proof-decode timeout; ADR-066): UnitHealth is unchanged and the first real load decides.
 * A unit skipped by the memory gate is in none of them.
 */
data class PrepareReport(
    val artifact: ArtifactId,
    val proven: Set<app.murmur.core.model.ComputeUnit>,
    val failed: Map<app.murmur.core.model.ComputeUnit, MurmurError>,
    val needsCpuArtifact: ArtifactId?,
    val untested: Map<app.murmur.core.model.ComputeUnit, MurmurError> = emptyMap(),
)

/**
 * In-app dictation target (onboarding practice field ob-ready, DESIGN 6.1): the orchestrator runs the normal pipeline
 * but reads the target from here and inserts here; no screenshot (privacy app_only), nothing stored in history.
 */
interface InAppTarget {
    val snapshot: A11ySnapshot
    fun insert(text: String)
}

/**
 * How dictated text joins the text before the cursor (WP-11, pure, unit-tested): a leading space when
 * [beforeCursor] ends with a letter, digit or closing punctuation and [text] starts with a letter or digit;
 * nothing at the start of a field or after whitespace/opening brackets. Lowercases the first letter only when
 * the field context is mid-sentence (before ends without . ! ? or newline) AND the guard accepted the text.
 */
interface JoinPolicy {
    /** [lowercaseAllowed] = the guard accepted the text (raw transcripts and fallbacks keep their casing). */
    fun join(beforeCursor: String?, text: String, lowercaseAllowed: Boolean = false): String
}
