package app.murmur.core.dictation

import app.murmur.core.model.ComputeUnit
import kotlin.math.ceil

/** Measured (benchmark) or default cleanup speed of the active (artifact, unit). */
data class LlmSpeed(val prefillTps: Double, val decodeTps: Double)

/**
 * Every timeout of the dictation flow (ARCHITECTURE.md 5.3). Defaults from DESIGN 5.3/5.5 and the spikes.
 * [timeoutScale] multiplies every timer (debug builds: Settings.debugTimeoutScale for the slow x86_64 emulator).
 */
data class DictationConfig(
    /** Tap -> first audio buffer; spike: 8-90 ms FGS + 67-209 ms first buffer. */
    val micStartTimeoutMs: Long = 2_000,
    /** D11 / BRIEF flow 2: screenshot 0.5-1 s after start. Must be >= 340 ms after any other capture (rate limit). */
    val screenshotDelayMs: Long = 600,
    /** NOTIFICATION start: wait at most this long for the shade to collapse before the target snapshot. */
    val shadeCollapseWaitMs: Long = 1_000,
    /** OCR result later than this after the screenshot is ignored (cleanup runs without screen). */
    val ocrBudgetMs: Long = 4_000,
    /** Transcript ready but context still pending: wait at most this long, then render without the screen. */
    val contextWaitMs: Long = 800,
    /** DESIGN 5.5: warn at 4:30, auto-stop at 5:00. */
    val warnAtMs: Long = 270_000,
    val maxDurationMs: Long = 300_000,
    /** DESIGN 5.3: Transcribing appears only if text is not ready within 300 ms of the stop. */
    val transcribingLabelMs: Long = 300,
    /** ASR tail: base + factor x tail audio (NPU 20 s window = one encoder call; CPU fallback slower). */
    val asrFinalBaseMs: Long = 4_000,
    val asrFinalPerAudioSec: Double = 0.5,
    /** Extra headroom on ASR_FINAL when the user taps stop while :asr is still cold-loading (matches 15 s loadAsr). */
    val asrLoadWaitMs: Long = 15_000,
    /** "Didn't catch that" below this much VAD speech (matches SegmenterConfig.minSpeechMs = 250 ms). */
    val minSpeechMs: Long = 250,
    /**
     * Transcript ready, LLM still loading (cold NPU load, or a prefix prefill after a prompt edit): wait at most this
     * long, then insert raw ("skipped:llm_not_ready"); the load continues so the next dictation is warm.
     */
    val llmReadyWaitMs: Long = 10_000,
    /** ADR-015 (revised by ADR-045): adaptive cleanup timeout = min(hardCap, base + safety x expected). */
    val llmBaseMs: Long = 4_000,
    val llmSafety: Double = 1.5,
    val llmHardCapMs: Long = 64_000,
    /** Cleanup whose expected time exceeds this is not attempted ("skipped:too_long"); owner-visible rule (ADR-045). */
    val llmMaxExpectedMs: Long = 40_000,
    /** Speeds used before any benchmark of the active (artifact, unit) exists (conservative, [I]). */
    val defaultSpeed: Map<ComputeUnit, LlmSpeed> = mapOf(
        ComputeUnit.NPU to LlmSpeed(prefillTps = 400.0, decodeTps = 20.0),
        ComputeUnit.GPU to LlmSpeed(prefillTps = 150.0, decodeTps = 15.0),
        ComputeUnit.CPU to LlmSpeed(prefillTps = 100.0, decodeTps = 15.0),
    ),
    val deliveryTimeoutMs: Long = 2_000,
    /**
     * The DELIVERY timer never runs shorter than this, whatever [timeoutScale] (ADR-090 item 4): the insertion ladder
     * (InsertionLadder.LADDER_BUDGET_MS, 1.5 s from its first read, not scaled) plus 400 ms for re-resolving the
     * target, the clipboard copy and the binder reads. A shorter timer would show Copied while a rung can still insert.
     */
    val deliveryMinMs: Long = 1_900,
    /** Result pill durations (DESIGN 5.3). */
    val insertedShowMs: Long = 1_200,
    val copiedShowMs: Long = 4_000,
    val errorShowMs: Long = 6_000,
    /** D5 idle timeout default (Settings.idleUnloadSec overrides). */
    val idleUnloadMs: Long = 60_000,
    /** Spare memory to keep when loading the LLM next to ASR (ARCHITECTURE.md 7). */
    val llmMemoryMarginMb: Int = 600,
    val timeoutScale: Double = 1.0,
    /** Row 10: a stop tap or hold release closer than this to the press is ignored (double-tap protection). */
    val doubleTapGuardMs: Long = 300,
    /** Settings.holdToTalk: row 2 only applies when it is on. */
    val holdToTalk: Boolean = true,
    /**
     * ASR_FINAL bound on the open tail: the segmenter closes a segment at least every maxWindowSec (NPU 20 s), so the
     * tail after a stop is never longer than this, however long the recording.
     */
    val asrMaxTailMs: Long = 30_000,
    /** Row 37 / ARCHITECTURE 4.4: after LlmEngine.cancel(), wait this long for the generation to end, then shutdown(:llm). */
    val llmCancelGraceMs: Long = 1_000,
    /** EnginePreparer.invalidatePrefix debounce (ARCHITECTURE 4.2). */
    val prefixRebuildDebounceMs: Long = 5_000,
    /** After CaptureHandle.stop(), how long the chunk stream may take to deliver its last samples. */
    val captureDrainMs: Long = 500,
    /**
     * A retry's "mic start": reading and decoding the stored recording (AudioStore.readPcm, Ogg/Opus, up to 5 min)
     * instead of MIC_START's 2 s; past it the retry fails with StorageFailed("audio_read"). [I] until DC-measured.
     */
    val storedAudioReadTimeoutMs: Long = 10_000,
    /** Row 6: free storage needed to start (LowStorage below). */
    val minFreeStorageBytes: Long = 50L * 1024 * 1024,
    /** Murmur's own package: only StartMode.IN_APP dictates into it (ARCHITECTURE 9.2). */
    val ownPackage: String = "app.murmur",
) {
    fun scaled(ms: Long): Long = (ms * timeoutScale).toLong()

    /** The DELIVERY timer: [deliveryTimeoutMs] scaled, but never below [deliveryMinMs] (ADR-090 item 4). */
    fun deliveryTimerMs(): Long = maxOf(scaled(deliveryTimeoutMs), deliveryMinMs)

    fun asrFinalTimeoutMs(tailAudioMs: Long): Long =
        scaled(asrFinalBaseMs + (tailAudioMs / 1000.0 * asrFinalPerAudioSec * 1000).toLong())

    /** Timer started together with RunCleanup (never during the LLM load). */
    fun llmTimeoutMs(expectedMs: Long): Long =
        scaled((llmBaseMs + expectedMs * llmSafety).toLong().coerceAtMost(llmHardCapMs))
}

/**
 * Does a cleanup request fit the model, and how long may it take (ADR-045, review 2026-11-19)? Pure; the reducer
 * runs it right before RunCleanup. Token counts are estimated in :main at [TOKENS_PER_CHAR] (conservative: the
 * ctx v2 goldens measure 0.31-0.34 tokens/char with the Qwen3.5 and LFM3 tokenizers); :llm re-checks after exact
 * tokenisation (LlmRequest.minTokens -> LlmContextOverflow).
 */
object CleanupBudget {
    const val TOKENS_PER_CHAR = 0.40
    /** Style line + <context> (1,400-char screen budget + 300 chars before the cursor + app/field lines) + tags. */
    const val CONTEXT_RESERVE_CHARS = 1_850
    /** guard.py R4: an output longer than 1.5 x input words is rejected anyway, so this is the useful maximum. */
    const val OUTPUT_FACTOR = 1.5
    const val MIN_OUTPUT_FACTOR = 1.1

    data class Plan(
        val fits: Boolean,
        /** "too_long" when the context or the time budget is exceeded. */
        val skipReason: String?,
        val maxTokens: Int,
        val minTokens: Int,
        val expectedMs: Long,
        val timeoutMs: Long,
    )

    fun estTokens(chars: Int): Int = ceil(chars * TOKENS_PER_CHAR).toInt()

    fun plan(transcriptChars: Int, prefixTokens: Int, nCtx: Int, speed: LlmSpeed, config: DictationConfig): Plan {
        val tTok = estTokens(transcriptChars)
        val suffixTok = tTok + estTokens(CONTEXT_RESERVE_CHARS)
        val minTokens = ceil(tTok * MIN_OUTPUT_FACTOR).toInt() + 16
        val room = nCtx - prefixTokens - suffixTok
        val maxTokens = minOf(ceil(tTok * OUTPUT_FACTOR).toInt() + 64, room)
        val expectedMs = (suffixTok / speed.prefillTps * 1000 + minTokens / speed.decodeTps * 1000).toLong()
        val fits = room >= minTokens && expectedMs <= config.llmMaxExpectedMs
        return Plan(fits, if (fits) null else "too_long", maxTokens.coerceAtLeast(0), minTokens, expectedMs, config.llmTimeoutMs(expectedMs))
    }
}
