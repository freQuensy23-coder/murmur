package app.murmur.core.dictation

import app.murmur.core.model.EntryId
import app.murmur.core.model.MurmurError

/**
 * What the dictation engine tells the bubble (DESIGN 5.3). While no dictation runs it is [Idle] and the
 * overlay's visibility policy decides Hidden / Rest / Ready; Dragging is overlay-local.
 */
sealed interface DictationBubble {
    data object Idle : DictationBubble
    /** [frozenAtMs] != null: stop received, the timer shows frozenAt - startedAt (first 300 ms of Finishing). */
    data class Listening(
        val startedAtMs: Long,
        val hold: Boolean,
        val warn: Boolean,
        val frozenAtMs: Long? = null,
        val hebrewAvailable: Boolean = false,
        val hebrewActive: Boolean = false,
    ) : DictationBubble
    data object Transcribing : DictationBubble
    data object Polishing : DictationBubble
    data class Inserted(val noCleanup: Boolean) : DictationBubble
    data class Copied(val entry: EntryId?) : DictationBubble
    data class Error(val reason: BubbleError, val action: BubbleAction?) : DictationBubble
}

/** DESIGN 5.5 error rows -> copy-deck keys (bubble_err_*). */
enum class BubbleError { NO_SPEECH, MIC_BUSY, NO_MODEL, LOAD_FAILED, TRANSCRIPTION_FAILED, GENERIC }

enum class BubbleAction { RETRY, OPEN_MODELS, OPEN_ENTRY }

/** The DESIGN 5.3/5.5 mapping, state -> bubble. Pure and total; unit-tested (BubbleMappingTest). */
object BubbleMapping {
    fun of(state: DictationState): DictationBubble = when (state) {
        DictationState.Idle -> DictationBubble.Idle
        // A retry has no live recording: it is decoding stored audio from the start.
        is DictationState.Arming -> if (state.retryOf != null) DictationBubble.Transcribing
            else DictationBubble.Listening(state.pressedAtMs, state.mode == StartMode.HOLD, warn = false,
                hebrewAvailable = state.plan?.hebrewAvailable == true && state.mode != StartMode.HOLD)
        is DictationState.Listening -> DictationBubble.Listening(
            state.startedAtMs, state.mode == StartMode.HOLD, state.warn,
            hebrewAvailable = state.plan.hebrewAvailable && state.mode != StartMode.HOLD, hebrewActive = state.hebrewActive,
        )
        is DictationState.Finishing -> if (state.showTranscribing) DictationBubble.Transcribing
            else DictationBubble.Listening(
                state.startedAtMs, state.mode == StartMode.HOLD, warn = false, frozenAtMs = state.stoppedAtMs,
                hebrewAvailable = false, hebrewActive = state.hebrewActive,
            )
        is DictationState.Polishing -> DictationBubble.Polishing
        // ARCHITECTURE 5.1: Delivering keeps whatever the previous working state showed (no Transcribing flash).
        is DictationState.Delivering -> state.shownAs
        is DictationState.Done -> when (val o = state.outcome) {
            is DictationOutcome.Inserted -> DictationBubble.Inserted(o.noCleanup)
            is DictationOutcome.Copied -> DictationBubble.Copied(o.entry)
            is DictationOutcome.Failed -> errorBubble(o.error)
            is DictationOutcome.Discarded -> DictationBubble.Idle
        }
    }

    fun errorBubble(e: MurmurError): DictationBubble.Error = when (e) {
        MurmurError.NoSpeech -> DictationBubble.Error(BubbleError.NO_SPEECH, BubbleAction.RETRY)
        MurmurError.MicBusy, MurmurError.MicSilenced, is MurmurError.MicStartFailed, is MurmurError.MicLost -> DictationBubble.Error(BubbleError.MIC_BUSY, null)
        MurmurError.NoAsrModel, MurmurError.NoVadModel -> DictationBubble.Error(BubbleError.NO_MODEL, BubbleAction.OPEN_MODELS)
        // NPU load failed and no CPU artifact installed: the hub offers "Download the CPU version" (ARCHITECTURE 4.4).
        is MurmurError.UnitUnavailable -> DictationBubble.Error(BubbleError.LOAD_FAILED, BubbleAction.OPEN_MODELS)
        is MurmurError.EngineLoadFailed, is MurmurError.OutOfMemory, is MurmurError.EngineRuntimeMissing ->
            DictationBubble.Error(BubbleError.LOAD_FAILED, BubbleAction.RETRY)
        is MurmurError.AsrFailed, is MurmurError.ProcessDied, is MurmurError.Timeout -> DictationBubble.Error(BubbleError.TRANSCRIPTION_FAILED, BubbleAction.OPEN_ENTRY)
        // AccessibilityOff: the overlay is gone with the service, so nothing is drawn; Dashboard/Settings show the banner.
        else -> DictationBubble.Error(BubbleError.GENERIC, null)
    }
}
