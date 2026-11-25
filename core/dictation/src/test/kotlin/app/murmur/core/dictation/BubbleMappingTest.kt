package app.murmur.core.dictation

import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.MurmurError
import app.murmur.core.model.SessionId
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class BubbleMappingTest {
    private val s = SessionId(1)
    private val plan = SessionPlan(cleanup = true, cleanupSkipReason = null, screenshot = true, asrUnit = ComputeUnit.CPU)

    @Test
    fun designStatesMap() {
        assertEquals(DictationBubble.Idle, BubbleMapping.of(DictationState.Idle))
        assertEquals(DictationBubble.Listening(5, hold = true, warn = false), BubbleMapping.of(DictationState.Arming(s, StartMode.HOLD, 5)))
        assertEquals(DictationBubble.Transcribing,
            BubbleMapping.of(DictationState.Finishing(s, StartMode.TAP, 1, 9, plan, EngineStatus(), showTranscribing = true)))
        assertEquals(DictationBubble.Inserted(noCleanup = true),
            BubbleMapping.of(DictationState.Done(s, DictationOutcome.Inserted(null, noCleanup = true))))
        assertEquals(DictationBubble.Error(BubbleError.NO_SPEECH, BubbleAction.RETRY),
            BubbleMapping.of(DictationState.Done(s, DictationOutcome.Failed(MurmurError.NoSpeech, null))))
        assertEquals(DictationBubble.Error(BubbleError.NO_MODEL, BubbleAction.OPEN_MODELS), BubbleMapping.errorBubble(MurmurError.NoAsrModel))
        assertEquals(DictationBubble.Error(BubbleError.LOAD_FAILED, BubbleAction.OPEN_MODELS),
            BubbleMapping.errorBubble(MurmurError.UnitUnavailable(ComputeUnit.NPU)))
    }

    @Test
    fun firstMomentsOfFinishingKeepTheHoldPillWithAFrozenTimer() {
        // review 2026-11-19: the timer jumped to 0:00 and a hold dictation showed tap visuals.
        val b = BubbleMapping.of(DictationState.Finishing(s, StartMode.HOLD, startedAtMs = 1_000, stoppedAtMs = 7_000, plan = plan, engines = EngineStatus()))
        assertEquals(DictationBubble.Listening(1_000, hold = true, warn = false, frozenAtMs = 7_000), b)
    }

    @Test
    fun listeningAndFinishingExposeHebrewAvailabilityAndActiveState() {
        val hePlan = plan.copy(hebrewAvailable = true, hebrewSpeechMeasured = false)
        val listening = BubbleMapping.of(DictationState.Listening(s, StartMode.TAP, 100, hePlan, EngineStatus(), hebrewActive = true))
        assertEquals(DictationBubble.Listening(100, hold = false, warn = false, hebrewAvailable = true, hebrewActive = true), listening)
        val holdListening = BubbleMapping.of(DictationState.Listening(s, StartMode.HOLD, 100, hePlan, EngineStatus(), hebrewActive = false))
        assertEquals(DictationBubble.Listening(100, hold = true, warn = false, hebrewAvailable = false, hebrewActive = false), holdListening)
        val finishing = BubbleMapping.of(DictationState.Finishing(s, StartMode.TAP, 100, 3_100, hePlan, EngineStatus(), hebrewActive = true))
        assertEquals(DictationBubble.Listening(100, hold = false, warn = false, frozenAtMs = 3_100, hebrewAvailable = false, hebrewActive = true), finishing)
    }

    @Test
    fun deliveringShowsThePreviousWorkingBubble() {
        assertEquals(DictationBubble.Polishing, BubbleMapping.of(DictationState.Delivering(s, 7_000, "x", cleaned = true, shownAs = DictationBubble.Polishing)))
        val listening = DictationBubble.Listening(1_000, hold = false, warn = false, frozenAtMs = 7_000)
        assertEquals(listening, BubbleMapping.of(DictationState.Delivering(s, 7_000, "x", cleaned = false, shownAs = listening)))
    }

    @Test
    fun cleanupBudgetFitsFiveMinutesInTheStockContextAndSkipsWhatCannotFit() {
        val c = DictationConfig()
        val cpu = LlmSpeed(prefillTps = 569.0, decodeTps = 83.8) // Qwen3.5 0.8B CPU, AI Hub S25 (ARCHITECTURE 6)
        // 5 min of Russian speech = about 5,000 characters. Stock Qwen3.5 default.v1: prefix 1,047 tokens, nCtx 7,168
        // (catalog 2026-11-20.2, ADR-085); the selectable default.v4 keeps 3,116 / 9,216 (ADR-045).
        for ((prefix, nCtx) in listOf(1_047 to 7_168, 3_116 to 9_216)) {
            val five = CleanupBudget.plan(5_000, prefixTokens = prefix, nCtx = nCtx, speed = cpu, config = c)
            assertTrue(five.toString(), five.fits)
            assertTrue(five.maxTokens >= five.minTokens)
        }
        // The old n_ctx 4,096 cannot hold a 60 s dictation with the default.v4 prefix (review blocker).
        assertFalse(CleanupBudget.plan(1_000, prefixTokens = 3_226, nCtx = 4_096, speed = cpu, config = c).fits)
        // Time budget: a slow unit makes a long dictation "too_long" instead of a guaranteed timeout.
        val slow = CleanupBudget.plan(5_000, 3_226, 9_216, LlmSpeed(100.0, 10.0), c)
        assertEquals("too_long", slow.skipReason)
        assertEquals(c.llmHardCapMs, CleanupBudget.plan(5_000, 3_226, 9_216, LlmSpeed(100.0, 10.0), c.copy(llmMaxExpectedMs = Long.MAX_VALUE)).timeoutMs)
    }

    @Test
    fun llmTimeoutIsAdaptiveCappedAndScaled() {
        val c = DictationConfig()
        assertEquals(4_000 + 3_000, c.llmTimeoutMs(2_000))
        assertEquals(c.llmHardCapMs, c.llmTimeoutMs(600_000))
        assertEquals(14_000, c.copy(timeoutScale = 2.0).llmTimeoutMs(2_000))
    }
}
