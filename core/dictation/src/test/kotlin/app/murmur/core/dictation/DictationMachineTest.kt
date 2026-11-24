package app.murmur.core.dictation

import app.murmur.core.engine.LoadInfo
import app.murmur.core.engine.Transcript
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.EntryId
import app.murmur.core.model.InferProcess
import app.murmur.core.model.ModelKind
import app.murmur.core.model.MurmurError
import app.murmur.core.model.SessionId
import app.murmur.core.model.Stage
import app.murmur.core.store.InsertMethod
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test


/** ARCHITECTURE.md 5.2: one test per row of the transition table (row number in the name). */
class DictationMachineTest {
    private val c = DictationConfig()
    private val s = SessionId(1)
    private val other = SessionId(9)
    private val plan = SessionPlan(cleanup = true, cleanupSkipReason = null, screenshot = true, asrUnit = ComputeUnit.CPU)
    private val rawPlan = SessionPlan(cleanup = false, cleanupSkipReason = "none", screenshot = false, asrUnit = ComputeUnit.CPU)
    private val asrInfo = LoadInfo(ComputeUnit.CPU, 5, "cpu")
    private val llmInfo = LoadInfo(ComputeUnit.CPU, 5, "llm-cpu")
    private val llmReady = LlmReadiness.Ready(llmInfo, prefixTokens = 3_129, nCtx = 9_216, speed = LlmSpeed(569.0, 83.8), fellBack = false)
    private val t = Transcript("привет это тестовая диктовка", "ru", 1, 3_000, 2_000, 10, ComputeUnit.CPU)
    private val entry = EntryId(7)

    private fun reduce(st: DictationState, e: DictationEvent, config: DictationConfig = c) = DictationMachine.reduce(st, e, config)
    private fun tap(at: Long, session: SessionId? = null) = DictationEvent.Input(UserInput.Tap, at, session)
    private fun input(i: UserInput, at: Long = 1_000) = DictationEvent.Input(i, at)

    private val readyEngines = EngineStatus(asr = AsrReadiness.Ready(asrInfo, false), llm = LlmReadiness.Loading, context = ContextReadiness.Pending)
    private fun arming(mode: StartMode = StartMode.TAP) = DictationState.Arming(s, mode, pressedAtMs = 0)
    private fun listening(mode: StartMode = StartMode.TAP, engines: EngineStatus = readyEngines, p: SessionPlan = plan) =
        DictationState.Listening(s, mode, startedAtMs = 100, plan = p, engines = engines)
    private fun finishing(engines: EngineStatus = readyEngines, p: SessionPlan = plan) =
        DictationState.Finishing(s, StartMode.TAP, startedAtMs = 100, stoppedAtMs = 3_100, plan = p, engines = engines)
    private fun polishing(engines: EngineStatus = readyEngines, started: Boolean = false) =
        DictationState.Polishing(s, stoppedAtMs = 3_100, transcript = t, plan = plan, engines = engines, cleanupStarted = started)
    private fun delivering(cleaned: Boolean = true, outcome: String = "accept") =
        DictationState.Delivering(s, 3_100, "Привет, это тестовая диктовка.", cleaned, DictationBubble.Polishing, outcome)

    private val row14Effects = listOf(
        Effect.StopCapture(s), Effect.FinishAsr(s),
        Effect.StartTimer(s, Timer.TRANSCRIBING_LABEL, 300), Effect.StartTimer(s, Timer.ASR_FINAL, c.asrFinalTimeoutMs(3_000)),
        Effect.CancelTimer(Timer.WARN), Effect.CancelTimer(Timer.MAX_DURATION), Effect.Haptic(HapticKind.CONFIRM),
    )

    // ------------------------------------------------------------------ rows 1-4

    @Test fun row01_idleTapArmsWithPreflightCaptureMicTimerAndHaptic() {
        val r = reduce(DictationState.Idle, tap(0, s))
        assertEquals(DictationState.Arming(s, StartMode.TAP, 0), r.state)
        assertEquals(listOf(Effect.Preflight(s, StartMode.TAP), Effect.StartCapture(s), Effect.StartTimer(s, Timer.MIC_START, 2_000),
            Effect.Haptic(HapticKind.CONFIRM)), r.effects)
    }

    @Test fun row02_idleHoldStartArmsHoldOnlyWithHoldToTalkOn() {
        val r = reduce(DictationState.Idle, DictationEvent.Input(UserInput.HoldStart, 0, s))
        assertEquals(DictationState.Arming(s, StartMode.HOLD, 0), r.state)
        assertTrue(Effect.Haptic(HapticKind.CONFIRM) in r.effects)
        assertEquals(Transition(DictationState.Idle), reduce(DictationState.Idle, DictationEvent.Input(UserInput.HoldStart, 0, s), c.copy(holdToTalk = false)))
    }

    @Test fun row03_notificationStartArmsWithoutHaptic() {
        val r = reduce(DictationState.Idle, DictationEvent.Input(UserInput.NotificationStart, 0, s))
        assertEquals(DictationState.Arming(s, StartMode.NOTIFICATION, 0), r.state)
        assertEquals(listOf(Effect.Preflight(s, StartMode.NOTIFICATION), Effect.StartCapture(s), Effect.StartTimer(s, Timer.MIC_START, 2_000)), r.effects)
    }

    @Test fun row04_startInAppArmsInApp() {
        val r = reduce(DictationState.Idle, DictationEvent.Input(UserInput.InAppStart, 0, s))
        assertEquals(DictationState.Arming(s, StartMode.IN_APP, 0), r.state)
        assertTrue(Effect.Preflight(s, StartMode.IN_APP) in r.effects)
        // Without a stamped session the reducer still numbers it (previous + 1).
        assertEquals(SessionId(1), reduce(DictationState.Idle, input(UserInput.InAppStart)).state.session)
    }

    // ------------------------------------------------------------------ rows 5-13

    @Test fun row05_preflightPassedPreparesEnginesAndContext() {
        val r = reduce(arming(), DictationEvent.PreflightPassed(s, plan))
        val st = r.state as DictationState.Arming
        assertEquals(plan, st.plan)
        assertEquals(LlmReadiness.Loading, st.engines.llm)
        assertEquals(ContextReadiness.Pending, st.engines.context)
        assertEquals(listOf(Effect.PrepareEngines(s, plan), Effect.CaptureContext(s, 600)), r.effects)
        val raw = reduce(arming(), DictationEvent.PreflightPassed(s, rawPlan))
        val rawArming = raw.state as DictationState.Arming
        assertEquals(LlmReadiness.NotRequested("none"), rawArming.engines.llm)
        assertEquals(ContextReadiness.NotRequested, rawArming.engines.context)
        assertEquals(listOf(Effect.PrepareEngines(s, rawPlan)), raw.effects)
    }

    @Test fun row06_preflightFailedEndsWithoutHistory() {
        val r = reduce(arming(), DictationEvent.PreflightFailed(s, MurmurError.NoAsrModel))
        assertEquals(DictationState.Done(s, DictationOutcome.Failed(MurmurError.NoAsrModel, null)), r.state)
        assertEquals(listOf(Effect.DiscardCapture(s), Effect.CancelTimer(Timer.MIC_START), Effect.CancelEngines(s),
            Effect.Haptic(HapticKind.REJECT), Effect.StartTimer(s, Timer.RESULT_DISMISS, 6_000)), r.effects)
        assertTrue(r.effects.none { it is Effect.RecordFailure })
    }

    @Test fun row07_micStartedRecordsTheFirstBuffer() {
        val r = reduce(arming(), DictationEvent.MicStarted(s, 80))
        assertEquals(arming().copy(micStartedAtMs = 80), r.state)
        assertEquals(listOf(Effect.CancelTimer(Timer.MIC_START)), r.effects)
    }

    @Test fun row08_armingCompletesToListeningInEitherOrder() {
        val warnMax = listOf(Effect.StartTimer(s, Timer.WARN, 270_000), Effect.StartTimer(s, Timer.MAX_DURATION, 300_000))
        val a = reduce(arming().copy(plan = plan, engines = readyEngines), DictationEvent.MicStarted(s, 80))
        assertEquals(DictationState.Listening(s, StartMode.TAP, 80, plan, readyEngines), a.state)
        assertEquals(listOf(Effect.CancelTimer(Timer.MIC_START)) + warnMax, a.effects)
        val b = reduce(arming().copy(micStartedAtMs = 80), DictationEvent.PreflightPassed(s, plan))
        assertTrue(b.state is DictationState.Listening)
        assertEquals((b.state as DictationState.Listening).startedAtMs, 80)
        assertEquals(listOf(Effect.PrepareEngines(s, plan), Effect.CaptureContext(s, 600)) + warnMax, b.effects)
    }

    @Test fun row09_completionWithStopRequestedGoesStraightToFinishing() {
        val st = arming().copy(plan = plan, engines = readyEngines, stopRequested = true, stopAtMs = 400)
        val r = reduce(st, DictationEvent.MicStarted(s, 500))
        val f = r.state as DictationState.Finishing
        assertEquals(500, f.startedAtMs)
        assertEquals(500, f.stoppedAtMs)   // the stop came before the first buffer
        assertTrue(r.effects.containsAll(listOf(Effect.StopCapture(s), Effect.FinishAsr(s), Effect.Haptic(HapticKind.CONFIRM))))
        assertTrue(r.effects.none { it is Effect.StartTimer && it.timer == Timer.WARN })
    }

    @Test fun row10_stopDuringArmingIsRememberedAfter300msAndIgnoredBefore() {
        assertEquals(Transition(arming()), reduce(arming(), tap(200)))
        assertEquals(Transition(arming().copy(stopRequested = true, stopAtMs = 400)), reduce(arming(), tap(400)))
        val hold = arming(StartMode.HOLD)
        assertEquals(Transition(hold.copy(stopRequested = true, stopAtMs = 450)), reduce(hold, input(UserInput.HoldRelease, 450)))
        assertEquals(Transition(hold), reduce(hold, input(UserInput.HoldRelease, 100)))
    }

    @Test fun row11_micFailureOrMicTimeoutFailsWithoutHistory() {
        val busy = reduce(arming(), DictationEvent.MicFailed(s, MurmurError.MicBusy))
        assertEquals(DictationState.Done(s, DictationOutcome.Failed(MurmurError.MicBusy, null)), busy.state)
        assertTrue(busy.effects.containsAll(listOf(Effect.CancelEngines(s), Effect.DiscardCapture(s), Effect.Haptic(HapticKind.REJECT))))
        val timeout = reduce(arming(), DictationEvent.TimerFired(s, Timer.MIC_START))
        assertEquals(DictationState.Done(s, DictationOutcome.Failed(MurmurError.MicStartFailed("timeout"), null)), timeout.state)
        assertTrue((busy.effects + timeout.effects).none { it is Effect.RecordFailure })
    }

    @Test fun row12_cancelOrSlideOffDiscardsAndKeepsProcessesWarm() {
        val a = reduce(arming(), input(UserInput.Cancel))
        assertEquals(DictationState.Idle, a.state)
        assertTrue(a.effects.containsAll(listOf(Effect.DiscardCapture(s), Effect.CancelEngines(s), Effect.ScheduleIdleUnload(60_000))))
        val l = reduce(listening(StartMode.HOLD), input(UserInput.HoldSlideOff))
        assertEquals(DictationState.Idle, l.state)
        assertTrue(l.effects.containsAll(listOf(Effect.DiscardCapture(s), Effect.CancelEngines(s), Effect.CancelTimer(Timer.WARN), Effect.CancelTimer(Timer.MAX_DURATION))))
        assertTrue((a.effects + l.effects).none { it is Effect.RecordFailure || it is Effect.Deliver })
    }

    @Test fun row13_externalStopDuringArmingFailsWithTheMappedError() {
        fun err(r: StopReason) = ((reduce(arming(), DictationEvent.ExternalStop(s, r)).state as DictationState.Done).outcome as DictationOutcome.Failed).error
        assertEquals(MurmurError.MicBusy, err(StopReason.CALL))
        assertEquals(MurmurError.AccessibilityOff, err(StopReason.A11Y_OFF))
        assertTrue(err(StopReason.MIC_LOST) is MurmurError.MicStartFailed)
        assertTrue(err(StopReason.SCREEN_OFF) is MurmurError.MicStartFailed)
        val r = reduce(arming(), DictationEvent.ExternalStop(s, StopReason.CALL))
        assertTrue(r.effects.containsAll(listOf(Effect.DiscardCapture(s), Effect.CancelEngines(s))))
    }

    // ------------------------------------------------------------------ rows 14-24

    @Test fun row14_tapFinishesListeningOrFailsWhenTheAsrFailed() {
        val r = reduce(listening(), tap(3_100))
        assertEquals(DictationState.Finishing(s, StartMode.TAP, 100, 3_100, plan, readyEngines), r.state)
        assertEquals(row14Effects, r.effects)
        val failed = readyEngines.copy(asr = AsrReadiness.Failed(MurmurError.UnitUnavailable(ComputeUnit.NPU)))
        val f = reduce(listening(engines = failed), tap(3_100))
        assertEquals(DictationState.Done(s, DictationOutcome.Failed(MurmurError.UnitUnavailable(ComputeUnit.NPU), null), recording = true), f.state)
        assertTrue(f.effects.containsAll(listOf(Effect.StopCapture(s), Effect.RecordFailure(s, MurmurError.UnitUnavailable(ComputeUnit.NPU), keepAudio = true))))
    }

    @Test fun row15_holdReleaseFinishesAHoldDictationOnly() {
        val r = reduce(listening(StartMode.HOLD), input(UserInput.HoldRelease, 3_100))
        assertEquals(DictationState.Finishing(s, StartMode.HOLD, 100, 3_100, plan, readyEngines), r.state)
        assertEquals(Transition(listening()), reduce(listening(), input(UserInput.HoldRelease, 3_100)))
    }

    @Test fun row16_warnTimerTurnsTheTimerWarn() {
        assertEquals(Transition(listening().copy(warn = true)), reduce(listening(), DictationEvent.TimerFired(s, Timer.WARN)))
    }

    @Test fun row17_maxTimerAutoStopsAtFiveMinutes() {
        val r = reduce(listening(), DictationEvent.TimerFired(s, Timer.MAX_DURATION))
        assertEquals(100 + 300_000L, (r.state as DictationState.Finishing).stoppedAtMs)
        assertTrue(r.effects.containsAll(listOf(Effect.StopCapture(s), Effect.FinishAsr(s))))
    }

    @Test fun row18_callOrMicLossFinishesWithTheAudioSoFar() {
        for (reason in listOf(StopReason.CALL, StopReason.MIC_LOST, StopReason.MIC_SILENCED, StopReason.SCREEN_OFF)) {
            val r = reduce(listening(), DictationEvent.ExternalStop(s, reason, atMs = 1_200))
            assertEquals(1_200L, (r.state as DictationState.Finishing).stoppedAtMs)
            assertTrue(r.effects.containsAll(listOf(Effect.StopCapture(s), Effect.FinishAsr(s))))
        }
    }

    @Test fun row19_serviceLostWhileListeningFailsAndKeepsTheAudio() {
        val r = reduce(listening(), DictationEvent.ExternalStop(s, StopReason.A11Y_OFF, 1_200))
        assertEquals(DictationState.Done(s, DictationOutcome.Failed(MurmurError.AccessibilityOff, null), recording = true), r.state)
        assertTrue(r.effects.containsAll(listOf(Effect.StopCapture(s), Effect.CancelEngines(s), Effect.RecordFailure(s, MurmurError.AccessibilityOff, true))))
        assertTrue(r.effects.none { it is Effect.Haptic })   // no overlay without the service
    }

    @Test fun row20_asrLoadFailedKeepsRecording() {
        val e = MurmurError.UnitUnavailable(ComputeUnit.NPU)
        val r = reduce(listening(), DictationEvent.AsrLoadFailed(s, e))
        assertEquals(listening(engines = readyEngines.copy(asr = AsrReadiness.Failed(e))), r.state)
        assertTrue(r.effects.isEmpty())
    }

    @Test fun row21_firstAsrDeathWhileListeningRestartsAndReplays() {
        val r = reduce(listening(), DictationEvent.ProcessDied(s, InferProcess.ASR))
        assertEquals(listening(engines = readyEngines.copy(asr = AsrReadiness.Loading, asrCrashes = 1)), r.state)
        assertEquals(listOf(Effect.RestartAsr(s, thenFinish = false)), r.effects)
    }

    @Test fun row22_secondAsrDeathWhileListeningMarksTheAsrFailed() {
        val r = reduce(listening(engines = readyEngines.copy(asrCrashes = 1)), DictationEvent.ProcessDied(s, InferProcess.ASR))
        assertEquals(AsrReadiness.Failed(MurmurError.ProcessDied(InferProcess.ASR)), (r.state as DictationState.Listening).engines.asr)
        assertTrue(r.effects.isEmpty())
    }

    @Test fun row23_firstLlmDeathBeforeCleanupReloads() {
        for (st in listOf<DictationState>(arming().copy(plan = plan, engines = readyEngines), listening(), finishing())) {
            val r = reduce(st, DictationEvent.ProcessDied(s, InferProcess.LLM))
            assertEquals(listOf(Effect.ReloadLlm(s)), r.effects)
        }
        val l = reduce(listening(engines = readyEngines.copy(llm = llmReady)), DictationEvent.ProcessDied(s, InferProcess.LLM)).state as DictationState.Listening
        assertEquals(LlmReadiness.Loading, l.engines.llm)
        assertEquals(1, l.engines.llmCrashes)
        // No cleanup planned: nothing to reload.
        assertEquals(Transition(listening(p = rawPlan)), reduce(listening(p = rawPlan), DictationEvent.ProcessDied(s, InferProcess.LLM)))
    }

    @Test fun row24_secondLlmDeathMarksTheLlmFailed() {
        val r = reduce(finishing(readyEngines.copy(llmCrashes = 1)), DictationEvent.ProcessDied(s, InferProcess.LLM))
        assertEquals(LlmReadiness.Failed(MurmurError.ProcessDied(InferProcess.LLM)), (r.state as DictationState.Finishing).engines.llm)
        assertTrue(r.effects.isEmpty())
    }

    // ------------------------------------------------------------------ rows 25-30

    @Test fun row25_transcribingLabelAfter300ms() {
        assertEquals(Transition(finishing().copy(showTranscribing = true)), reduce(finishing(), DictationEvent.TimerFired(s, Timer.TRANSCRIBING_LABEL)))
    }

    @Test fun row26_shortBlankOrPrecheckedTranscriptIsNoSpeechWithAudio() {
        val cases = listOf(
            DictationEvent.TranscriptReady(s, t.copy(speechMs = 150), false),
            DictationEvent.TranscriptReady(s, t.copy(text = "  "), false),
            DictationEvent.TranscriptReady(s, t, precheckSkip = true),
        )
        for (e in cases) {
            val r = reduce(finishing(), e)
            assertEquals(DictationState.Done(s, DictationOutcome.Failed(MurmurError.NoSpeech, null), recording = true), r.state)
            assertTrue(r.effects.containsAll(listOf(Effect.CancelTimer(Timer.ASR_FINAL), Effect.RecordFailure(s, MurmurError.NoSpeech, keepAudio = true))))
        }
        // A short but real spoken reply (>= 250 ms) is accepted and not discarded as NoSpeech.
        assertTrue(reduce(finishing(), DictationEvent.TranscriptReady(s, t.copy(speechMs = 300), false)).state is DictationState.Polishing)
        // A VAD-fed (batch) model measures speech: 0 ms is no speech, as the table says, whatever text it returned.
        assertEquals(DictationState.Done(s, DictationOutcome.Failed(MurmurError.NoSpeech, null), recording = true),
            reduce(finishing(), DictationEvent.TranscriptReady(s, t.copy(speechMs = 0), false)).state)
        // A streaming model has no VAD (Transcript.speechMs = 0, "not measured"): the text and the precheck decide.
        val streaming = finishing().copy(plan = plan.copy(speechMeasured = false))
        assertTrue(reduce(streaming, DictationEvent.TranscriptReady(s, t.copy(speechMs = 0), false)).state is DictationState.Polishing)
        assertTrue(reduce(streaming, DictationEvent.TranscriptReady(s, t.copy(speechMs = 0, text = " "), false)).state is DictationState.Done)
    }

    @Test fun row27_transcriptWithoutUsableLlmIsDeliveredRawWithTheReason() {
        fun outcome(llm: LlmReadiness): String {
            val r = reduce(finishing(readyEngines.copy(llm = llm)), DictationEvent.TranscriptReady(s, t, false))
            val d = r.state as DictationState.Delivering
            assertEquals(t.text, d.text)
            assertFalse(d.cleaned)
            assertEquals(DictationBubble.Listening(100, false, false, frozenAtMs = 3_100), d.shownAs)
            return (r.effects.single { it is Effect.Deliver } as Effect.Deliver).cleanupOutcome
        }
        assertEquals("skipped:none", outcome(LlmReadiness.NotRequested("none")))
        assertEquals("skipped:password", outcome(LlmReadiness.NotRequested("password")))
        assertEquals("skipped:low_memory", outcome(LlmReadiness.Failed(MurmurError.OutOfMemory(ModelKind.LLM))))
        assertEquals("skipped:llm_died", outcome(LlmReadiness.Failed(MurmurError.ProcessDied(InferProcess.LLM))))
        assertEquals("skipped:llm_load", outcome(LlmReadiness.Failed(MurmurError.EngineLoadFailed(ModelKind.LLM, null))))
        val r = reduce(finishing(readyEngines.copy(llm = LlmReadiness.NotRequested("none"))), DictationEvent.TranscriptReady(s, t, false))
        assertTrue(r.effects.containsAll(listOf(Effect.CancelTimer(Timer.ASR_FINAL), Effect.StartTimer(s, Timer.DELIVERY, 2_000))))
    }

    @Test fun row28_transcriptWithLoadingLlmWaitsForContextAndReadiness() {
        val r = reduce(finishing(), DictationEvent.TranscriptReady(s, t, false))
        assertEquals(DictationState.Polishing(s, 3_100, t, plan, readyEngines), r.state)
        assertEquals(listOf(Effect.CancelTimer(Timer.ASR_FINAL), Effect.CancelTimer(Timer.TRANSCRIBING_LABEL),
            Effect.StartTimer(s, Timer.CONTEXT_WAIT, 800), Effect.StartTimer(s, Timer.LLM_READY, 10_000)), r.effects)
    }

    @Test fun row29_firstAsrDeathOrTimeoutInFinishingRestartsAndFinishes() {
        val events = listOf(DictationEvent.AsrFailed(s, MurmurError.ProcessDied(InferProcess.ASR)), DictationEvent.AsrFailed(s, MurmurError.Timeout(Stage.ASR_FINAL)),
            DictationEvent.TimerFired(s, Timer.ASR_FINAL), DictationEvent.ProcessDied(s, InferProcess.ASR))
        for (e in events) {
            val r = reduce(finishing(), e)
            assertEquals(1, (r.state as DictationState.Finishing).engines.asrCrashes)
            assertEquals(listOf(Effect.RestartAsr(s, thenFinish = true), Effect.StartTimer(s, Timer.ASR_FINAL, c.asrFinalTimeoutMs(3_000))), r.effects)
        }
    }

    @Test fun row30_secondAsrFailureOrAnyOtherAsrErrorFailsWithAudio() {
        val twice = reduce(finishing(readyEngines.copy(asrCrashes = 1)), DictationEvent.ProcessDied(s, InferProcess.ASR))
        assertEquals(DictationState.Done(s, DictationOutcome.Failed(MurmurError.ProcessDied(InferProcess.ASR), null), recording = true), twice.state)
        assertTrue(twice.effects.containsAll(listOf(Effect.RecordFailure(s, MurmurError.ProcessDied(InferProcess.ASR), true), Effect.CancelEngines(s))))
        val other = reduce(finishing(), DictationEvent.AsrFailed(s, MurmurError.AsrFailed("decode")))
        assertEquals(DictationState.Done(s, DictationOutcome.Failed(MurmurError.AsrFailed("decode"), null), recording = true), other.state)
        val timeout = reduce(finishing(readyEngines.copy(asrCrashes = 1)), DictationEvent.TimerFired(s, Timer.ASR_FINAL))
        assertEquals(MurmurError.Timeout(Stage.ASR_FINAL), ((timeout.state as DictationState.Done).outcome as DictationOutcome.Failed).error)
    }

    // ------------------------------------------------------------------ rows 31-38

    @Test fun row31_startRuleRunsCleanupWithTheBudgetTimerOrSkipsWhatDoesNotFit() {
        val ready = readyEngines.copy(llm = llmReady, context = ContextReadiness.Ready(true, 12))
        val r = reduce(finishing(ready), DictationEvent.TranscriptReady(s, t, false))
        assertEquals(polishing(ready, started = true), r.state)
        val budget = CleanupBudget.plan(t.text.length, 3_129, 9_216, LlmSpeed(569.0, 83.8), c)
        assertEquals(listOf(Effect.CancelTimer(Timer.ASR_FINAL), Effect.CancelTimer(Timer.TRANSCRIBING_LABEL),
            Effect.RunCleanup(s, t, budget), Effect.StartTimer(s, Timer.LLM, budget.timeoutMs),
            Effect.CancelTimer(Timer.CONTEXT_WAIT), Effect.CancelTimer(Timer.LLM_READY)), r.effects)
        // No benchmark speed: the unit default is used; a 5,000-char transcript on a slow unit does not fit.
        val slow = ready.copy(llm = llmReady.copy(speed = LlmSpeed(100.0, 10.0)))
        val long = t.copy(text = "слово ".repeat(850))
        val skip = reduce(finishing(slow), DictationEvent.TranscriptReady(s, long, false))
        assertTrue(skip.state is DictationState.Delivering)
        assertEquals("skipped:too_long", (skip.effects.single { it is Effect.Deliver } as Effect.Deliver).cleanupOutcome)
        assertTrue(skip.effects.none { it is Effect.RunCleanup })
    }

    @Test fun row32_contextWaitOverRendersWithoutTheScreen() {
        val st = polishing(readyEngines.copy(llm = llmReady))
        val r = reduce(st, DictationEvent.TimerFired(s, Timer.CONTEXT_WAIT))
        val p = r.state as DictationState.Polishing
        assertTrue(p.contextWaitOver)
        assertTrue(p.cleanupStarted)
        assertTrue(r.effects.any { it is Effect.RunCleanup })
        // LLM still loading: only the flag changes.
        val w = reduce(polishing(), DictationEvent.TimerFired(s, Timer.CONTEXT_WAIT))
        assertEquals(polishing().copy(contextWaitOver = true), w.state)
        assertTrue(w.effects.isEmpty())
    }

    @Test fun row33_contextReadyOrLlmReadyAppliesTheStartRule() {
        val ctx = reduce(polishing(readyEngines.copy(llm = llmReady)), DictationEvent.ContextReady(s, true, 5))
        assertTrue((ctx.state as DictationState.Polishing).cleanupStarted)
        val llm = reduce(polishing(readyEngines.copy(context = ContextReadiness.Ready(false, 0))), DictationEvent.LlmReady(s, llmInfo, 3_129, 9_216, null, false))
        assertTrue((llm.state as DictationState.Polishing).cleanupStarted)
        val waiting = reduce(polishing(), DictationEvent.LlmReady(s, llmInfo, 3_129, 9_216, null, false))
        assertFalse((waiting.state as DictationState.Polishing).cleanupStarted)   // context still pending, CONTEXT_WAIT not over
    }

    @Test fun row34_llmNotReadyInTimeInsertsRaw() {
        val r = reduce(polishing(), DictationEvent.TimerFired(s, Timer.LLM_READY))
        assertEquals("skipped:llm_not_ready", (r.effects.single { it is Effect.Deliver } as Effect.Deliver).cleanupOutcome)
        assertTrue(r.state is DictationState.Delivering)
        assertTrue(r.effects.none { it is Effect.CancelEngines })   // the load continues: next dictation is warm
    }

    @Test fun row35_llmLoadFailureOrDeathBeforeCleanupInsertsRaw() {
        val load = reduce(polishing(), DictationEvent.LlmLoadFailed(s, MurmurError.EngineLoadFailed(ModelKind.LLM, null)))
        assertEquals("skipped:llm_load", (load.effects.single { it is Effect.Deliver } as Effect.Deliver).cleanupOutcome)
        val oom = reduce(polishing(), DictationEvent.LlmLoadFailed(s, MurmurError.OutOfMemory(ModelKind.LLM)))
        assertEquals("skipped:low_memory", (oom.effects.single { it is Effect.Deliver } as Effect.Deliver).cleanupOutcome)
        val died = reduce(polishing(), DictationEvent.ProcessDied(s, InferProcess.LLM))
        assertEquals("skipped:llm_died", (died.effects.single { it is Effect.Deliver } as Effect.Deliver).cleanupOutcome)
    }

    @Test fun row36_cleanupDoneDeliversTheCleanedText() {
        val r = reduce(polishing(started = true), DictationEvent.CleanupDone(s, "Привет.", cleaned = true, outcome = "accept"))
        assertEquals(DictationState.Delivering(s, 3_100, "Привет.", true, DictationBubble.Polishing, "accept"), r.state)
        assertEquals(listOf(Effect.CancelTimer(Timer.LLM), Effect.Deliver(s, "Привет.", t.text, true, "accept"),
            Effect.StartTimer(s, Timer.DELIVERY, 2_000)), r.effects)
    }

    @Test fun row37_llmTimeoutDeliversRaw() {
        val r = reduce(polishing(started = true), DictationEvent.TimerFired(s, Timer.LLM))
        assertEquals(DictationState.Delivering(s, 3_100, t.text, false, DictationBubble.Polishing, "skipped:timeout"), r.state)
        assertTrue(Effect.Deliver(s, t.text, t.text, false, "skipped:timeout") in r.effects)
    }

    @Test fun row38_llmDeathDuringCleanupDeliversRaw() {
        val r = reduce(polishing(started = true), DictationEvent.ProcessDied(s, InferProcess.LLM))
        assertTrue(Effect.Deliver(s, t.text, t.text, false, "skipped:llm_died") in r.effects)
        assertTrue(Effect.CancelTimer(Timer.LLM) in r.effects)
    }

    // ------------------------------------------------------------------ rows 39-45

    @Test fun row39_insertedShowsInsertedAndSchedulesTheIdleUnload() {
        val r = reduce(delivering(), DictationEvent.Delivered(s, DeliveryResult.Inserted(entry, InsertMethod.COMMIT_TEXT)))
        assertEquals(DictationState.Done(s, DictationOutcome.Inserted(entry, noCleanup = false)), r.state)
        assertEquals(listOf(Effect.CancelTimer(Timer.DELIVERY), Effect.Haptic(HapticKind.CLOCK_TICK), Effect.StartTimer(s, Timer.RESULT_DISMISS, 1_200),
            Effect.ScheduleIdleUnload(60_000)), r.effects)
        val raw = reduce(delivering(cleaned = false, outcome = "skipped:timeout"), DictationEvent.Delivered(s, DeliveryResult.Inserted(entry, InsertMethod.PASTE)))
        assertEquals(DictationState.Done(s, DictationOutcome.Inserted(entry, noCleanup = true, skipReason = "skipped:timeout")), raw.state)
        val none = reduce(delivering(cleaned = false, outcome = "skipped:none"), DictationEvent.Delivered(s, DeliveryResult.Inserted(entry, InsertMethod.PASTE)))
        assertEquals(DictationState.Done(s, DictationOutcome.Inserted(entry, noCleanup = false, skipReason = "skipped:none")), none.state)
    }

    @Test fun row40_copiedOnlyShowsCopiedFor4s() {
        val r = reduce(delivering(), DictationEvent.Delivered(s, DeliveryResult.CopiedOnly(entry, MurmurError.TargetChanged)))
        assertEquals(DictationState.Done(s, DictationOutcome.Copied(entry)), r.state)
        assertTrue(r.effects.containsAll(listOf(Effect.StartTimer(s, Timer.RESULT_DISMISS, 4_000), Effect.ScheduleIdleUnload(60_000))))
    }

    @Test fun row41_passwordAtDeliveryGoesIdleWithNothingKept() {
        val r = reduce(delivering(), DictationEvent.Delivered(s, DeliveryResult.DiscardedPassword))
        assertEquals(DictationState.Idle, r.state)
        assertEquals(listOf(Effect.CancelTimer(Timer.DELIVERY), Effect.ScheduleIdleUnload(60_000)), r.effects)
    }

    @Test fun row42_nothingToInsertIsNoSpeechWithTheEntry() {
        val r = reduce(delivering(), DictationEvent.Delivered(s, DeliveryResult.NothingToInsert(entry)))
        assertEquals(DictationState.Done(s, DictationOutcome.Failed(MurmurError.NoSpeech, entry)), r.state)
        assertTrue(r.effects.containsAll(listOf(Effect.StartTimer(s, Timer.RESULT_DISMISS, 6_000), Effect.ScheduleIdleUnload(60_000))))
    }

    @Test fun row43_notDeliveredKeepsTheEntryAndShowsTheError() {
        val r = reduce(delivering(), DictationEvent.Delivered(s, DeliveryResult.NotDelivered(entry, MurmurError.AccessibilityOff)))
        assertEquals(DictationState.Done(s, DictationOutcome.Failed(MurmurError.AccessibilityOff, entry)), r.state)
        assertTrue(r.effects.containsAll(listOf(Effect.StartTimer(s, Timer.RESULT_DISMISS, 6_000), Effect.ScheduleIdleUnload(60_000))))
        assertTrue(r.effects.none { it is Effect.Haptic })
    }

    @Test fun row44_deliveryTimeoutShowsCopied() {
        val r = reduce(delivering(), DictationEvent.TimerFired(s, Timer.DELIVERY))
        assertEquals(DictationState.Done(s, DictationOutcome.Copied(null)), r.state)
        assertTrue(Effect.StartTimer(s, Timer.RESULT_DISMISS, 4_000) in r.effects)
        // The late delivery result still gives the pill its entry.
        val late = reduce(r.state, DictationEvent.Delivered(s, DeliveryResult.Inserted(entry, InsertMethod.COMMIT_TEXT)))
        assertEquals(DictationState.Done(s, DictationOutcome.Copied(entry)), late.state)
    }

    @Test fun row45_resultDismissTimerReturnsToIdle() {
        assertEquals(Transition(DictationState.Idle), reduce(DictationState.Done(s, DictationOutcome.Inserted(entry, false)), DictationEvent.TimerFired(s, Timer.RESULT_DISMISS)))
    }

    // ------------------------------------------------------------------ rows 46-54

    @Test fun row46_tapOrHoldOnAResultPillStartsANewSession() {
        val done = DictationState.Done(s, DictationOutcome.Inserted(entry, false))
        val r = reduce(done, tap(5_000, SessionId(2)))
        assertEquals(DictationState.Arming(SessionId(2), StartMode.TAP, 5_000), r.state)
        assertTrue(Effect.Preflight(SessionId(2), StartMode.TAP) in r.effects)
        assertEquals(DictationState.Arming(SessionId(2), StartMode.HOLD, 5_000), reduce(done, DictationEvent.Input(UserInput.HoldStart, 5_000)).state)
    }

    @Test fun row47_copiedPillTapOpensTheEntry() {
        assertEquals(Transition(DictationState.Idle, listOf(Effect.Open(OpenTarget.Entry(entry)))), reduce(DictationState.Done(s, DictationOutcome.Copied(entry)), input(UserInput.ResultTap)))
        assertEquals(Transition(DictationState.Idle), reduce(DictationState.Done(s, DictationOutcome.Copied(null)), input(UserInput.ResultTap)))   // retention 0
    }

    @Test fun row48_errorPillTapDismisses() {
        assertEquals(Transition(DictationState.Idle), reduce(DictationState.Done(s, DictationOutcome.Failed(MurmurError.MicBusy, null)), input(UserInput.ResultTap)))
    }

    @Test fun row49_retryStartsANewDictationForNoSpeechElseRetranscribesTheEntry() {
        val action = DictationEvent.Input(UserInput.ResultAction, 9_000, SessionId(2))
        val noSpeech = reduce(DictationState.Done(s, DictationOutcome.Failed(MurmurError.NoSpeech, entry, audioKept = true)), action)
        assertEquals(DictationState.Arming(SessionId(2), StartMode.TAP, 9_000), noSpeech.state)
        // A load failure with the recording kept: the stored audio runs again from Finishing as session 2.
        val loadError = MurmurError.EngineLoadFailed(ModelKind.ASR, null)
        val load = reduce(DictationState.Done(s, DictationOutcome.Failed(loadError, entry, audioKept = true)), action)
        assertEquals(DictationState.Arming(SessionId(2), StartMode.TAP, 9_000, stopRequested = true, retryOf = entry), load.state)
        assertEquals(listOf(Effect.Preflight(SessionId(2), StartMode.TAP), Effect.StartCapture(SessionId(2)), Effect.StartTimer(SessionId(2), Timer.MIC_START, 10_000)), load.effects)
        // Tapped before RecordFailure returned: remembered, then run with the entry.
        val pending = reduce(DictationState.Done(s, DictationOutcome.Failed(loadError, null), recording = true), action)
        assertEquals(Transition(DictationState.Done(s, DictationOutcome.Failed(loadError, null), recording = true, retryAtMs = 9_000)), pending)
        val ran = reduce(pending.state, DictationEvent.FailureRecorded(s, entry, audioKept = true))
        assertEquals(DictationState.Arming(SessionId(2), StartMode.TAP, 9_000, stopRequested = true, retryOf = entry), ran.state)
        // Nothing kept to re-run (recordings off, retention 0): Retry is a new dictation that retries the load.
        val nothing = reduce(pending.state, DictationEvent.FailureRecorded(s, entry, audioKept = false))
        assertEquals(DictationState.Arming(SessionId(2), StartMode.TAP, 9_000), nothing.state)
        assertEquals(DictationState.Arming(SessionId(2), StartMode.TAP, 9_000), reduce(pending.state, DictationEvent.FailureRecorded(s, null)).state)
    }

    @Test fun row50_openActionsOpenModelsOrTheEntry() {
        assertEquals(Transition(DictationState.Idle, listOf(Effect.Open(OpenTarget.Models(ModelKind.ASR)))),
            reduce(DictationState.Done(s, DictationOutcome.Failed(MurmurError.NoAsrModel, null)), input(UserInput.ResultAction)))
        assertEquals(Transition(DictationState.Idle, listOf(Effect.Open(OpenTarget.Entry(entry)))),
            reduce(DictationState.Done(s, DictationOutcome.Failed(MurmurError.ProcessDied(InferProcess.ASR), entry)), input(UserInput.ResultAction)))
    }

    @Test fun row51_inputsAreIgnoredAfterTheStop() {
        for (st in listOf(finishing(), polishing(), delivering())) {
            for (i in listOf(UserInput.Tap, UserInput.Cancel, UserInput.HoldRelease, UserInput.HoldSlideOff, UserInput.HoldStart)) {
                assertEquals(Transition(st), reduce(st, input(i)))
            }
        }
    }

    @Test fun row52_externalStopsAreIgnoredAfterTheStop() {
        for (st in listOf(finishing(), polishing(), delivering())) {
            for (r in StopReason.entries) assertEquals(Transition(st), reduce(st, DictationEvent.ExternalStop(s, r, 5_000)))
        }
    }

    @Test fun row53_idleAndDoneDropDeathsReadinessAndTimers() {
        val done = DictationState.Done(s, DictationOutcome.Inserted(entry, false))
        val events = listOf(DictationEvent.ProcessDied(s, InferProcess.ASR), DictationEvent.ProcessDied(s, InferProcess.LLM), DictationEvent.AsrReady(s, asrInfo, false),
            DictationEvent.LlmReady(s, llmInfo, 1, 2, null, false), DictationEvent.ContextReady(s, true, 3), DictationEvent.TimerFired(s, Timer.LLM), DictationEvent.TimerFired(s, Timer.WARN))
        for (e in events) {
            assertEquals(Transition(DictationState.Idle), reduce(DictationState.Idle, e))
            assertEquals(Transition(done), reduce(done, e))
        }
    }

    @Test fun row54_eventsOfAnotherSessionAreDropped() {
        val events = listOf(DictationEvent.PreflightPassed(other, plan), DictationEvent.MicStarted(other, 1), DictationEvent.TranscriptReady(other, t, false),
            DictationEvent.CleanupDone(other, "x", true, "accept"), DictationEvent.Delivered(other, DeliveryResult.DiscardedPassword), DictationEvent.TimerFired(other, Timer.MAX_DURATION),
            DictationEvent.ExternalStop(other, StopReason.CALL), DictationEvent.ProcessDied(other, InferProcess.ASR), DictationEvent.FailureRecorded(other, entry))
        for (st in listOf(arming(), listening(), finishing(), polishing(), delivering(), DictationState.Done(s, DictationOutcome.Failed(MurmurError.NoSpeech, null)))) {
            for (e in events) assertEquals("$st / $e", Transition(st), reduce(st, e))
        }
    }

    // ------------------------------------------------------------------ beyond the rows

    @Test fun failureRecordedGivesTheErrorPillItsEntry() {
        val error = MurmurError.EngineLoadFailed(ModelKind.ASR, null)
        val failed = reduce(finishing(), DictationEvent.AsrFailed(s, error))
        assertEquals(DictationState.Done(s, DictationOutcome.Failed(error, null), recording = true), failed.state)
        val r = reduce(failed.state, DictationEvent.FailureRecorded(s, entry, audioKept = true))
        assertEquals(Transition(DictationState.Done(s, DictationOutcome.Failed(error, entry, audioKept = true))), r)
        // Failures that record nothing (row 6) do not wait for an entry.
        assertEquals(DictationState.Done(s, DictationOutcome.Failed(MurmurError.LowStorage, null)), reduce(arming(), DictationEvent.PreflightFailed(s, MurmurError.LowStorage)).state)
    }

    @Test fun retryRequestedRunsTheStoredAudioFromFinishing() {
        val r = reduce(DictationState.Idle, DictationEvent.RetryRequested(entry, 1_000, s))
        val armed = DictationState.Arming(s, StartMode.TAP, 1_000, stopRequested = true, retryOf = entry)
        assertEquals(armed, r.state)
        assertEquals(DictationBubble.Transcribing, BubbleMapping.of(armed))
        assertFalse(r.effects.any { it is Effect.Haptic })
        // A tap while it arms is ignored (it is already "stopped").
        assertEquals(Transition(armed), reduce(armed, tap(1_500)))
        // The stored 300 s recording arrives with MicStarted: Finishing spans it and ASR_FINAL covers all of it.
        val withPlan = reduce(armed, DictationEvent.PreflightPassed(s, plan.copy(screenshot = false))).state
        val f = reduce(withPlan, DictationEvent.MicStarted(s, 1_200, recordedMs = 300_000))
        val fin = f.state as DictationState.Finishing
        assertEquals(1_200L - 300_000, fin.startedAtMs)
        assertEquals(1_200L, fin.stoppedAtMs)
        assertTrue(fin.showTranscribing)
        assertEquals(DictationBubble.Transcribing, BubbleMapping.of(fin))
        assertTrue(Effect.FinishAsr(s) in f.effects)
        assertTrue(Effect.StartTimer(s, Timer.ASR_FINAL, c.asrFinalTimeoutMs(300_000) + c.scaled(c.asrLoadWaitMs)) in f.effects)
        // Once cold ASR finishes loading during a stored-audio retry, ASR_FINAL preserves the full-recording budget.
        val readyAfterStop = reduce(fin, DictationEvent.AsrReady(s, asrInfo, fellBack = false))
        assertTrue(Effect.StartTimer(s, Timer.ASR_FINAL, c.asrFinalTimeoutMs(300_000)) in readyAfterStop.effects)
        // Crash replay in Finishing also preserves the full-recording budget when ASR comes back ready.
        val longLive = DictationState.Finishing(s, StartMode.TAP, 0, 300_000, plan, readyEngines)
        val restarted = reduce(longLive, DictationEvent.ProcessDied(s, InferProcess.ASR)).state as DictationState.Finishing
        val readyAfterCrash = reduce(restarted, DictationEvent.AsrReady(s, asrInfo, fellBack = false))
        assertTrue(Effect.StartTimer(s, Timer.ASR_FINAL, c.asrFinalTimeoutMs(300_000)) in readyAfterCrash.effects)
        // A normal live dictation with cold ASR loading tightens to the tail budget on AsrReady.
        val coldLive = DictationState.Finishing(s, StartMode.TAP, 0, 60_000, plan, readyEngines.copy(asr = AsrReadiness.Loading))
        val readyColdLive = reduce(coldLive, DictationEvent.AsrReady(s, asrInfo, fellBack = false))
        assertTrue(Effect.StartTimer(s, Timer.ASR_FINAL, c.asrFinalTimeoutMs(c.asrMaxTailMs)) in readyColdLive.effects)
        // Reading the stored file has its own bound; past it the retry is a read failure, not "Microphone is busy".
        val timedOut = reduce(armed, DictationEvent.TimerFired(s, Timer.MIC_START))
        assertEquals(DictationState.Done(s, DictationOutcome.Failed(MurmurError.StorageFailed("audio_read"), null)), timedOut.state)
        // A History retry replaces a result pill like a press (row 46).
        val fromDone = reduce(DictationState.Done(s, DictationOutcome.Inserted(entry, false)), DictationEvent.RetryRequested(entry, 2_000))
        assertEquals(DictationState.Arming(SessionId(2), StartMode.TAP, 2_000, stopRequested = true, retryOf = entry), fromDone.state)
        // Working states ignore it (the orchestrator waits for the running dictation first).
        for (st in listOf(arming(), listening(), finishing(), polishing(), delivering())) assertEquals(Transition(st), reduce(st, DictationEvent.RetryRequested(entry, 2_000)))
        // A live start ignores recordedMs.
        val live = reduce(arming().copy(plan = plan), DictationEvent.MicStarted(s, 400, recordedMs = 5_000)).state as DictationState.Listening
        assertEquals(400L, live.startedAtMs)
    }

    @Test fun timeoutScaleMultipliesEveryTimer() {
        val scaled = c.copy(timeoutScale = 3.0)
        val r = reduce(DictationState.Idle, tap(0, s), scaled)
        assertTrue(Effect.StartTimer(s, Timer.MIC_START, 6_000) in r.effects)
        val l = reduce(arming().copy(plan = plan, engines = readyEngines), DictationEvent.MicStarted(s, 0), scaled)
        assertTrue(Effect.StartTimer(s, Timer.MAX_DURATION, 900_000) in l.effects)
    }

    /**
     * ADR-090 item 4: the DELIVERY timer scales up with the slow emulator but never drops below the insertion ladder's
     * unscaled 1.5 s budget plus margin (DC8's tmo10 runs at scale 0.1: 200 ms would show Copied while a rung inserts).
     */
    @Test fun deliveryTimerNeverDropsBelowTheLadderBudget() {
        val short = c.copy(timeoutScale = 0.1)
        val done = reduce(polishing(started = true), DictationEvent.CleanupDone(s, "Привет.", cleaned = true, outcome = "accept"), short)
        assertTrue(Effect.StartTimer(s, Timer.DELIVERY, 1_900) in done.effects)
        val raw = reduce(polishing(started = true), DictationEvent.TimerFired(s, Timer.LLM), short)
        assertTrue(Effect.StartTimer(s, Timer.DELIVERY, 1_900) in raw.effects)
        val slow = reduce(polishing(started = true), DictationEvent.CleanupDone(s, "Привет.", cleaned = true, outcome = "accept"), c.copy(timeoutScale = 3.0))
        assertTrue(Effect.StartTimer(s, Timer.DELIVERY, 6_000) in slow.effects)
        assertEquals(2_000L, c.deliveryTimerMs())
    }

    @Test fun reducerIsTotal() {
        val states = listOf(DictationState.Idle, arming(), arming().copy(plan = plan, engines = readyEngines), listening(), finishing(), polishing(),
            polishing(started = true), delivering(), DictationState.Done(s, DictationOutcome.Copied(entry)), DictationState.Done(s, DictationOutcome.Failed(MurmurError.NoSpeech, entry)),
            DictationState.Done(s, DictationOutcome.Failed(MurmurError.EngineLoadFailed(ModelKind.ASR, null), null), recording = true, retryAtMs = 500),
            DictationState.Arming(s, StartMode.TAP, 0, stopRequested = true, retryOf = entry))
        val inputs = listOf(UserInput.Tap, UserInput.HoldStart, UserInput.HoldRelease, UserInput.HoldSlideOff, UserInput.Cancel,
            UserInput.NotificationStart, UserInput.InAppStart, UserInput.ResultTap, UserInput.ResultAction).map { DictationEvent.Input(it, 1_000) }
        val events = inputs + Timer.entries.map { DictationEvent.TimerFired(s, it) } + StopReason.entries.map { DictationEvent.ExternalStop(s, it, 1_000) } + listOf(
            DictationEvent.PreflightPassed(s, plan), DictationEvent.PreflightFailed(s, MurmurError.LowStorage), DictationEvent.MicStarted(s, 1), DictationEvent.MicFailed(s, MurmurError.MicBusy),
            DictationEvent.AsrReady(s, asrInfo, true), DictationEvent.AsrLoadFailed(s, MurmurError.NoVadModel), DictationEvent.LlmReady(s, llmInfo, 3_129, 9_216, null, false),
            DictationEvent.LlmLoadFailed(s, MurmurError.OutOfMemory(ModelKind.LLM)), DictationEvent.ContextReady(s, false, 0), DictationEvent.TranscriptReady(s, t, false),
            DictationEvent.AsrFailed(s, MurmurError.NoSpeech), DictationEvent.CleanupDone(s, "x", true, "accept"), DictationEvent.Delivered(s, DeliveryResult.NothingToInsert(null)),
            DictationEvent.ProcessDied(s, InferProcess.ASR), DictationEvent.ProcessDied(s, InferProcess.LLM), DictationEvent.FailureRecorded(s, entry, audioKept = true),
            DictationEvent.RetryRequested(entry, 1_000), DictationEvent.MicStarted(s, 1, recordedMs = 2_000),
        )
        for (st in states) for (e in events) {
            val r = DictationMachine.reduce(st, e, c)
            // Only a start input, a retry request or a remembered Retry may open another session; nothing else ever
            // leaves the session it belongs to.
            val pendingRetry = st is DictationState.Done && st.retryAtMs != null && e is DictationEvent.FailureRecorded
            if (e !is DictationEvent.Input && e !is DictationEvent.RetryRequested && !pendingRetry) {
                assertTrue("$st / $e", r.state.session == null || r.state.session == st.session)
            }
            if (r.state.session != st.session && r.state.session != null) assertTrue("$st / $e", r.state is DictationState.Arming)
        }
    }
}
