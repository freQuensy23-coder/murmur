package app.murmur.core.dictation

import app.murmur.core.engine.Transcript
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.EntryId
import app.murmur.core.model.InferProcess
import app.murmur.core.model.ModelKind
import app.murmur.core.model.MurmurError
import app.murmur.core.model.SessionId
import app.murmur.core.model.Stage


/**
 * The pure transition function: ARCHITECTURE.md 5.2, one branch per row (row numbers in the comments). Total over
 * (state, event): anything a row does not mention leaves the state unchanged with no effects. Every timer value is
 * scaled by [DictationConfig.timeoutScale] here, except the LLM timer, which [CleanupBudget.plan] already scaled, and
 * DELIVERY, which never drops below [DictationConfig.deliveryMinMs] (ADR-090 item 4).
 *
 * Decisions beyond the table (WP-11 report): row 39 shows plain Inserted for a user-chosen raw insert (see noCleanup);
 * Discarded goes straight to Idle (the bubble maps it to Idle anyway);
 * every Done(Failed) starts the 6 s error dismiss timer and plays REJECT (DESIGN 5.3) unless the accessibility
 * service is gone (no overlay); Copied plays CLOCK_TICK (DESIGN 5.3); DELIVERY is started with every Deliver;
 * an ASR death in Arming after the preflight follows rows 21/22; a Tap also finishes a HOLD pill (recovers from a
 * release ignored by row 10); a start input in Done also accepts NotificationStart / InAppStart; row 26's speech
 * test applies when the ASR measures speech (SessionPlan.speechMeasured): a streaming model reports speechMs = 0
 * ("not measured", Transcript KDoc), so for it only blank text or the precheck make NoSpeech.
 * Row 49's Retry of stored audio (WP-11 Deliver 4, "normal flow from Finishing") opens a new session at once instead of
 * Idle + a fire-and-forget effect: Arming(retryOf) is born stopped, MicStarted brings the stored recording, row 9 takes
 * it to Finishing (Transcribing), then Polishing and Delivering as usual; a Retry tapped while RecordFailure is still
 * writing waits for FailureRecorded; with no recording kept, Retry is a new dictation (as for NoSpeech).
 */
object DictationMachine {

    fun reduce(state: DictationState, event: DictationEvent, config: DictationConfig): Transition {
        // Row 54 (and row 53 for Idle): completion events of another session are dropped (I2).
        val es = event.sessionOf()
        if (es != null && es != state.session) return Transition(state)
        return when (state) {
            DictationState.Idle -> idle(event, config)
            is DictationState.Arming -> arming(state, event, config)
            is DictationState.Listening -> listening(state, event, config)
            is DictationState.Finishing -> finishing(state, event, config)
            is DictationState.Polishing -> polishing(state, event, config)
            is DictationState.Delivering -> delivering(state, event, config)
            is DictationState.Done -> done(state, event, config)
        }
    }

    // ---------------------------------------------------------------- Idle (rows 1-4)

    private fun idle(e: DictationEvent, c: DictationConfig): Transition = when (e) {
        is DictationEvent.Input -> startFor(e, DictationState.Idle, c) ?: Transition(DictationState.Idle)
        is DictationEvent.RetryRequested -> startRetry(e.entry, e.atMs, e.session ?: SessionId(1), c)
        else -> Transition(DictationState.Idle)
    }

    /** Rows 1-4 (and 46 / 49 from Done): a start input opens a new session; null = not a start input. */
    private fun startFor(e: DictationEvent.Input, from: DictationState, c: DictationConfig): Transition? {
        val mode = when (e.input) {
            UserInput.Tap -> StartMode.TAP
            UserInput.HoldStart -> if (c.holdToTalk) StartMode.HOLD else return null
            UserInput.NotificationStart -> StartMode.NOTIFICATION
            UserInput.InAppStart -> StartMode.IN_APP
            else -> return null
        }
        return start(mode, e.atMs, e.session ?: SessionId((from.session?.value ?: 0) + 1), c)
    }

    private fun start(mode: StartMode, atMs: Long, s: SessionId, c: DictationConfig): Transition {
        val effects = buildList {
            add(Effect.Preflight(s, mode))
            add(Effect.StartCapture(s))
            add(Effect.StartTimer(s, Timer.MIC_START, c.scaled(c.micStartTimeoutMs)))
            // Row 3: no haptic for the notification start (the phone is not in the hand on the bubble).
            if (mode != StartMode.NOTIFICATION) add(Effect.Haptic(HapticKind.CONFIRM))
        }
        return Transition(DictationState.Arming(s, mode, atMs), effects)
    }

    /**
     * Retry of a Failed entry (row 49 / RetryRequested; WP-11 Deliver 4 "normal flow from Finishing"): a new session whose
     * capture is the stored audio. Preflight checks the current target; MicStarted brings the recording, and the
     * remembered stop takes Arming straight to Finishing (row 9).
     */
    private fun startRetry(entry: EntryId, atMs: Long, s: SessionId, c: DictationConfig) = Transition(
        DictationState.Arming(s, StartMode.TAP, atMs, stopRequested = true, retryOf = entry),
        listOf(Effect.Preflight(s, StartMode.TAP), Effect.StartCapture(s), Effect.StartTimer(s, Timer.MIC_START, c.scaled(c.storedAudioReadTimeoutMs))),
    )

    // ---------------------------------------------------------------- Arming (rows 5-13, 21-24)

    private fun arming(st: DictationState.Arming, e: DictationEvent, c: DictationConfig): Transition {
        val s = st.session
        return when (e) {
            // Row 5.
            is DictationEvent.PreflightPassed -> {
                if (st.plan != null) return Transition(st)
                val plan = e.plan
                val engines = st.engines.copy(
                    llm = if (plan.cleanup) LlmReadiness.Loading else LlmReadiness.NotRequested(plan.cleanupSkipReason ?: "none"),
                    context = if (plan.screenshot) ContextReadiness.Pending else ContextReadiness.NotRequested,
                )
                val effects = buildList {
                    add(Effect.PrepareEngines(s, plan))
                    if (plan.screenshot) add(Effect.CaptureContext(s, c.screenshotDelayMs))
                }
                complete(st.copy(plan = plan, engines = engines), effects, c)
            }
            // Row 6.
            is DictationEvent.PreflightFailed -> failed(s, e.error, null, c,
                Effect.DiscardCapture(s), Effect.CancelTimer(Timer.MIC_START), Effect.CancelEngines(s))
            // Row 7.
            is DictationEvent.MicStarted -> {
                if (st.micStartedAtMs != null) return Transition(st)
                val recorded = if (st.retryOf != null) e.recordedMs.coerceAtLeast(0) else 0
                val next = if (recorded > 0) st.copy(micStartedAtMs = e.atMs - recorded, stopAtMs = e.atMs) else st.copy(micStartedAtMs = e.atMs)
                complete(next, listOf(Effect.CancelTimer(Timer.MIC_START)), c)
            }
            // Row 11.
            is DictationEvent.MicFailed -> failed(s, e.error, null, c,
                Effect.CancelTimer(Timer.MIC_START), Effect.CancelEngines(s), Effect.DiscardCapture(s), Effect.ScheduleIdleUnload(c.idleUnloadMs))
            is DictationEvent.TimerFired -> when (e.timer) {
                // A retry's capture is the stored file: its timeout is a read failure, not a busy microphone.
                Timer.MIC_START -> failed(s, if (st.retryOf != null) MurmurError.StorageFailed("audio_read") else MurmurError.MicStartFailed("timeout"), null, c,
                    Effect.CancelEngines(s), Effect.DiscardCapture(s), Effect.ScheduleIdleUnload(c.idleUnloadMs))
                else -> Transition(st)
            }
            is DictationEvent.Input -> when (e.input) {
                // Row 10: the stop is remembered and applied when Arming completes (row 9); a stop closer than
                // doubleTapGuardMs to the press is ignored.
                UserInput.Tap, UserInput.HoldRelease -> {
                    val forMode = if (st.mode == StartMode.HOLD) e.input == UserInput.HoldRelease else e.input == UserInput.Tap
                    if (!forMode || st.stopRequested || e.atMs - st.pressedAtMs < c.doubleTapGuardMs) Transition(st)
                    else Transition(st.copy(stopRequested = true, stopAtMs = e.atMs))
                }
                // Row 12.
                UserInput.Cancel, UserInput.HoldSlideOff -> discard(s, c, Effect.CancelTimer(Timer.MIC_START))
                else -> Transition(st)
            }
            // Row 13.
            is DictationEvent.ExternalStop -> failed(s, armingStopError(e.reason), null, c,
                Effect.DiscardCapture(s), Effect.CancelEngines(s), Effect.CancelTimer(Timer.MIC_START), Effect.ScheduleIdleUnload(c.idleUnloadMs))
            // Rows 21/22 (after the preflight: the ASR is being prepared).
            is DictationEvent.ProcessDied -> when {
                e.process == InferProcess.ASR && st.plan != null -> {
                    val (eng, eff) = asrDied(s, st.engines, thenFinish = false)
                    Transition(st.copy(engines = eng), eff)
                }
                e.process == InferProcess.LLM && st.plan?.cleanup == true -> {
                    val (eng, eff) = llmDied(s, st.engines)
                    Transition(st.copy(engines = eng), eff)
                }
                else -> Transition(st)
            }
            else -> readiness(st.engines, e)?.let { Transition(st.copy(engines = it)) } ?: Transition(st)
        }
    }

    /** Rows 8 and 9: Arming completes once both the plan and the first buffer are known. */
    private fun complete(st: DictationState.Arming, effects: List<Effect>, c: DictationConfig): Transition {
        val plan = st.plan
        val mic = st.micStartedAtMs
        if (plan == null || mic == null) return Transition(st, effects)
        if (!st.stopRequested) {
            return Transition(
                DictationState.Listening(st.session, st.mode, mic, plan, st.engines),
                effects + listOf(
                    Effect.StartTimer(st.session, Timer.WARN, c.scaled(c.warnAtMs)),
                    Effect.StartTimer(st.session, Timer.MAX_DURATION, c.scaled(c.maxDurationMs)),
                ),
            )
        }
        val retry = st.retryOf != null
        val stop = stop(st.session, st.mode, mic, maxOf(mic, st.stopAtMs ?: mic), plan, st.engines, c, wholeRecording = retry)
        // A retry decodes stored audio: there is no live recording to freeze, the bubble shows Transcribing at once.
        val state = (stop.state as? DictationState.Finishing)?.takeIf { retry }?.copy(showTranscribing = true) ?: stop.state
        return Transition(state, effects + stop.effects)
    }

    private fun armingStopError(r: StopReason): MurmurError = when (r) {
        StopReason.CALL -> MurmurError.MicBusy
        StopReason.A11Y_OFF -> MurmurError.AccessibilityOff
        StopReason.MIC_LOST, StopReason.MIC_SILENCED, StopReason.SCREEN_OFF -> MurmurError.MicStartFailed(r.name.lowercase())
    }

    // ---------------------------------------------------------------- Listening (rows 12, 14-24)

    private fun listening(st: DictationState.Listening, e: DictationEvent, c: DictationConfig): Transition {
        val s = st.session
        return when (e) {
            is DictationEvent.Input -> when (e.input) {
                // Row 14 (TAP / NOTIFICATION / IN_APP; a tap on a HOLD pill too), row 15 (HOLD release).
                UserInput.Tap -> stop(s, st.mode, st.startedAtMs, e.atMs, st.plan, st.engines, c)
                UserInput.HoldRelease -> if (st.mode == StartMode.HOLD) stop(s, st.mode, st.startedAtMs, e.atMs, st.plan, st.engines, c) else Transition(st)
                // Row 12.
                UserInput.Cancel, UserInput.HoldSlideOff -> discard(s, c, Effect.CancelTimer(Timer.WARN), Effect.CancelTimer(Timer.MAX_DURATION))
                else -> Transition(st)
            }
            is DictationEvent.TimerFired -> when (e.timer) {
                // Row 16.
                Timer.WARN -> Transition(st.copy(warn = true))
                // Row 17: auto stop at 5:00.
                Timer.MAX_DURATION -> stop(s, st.mode, st.startedAtMs, st.startedAtMs + c.scaled(c.maxDurationMs), st.plan, st.engines, c)
                else -> Transition(st)
            }
            is DictationEvent.ExternalStop -> if (e.reason == StopReason.A11Y_OFF) {
                // Row 19.
                failed(s, MurmurError.AccessibilityOff, null, c,
                    Effect.StopCapture(s), Effect.CancelEngines(s), Effect.CancelTimer(Timer.WARN), Effect.CancelTimer(Timer.MAX_DURATION),
                    Effect.RecordFailure(s, MurmurError.AccessibilityOff, keepAudio = true), Effect.ScheduleIdleUnload(c.idleUnloadMs))
            } else {
                // Row 18: the audio so far is processed.
                stop(s, st.mode, st.startedAtMs, maxOf(st.startedAtMs, e.atMs ?: st.startedAtMs), st.plan, st.engines, c)
            }
            is DictationEvent.ProcessDied -> when (e.process) {
                // Rows 21/22.
                InferProcess.ASR -> {
                    val (eng, eff) = asrDied(s, st.engines, thenFinish = false)
                    Transition(st.copy(engines = eng), eff)
                }
                // Rows 23/24.
                InferProcess.LLM -> if (st.plan.cleanup) {
                    val (eng, eff) = llmDied(s, st.engines)
                    Transition(st.copy(engines = eng), eff)
                } else Transition(st)
            }
            // Row 20 and the readiness rows.
            else -> readiness(st.engines, e)?.let { Transition(st.copy(engines = it)) } ?: Transition(st)
        }
    }

    /**
     * Row 14 (shared by 9, 15, 17, 18). ASR_FINAL covers the open tail (at most asrMaxTailMs: the segmenter decodes
     * while recording), or the [wholeRecording] for a retry, which feeds all of it at the stop.
     */
    private fun stop(
        s: SessionId, mode: StartMode, startedAt: Long, stoppedAt: Long, plan: SessionPlan, engines: EngineStatus, c: DictationConfig,
        wholeRecording: Boolean = false,
    ): Transition {
        val asr = engines.asr
        if (asr is AsrReadiness.Failed) {
            return failed(s, asr.error, null, c,
                Effect.StopCapture(s), Effect.CancelTimer(Timer.WARN), Effect.CancelTimer(Timer.MAX_DURATION),
                Effect.RecordFailure(s, asr.error, keepAudio = true), Effect.CancelEngines(s), Effect.ScheduleIdleUnload(c.idleUnloadMs))
        }
        val recorded = (stoppedAt - startedAt).coerceAtLeast(0)
        val tail = if (wholeRecording) recorded else minOf(recorded, c.asrMaxTailMs)
        val finalBudget = c.asrFinalTimeoutMs(tail) + if (asr == AsrReadiness.Loading) c.scaled(c.asrLoadWaitMs) else 0L
        return Transition(
            DictationState.Finishing(s, mode, startedAt, stoppedAt, plan, engines, wholeRecording = wholeRecording),
            listOf(
                Effect.StopCapture(s),
                Effect.FinishAsr(s),
                Effect.StartTimer(s, Timer.TRANSCRIBING_LABEL, c.scaled(c.transcribingLabelMs)),
                Effect.StartTimer(s, Timer.ASR_FINAL, finalBudget),
                Effect.CancelTimer(Timer.WARN),
                Effect.CancelTimer(Timer.MAX_DURATION),
                Effect.Haptic(HapticKind.CONFIRM),
            ),
        )
    }

    // ---------------------------------------------------------------- Finishing (rows 23-30, 51, 52)

    private fun finishing(st: DictationState.Finishing, e: DictationEvent, c: DictationConfig): Transition {
        val s = st.session
        return when (e) {
            is DictationEvent.TimerFired -> when (e.timer) {
                // Row 25.
                Timer.TRANSCRIBING_LABEL -> Transition(st.copy(showTranscribing = true))
                // Rows 29/30.
                Timer.ASR_FINAL -> asrFinalFailed(st, MurmurError.Timeout(Stage.ASR_FINAL), retryable = true, c)
                else -> Transition(st)
            }
            is DictationEvent.TranscriptReady -> transcript(st, e.transcript, e.precheckSkip, c)
            // Rows 29/30.
            is DictationEvent.AsrFailed -> asrFinalFailed(st, e.error, retryable = e.error is MurmurError.ProcessDied || e.error is MurmurError.Timeout, c)
            is DictationEvent.ProcessDied -> when (e.process) {
                InferProcess.ASR -> asrFinalFailed(st, MurmurError.ProcessDied(InferProcess.ASR), retryable = true, c)
                // Rows 23/24.
                InferProcess.LLM -> if (st.plan.cleanup) {
                    val (eng, eff) = llmDied(s, st.engines)
                    Transition(st.copy(engines = eng), eff)
                } else Transition(st)
            }
            // Rows 51/52: inputs and external stops are ignored once the stop was received.
            is DictationEvent.Input, is DictationEvent.ExternalStop -> Transition(st)
            is DictationEvent.AsrReady -> {
                val eng = readiness(st.engines, e) ?: return Transition(st)
                val effects = if (st.engines.asr == AsrReadiness.Loading) {
                    val recorded = (st.stoppedAtMs - st.startedAtMs).coerceAtLeast(0)
                    val audioMs = if (st.wholeRecording) recorded else minOf(recorded, c.asrMaxTailMs)
                    listOf(Effect.StartTimer(s, Timer.ASR_FINAL, c.asrFinalTimeoutMs(audioMs)))
                } else emptyList()
                Transition(st.copy(engines = eng), effects)
            }
            else -> readiness(st.engines, e)?.let { Transition(st.copy(engines = it)) } ?: Transition(st)
        }
    }

    private fun asrFinalFailed(st: DictationState.Finishing, error: MurmurError, retryable: Boolean, c: DictationConfig): Transition {
        val s = st.session
        if (retryable && st.engines.asrCrashes == 0) {
            // Row 29: re-bind, replay the full recording to a fresh session, finish again.
            val full = (st.stoppedAtMs - st.startedAtMs).coerceAtLeast(0)
            return Transition(
                st.copy(engines = st.engines.copy(asr = AsrReadiness.Loading, asrCrashes = 1), wholeRecording = true),
                listOf(Effect.RestartAsr(s, thenFinish = true), Effect.StartTimer(s, Timer.ASR_FINAL, c.asrFinalTimeoutMs(full))),
            )
        }
        // Row 30: "Try again" keeps the audio.
        return failed(s, error, null, c,
            Effect.CancelTimer(Timer.ASR_FINAL), Effect.CancelTimer(Timer.TRANSCRIBING_LABEL),
            Effect.RecordFailure(s, error, keepAudio = true), Effect.CancelEngines(s), Effect.ScheduleIdleUnload(c.idleUnloadMs))
    }

    /** Rows 26-28. */
    private fun transcript(st: DictationState.Finishing, t: Transcript, precheckSkip: Boolean, c: DictationConfig): Transition {
        val s = st.session
        val stopTimers = listOf(Effect.CancelTimer(Timer.ASR_FINAL), Effect.CancelTimer(Timer.TRANSCRIBING_LABEL))
        // Row 26.
        if (isNoSpeech(t, precheckSkip, st.plan, c)) {
            return failed(s, MurmurError.NoSpeech, null, c,
                *stopTimers.toTypedArray(), Effect.RecordFailure(s, MurmurError.NoSpeech, keepAudio = true), Effect.ScheduleIdleUnload(c.idleUnloadMs))
        }
        return when (val llm = st.engines.llm) {
            // Row 27.
            is LlmReadiness.NotRequested, is LlmReadiness.Failed -> {
                val reason = if (llm is LlmReadiness.NotRequested) llm.reason else skipReasonOf((llm as LlmReadiness.Failed).error)
                deliverRaw(s, st.stoppedAtMs, t, "skipped:$reason", BubbleMapping.of(st), c, stopTimers)
            }
            // Row 28, then the start rule (row 31).
            LlmReadiness.Loading, is LlmReadiness.Ready -> {
                val effects = buildList {
                    addAll(stopTimers)
                    if (st.engines.context == ContextReadiness.Pending) add(Effect.StartTimer(s, Timer.CONTEXT_WAIT, c.scaled(c.contextWaitMs)))
                    if (llm == LlmReadiness.Loading) add(Effect.StartTimer(s, Timer.LLM_READY, c.scaled(c.llmReadyWaitMs)))
                }
                startRule(DictationState.Polishing(s, st.stoppedAtMs, t, st.plan, st.engines), effects, c)
            }
        }
    }

    // ---------------------------------------------------------------- Polishing (rows 31-38)

    private fun polishing(st: DictationState.Polishing, e: DictationEvent, c: DictationConfig): Transition {
        val s = st.session
        if (st.cleanupStarted) {
            return when (e) {
                // Row 36.
                is DictationEvent.CleanupDone -> Transition(
                    DictationState.Delivering(s, st.stoppedAtMs, e.text, e.cleaned, DictationBubble.Polishing, e.outcome),
                    listOf(
                        Effect.CancelTimer(Timer.LLM),
                        Effect.Deliver(s, e.text, st.transcript.text, e.cleaned, e.outcome),
                        Effect.StartTimer(s, Timer.DELIVERY, c.deliveryTimerMs()),
                    ),
                )
                // Row 37.
                is DictationEvent.TimerFired -> if (e.timer == Timer.LLM) deliverRaw(s, st.stoppedAtMs, st.transcript, "skipped:timeout", DictationBubble.Polishing, c, emptyList())
                    else Transition(st)
                // Row 38.
                is DictationEvent.ProcessDied -> if (e.process == InferProcess.LLM)
                    deliverRaw(s, st.stoppedAtMs, st.transcript, "skipped:llm_died", DictationBubble.Polishing, c, listOf(Effect.CancelTimer(Timer.LLM)))
                    else Transition(st)
                else -> Transition(st)
            }
        }
        val waitTimers = listOf(Effect.CancelTimer(Timer.CONTEXT_WAIT), Effect.CancelTimer(Timer.LLM_READY))
        return when (e) {
            is DictationEvent.TimerFired -> when (e.timer) {
                // Row 32.
                Timer.CONTEXT_WAIT -> startRule(st.copy(contextWaitOver = true), emptyList(), c)
                // Row 34: the load continues so that the next dictation is warm.
                Timer.LLM_READY -> if (st.engines.llm == LlmReadiness.Loading)
                    deliverRaw(s, st.stoppedAtMs, st.transcript, "skipped:llm_not_ready", DictationBubble.Polishing, c, waitTimers)
                    else Transition(st)
                else -> Transition(st)
            }
            // Row 35.
            is DictationEvent.LlmLoadFailed -> deliverRaw(s, st.stoppedAtMs, st.transcript, "skipped:${skipReasonOf(e.error)}", DictationBubble.Polishing, c, waitTimers)
            is DictationEvent.ProcessDied -> if (e.process == InferProcess.LLM)
                deliverRaw(s, st.stoppedAtMs, st.transcript, "skipped:llm_died", DictationBubble.Polishing, c, waitTimers)
                else Transition(st)
            // Row 33 (ContextReady / LlmReady), other readiness rows update only.
            else -> readiness(st.engines, e)?.let { startRule(st.copy(engines = it), emptyList(), c) } ?: Transition(st)
        }
    }

    /** Row 31: the start rule. */
    private fun startRule(st: DictationState.Polishing, effects: List<Effect>, c: DictationConfig): Transition {
        val llm = st.engines.llm
        val contextSettled = st.engines.context != ContextReadiness.Pending || st.contextWaitOver
        if (!contextSettled || llm !is LlmReadiness.Ready) return Transition(st, effects)
        val speed = llm.speed ?: c.defaultSpeed[llm.info.unit] ?: c.defaultSpeed.getValue(ComputeUnit.CPU)
        val plan = CleanupBudget.plan(st.transcript.text.length, llm.prefixTokens, llm.nCtx, speed, c)
        val s = st.session
        val waitTimers = listOf(Effect.CancelTimer(Timer.CONTEXT_WAIT), Effect.CancelTimer(Timer.LLM_READY))
        if (!plan.fits) {
            return deliverRaw(s, st.stoppedAtMs, st.transcript, "skipped:${plan.skipReason ?: "too_long"}", DictationBubble.Polishing, c, effects + waitTimers)
        }
        return Transition(
            st.copy(cleanupStarted = true),
            effects + listOf(Effect.RunCleanup(s, st.transcript, plan), Effect.StartTimer(s, Timer.LLM, plan.timeoutMs)) + waitTimers,
        )
    }

    // ---------------------------------------------------------------- Delivering (rows 39-44, 51, 52)

    private fun delivering(st: DictationState.Delivering, e: DictationEvent, c: DictationConfig): Transition {
        val s = st.session
        return when (e) {
            is DictationEvent.Delivered -> {
                val cancel = Effect.CancelTimer(Timer.DELIVERY)
                val idle = Effect.ScheduleIdleUnload(c.idleUnloadMs)
                when (val r = e.result) {
                    // Row 39.
                    is DeliveryResult.Inserted -> Transition(
                        DictationState.Done(s, DictationOutcome.Inserted(r.entry, noCleanup = noCleanup(st), skipReason = if (st.cleaned) null else st.cleanupOutcome)),
                        listOf(cancel, Effect.Haptic(HapticKind.CLOCK_TICK), Effect.StartTimer(s, Timer.RESULT_DISMISS, c.scaled(c.insertedShowMs)), idle),
                    )
                    // Row 40.
                    is DeliveryResult.CopiedOnly -> copied(s, r.entry, c, cancel)
                    // Row 41: nothing stored, nothing copied.
                    DeliveryResult.DiscardedPassword -> Transition(DictationState.Idle, listOf(cancel, idle))
                    // Row 42.
                    is DeliveryResult.NothingToInsert -> failed(s, MurmurError.NoSpeech, r.entry, c, cancel, idle)
                    // Row 43.
                    is DeliveryResult.NotDelivered -> failed(s, r.error, r.entry, c, cancel, idle)
                }
            }
            // Row 44: the text is already on the clipboard (I5); the executor still writes the history row.
            is DictationEvent.TimerFired -> if (e.timer == Timer.DELIVERY) copied(s, null, c) else Transition(st)
            else -> Transition(st)
        }
    }

    /**
     * DESIGN 5.3: "Inserted, no cleanup" is for a cleanup that failed, timed out, did not fit or had no memory; a
     * dictation the user chose to insert as recognised (cleanup None) or a password-context one shows plain Inserted.
     */
    private fun noCleanup(st: DictationState.Delivering): Boolean =
        !st.cleaned && st.cleanupOutcome != "skipped:none" && st.cleanupOutcome != "skipped:password"

    private fun copied(s: SessionId, entry: EntryId?, c: DictationConfig, vararg extra: Effect) = Transition(
        DictationState.Done(s, DictationOutcome.Copied(entry)),
        extra.toList() + listOf(Effect.Haptic(HapticKind.CLOCK_TICK), Effect.StartTimer(s, Timer.RESULT_DISMISS, c.scaled(c.copiedShowMs)),
            Effect.ScheduleIdleUnload(c.idleUnloadMs)),
    )

    // ---------------------------------------------------------------- Done (rows 45-50, 53)

    private fun done(st: DictationState.Done, e: DictationEvent, c: DictationConfig): Transition {
        val s = st.session
        return when (e) {
            // Row 45.
            is DictationEvent.TimerFired -> if (e.timer == Timer.RESULT_DISMISS) Transition(DictationState.Idle) else Transition(st)
            // A History retry replaces the pill like a new press (row 46).
            is DictationEvent.RetryRequested -> startRetry(e.entry, e.atMs, e.session ?: SessionId(s.value + 1), c)
            is DictationEvent.Input -> startFor(e, st, c) ?: when (e.input) {
                // Rows 47/48 (an Inserted pill tap dismisses too).
                UserInput.ResultTap -> when (val o = st.outcome) {
                    is DictationOutcome.Copied -> Transition(DictationState.Idle, listOfNotNull(o.entry?.let { Effect.Open(OpenTarget.Entry(it)) }))
                    else -> Transition(DictationState.Idle)
                }
                // Rows 49/50.
                UserInput.ResultAction -> resultAction(st, e, c)
                else -> Transition(st)
            }
            // The entry written by RecordFailure arrives after the pill is shown; a Retry tapped meanwhile runs now.
            is DictationEvent.FailureRecorded -> {
                val o = st.outcome
                if (o !is DictationOutcome.Failed || !st.recording) return Transition(st)
                val recorded = st.copy(outcome = o.copy(entry = e.entry, audioKept = e.entry != null && e.audioKept), recording = false, retryAtMs = null)
                st.retryAtMs?.let { at -> retry(recorded, at, null, c) } ?: Transition(recorded)
            }
            // A delivery that finished after row 44: the Copied pill now opens its entry.
            is DictationEvent.Delivered -> {
                val o = st.outcome
                val entry = when (val r = e.result) {
                    is DeliveryResult.Inserted -> r.entry
                    is DeliveryResult.CopiedOnly -> r.entry
                    else -> null
                }
                if (o is DictationOutcome.Copied && o.entry == null && entry != null) Transition(st.copy(outcome = DictationOutcome.Copied(entry))) else Transition(st)
            }
            // Row 53.
            else -> Transition(st)
        }
    }

    private fun resultAction(st: DictationState.Done, e: DictationEvent.Input, c: DictationConfig): Transition {
        val o = st.outcome as? DictationOutcome.Failed ?: return Transition(st)
        return when (BubbleMapping.errorBubble(o.error).action) {
            BubbleAction.RETRY -> retry(st, e.atMs, e.session, c)
            // Row 50.
            BubbleAction.OPEN_MODELS -> Transition(DictationState.Idle, listOf(Effect.Open(OpenTarget.Models(ModelKind.ASR))))
            BubbleAction.OPEN_ENTRY -> Transition(DictationState.Idle, listOfNotNull(o.entry?.let { Effect.Open(OpenTarget.Entry(it)) }))
            null -> Transition(DictationState.Idle)
        }
    }

    /**
     * Row 49, RETRY: NoSpeech = a new dictation (row 1); otherwise the stored audio runs again from Finishing. While
     * the entry is still being written, the tap is remembered; with nothing kept (recordings off, retention 0) a new
     * dictation retries the load instead.
     */
    private fun retry(st: DictationState.Done, atMs: Long, session: SessionId?, c: DictationConfig): Transition {
        val o = st.outcome as DictationOutcome.Failed
        val next = session ?: SessionId(st.session.value + 1)
        return when {
            o.error == MurmurError.NoSpeech -> start(StartMode.TAP, atMs, next, c)
            st.recording -> Transition(st.copy(retryAtMs = atMs))
            o.entry != null && o.audioKept -> startRetry(o.entry, atMs, next, c)
            else -> start(StartMode.TAP, atMs, next, c)
        }
    }

    // ---------------------------------------------------------------- shared pieces

    /** Rows 6, 11, 13, 14, 19, 26, 30, 42, 43: the error pill (DESIGN 5.3: REJECT, 6 s), no haptic without the overlay. */
    private fun failed(s: SessionId, error: MurmurError, entry: EntryId?, c: DictationConfig, vararg effects: Effect): Transition {
        val pill = buildList {
            if (error != MurmurError.AccessibilityOff) add(Effect.Haptic(HapticKind.REJECT))
            add(Effect.StartTimer(s, Timer.RESULT_DISMISS, c.scaled(c.errorShowMs)))
        }
        val recording = effects.any { it is Effect.RecordFailure }
        return Transition(DictationState.Done(s, DictationOutcome.Failed(error, entry), recording = recording), effects.toList() + pill)
    }

    /** Row 12: discard; processes stay warm. */
    private fun discard(s: SessionId, c: DictationConfig, vararg timers: Effect) = Transition(
        DictationState.Idle,
        timers.toList() + listOf(Effect.DiscardCapture(s), Effect.CancelEngines(s), Effect.ScheduleIdleUnload(c.idleUnloadMs)),
    )

    private fun deliverRaw(s: SessionId, stoppedAt: Long, t: Transcript, outcome: String, shownAs: DictationBubble, c: DictationConfig, effects: List<Effect>) =
        Transition(
            DictationState.Delivering(s, stoppedAt, t.text, cleaned = false, shownAs = shownAs, cleanupOutcome = outcome),
            effects + listOf(Effect.Deliver(s, t.text, t.text, cleaned = false, cleanupOutcome = outcome),
                Effect.StartTimer(s, Timer.DELIVERY, c.deliveryTimerMs())),
        )

    /** Rows 21/22. */
    private fun asrDied(s: SessionId, eng: EngineStatus, thenFinish: Boolean): Pair<EngineStatus, List<Effect>> =
        if (eng.asrCrashes == 0) eng.copy(asr = AsrReadiness.Loading, asrCrashes = 1) to listOf(Effect.RestartAsr(s, thenFinish))
        else eng.copy(asr = AsrReadiness.Failed(MurmurError.ProcessDied(InferProcess.ASR)), asrCrashes = eng.asrCrashes + 1) to emptyList()

    /** Rows 23/24. */
    private fun llmDied(s: SessionId, eng: EngineStatus): Pair<EngineStatus, List<Effect>> =
        if (eng.llmCrashes == 0) eng.copy(llm = LlmReadiness.Loading, llmCrashes = 1) to listOf(Effect.ReloadLlm(s))
        else eng.copy(llm = LlmReadiness.Failed(MurmurError.ProcessDied(InferProcess.LLM)), llmCrashes = eng.llmCrashes + 1) to emptyList()

    /** The readiness rows: update [EngineStatus] only; null = not a readiness event. */
    private fun readiness(eng: EngineStatus, e: DictationEvent): EngineStatus? = when (e) {
        is DictationEvent.AsrReady -> eng.copy(asr = AsrReadiness.Ready(e.info, e.fellBack))
        is DictationEvent.AsrLoadFailed -> eng.copy(asr = AsrReadiness.Failed(e.error))
        is DictationEvent.LlmReady -> if (eng.llm is LlmReadiness.NotRequested) eng
            else eng.copy(llm = LlmReadiness.Ready(e.info, e.prefixTokens, e.nCtx, e.speed, e.fellBack))
        is DictationEvent.LlmLoadFailed -> if (eng.llm is LlmReadiness.NotRequested) eng else eng.copy(llm = LlmReadiness.Failed(e.error))
        is DictationEvent.ContextReady -> if (eng.context == ContextReadiness.NotRequested) eng else eng.copy(context = ContextReadiness.Ready(e.hasScreen, e.contextWords))
        else -> null
    }

    /** Row 26: the transcript is no speech (failed, audio kept); the executor asks the same before it loads an LLM. */
    fun isNoSpeech(t: Transcript, precheckSkip: Boolean, plan: SessionPlan, c: DictationConfig): Boolean =
        precheckSkip || t.text.isBlank() || (plan.speechMeasured && t.speechMs < c.minSpeechMs)

    /** cleanup_outcome code for an LLM that is not usable (ARCHITECTURE 15). */
    fun skipReasonOf(e: MurmurError): String = when (e) {
        is MurmurError.OutOfMemory -> "low_memory"
        is MurmurError.ProcessDied -> "llm_died"
        is MurmurError.LlmContextOverflow -> "too_long"
        else -> "llm_load"
    }

    private fun DictationEvent.sessionOf(): SessionId? = when (this) {
        is DictationEvent.Input -> null
        is DictationEvent.PreflightPassed -> session
        is DictationEvent.PreflightFailed -> session
        is DictationEvent.MicStarted -> session
        is DictationEvent.MicFailed -> session
        is DictationEvent.AsrReady -> session
        is DictationEvent.AsrLoadFailed -> session
        is DictationEvent.LlmReady -> session
        is DictationEvent.LlmLoadFailed -> session
        is DictationEvent.ContextReady -> session
        is DictationEvent.TranscriptReady -> session
        is DictationEvent.AsrFailed -> session
        is DictationEvent.CleanupDone -> session
        is DictationEvent.Delivered -> session
        is DictationEvent.TimerFired -> session
        is DictationEvent.ProcessDied -> session
        is DictationEvent.ExternalStop -> session
        is DictationEvent.FailureRecorded -> session
        is DictationEvent.RetryRequested -> null
    }
}
