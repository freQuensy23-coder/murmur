package app.murmur.core.dictation

import app.murmur.core.dictation.DictationBubble.Inserted
import app.murmur.core.dictation.DictationBubble.Listening
import app.murmur.core.dictation.DictationBubble.Polishing
import app.murmur.core.dictation.DictationBubble.Transcribing
import app.murmur.core.context.CaptureKind
import app.murmur.core.context.ScreenOutcome
import app.murmur.core.model.AudioKey
import app.murmur.core.model.ModelKind
import app.murmur.core.model.MurmurError
import app.murmur.core.model.SessionId
import app.murmur.core.store.CleanupChoice
import app.murmur.core.store.EntryStatus
import app.murmur.core.store.InsertMethod
import app.murmur.core.store.Tone
import app.murmur.testing.FakeInAppTarget
import app.murmur.testing.FakeTargetReader
import app.murmur.testing.syntheticSpeech
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.Instant

/** Orchestrator on FakeWorld + virtual time: the normal flow, delivery rules, context, modes (WP-11 acceptance). */
class OrchestratorFlowTest {
    private val origin = Instant.parse("2026-11-18T12:00:00Z")
    private val cleaned = "Привет это тестовая диктовка."
    private val raw = "привет это тестовая диктовка"

    @Test fun happyPathInsertsWritesHistoryAndMetricsAndSchedulesTheIdleUnload() = runTest {
        val h = harness()
        h.dictate(stopAt = 2_800)
        val e = h.only()
        assertEquals(EntryStatus.INSERTED, e.status)
        assertEquals(cleaned, e.finalText)
        assertEquals(raw, e.rawText)
        assertEquals("accept", e.cleanup?.outcome)
        assertEquals("default.v1", e.cleanup?.promptVersion)
        assertEquals(InsertMethod.COMMIT_TEXT, e.insertMethod)
        assertEquals(2, e.contextWords)                                     // "Лёша Козлов": the count only
        assertEquals(2_800L, e.timings.recordMs)
        assertNotNull(e.timings.stopToDoneMs)
        // Audio file named by AudioKey = createdAt-session (ADR-053), and it is the entry's audio.
        val key = AudioKey.of(origin.toEpochMilli(), SessionId(1)).value
        assertEquals(listOf(key), h.w.audio.audioKeys)
        assertEquals("mem://audio/$key.ogg", e.audio?.path)
        assertEquals(1, h.w.metrics.snapshot().size)
        val m = h.w.metrics.snapshot().single()
        assertEquals(4, m.wordsFinal)
        assertTrue(m.cleanupApplied)
        // I5: clipboard first, then the ladder.
        assertEquals(listOf("clipboard", "insert"), h.w.inserter.calls)
        assertEquals(listOf(cleaned), h.w.inserter.inserted)
        // D5: both processes stay warm 60 s.
        assertEquals(60_000L, h.w.engines.asrHost.idleExitMs)
        assertEquals(60_000L, h.w.engines.llmHost.idleExitMs)
        // ctx v2 reached the LLM with the screen text.
        assertTrue(h.w.engines.llm.requests.single().suffix.contains("Лёша Козлов"))
        assertEquals(DictationState.Idle, h.state)
        h.assertNoInternalErrors()
    }

    /** ADR-088: Tone per app reaches the renderer as the style override; "All other apps" only for apps without a category. */
    @Test fun toneOfTheAppAndTheDefaultToneReachTheCleanupAndTheHistoryRow() = runTest {
        val texts = mapOf("formal" to "Formal register.", "casual" to "Casual register.")
        // Telegram without a category of its own: the default tone applies.
        val noCategory = harness { renderer.toneTexts = texts; settings.edit { it.copy(defaultTone = Tone.FORMAL) } }
        noCategory.dictate()
        assertEquals("Formal register.", noCategory.w.renderer.inputs.last().styleOverride)
        assertEquals("formal", noCategory.only().cleanup?.tone)
        // With a category (messaging): the category line, not the default tone.
        val withCategory = harness {
            renderer.toneTexts = texts; context.categories = mapOf("org.telegram.messenger" to "messaging")
            settings.edit { it.copy(defaultTone = Tone.FORMAL) }
        }
        withCategory.dictate()
        assertNull(withCategory.w.renderer.inputs.last().styleOverride)
        assertNull(withCategory.only().cleanup?.tone)
        // The app's own row wins over both.
        val own = harness {
            renderer.toneTexts = texts; context.categories = mapOf("org.telegram.messenger" to "messaging")
            settings.edit { it.copy(defaultTone = Tone.FORMAL, tonePerApp = mapOf("org.telegram.messenger" to Tone.CASUAL)) }
        }
        own.dictate()
        assertEquals("Casual register.", own.w.renderer.inputs.last().styleOverride)
        assertEquals("casual", own.only().cleanup?.tone)
        // A tone without a named text (tones.v1.json today): no override, the tone is still recorded.
        val noText = harness { settings.edit { it.copy(tonePerApp = mapOf("org.telegram.messenger" to Tone.VERBATIM)) } }
        noText.dictate()
        assertNull(noText.w.renderer.inputs.last().styleOverride)
        assertEquals("verbatim", noText.only().cleanup?.tone)
        listOf(noCategory, withCategory, own, noText).forEach { it.assertNoInternalErrors() }
    }

    /** ADR-088: Re-polish and Try it have no ctx and keep the category line, even when the app has its own tone row. */
    @Test fun rePolishAndTryItKeepTheCategoryLineEvenWithAToneRowForTheApp() = runTest {
        val h = harness {
            renderer.toneTexts = mapOf("formal" to "Formal register.")
            settings.edit { it.copy(defaultTone = Tone.FORMAL, tonePerApp = mapOf("org.telegram.messenger" to Tone.FORMAL)) }
        }
        h.dictate()
        assertEquals("the dictation uses the row", "Formal register.", h.w.renderer.inputs.last().styleOverride)
        val renders = h.w.renderer.inputs.size
        h.o.repolish(h.only().id)
        assertTrue("Re-polish rendered", h.w.renderer.inputs.size > renders)
        assertNull("Re-polish: category line", h.w.renderer.inputs.last().styleOverride)
        val afterRepolish = h.w.renderer.inputs.size
        h.o.tryCleanup("привет мир", null, null)
        assertTrue("Try it rendered", h.w.renderer.inputs.size > afterRepolish)
        assertNull("Try it: category line", h.w.renderer.inputs.last().styleOverride)
        h.assertNoInternalErrors()
    }

    @Test fun bubbleSequenceOfTheHappyPathWithASlowTailIsExact() = runTest {
        val slow = harness { engines.asr.finishDelayMs = 500 }
        slow.dictate(stopAt = 2_800)
        assertEquals(listOf(DictationBubble.Idle, Listening(0, hold = false, warn = false), Listening(0, false, false, frozenAtMs = 2_800),
            Transcribing, Polishing, Inserted(noCleanup = false), DictationBubble.Idle), slow.bubbles)
        slow.assertNoInternalErrors()
    }

    @Test fun bubbleSequenceOfTheHappyPathWithAFastTailHasNoTranscribing() = runTest {
        val fast = harness()
        fast.dictate(stopAt = 2_800)
        assertEquals(listOf(DictationBubble.Idle, Listening(0, false, false), Listening(0, false, false, frozenAtMs = 2_800),
            Polishing, Inserted(noCleanup = false), DictationBubble.Idle), fast.bubbles)
        fast.assertNoInternalErrors()
    }

    @Test fun deliveringKeepsTheFrozenListeningBubble() = runTest {
        // Raw path: Delivering keeps the previous working bubble (here the frozen Listening pill): no flash.
        val quick = harness { settings.edit { it.copy(cleanup = CleanupChoice.None) } }
        quick.dictate(stopAt = 2_800)
        assertEquals(listOf(DictationBubble.Idle, Listening(0, false, false), Listening(0, false, false, frozenAtMs = 2_800),
            Inserted(noCleanup = false), DictationBubble.Idle), quick.bubbles)
    }

    @Test fun deliveringKeepsTranscribingAfterASlowTail() = runTest {
        val slow = harness { settings.edit { it.copy(cleanup = CleanupChoice.None) }; engines.asr.finishDelayMs = 500 }
        slow.dictate(stopAt = 2_800)
        assertEquals(listOf(DictationBubble.Idle, Listening(0, false, false), Listening(0, false, false, frozenAtMs = 2_800),
            Transcribing, Inserted(noCleanup = false), DictationBubble.Idle), slow.bubbles)
    }

    @Test fun cleanupNoneMakesNoLlmCallNeverShowsPolishingAndRecordsSkippedNone() = runTest {
        val spoken = "So like I can sign in now thenks for your help"
        val h = harness {
            settings.edit { it.copy(cleanup = CleanupChoice.None) }
            engines.asr.script = spoken
            engines.llm.output = { "I can sign in now." }
        }
        h.dictate()
        assertTrue(h.w.engines.llm.loadAttempts.isEmpty())
        assertTrue(h.w.engines.llm.requests.isEmpty())
        assertFalse(Polishing in h.bubbles)
        assertEquals(spoken, h.only().finalText)
        assertEquals(listOf(spoken), h.w.inserter.inserted)
        h.assertInserted("skipped:none")
    }

    /** I4 (review F4): the field text, window title and transcript do not outlive the dictation in executor memory. */
    @Test fun theSessionMemoryIsDroppedOnceTheDictationIsWrittenOut() = runTest {
        val h = harness { target.target = FakeTargetReader.defaultTarget().let { it.copy(field = it.field?.copy(beforeCursor = "секрет")) } }
        h.at(0); h.tap(); h.at(1_000)
        assertEquals("секрет", h.o.sessionMemory?.target?.field?.beforeCursor)
        h.at(2_800); h.tap(); h.at(2_900)
        assertTrue(h.state is DictationState.Done)
        assertNull(h.o.sessionMemory)
        assertEquals(EntryStatus.INSERTED, h.only().status)
        h.assertNoInternalErrors()
    }

    /** Row 44: Copied at the DELIVERY timer; the ladder still finishes and writes the row, then the memory goes. */
    @Test fun aSlowDeliveryIsCopiedAtTheTimerAndStillWritesTheEntry() = runTest {
        val h = harness { inserter.insertDelayMs = 3_000 }
        h.at(0); h.tap(); h.at(2_800); h.tap()
        while (h.state !is DictationState.Delivering) h.at(h.now + 1)
        val t0 = h.now
        h.at(t0 + 1_999)
        assertTrue(h.state is DictationState.Delivering)
        h.at(t0 + 2_000)
        assertEquals(DictationState.Done(SessionId(1), DictationOutcome.Copied(null)), h.state)
        assertEquals(DictationBubble.Copied(null), h.o.bubble.value)
        assertNotNull(h.o.sessionMemory?.target)                             // the delivery coroutine still needs it
        h.at(t0 + 3_000)
        val e = h.only()
        assertEquals(EntryStatus.INSERTED, e.status)
        assertEquals(DictationState.Done(SessionId(1), DictationOutcome.Copied(e.id)), h.state)
        assertNull(h.o.sessionMemory)
        h.press(UserInput.ResultTap)
        assertEquals(listOf<OpenTarget>(OpenTarget.Entry(e.id)), h.w.navigator.opened)
        h.assertNoInternalErrors()
    }

    @Test fun passwordAtDeliveryStoresAndCopiesNothing() = runTest {
        val password = FakeTargetReader.defaultTarget().let {
            it.copy(field = it.field!!.copy(inputType = 0x00000081, isPassword = true))
        }
        val h = harness { target.currentOverride = password }
        h.dictate()
        assertEquals(DictationState.Idle, h.state)
        assertTrue(h.w.inserter.clipboard.isEmpty())
        assertTrue(h.w.inserter.inserted.isEmpty())
        assertTrue(h.history().isEmpty())
        assertTrue(h.w.metrics.snapshot().isEmpty())
        assertTrue(h.w.audioStore.files.isEmpty())                          // the recording is deleted too
        assertTrue(h.outcomes().isEmpty())
        h.assertNoInternalErrors()
    }

    @Test fun targetChangedIsCopied() = runTest {
        val h = harness { target.currentOverride = FakeTargetReader.defaultTarget("com.other.app") }
        h.dictate(stopAt = 2_800)
        assertEquals(listOf(cleaned), h.w.inserter.clipboard)
        assertTrue(h.w.inserter.inserted.isEmpty())
        val e = h.only()
        assertEquals(EntryStatus.COPIED, e.status)
        assertEquals(InsertMethod.CLIPBOARD_ONLY, e.insertMethod)
        assertEquals(DictationOutcome.Copied(e.id), h.outcomes().single())
        h.assertNoInternalErrors()
    }

    @Test fun copiedPillTapOpensTheEntry() = runTest {
        val h = harness { target.currentOverride = FakeTargetReader.defaultTarget("com.other.app") }
        h.at(0); h.tap(); h.at(2_800); h.tap(); h.at(3_500)
        assertTrue(h.state is DictationState.Done)
        h.press(UserInput.ResultTap)
        assertEquals(listOf(OpenTarget.Entry(h.only().id)), h.w.navigator.opened)
        assertEquals(DictationState.Idle, h.state)
        h.assertNoInternalErrors()
    }

    @Test fun insertionRejectedIsCopied() = runTest {
        val h = harness { inserter.result = InsertResult.Failed(MurmurError.InsertRejected) }
        h.dictate()
        assertEquals(listOf(cleaned), h.w.inserter.clipboard)
        assertEquals(EntryStatus.COPIED, h.only().status)
        assertTrue(h.outcomes().single() is DictationOutcome.Copied)
        h.assertNoInternalErrors()
    }

    @Test fun serviceLostWhileListeningFailsWithTheAudioKept() = runTest {
        val h = harness()
        h.at(0); h.tap(); h.at(1_000)
        h.o.onAccessibilityLost()
        h.scope.testScheduler.advanceUntilIdle()
        assertEquals(DictationOutcome.Failed(MurmurError.AccessibilityOff, h.only().id, audioKept = true), h.outcomes().single())
        val e = h.only()
        assertEquals(EntryStatus.FAILED, e.status)
        assertEquals("perm.a11y", e.failureCode)
        assertNotNull(e.audio)
        assertTrue(h.w.audioStore.files.containsKey(e.audio!!.path))
        assertTrue(h.w.inserter.clipboard.isEmpty())
        h.assertNoInternalErrors()
    }

    @Test fun serviceLostInPolishingStoresTheTextButDeliversNothing() = runTest {
        val h = harness { engines.llm.delayPerPieceMs = 500 }
        h.at(0); h.tap(); h.at(2_800); h.tap(); h.at(3_000)
        assertTrue(h.state is DictationState.Polishing)
        h.o.onAccessibilityLost()
        h.scope.testScheduler.advanceUntilIdle()
        val e = h.only()
        assertEquals(EntryStatus.FAILED, e.status)
        assertEquals(cleaned, e.finalText)
        assertEquals(DictationOutcome.Failed(MurmurError.AccessibilityOff, e.id), h.outcomes().single())
        assertTrue(h.w.inserter.clipboard.isEmpty())
        assertTrue(h.w.inserter.inserted.isEmpty())
        h.assertNoInternalErrors()
    }

    @Test fun callSignalWhileListeningFinishesWithTheAudioSoFar() = runTest {
        val h = harness()
        h.at(0); h.tap(); h.at(1_200)
        h.w.audio.signal(CaptureSignal.CallStarted)
        h.scope.testScheduler.advanceUntilIdle()
        val e = h.only()
        assertEquals(EntryStatus.INSERTED, e.status)
        assertEquals(1_200L, e.timings.recordMs)
        assertEquals(1_200L * 16, h.w.engines.asr.sessions.single().samples)
        assertTrue(h.states.any { it is DictationState.Finishing && it.stoppedAtMs == 1_200L })
        h.assertNoInternalErrors()
    }

    @Test fun autoStopAtFiveMinutesAfterTheWarnAtFourThirty() = runTest {
        val h = harness { audio.pcm = syntheticSpeech(totalMs = 320_000, speechMs = 310_000) }
        h.at(0); h.tap()
        h.at(269_000)
        assertFalse((h.state as DictationState.Listening).warn)
        h.at(270_000)
        assertTrue((h.state as DictationState.Listening).warn)
        assertEquals(Listening(0, false, warn = true), h.bubbles.last())
        h.at(300_000)
        assertTrue(h.state is DictationState.Finishing || h.state is DictationState.Polishing || h.state is DictationState.Delivering || h.state is DictationState.Done)
        h.scope.testScheduler.advanceUntilIdle()
        assertEquals(300_000L, h.only().timings.recordMs)
        assertEquals(EntryStatus.INSERTED, h.only().status)
        h.assertNoInternalErrors()
    }

    @Test fun cancelDuringListeningStoresNothing() = runTest {
        val h = harness()
        h.at(0); h.tap(); h.at(1_000)
        h.press(UserInput.Cancel)
        h.scope.testScheduler.advanceUntilIdle()
        assertEquals(DictationState.Idle, h.state)
        assertTrue(h.history().isEmpty())
        assertTrue(h.w.metrics.snapshot().isEmpty())
        assertTrue(h.w.audioStore.files.isEmpty())
        assertTrue(h.w.audio.current!!.cancelled)
        assertTrue(h.w.engines.asr.sessions.single().cancelled)
        assertTrue(h.w.engines.llm.requests.isEmpty())
        assertEquals(60_000L, h.w.engines.asrHost.idleExitMs)               // processes stay warm
        h.assertNoInternalErrors()
    }

    @Test fun idleExitIsHeldForTheWholeDictationAndReArmedAtDone() = runTest {
        // ADR-086 (WP-20 DC8 dry run: a warm :llm exited 60 s after the previous Done while the next dictation was
        // still Listening). A press holds both processes for the longest session; Done re-arms the 60 s timeout.
        val h = harness()
        h.dictate()
        h.at(h.now + 50_000)
        h.tap(); h.scope.testScheduler.runCurrent()
        for (host in listOf(h.w.engines.asrHost, h.w.engines.llmHost)) {
            val hold = host.idleExitMs!!
            assertTrue("hold $hold ms", hold >= 300_000L + 60_000L)   // 5 min recording + the idle timeout, at least
            assertTrue("hold $hold ms", hold < 24L * 3_600_000)        // finite: a missed Done cannot pin the processes
        }
        h.at(h.now + 30_000); h.tap()
        h.scope.testScheduler.advanceUntilIdle()
        assertEquals(2, h.history().size)
        for (host in listOf(h.w.engines.asrHost, h.w.engines.llmHost)) {
            assertEquals(60_000L, host.idleExitMs)
            // per dictation: the hold at the press, then the normal timeout after Done
            assertTrue(host.idleExitHistory.toString(), host.idleExitHistory.count { it > 60_000L } >= 2)
        }
        h.assertNoInternalErrors()
    }

    @Test fun theIdleExitAfterAProbeAppliesOnlyWhileNoDictationRuns() = runTest {
        // ADR-093: a device probe's short exit must never replace the ADR-086 session hold.
        val h = harness()
        h.o.releaseAfterProbe(h.w.engines.llmHost, 5_000); h.scope.testScheduler.runCurrent()
        assertEquals(5_000L, h.w.engines.llmHost.idleExitMs)
        h.at(h.now + 1_000); h.tap(); h.scope.testScheduler.runCurrent()
        val hold = h.w.engines.llmHost.idleExitMs!!
        assertTrue("hold $hold", hold > 300_000L)
        h.o.releaseAfterProbe(h.w.engines.llmHost, 5_000); h.scope.testScheduler.runCurrent()
        assertEquals(hold, h.w.engines.llmHost.idleExitMs)
        h.at(h.now + 3_000); h.tap()
        h.scope.testScheduler.advanceUntilIdle()
        assertEquals(60_000L, h.w.engines.llmHost.idleExitMs)
        h.assertNoInternalErrors()
    }

    @Test fun tapDuringArmingAfter300msStopsRightAfterTheMicStarts() = runTest {
        val h = harness { audio.startDelayMs = 500 }
        h.at(0); h.tap()
        h.at(200); h.tap()                                                   // < 300 ms: double-tap protection
        h.at(400); h.tap()                                                   // remembered
        assertTrue((h.state as DictationState.Arming).stopRequested)
        h.at(500)
        val f = h.states.filterIsInstance<DictationState.Finishing>().first()
        assertEquals(500L, f.stoppedAtMs)
        assertTrue(h.states.none { it is DictationState.Listening })
        assertTrue(h.w.audio.current!!.stopped)
        h.scope.testScheduler.advanceUntilIdle()
        h.assertNoInternalErrors()
    }

    @Test fun earlyTapIsIgnoredAndTheDictationKeepsListening() = runTest {
        val h = harness { audio.startDelayMs = 500 }
        h.at(0); h.tap(); h.at(200); h.tap(); h.at(600)
        assertTrue(h.state is DictationState.Listening)
        h.at(2_800); h.tap(); h.scope.testScheduler.advanceUntilIdle()
        assertEquals(EntryStatus.INSERTED, h.only().status)
    }

    @Test fun notificationStartWaitsForTheShadeBeforeTheSnapshot() = runTest {
        val h = harness { target.shadeDelayMs = 400 }
        h.at(0); h.o.onInput(UserInput.NotificationStart); h.at(300)
        assertTrue(h.state is DictationState.Arming)
        assertTrue(h.w.feedback.haptics.isEmpty())                           // row 3: no haptic
        h.at(400)
        assertTrue(h.state is DictationState.Listening)
        assertEquals(1, h.w.target.shadeWaits)
        assertEquals(1_000L, h.w.target.lastShadeTimeoutMs)
        assertEquals(0, h.w.target.snapshots)
        h.at(2_800); h.tap(); h.scope.testScheduler.advanceUntilIdle()
        assertEquals(EntryStatus.INSERTED, h.only().status)
        h.assertNoInternalErrors()
    }

    @Test fun inAppPracticeInsertsIntoTheTargetAndStoresNothing() = runTest {
        val h = harness()
        val field = FakeInAppTarget()
        h.at(0); h.o.startInApp(field); h.at(2_800); h.tap(); h.scope.testScheduler.advanceUntilIdle()
        assertEquals(listOf(cleaned), field.inserted)
        assertTrue(h.history().isEmpty())
        assertTrue(h.w.metrics.snapshot().isEmpty())
        assertTrue(h.w.inserter.clipboard.isEmpty())
        assertEquals(listOf(false), h.w.audio.keepFlags)
        assertEquals(0, h.w.target.screenshots)                             // privacy app_only: no screenshot
        assertEquals(DictationOutcome.Inserted(null, noCleanup = false), h.outcomes().single())
        h.assertNoInternalErrors()
    }

    @Test fun staleEventsOfAPreviousSessionAreIgnored() = runTest {
        val h = harness { models.delayMs = 1_000 }
        h.at(0); h.tap()                                                     // session 1: preflight answers at 1,000
        h.at(400); h.press(UserInput.Cancel)
        h.at(500); h.tap()                                                   // session 2: preflight answers at 1,500
        h.at(1_200)
        val st = h.state as DictationState.Arming
        assertEquals(SessionId(2), st.session)
        assertNull(st.plan)                                                  // session 1's PreflightPassed was dropped
        h.at(3_500); h.tap(); h.scope.testScheduler.advanceUntilIdle()
        assertEquals(EntryStatus.INSERTED, h.only().status)
        assertTrue(h.only().audio!!.path.endsWith("-2.ogg"))
        h.assertNoInternalErrors()
    }

    @Test fun retentionZeroWritesNoRowAndNoAudio() = runTest {
        val h = harness { settings.edit { it.copy(retentionDays = 0) } }
        h.dictate()
        assertTrue(h.history().isEmpty())
        assertTrue(h.w.audioStore.files.isEmpty())
        assertEquals(listOf(false), h.w.audio.keepFlags)
        assertEquals(1, h.w.metrics.snapshot().size)                         // numbers-only stats are independent (ADR-035)
        assertEquals(DictationOutcome.Inserted(null, noCleanup = false), h.outcomes().single())
        h.assertNoInternalErrors()
    }

    @Test fun ocrInTimeIsUsedByTheCleanup() = runTest {
        val inTime = harness()
        inTime.dictate(stopAt = 1_500)
        assertTrue(inTime.w.engines.llm.requests.single().suffix.contains("Лёша Козлов"))
        assertEquals(2, inTime.only().contextWords)
        inTime.assertNoInternalErrors()
    }

    @Test fun lateOcrIsDroppedAfterContextWait() = runTest {
        // OCR answers 3 s after the 0.6 s screenshot; the transcript is ready at 1.5 s: wait 800 ms, then no screen.
        val late = harness { engines.ocr.delayMs = 3_000 }
        late.at(0); late.tap(); late.at(1_500); late.tap(); late.at(2_299)
        assertTrue(late.state is DictationState.Polishing && !(late.state as DictationState.Polishing).cleanupStarted)
        late.at(2_300)
        late.scope.testScheduler.advanceUntilIdle()
        assertFalse(late.w.engines.llm.requests.single().suffix.contains("Лёша Козлов"))
        assertNull(late.only().contextWords)
        assertEquals("accept", late.only().cleanup?.outcome)
        late.assertNoInternalErrors()
    }

    /** The capturer's path (ScreenImage.kind) reaches the assembler, which places field.box with it (CaptureFrame). */
    @Test fun theCaptureKindReachesTheContextAssembler() = runTest {
        for (kind in listOf(CaptureKind.WINDOW, CaptureKind.DISPLAY, null)) {
            val h = harness { target.captureKind = kind }
            h.dictate(stopAt = h.now + 2_800) // one test scheduler for all three runs
            assertEquals(kind, (h.w.context.lastScreen as ScreenOutcome.Text).capture)
            h.assertInserted("accept")
        }
    }

    @Test fun secureWindowRunsCleanupWithoutTheScreen() = runTest {
        val h = harness { target.screenshotError = MurmurError.SecureWindow }
        h.dictate(stopAt = 2_800)
        assertEquals(0, h.w.engines.ocr.calls)
        assertFalse(h.w.engines.llm.requests.single().suffix.contains("Лёша Козлов"))
        h.assertInserted("accept")
    }

    @Test fun micBusyFailsWithoutHistory() = runTest {
        val h = harness { audio.startError = MurmurError.MicBusy }
        h.at(0); h.tap(); h.at(100)
        assertEquals(DictationBubble.Error(BubbleError.MIC_BUSY, null), h.o.bubble.value)
        h.scope.testScheduler.advanceUntilIdle()
        assertEquals(DictationOutcome.Failed(MurmurError.MicBusy, null), h.outcomes().single())
        assertTrue(h.history().isEmpty())
        assertTrue(HapticKind.REJECT in h.w.feedback.haptics)
        h.assertNoInternalErrors()
    }

    @Test fun noAsrModelShowsOpenModelsAndTheActionOpensIt() = runTest { openModels(harness { models.noAsr = true }, MurmurError.NoAsrModel) }

    @Test fun noVadModelShowsOpenModelsAndTheActionOpensIt() = runTest { openModels(harness { models.noVad = true }, MurmurError.NoVadModel) }

    private fun openModels(h: Harness, error: MurmurError) {
        h.at(0); h.tap(); h.at(100)
        assertEquals(DictationBubble.Error(BubbleError.NO_MODEL, BubbleAction.OPEN_MODELS), h.o.bubble.value)
        assertEquals(DictationOutcome.Failed(error, null), (h.state as DictationState.Done).outcome)
        h.press(UserInput.ResultAction)
        assertEquals(listOf(OpenTarget.Models(ModelKind.ASR)), h.w.navigator.opened)
        assertEquals(DictationState.Idle, h.state)
        assertTrue(h.history().isEmpty())
        h.assertNoInternalErrors()
    }

    @Test fun noSpeechIsStoredWithAudioAndRetryStartsANewDictation() = runTest {
        val h = harness { audio.pcm = ShortArray(48_000) }
        h.at(0); h.tap(); h.at(2_800); h.tap(); h.at(3_000)
        assertEquals(DictationBubble.Error(BubbleError.NO_SPEECH, BubbleAction.RETRY), h.o.bubble.value)
        val e = h.only()
        assertEquals(EntryStatus.FAILED, e.status)
        assertEquals("audio.no_speech", e.failureCode)
        assertNotNull(e.audio)
        assertEquals(DictationOutcome.Failed(MurmurError.NoSpeech, e.id, audioKept = true), (h.state as DictationState.Done).outcome)
        h.press(UserInput.ResultAction); h.at(3_100)
        assertEquals(SessionId(2), h.state.session)
        assertTrue(h.state is DictationState.Listening)
        assertEquals(2, h.w.audio.starts)
        h.assertNoInternalErrors()
    }

    @Test fun holdToTalkFinishesOnRelease() = runTest {
        val h = harness()
        h.at(0); h.o.onInput(UserInput.HoldStart); h.at(2_000)
        assertEquals(Listening(0, hold = true, warn = false), h.o.bubble.value)
        h.o.onInput(UserInput.HoldRelease); h.scope.testScheduler.advanceUntilIdle()
        assertEquals(EntryStatus.INSERTED, h.only().status)
        assertTrue(h.bubbles.contains(Listening(0, hold = true, warn = false, frozenAtMs = 2_000)))
        h.assertNoInternalErrors()
    }

    @Test fun joinPolicyIsAppliedAgainstTheCurrentField() = runTest {
        val before = FakeTargetReader.defaultTarget().let { it.copy(field = it.field!!.copy(beforeCursor = "Смотри, ")) }
        val h = harness { target.target = before }
        h.dictate()
        assertEquals(listOf("привет это тестовая диктовка."), h.w.inserter.inserted)   // mid-sentence, accepted: lowercased
        assertEquals(cleaned, h.only().finalText)                                        // history keeps the text itself
        val word = FakeTargetReader.defaultTarget().let { it.copy(field = it.field!!.copy(beforeCursor = "Смотри")) }
        val h2 = harness { target.target = word; settings.edit { it.copy(cleanup = CleanupChoice.None) } }
        h2.dictate()
        assertEquals(listOf(" $raw"), h2.w.inserter.inserted)
    }

    @Test fun toggleHebrewMidRecordingSwapsAsrAndReplaysFromSampleZeroWithoutStoppingMic() = runTest {
        val h = harness {
            settings.edit { it.copy(languages = listOf("ru", "en", "he")) }
            models.hebrewAsrInstalled = true
            engines.asr.scripts.addLast("привет это тест")
            engines.asr.scripts.addLast("שלום עולם מה שלומך")
            engines.llm.output = { it }
        }
        h.at(0); h.tap(); h.at(1_200)
        assertEquals(Listening(0, hold = false, warn = false, hebrewAvailable = true, hebrewActive = false), h.o.bubble.value)
        // Swap to Hebrew mid-recording: mic keeps recording continuously and replays from sample 0.
        h.press(UserInput.ToggleHebrew)
        h.at(1_250)
        assertEquals(Listening(0, hold = false, warn = false, hebrewAvailable = true, hebrewActive = true), h.o.bubble.value)
        h.at(2_800); h.tap()
        h.scope.testScheduler.advanceUntilIdle()
        assertEquals(1, h.w.audio.starts)
        assertEquals(listOf(h.w.models.asrCpu.artifactId, h.w.models.hebrewAsrSpec.artifactId), h.w.engines.asr.loadedSpecs.map { it.artifactId })
        assertEquals(2, h.w.engines.asr.sessions.size)
        assertTrue(h.w.engines.asr.sessions[0].cancelled)
        assertEquals("all 2.8s replayed from sample 0", 2_800L * 16, h.w.engines.asr.sessions[1].samples)
        val e = h.only()
        assertEquals(EntryStatus.INSERTED, e.status)
        assertEquals("שלום עולם מה שלומך", e.finalText)
        assertEquals("he", e.language)
        assertEquals(h.w.models.hebrewAsrSpec.modelId, e.asr?.modelId)
        assertEquals("he", h.w.renderer.inputs.last().ctx?.language?.asr)
        h.assertNoInternalErrors()
    }

    @Test fun togglingHebrewOffMidRecordingSwapsBackToPrimaryAsrAndReplaysAllAudio() = runTest {
        val h = harness {
            settings.edit { it.copy(languages = listOf("ru", "en", "he")) }
            models.hebrewAsrInstalled = true
        }
        h.at(0); h.tap(); h.at(1_000)
        h.press(UserInput.ToggleHebrew); h.at(1_800)
        h.press(UserInput.ToggleHebrew); h.at(2_800); h.tap()
        h.scope.testScheduler.advanceUntilIdle()
        assertEquals(1, h.w.audio.starts)
        assertEquals(
            listOf(h.w.models.asrCpu.artifactId, h.w.models.hebrewAsrSpec.artifactId, h.w.models.asrCpu.artifactId),
            h.w.engines.asr.loadedSpecs.map { it.artifactId },
        )
        assertEquals(3, h.w.engines.asr.sessions.size)
        assertEquals(2_800L * 16, h.w.engines.asr.sessions[2].samples)
        assertEquals(EntryStatus.INSERTED, h.only().status)
        h.assertNoInternalErrors()
    }

}
