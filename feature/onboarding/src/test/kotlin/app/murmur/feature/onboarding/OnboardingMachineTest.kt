package app.murmur.feature.onboarding

import app.murmur.core.dictation.PermissionState
import app.murmur.feature.onboarding.OnboardingEvent as E
import app.murmur.feature.onboarding.OnboardingStep as S
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class OnboardingMachineTest {
    private val none = PermissionState(
        micGranted = false, notificationsGranted = false, a11yEnabled = false, a11yConnected = false,
        a11yEverConnected = false, batteryUnrestricted = false, restrictedLikely = false,
    )

    private fun run(vararg events: E, from: OnboardingState = OnboardingState()): OnboardingState =
        events.fold(from) { s, e -> OnboardingMachine.reduce(s, e) }

    private fun at(step: S, perms: PermissionState = none, extra: (OnboardingState) -> OnboardingState = { it }) =
        extra(OnboardingState(step = step, perms = perms))

    @Test
    fun welcomeStartsAtTheFirstMissingPermission() {
        assertEquals(S.MIC, run(E.Permissions(none), E.Start).step)
        assertEquals(S.ACCESSIBILITY, run(E.Permissions(none.copy(micGranted = true)), E.Start).step)
        val allGranted = none.copy(micGranted = true, a11yConnected = true, a11yEnabled = true, notificationsGranted = true, batteryUnrestricted = true)
        assertEquals("every permission granted: straight to the models", S.MODELS, run(E.Permissions(allGranted), E.Start).step)
        assertEquals("no reading yet: the first step", S.MIC, run(E.Start).step)
    }

    @Test
    fun micGrantedByTheDialogOrOnReturnAdvances() {
        assertEquals(S.ACCESSIBILITY, run(E.MicResult(granted = true, canAskAgain = true), from = at(S.MIC)).step)
        // Granted outside the dialog (pm grant, app settings): the refresh on resume advances.
        assertEquals(S.ACCESSIBILITY, run(E.Permissions(none.copy(micGranted = true)), from = at(S.MIC)).step)
    }

    @Test
    fun micDeniedTwiceOrDontAskAgainTurnsTheButtonIntoAppSettings() {
        val once = run(E.MicResult(granted = false, canAskAgain = true), from = at(S.MIC))
        assertEquals(S.MIC, once.step)
        assertEquals(1, once.micDenials)
        assertFalse(once.micBlocked)
        val twice = run(E.MicResult(granted = false, canAskAgain = true), from = once)
        assertTrue("DESIGN 6.1: denied twice -> Open app settings", twice.micBlocked)
        val never = run(E.MicResult(granted = false, canAskAgain = false), from = at(S.MIC))
        assertTrue("Don't ask again on the first denial", never.micBlocked)
        // Granted later in the app settings: the resume refresh still advances.
        assertEquals(S.ACCESSIBILITY, run(E.Permissions(none.copy(micGranted = true)), from = twice).step)
    }

    @Test
    fun micCannotBeSkippedAndBackExits() {
        assertEquals(S.MIC, run(E.Skip, from = at(S.MIC)).step)
        assertTrue(run(E.Back, from = at(S.MIC)).exited)
    }

    @Test
    fun accessibilityAdvancesOnlyWhenTheServiceIsConnected() {
        val opened = run(E.A11ySettingsOpened, from = at(S.ACCESSIBILITY, none.copy(micGranted = true)))
        assertTrue(opened.awaitingA11yReturn)
        // Enabled in Settings.Secure but not connected yet: stay.
        val enabled = run(E.Permissions(none.copy(micGranted = true, a11yEnabled = true)), from = opened)
        assertEquals(S.ACCESSIBILITY, enabled.step)
        val connected = run(E.Permissions(none.copy(micGranted = true, a11yEnabled = true, a11yConnected = true)), from = enabled)
        assertEquals(S.NOTIFICATIONS, connected.step)
        assertFalse(connected.awaitingA11yReturn)
        assertEquals(S.ACCESSIBILITY, run(E.Skip, from = at(S.ACCESSIBILITY)).step)
        assertTrue(run(E.Back, from = at(S.ACCESSIBILITY)).exited)
    }

    @Test
    fun restrictedBranchOnlyAfterReturningFromTheSettingsScreen() {
        val restricted = none.copy(micGranted = true, restrictedLikely = true)
        // restrictedLikely without having opened the settings in this run: not shown.
        assertEquals(S.ACCESSIBILITY, run(E.Permissions(restricted), from = at(S.ACCESSIBILITY, none.copy(micGranted = true))).step)
        val back = run(E.A11ySettingsOpened, E.Permissions(restricted), from = at(S.ACCESSIBILITY, none.copy(micGranted = true)))
        assertEquals(S.RESTRICTED, back.step)
        // "I've done this" returns to the accessibility step; opening and coming back connected advances.
        val done = run(E.RestrictedDone, from = back)
        assertEquals(S.ACCESSIBILITY, done.step)
        val ok = run(E.A11ySettingsOpened, E.Permissions(none.copy(micGranted = true, a11yEnabled = true, a11yConnected = true)), from = done)
        assertEquals(S.NOTIFICATIONS, ok.step)
        // Allowed while on the recovery screen (the user went to App info and enabled the service): advances too.
        assertEquals(S.NOTIFICATIONS, run(E.Permissions(none.copy(micGranted = true, a11yConnected = true)), from = back).step)
        // Back on the recovery screen goes to the accessibility step, not out of setup.
        assertEquals(S.ACCESSIBILITY, run(E.Back, from = back).step)
        // Not file-installed (restrictedLikely false): the step stays and keeps waiting.
        val plain = run(E.A11ySettingsOpened, E.Permissions(none.copy(micGranted = true)), from = at(S.ACCESSIBILITY, none.copy(micGranted = true)))
        assertEquals(S.ACCESSIBILITY, plain.step)
    }

    @Test
    fun aFailedTripRecordedBeforeThisRunShowsTheRestrictedScreen() {
        // A cold start (or a restore after process death) whose first reading already says restrictedLikely:
        // A11yStatus kept the trip, so the recovery screen follows at once instead of after a second trip.
        val restricted = none.copy(micGranted = true, restrictedLikely = true)
        assertEquals(S.RESTRICTED, run(E.Permissions(restricted), E.Start).step)
        // The microphone first; the restricted screen once it is granted.
        val atMic = run(E.Permissions(none.copy(restrictedLikely = true)), E.Start)
        assertEquals(S.MIC, atMic.step)
        assertEquals(S.RESTRICTED, run(E.Permissions(restricted), from = atMic).step)
        // A restored ACCESSIBILITY step with the trip pending (SavedStateHandle) and no reading yet.
        val restored = OnboardingState(step = S.ACCESSIBILITY, awaitingA11yReturn = true)
        assertEquals(S.RESTRICTED, run(E.Permissions(restricted), from = restored).step)
        // "I've done this" goes back to the step and stays there with the same reading (no bounce).
        val done = run(E.Permissions(restricted), E.Start, E.RestrictedDone)
        assertEquals(S.ACCESSIBILITY, done.step)
        assertEquals(S.ACCESSIBILITY, run(E.Permissions(restricted.copy(notificationsGranted = true)), from = done).step)
        // Connected after all: no recovery screen.
        assertEquals(S.NOTIFICATIONS, run(E.Permissions(restricted.copy(a11yConnected = true)), E.Start).step)
    }

    @Test
    fun notificationsGrantedDeniedOrSkipped() {
        val base = none.copy(micGranted = true, a11yConnected = true)
        assertEquals(S.BATTERY, run(E.NotificationsResult(granted = true), from = at(S.NOTIFICATIONS, base)).step)
        val denied = run(E.NotificationsResult(granted = false), from = at(S.NOTIFICATIONS, base))
        assertEquals("a refusal counts as skipping (the step is skippable)", S.BATTERY, denied.step)
        assertTrue(S.NOTIFICATIONS in denied.skipped)
        val skipped = run(E.Skip, from = at(S.NOTIFICATIONS, base))
        assertEquals(S.BATTERY, skipped.step)
        assertTrue(S.NOTIFICATIONS in skipped.skipped)
        // Granted from the system settings while on the step.
        assertEquals(S.BATTERY, run(E.Permissions(base.copy(notificationsGranted = true)), from = at(S.NOTIFICATIONS, base)).step)
    }

    @Test
    fun batteryGrantedOnReturnOrSkipped() {
        val base = none.copy(micGranted = true, a11yConnected = true, notificationsGranted = true)
        assertEquals(S.MODELS, run(E.Permissions(base.copy(batteryUnrestricted = true)), from = at(S.BATTERY, base)).step)
        // Declined in the system dialog: the step stays so the user can try again or skip.
        assertEquals(S.BATTERY, run(E.Permissions(base), from = at(S.BATTERY, base)).step)
        val skipped = run(E.Skip, from = at(S.BATTERY, base))
        assertEquals(S.MODELS, skipped.step)
        assertTrue(S.BATTERY in skipped.skipped)
    }

    @Test
    fun backFromLaterStepsGoesToTheLastOpenStep() {
        val base = none.copy(micGranted = true, a11yConnected = true)
        val battery = at(S.BATTERY, base)
        assertEquals("notifications not granted: Back reopens it", S.NOTIFICATIONS, run(E.Back, from = battery).step)
        val skippedNotif = battery.copy(skipped = setOf(S.NOTIFICATIONS))
        assertTrue("nothing open before it: Back leaves setup", run(E.Back, from = skippedNotif).exited)
        assertEquals(S.BATTERY, run(E.Back, from = at(S.MODELS, base.copy(notificationsGranted = true))).step)
    }

    @Test
    fun modelsDownloadPauseRetryAndFinish() {
        val base = none.copy(micGranted = true, a11yConnected = true, notificationsGranted = true, batteryUnrestricted = true)
        val started = run(E.DownloadStarted, from = at(S.MODELS, base))
        assertEquals(S.DOWNLOADING, started.step)
        val stopped = run(E.Downloads(ModelsPhase.STOPPED, speechReady = true), from = started)
        assertEquals(S.DOWNLOAD_PAUSED, stopped.step)
        assertEquals(S.DOWNLOADING, run(E.RetryDownload, from = stopped).step)
        assertEquals("resumed elsewhere", S.DOWNLOADING, run(E.Downloads(ModelsPhase.ACTIVE, true), from = stopped).step)
        assertTrue("downloads done finishes onboarding directly", run(E.Downloads(ModelsPhase.DONE, true), from = started).finished)
        assertTrue("continue while downloading finishes directly", run(E.Continue, from = started).finished)
        assertTrue("continue without cleanup once speech works finishes directly", run(E.Continue, from = stopped).finished)
        val noSpeech = run(E.Downloads(ModelsPhase.STOPPED, speechReady = false), from = started)
        assertFalse("without a speech model continue does not finish", run(E.Continue, from = noSpeech).finished)
        assertEquals(S.DOWNLOAD_PAUSED, run(E.Continue, from = noSpeech).step)
    }

    @Test
    fun modelsFromAnEarlierRunAreNotOfferedAgain() {
        assertTrue(run(E.Downloads(ModelsPhase.DONE, true), from = at(S.MODELS)).finished)
        assertEquals(S.DOWNLOADING, run(E.Downloads(ModelsPhase.ACTIVE, false), from = at(S.MODELS)).step)
        assertEquals(S.DOWNLOAD_PAUSED, run(E.Downloads(ModelsPhase.STOPPED, false), from = at(S.MODELS)).step)
        assertEquals(S.MODELS, run(E.Downloads(ModelsPhase.NONE, false), from = at(S.MODELS)).step)
    }

    @Test
    fun continueOnlyFinishesFromDownloadSteps() {
        assertFalse(run(E.Continue, from = at(S.MODELS)).finished)
        assertFalse(run(E.Continue, from = at(S.WELCOME)).finished)
    }

    @Test
    fun dotsFollowTheFiveSteps() {
        assertEquals(listOf(0, 0, 1, 1, 2, 3, 4, 4, 4), S.entries.map { it.dotIndex })
    }
}
