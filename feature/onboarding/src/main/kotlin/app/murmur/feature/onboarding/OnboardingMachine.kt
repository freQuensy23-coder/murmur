package app.murmur.feature.onboarding

import app.murmur.core.dictation.PermissionState

/**
 * The first-run steps of DESIGN 6.1. WELCOME, then the four permission steps, then the models (choice, download,
 * paused download). RESTRICTED is the recovery screen of step 2 (ob-restricted); there is no overlay step (ADR-034).
 * Finishing the models download (or continuing while downloading / without cleanup) completes onboarding directly.
 */
enum class OnboardingStep { WELCOME, MIC, ACCESSIBILITY, RESTRICTED, NOTIFICATIONS, BATTERY, MODELS, DOWNLOADING, DOWNLOAD_PAUSED }

/** Index of the step in the 5-dot progress (DESIGN 6.1 "n of 5"); WELCOME = no dots. */
val OnboardingStep.dotIndex: Int
    get() = when (this) {
        OnboardingStep.WELCOME, OnboardingStep.MIC -> 0
        OnboardingStep.ACCESSIBILITY, OnboardingStep.RESTRICTED -> 1
        OnboardingStep.NOTIFICATIONS -> 2
        OnboardingStep.BATTERY -> 3
        OnboardingStep.MODELS, OnboardingStep.DOWNLOADING, OnboardingStep.DOWNLOAD_PAUSED -> 4
    }

/** What the models part knows about the downloads of the chosen pack (see [DownloadSummary]). */
enum class ModelsPhase {
    /** Nothing chosen or queued yet. */
    NONE,
    /** Queued, downloading, verifying, extracting or preparing. */
    ACTIVE,
    /** Stopped: paused by the user, no network, full disk, or failed. */
    STOPPED,
    /** Every artifact of the pack is installed. */
    DONE,
}

/**
 * The onboarding state. [perms] is the latest [PermissionState] seen; [micDenials] counts refused runtime dialogs and
 * [micBlocked] says Android will not show the dialog again (DESIGN 6.1: "If it is denied twice, the button becomes
 * Open app settings"). [awaitingA11yReturn] is true between opening the accessibility settings and coming back.
 * [skipped] holds the skippable steps the user passed without granting (Settings shows them later as banners).
 */
data class OnboardingState(
    val step: OnboardingStep = OnboardingStep.WELCOME,
    val perms: PermissionState? = null,
    val micDenials: Int = 0,
    val micBlocked: Boolean = false,
    val awaitingA11yReturn: Boolean = false,
    val skipped: Set<OnboardingStep> = emptySet(),
    val models: ModelsPhase = ModelsPhase.NONE,
    /** The speech model of the chosen pack is installed (dictation works without cleanup). */
    val speechReady: Boolean = false,
    /** Set when the user leaves setup with Back on a step that cannot be skipped. */
    val exited: Boolean = false,
    /** Set when onboarding finishes (downloads complete or user continues): the shell writes Settings.onboardingDone. */
    val finished: Boolean = false,
)

sealed interface OnboardingEvent {
    /** "Set up Murmur" on the welcome screen. */
    data object Start : OnboardingEvent
    /** A new permission reading (PermissionMonitor.state; refresh() runs on every resume). */
    data class Permissions(val state: PermissionState) : OnboardingEvent
    /** The microphone dialog answered. [canAskAgain] = shouldShowRequestPermissionRationale after a denial. */
    data class MicResult(val granted: Boolean, val canAskAgain: Boolean) : OnboardingEvent
    /** "Open Accessibility settings" launched the system screen. */
    data object A11ySettingsOpened : OnboardingEvent
    /** "I've done this" on the restricted-settings screen. */
    data object RestrictedDone : OnboardingEvent
    /** The notification dialog answered (a refusal counts as skipping, the step is skippable). */
    data class NotificationsResult(val granted: Boolean) : OnboardingEvent
    /** "Not now" / "Skip for now". */
    data object Skip : OnboardingEvent
    /** The download of a pack was queued. */
    data object DownloadStarted : OnboardingEvent
    /** A new reading of the pack's downloads. */
    data class Downloads(val phase: ModelsPhase, val speechReady: Boolean) : OnboardingEvent
    /** "Try again" on the paused screen. */
    data object RetryDownload : OnboardingEvent
    /** "Continue while downloading" / "Continue without cleanup": finishes onboarding directly. */
    data object Continue : OnboardingEvent
    /** System Back. */
    data object Back : OnboardingEvent
}

/**
 * The step machine (pure; [OnboardingMachineTest]). Every step whose permission is already granted is passed over, so
 * a second run (after a reinstall, or after Back) starts where something is still missing; on return from each system
 * screen the new [PermissionState] advances the flow by itself (DESIGN 6.1). Restricted settings are detected only
 * through [PermissionState.restrictedLikely] (ADR-027/054): after a return from the accessibility settings with the
 * service still not connected.
 */
object OnboardingMachine {
    private val ORDER = listOf(
        OnboardingStep.MIC, OnboardingStep.ACCESSIBILITY, OnboardingStep.NOTIFICATIONS, OnboardingStep.BATTERY, OnboardingStep.MODELS,
    )

    fun reduce(s: OnboardingState, e: OnboardingEvent): OnboardingState = when (e) {
        OnboardingEvent.Start -> if (s.step == OnboardingStep.WELCOME) failedTrip(s.copy(step = firstOpen(s, from = 0))) else s
        is OnboardingEvent.Permissions -> onPermissions(
            s.copy(
                perms = e.state,
                // The first reading of this controller (a cold start, or a restore after process death): a recorded
                // trip to the accessibility settings that came back without the service (A11yStatus keeps it across
                // process death, ADR-054) is the trip of this run, so the restricted screen follows at once.
                awaitingA11yReturn = s.awaitingA11yReturn || (s.perms == null && e.state.restrictedLikely && !e.state.a11yConnected),
            ),
        )
        is OnboardingEvent.MicResult -> when {
            s.step != OnboardingStep.MIC -> s
            e.granted -> failedTrip(s.copy(step = next(s, OnboardingStep.MIC)))
            else -> {
                val denials = s.micDenials + 1
                s.copy(micDenials = denials, micBlocked = !e.canAskAgain || denials >= 2)
            }
        }
        OnboardingEvent.A11ySettingsOpened ->
            if (s.step == OnboardingStep.ACCESSIBILITY || s.step == OnboardingStep.RESTRICTED) s.copy(awaitingA11yReturn = true) else s
        OnboardingEvent.RestrictedDone ->
            if (s.step == OnboardingStep.RESTRICTED) s.copy(step = OnboardingStep.ACCESSIBILITY, awaitingA11yReturn = false) else s
        is OnboardingEvent.NotificationsResult -> when {
            s.step != OnboardingStep.NOTIFICATIONS -> s
            e.granted -> s.copy(step = next(s, OnboardingStep.NOTIFICATIONS))
            else -> s.copy(step = next(s, OnboardingStep.NOTIFICATIONS), skipped = s.skipped + OnboardingStep.NOTIFICATIONS)
        }
        OnboardingEvent.Skip -> when (s.step) {
            OnboardingStep.NOTIFICATIONS, OnboardingStep.BATTERY -> s.copy(step = next(s, s.step), skipped = s.skipped + s.step)
            else -> s
        }
        OnboardingEvent.DownloadStarted ->
            if (s.step == OnboardingStep.MODELS || s.step == OnboardingStep.DOWNLOAD_PAUSED) s.copy(step = OnboardingStep.DOWNLOADING, models = ModelsPhase.ACTIVE) else s
        is OnboardingEvent.Downloads -> onDownloads(s.copy(models = e.phase, speechReady = e.speechReady))
        OnboardingEvent.RetryDownload ->
            if (s.step == OnboardingStep.DOWNLOAD_PAUSED) s.copy(step = OnboardingStep.DOWNLOADING, models = ModelsPhase.ACTIVE) else s
        OnboardingEvent.Continue -> when (s.step) {
            OnboardingStep.DOWNLOADING -> s.copy(finished = true)
            // "Continue without cleanup" is offered only once the speech model works.
            OnboardingStep.DOWNLOAD_PAUSED -> if (s.speechReady) s.copy(finished = true) else s
            else -> s
        }
        OnboardingEvent.Back -> onBack(s)
    }

    private fun onPermissions(s: OnboardingState): OnboardingState {
        val p = s.perms ?: return s
        return when (s.step) {
            OnboardingStep.MIC -> if (p.micGranted) failedTrip(s.copy(step = next(s, OnboardingStep.MIC))) else s
            OnboardingStep.ACCESSIBILITY, OnboardingStep.RESTRICTED ->
                if (p.a11yConnected) s.copy(step = next(s, OnboardingStep.ACCESSIBILITY), awaitingA11yReturn = false) else failedTrip(s)
            OnboardingStep.NOTIFICATIONS -> if (p.notificationsGranted) s.copy(step = next(s, OnboardingStep.NOTIFICATIONS)) else s
            OnboardingStep.BATTERY -> if (p.batteryUnrestricted) s.copy(step = next(s, OnboardingStep.BATTERY)) else s
            else -> s
        }
    }

    /**
     * Back from the settings screen ([OnboardingState.awaitingA11yReturn]), still off, installed from a file
     * ([PermissionState.restrictedLikely]): Android refused the toggle, so the accessibility step becomes the recovery
     * screen. Checked on every way into the step, so a trip recorded before this run (cold start) is not missed.
     */
    private fun failedTrip(s: OnboardingState): OnboardingState {
        val p = s.perms ?: return s
        return if (s.step == OnboardingStep.ACCESSIBILITY && s.awaitingA11yReturn && p.restrictedLikely && !p.a11yConnected) {
            s.copy(step = OnboardingStep.RESTRICTED, awaitingA11yReturn = false)
        } else {
            s
        }
    }

    private fun onDownloads(s: OnboardingState): OnboardingState = when (s.step) {
        OnboardingStep.DOWNLOADING -> when (s.models) {
            ModelsPhase.DONE -> s.copy(finished = true)
            ModelsPhase.STOPPED -> s.copy(step = OnboardingStep.DOWNLOAD_PAUSED)
            else -> s
        }
        OnboardingStep.DOWNLOAD_PAUSED -> when (s.models) {
            ModelsPhase.DONE -> s.copy(finished = true)
            // Resumed elsewhere (the hub, or the job came back with the network).
            ModelsPhase.ACTIVE -> s.copy(step = OnboardingStep.DOWNLOADING)
            else -> s
        }
        // A pack downloaded or downloading from an earlier run: no second choice.
        OnboardingStep.MODELS -> when (s.models) {
            ModelsPhase.DONE -> s.copy(finished = true)
            ModelsPhase.ACTIVE -> s.copy(step = OnboardingStep.DOWNLOADING)
            ModelsPhase.STOPPED -> s.copy(step = OnboardingStep.DOWNLOAD_PAUSED)
            ModelsPhase.NONE -> s
        }
        else -> s
    }

    private fun onBack(s: OnboardingState): OnboardingState = when (s.step) {
        OnboardingStep.RESTRICTED -> s.copy(step = OnboardingStep.ACCESSIBILITY, awaitingA11yReturn = false)
        // Microphone and Accessibility cannot be skipped: Back exits setup (DESIGN 6.1). So do the welcome screen and
        // a running download (it continues in the background).
        OnboardingStep.WELCOME, OnboardingStep.MIC, OnboardingStep.ACCESSIBILITY,
        OnboardingStep.DOWNLOADING, OnboardingStep.DOWNLOAD_PAUSED -> s.copy(exited = true)
        OnboardingStep.NOTIFICATIONS, OnboardingStep.BATTERY, OnboardingStep.MODELS -> {
            val i = ORDER.indexOf(s.step)
            val prev = (i - 1 downTo 0).map { ORDER[it] }.firstOrNull { !satisfied(s, it) }
            if (prev == null) s.copy(exited = true) else s.copy(step = prev, awaitingA11yReturn = false)
        }
    }

    /** The step after [step] in [ORDER] that still needs the user. */
    private fun next(s: OnboardingState, step: OnboardingStep): OnboardingStep = firstOpen(s, from = ORDER.indexOf(step) + 1)

    private fun firstOpen(s: OnboardingState, from: Int): OnboardingStep {
        for (i in from until ORDER.size) {
            val step = ORDER[i]
            if (!satisfied(s, step)) return step
        }
        return OnboardingStep.MODELS
    }

    /**
     * Whether a step needs nothing from the user. Skipped steps stay skipped within this run. The models step is never
     * passed over by permissions: it resolves through [OnboardingEvent.Downloads] (a pack downloaded earlier jumps on).
     */
    internal fun satisfied(s: OnboardingState, step: OnboardingStep): Boolean {
        val p = s.perms
        return when (step) {
            OnboardingStep.MIC -> p?.micGranted == true
            OnboardingStep.ACCESSIBILITY -> p?.a11yConnected == true
            OnboardingStep.NOTIFICATIONS -> p?.notificationsGranted == true || step in s.skipped
            OnboardingStep.BATTERY -> p?.batteryUnrestricted == true || step in s.skipped
            OnboardingStep.MODELS -> false
            else -> true
        }
    }
}
