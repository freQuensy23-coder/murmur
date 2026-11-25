package app.murmur.feature.onboarding

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import android.content.Intent
import android.net.Uri
import android.provider.Settings as AndroidSettings
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.LifecycleResumeEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.createSavedStateHandle
import androidx.lifecycle.viewmodel.compose.viewModel
import app.murmur.core.catalog.CatalogRepository
import app.murmur.core.catalog.DeviceProfileRepository
import app.murmur.core.dictation.DictationBubble
import app.murmur.core.dictation.InAppTarget
import app.murmur.core.dictation.PermissionMonitor
import app.murmur.core.dictation.UserInput
import app.murmur.core.model.MurmurClock
import app.murmur.core.store.InstalledModelsRepository
import app.murmur.core.store.SettingsRepository
import app.murmur.ui.design.rememberIsWide
import kotlinx.coroutines.flow.StateFlow

/** Onboarding (DESIGN 6.1): mic, accessibility (+restricted settings recovery), notifications, battery, models, try it. */
class OnboardingDeps(
    val settings: SettingsRepository,
    val catalog: CatalogRepository,
    val installed: InstalledModelsRepository,
    val serviceConnected: StateFlow<Boolean>,
    /** mic / notifications / a11y enabled+connected / battery / restrictedLikely; refresh() on return from system screens. */
    val permissions: PermissionMonitor,
    /** Pack sizes and which artifact each pack item resolves to: await device.ensure() before showing sizes. */
    val device: DeviceProfileRepository,
    /** ob-ready practice field: DictationOrchestrator.startInApp(target) behind the in-app bubble. */
    val practice: (InAppTarget) -> Unit,
    /** The orchestrator's bubble state and level bars, drawn by the in-app practice bubble (DESIGN 5.3). */
    val bubble: StateFlow<DictationBubble>,
    val levels: StateFlow<FloatArray>,
    /** Taps on the practice bubble while it runs (finish = UserInput.Tap): DictationOrchestrator.onInput. */
    val bubbleInput: (UserInput) -> Unit,
    val clock: MurmurClock,
    /**
     * Called when the accessibility settings screen is opened: the permission monitor then treats the next resume as
     * "returned from the accessibility settings", the input of RestrictedSettingsProbe (ADR-054).
     */
    val accessibilitySettingsOpened: () -> Unit,
)

/**
 * Entry point used by the app shell while Settings.onboardingDone is false. [onDone] runs after Done on the practice
 * step has stored onboardingDone. Back on a step that cannot be skipped leaves the app (DESIGN 6.1).
 */
@Composable
fun OnboardingScreen(deps: OnboardingDeps, onDone: () -> Unit, modifier: Modifier = Modifier) {
    val vm: OnboardingViewModel = viewModel { OnboardingViewModel(deps, createSavedStateHandle()) }
    val ctl = vm.controller
    val state by ctl.state.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val activity = context.findActivity()

    // DESIGN 6.1: "On return from each system screen, Murmur re-checks the permission and moves to the next step".
    LifecycleResumeEffect(Unit) {
        deps.permissions.refresh()
        onPauseOrDispose { }
    }
    BackHandler { ctl.dispatch(OnboardingEvent.Back) }
    LaunchedEffect(state.exited) { if (state.exited) activity?.finish() }
    LaunchedEffect(state.finished) {
        if (state.finished) {
            deps.settings.update { it.copy(onboardingDone = true) }
            onDone()
        }
    }

    val mic = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        val canAsk = activity?.shouldShowRequestPermissionRationale(Manifest.permission.RECORD_AUDIO) == true
        ctl.dispatch(OnboardingEvent.MicResult(granted, canAsk))
        deps.permissions.refresh()
    }
    val notifications = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        ctl.dispatch(OnboardingEvent.NotificationsResult(granted))
        deps.permissions.refresh()
    }

    val actions = ObActions(
        start = { ctl.dispatch(OnboardingEvent.Start) },
        requestMic = {
            if (state.micBlocked) context.startSafely(appDetails(context)) else mic.launch(Manifest.permission.RECORD_AUDIO)
        },
        openAccessibility = {
            deps.accessibilitySettingsOpened()
            ctl.dispatch(OnboardingEvent.A11ySettingsOpened)
            context.startSafely(Intent(AndroidSettings.ACTION_ACCESSIBILITY_SETTINGS))
        },
        openAppInfo = { context.startSafely(appDetails(context)) },
        restrictedDone = { ctl.dispatch(OnboardingEvent.RestrictedDone) },
        requestNotifications = { notifications.launch(Manifest.permission.POST_NOTIFICATIONS) },
        requestBattery = {
            @Suppress("BatteryLife") // DESIGN 6.1 step 4: the user is asked once, with the reason on screen.
            val ask = Intent(AndroidSettings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, Uri.parse("package:${context.packageName}"))
            if (!context.startSafely(ask)) context.startSafely(Intent(AndroidSettings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS))
        },
        skip = { ctl.dispatch(OnboardingEvent.Skip) },
        select = ctl::select,
        setLanguage = ctl::setLanguage,
        setWifiOnly = ctl::setWifiOnly,
        download = ctl::download,
        pause = ctl::pause,
        retry = ctl::retry,
        continueSetup = { ctl.dispatch(OnboardingEvent.Continue) },
        startPractice = ctl::startPractice,
        bubbleTap = { deps.bubbleInput(UserInput.Tap) },
        done = { ctl.finish(onDone) },
    )
    OnboardingContent(state, ctl, actions, rememberIsWide(), modifier)
}

/**
 * "Try a dictation" from the app (History and Dashboard empty states): the practice field and in-app bubble of the
 * last onboarding step (DESIGN 6.1 ob-ready) without the onboarding chrome (no dots, no checklist, no "All set"). The
 * shell pushes it like a detail screen; Done and Back call [onClose]. Nothing is stored (ADR-070).
 */
@Composable
fun PracticeScreen(deps: OnboardingDeps, onClose: () -> Unit, modifier: Modifier = Modifier) {
    val vm: OnboardingViewModel = viewModel(key = PRACTICE_VM) { OnboardingViewModel(deps, createSavedStateHandle()) }
    val ctl = vm.controller
    val practice by ctl.practice.collectAsStateWithLifecycle()
    val bubble by deps.bubble.collectAsStateWithLifecycle()
    val levels by deps.levels.collectAsStateWithLifecycle()
    BackHandler(onBack = onClose)
    // The result line belongs to this visit only (in the shell the view model is the back-stack entry's, but a host
    // without entry-scoped view models keeps it for the Activity).
    DisposableEffect(ctl) { onDispose { ctl.clearPractice() } }
    val none = {}
    val actions = ObActions(
        start = none, requestMic = none, openAccessibility = none, openAppInfo = none, restrictedDone = none,
        requestNotifications = none, requestBattery = none, skip = none, select = {}, setLanguage = { _, _ -> }, setWifiOnly = {}, download = none,
        pause = none, retry = none, continueSetup = none,
        startPractice = ctl::startPractice,
        bubbleTap = { deps.bubbleInput(UserInput.Tap) },
        done = onClose,
    )
    Box(modifier) { Ready(rememberIsWide(), bubble, levels, deps.clock, practice, actions, chrome = false) }
}

private const val PRACTICE_VM = "app.murmur.feature.onboarding.practice"

/** Everything a step can do; plain lambdas so that screenshot tests can render the steps without an Activity. */
internal class ObActions(
    val start: () -> Unit,
    val requestMic: () -> Unit,
    val openAccessibility: () -> Unit,
    val openAppInfo: () -> Unit,
    val restrictedDone: () -> Unit,
    val requestNotifications: () -> Unit,
    val requestBattery: () -> Unit,
    val skip: () -> Unit,
    val select: (String) -> Unit,
    val setLanguage: (String, Boolean) -> Unit,
    val setWifiOnly: (Boolean) -> Unit,
    val download: () -> Unit,
    val pause: () -> Unit,
    val retry: () -> Unit,
    val continueSetup: () -> Unit,
    val startPractice: (InAppTarget) -> Unit,
    val bubbleTap: () -> Unit,
    val done: () -> Unit,
)

private fun appDetails(context: Context) =
    Intent(AndroidSettings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", context.packageName, null))

/** startActivity that reports false instead of throwing when no activity handles the intent. */
private fun Context.startSafely(intent: Intent): Boolean = try {
    if (this !is Activity) intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    startActivity(intent)
    true
} catch (e: android.content.ActivityNotFoundException) {
    false
}

private tailrec fun Context.findActivity(): Activity? = when (this) {
    is Activity -> this
    is ContextWrapper -> baseContext.findActivity()
    else -> null
}
