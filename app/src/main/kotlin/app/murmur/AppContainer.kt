package app.murmur

import android.app.ActivityManager
import android.content.Context
import app.murmur.core.benchmark.BenchmarkRunner
import app.murmur.core.benchmark.DefaultBenchmarkRunner
import app.murmur.core.catalog.CatalogRepository
import app.murmur.core.catalog.CustomModelResolver
import app.murmur.core.catalog.DeviceProfileRepository
import app.murmur.core.catalog.UnitHealthRepository
import app.murmur.core.context.DefaultContextAssembler
import app.murmur.core.dictation.AppNavigator
import app.murmur.core.dictation.AudioCapture
import app.murmur.core.dictation.DictationDeps
import app.murmur.core.dictation.DictationOrchestrator
import app.murmur.core.dictation.Feedback
import app.murmur.core.dictation.MemoryGate
import app.murmur.core.dictation.ModelResolver
import app.murmur.core.dictation.PermissionMonitor
import app.murmur.core.dictation.SystemConditions
import app.murmur.core.engine.Engines
import app.murmur.core.engine.ProcessExits
import app.murmur.core.model.ModelKind
import app.murmur.core.model.MurmurClock
import app.murmur.core.model.SystemMurmurClock
import app.murmur.core.prompt.ClasspathPromptAssets
import app.murmur.core.prompt.GuardPyPort
import app.murmur.core.prompt.OutputGuard
import app.murmur.core.prompt.PromptRenderer
import app.murmur.core.prompt.RenderPyPort
import app.murmur.core.store.AudioStore
import app.murmur.core.store.BenchmarkRepository
import app.murmur.core.store.DefaultDeliveryMetrics
import app.murmur.core.store.HistoryRepository
import app.murmur.core.store.InstalledModelsRepository
import app.murmur.core.store.MetricsRepository
import app.murmur.core.store.SettingsRepository
import app.murmur.data.db.DbModule
import app.murmur.data.db.Retention
import app.murmur.data.models.DefaultCatalogRepository
import app.murmur.data.models.DefaultDeviceProfileRepository
import app.murmur.data.models.DeviceProfileProber
import app.murmur.data.models.DefaultInstalledModelsRepository
import app.murmur.data.models.DefaultModelResolver
import app.murmur.data.models.FileUnitHealthRepository
import app.murmur.data.models.HfCustomModelResolver
import app.murmur.data.models.ModelDownloads
import app.murmur.data.models.ModelsModule
import app.murmur.data.models.RoomCustomModelStore
import app.murmur.feature.dashboard.DashboardDeps
import app.murmur.feature.history.HistoryDeps
import app.murmur.feature.models.ModelsDeps
import app.murmur.feature.onboarding.OnboardingDeps
import app.murmur.feature.settings.SettingsDeps
import app.murmur.infer.api.BinderEngines
import app.murmur.infer.api.BinderTimeouts
import app.murmur.platform.a11y.A11yBridge
import app.murmur.platform.a11y.A11yStatus
import app.murmur.platform.a11y.BridgedInserter
import app.murmur.platform.a11y.BridgedTargetReader
import app.murmur.platform.audio.AndroidAudioCapture
import app.murmur.platform.audio.AndroidStatusNotifier
import app.murmur.platform.audio.OpusAudioStore
import app.murmur.platform.overlay.OverlayFeedback
import app.murmur.shell.IntentAppNavigator
import app.murmur.ui.design.AppInfoProvider
import app.murmur.wiring.AndroidPermissionMonitor
import app.murmur.wiring.FakeData
import app.murmur.wiring.GitHubAppUpdater
import app.murmur.wiring.PackageAppInfoProvider
import app.murmur.wiring.StatusNotificationController
import app.murmur.wiring.StoredPrefillSpeeds
import app.murmur.wiring.binderScale
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

/**
 * The :main object graph (ADR-010: manual constructor injection). Everything is lazy: the container is built in
 * Application.onCreate and a part is constructed when first used. Debug builds can swap inference or the whole data
 * side for :testing fakes ([DebugWiring], D6); release builds use the real modules only.
 * Owner: WP-15 in wave 2 (feature deps, ports, fake wiring for screens), WP-20 from wave 3 (real modules); docs/wp/README.
 */
class AppContainer(private val context: Context) {
    val clock: MurmurClock = SystemMurmurClock
    /** Application scope; the orchestrator gets its own serial view of it (threading policy, ARCHITECTURE.md 14). */
    val appScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    /** onServiceConnected / onUnbind of the accessibility service (A11yServiceHost.onServiceState). */
    val serviceConnected = MutableStateFlow(false)

    private val debug = DebugWiring(context, appScope)
    private val replacements by lazy { debug.replacements() }
    private val fake: FakeData? get() = replacements.data
    /** True when the debug fake container replaces the data side (shell and screenshot tests). */
    val fakeContainer: Boolean get() = fake != null

    /** Binder clients bind from the accessibility service when connected (process importance), else from the app. */
    private val serviceOrApp: () -> Context = { A11yBridge.service.value ?: context }

    // --- data (WP-12): one database and one settings DataStore per process (ADR-072)
    private val db by lazy { DbModule.database(context) }
    val audioStore: AudioStore by lazy { fake?.audioStore ?: OpusAudioStore(context) }
    val history: HistoryRepository by lazy { fake?.history ?: DbModule.history(db, audioStore) }
    val metrics: MetricsRepository by lazy { fake?.metrics ?: DbModule.metrics(db) }
    /** Fake engines with the real data side keep their records apart (ADR-089, [Replacements.engineRecords]). */
    private val engineRecords get() = replacements.engineRecords
    val benchmarks: BenchmarkRepository by lazy { fake?.benchmarks ?: engineRecords?.benchmarks ?: DbModule.benchmarks(db) }
    val settings: SettingsRepository by lazy { fake?.settings ?: DbModule.settings(context, appScope) }
    /** Retention now (Settings "Keep history" shortening, WP-12 runNow); the scheduler only for the real database. */
    private val retentionScheduler by lazy { DbModule.retention(context, db, history, metrics, audioStore, settings, clock, appScope) }
    private val fakeRetention by lazy {
        Retention(history, metrics, audioStore, clock) { history.entries().first().mapNotNull { it.audio?.path }.toSet() }
    }
    private val retentionNow: suspend (Int) -> Int = { days -> if (fake != null) fakeRetention.runNow(days) else retentionScheduler.runNow(days) }

    // --- models (WP-13)
    val catalog: CatalogRepository by lazy { fake?.catalog ?: DefaultCatalogRepository(store = RoomCustomModelStore(db), scope = appScope) }
    /** Downloads of this process; MurmurApp hands it to ModelDownloadJob (ModelJobHost). Preparing = the orchestrator. */
    val modelDownloads: ModelDownloads by lazy { ModelsModule.downloads(context, catalog, scope = appScope) { orchestrator } }
    val installed: InstalledModelsRepository by lazy { fake?.installed ?: DefaultInstalledModelsRepository(modelDownloads) }
    // --- engines (real binder clients, or :testing fakes in debug)
    val engines: Engines by lazy { replacements.engines ?: BinderEngines(serviceOrApp, ::binderTimeouts, prefillSpeeds) }
    /**
     * Read at every binder call (ARCHITECTURE 4.3). Debug builds lengthen them with Settings.debugTimeoutScale for the
     * slow x86_64 emulator (ADR-049, ADR-063), never below 1 ([binderScale], ADR-090 item 2); release builds always use 1.
     */
    private fun binderTimeouts(): BinderTimeouts {
        val scale = binderScale(BuildConfig.DEBUG, settings.settings.value.debugTimeoutScale)
        return if (scale == timeouts.scale) timeouts else BinderTimeouts(scale = scale).also { timeouts = it }
    }
    @Volatile private var timeouts = BinderTimeouts()
    /** The stored benchmarks' prefill speed scales the prefix-prefill timeout (ADR-071); watched from [start]. */
    private val prefillSpeeds by lazy { StoredPrefillSpeeds(benchmarks, appScope) }
    /** The only owner of the probe (WP-13, ARCHITECTURE 8.4); never throws (CPU-only profile on failure). */
    val deviceProfile: DeviceProfileRepository by lazy {
        fake?.device ?: engineRecords?.device ?: DefaultDeviceProfileRepository(
            context, engines.asrHost, engines.llmHost,
            // ADR-093: a cpuOnly fallback is retried outside dictations (the background slot; a press preempts it), and
            // the short idle exit after a probe never replaces a dictation's hold (ADR-086).
            retryScope = appScope,
            runRetry = { orchestrator.whenNoDictation(it) },
            releaseAfterProbe = { orchestrator.releaseAfterProbe(it, DeviceProfileProber.IDLE_AFTER_PROBE_MS) },
        )
    }
    val unitHealth: UnitHealthRepository by lazy { fake?.unitHealth ?: engineRecords?.unitHealth ?: FileUnitHealthRepository(ModelsModule.storage(context), ModelsModule.appVersion(context)) }
    /** The production resolver; kept as its own type for the benchmark runner's specs (ADR-080, `resolver.benchSpecs`). */
    private val resolver by lazy { DefaultModelResolver(catalog, installed, deviceProfile, unitHealth, benchmarks) { ModelsModule.storage(context).dir(it).path } }
    /**
     * What the orchestrator and the status notification resolve with: the debug fake container's or the engines-only
     * mode's fake resolver ([Replacements.resolver]: the fake engines need no installed pack), else [resolver].
     */
    private val models: ModelResolver by lazy { fake?.resolver ?: replacements.resolver ?: resolver }
    private val customModels: CustomModelResolver by lazy { HfCustomModelResolver() }
    /**
     * WP-14 runner over the production engines, specs from the production resolver's loadSpec (ADR-080; [resolver] stays
     * a [DefaultModelResolver] reference for `benchSpecs` even when [models] is a fake). The :testing runner exists only
     * inside the debug fake container ([FakeData.benchmarkRunner], over the in-memory [benchmarks]). With fake engines
     * and the real data side, this runner runs over the fake engines and [benchmarks] / [unitHealth] are in memory
     * ([Replacements.engineRecords], ADR-089): no debug mode stores made-up records in the real database (ADR-087).
     */
    val benchmarkRunner: BenchmarkRunner by lazy {
        fake?.benchmarkRunner ?: DefaultBenchmarkRunner(
            engines = engines, specs = resolver.benchSpecs, records = benchmarks, health = unitHealth, clock = clock,
            environment = AndroidBenchEnvironment(context), renderer = renderer, guard = guard, assets = prompts,
            exits = exits,
        )
    }

    /** ADR-083: exit records of :asr / :llm (a low-memory kill during a load is OutOfMemory, not a unit failure). */
    private val exits: ProcessExits by lazy { AndroidProcessExits(context) }

    // --- prompt (WP-04)
    val prompts by lazy { ClasspathPromptAssets() }
    val renderer: PromptRenderer by lazy { replacements.renderer ?: RenderPyPort(prompts) }
    val guard: OutputGuard by lazy { replacements.guard ?: GuardPyPort() }

    // --- platform ports implemented here (WP-15)
    /** The one permission monitor (A11yStatus-based); MurmurApp and MainActivity refresh it (see its KDoc). */
    val permissions: PermissionMonitor by lazy { AndroidPermissionMonitor(context) }
    val apps: AppInfoProvider by lazy { PackageAppInfoProvider(context) }
    private val navigator: AppNavigator by lazy { IntentAppNavigator(context) }
    private val statusNotifier by lazy { AndroidStatusNotifier(context) }
    private val audioCapture: AudioCapture by lazy {
        // The microphone FGS starts from the accessibility-service context at press time (WP-09, platform spike b).
        // Debug builds: the clip in files/debug-capture.wav instead of the microphone when it exists ([Replacements.capture]).
        fake?.audio ?: AndroidAudioCapture(context, audioStore, fgsContext = serviceOrApp).let { mic ->
            replacements.capture?.wrap(mic, audioStore) ?: mic
        }
    }
    private val memory: MemoryGate by lazy {
        fake?.memory ?: MemoryGate {
            val mi = ActivityManager.MemoryInfo()
            (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(mi)
            mi.availMem / (1024 * 1024)
        }
    }

    /** Preflight row 6: free storage (filesDir volume) and thermal status (ADR-068). */
    private val system = object : SystemConditions {
        override fun freeStorageBytes(): Long = android.os.StatFs(context.filesDir.path).availableBytes
        override fun thermalCritical(): Boolean =
            (context.getSystemService(Context.POWER_SERVICE) as android.os.PowerManager).currentThermalStatus >=
                android.os.PowerManager.THERMAL_STATUS_CRITICAL
    }

    /** WP-03: haptics from the bubble view per Settings.feedback (ADR-044); earcons silent until designed (ADR-062). */
    private val feedback: Feedback = OverlayFeedback { settings.settings.value.feedback }

    // --- dictation (WP-11)
    val orchestrator: DictationOrchestrator by lazy {
        DictationOrchestrator(
            DictationDeps(
                clock = clock, settings = settings, models = models, engines = engines,
                audio = audioCapture, target = BridgedTargetReader(), inserter = BridgedInserter(),
                context = DefaultContextAssembler(), renderer = renderer, guard = guard, prompts = prompts,
                history = history, metrics = metrics, audioStore = audioStore, feedback = feedback, memory = memory,
                unitHealth = unitHealth, navigator = navigator, permissions = permissions, deliveryMetrics = DefaultDeliveryMetrics(),
                system = system, exits = exits,
            ),
            appScope,
            onInternalError = { android.util.Log.e("MurmurDictation", internalErrorLine(it)) },
        )
    }

    // --- feature deps (every member listed in each feature file)
    val historyDeps: HistoryDeps by lazy {
        HistoryDeps(
            history = history, settings = settings, audio = audioStore, clock = clock,
            retry = { orchestrator.retry(it) }, repolish = { orchestrator.repolish(it) }, apps = apps,
            modelName = { id -> catalog.catalog.value.model(id)?.displayName }, appScope = appScope,
        )
    }
    val dashboardDeps: DashboardDeps by lazy {
        DashboardDeps(metrics = metrics, settings = settings, stats = DbModule.stats(), clock = clock, serviceConnected = serviceConnected, apps = apps)
    }
    val modelsDeps: ModelsDeps by lazy {
        ModelsDeps(
            catalog = catalog, installed = installed, benchmarks = benchmarks, runner = benchmarkRunner, settings = settings,
            custom = customModels, device = deviceProfile.profile, health = unitHealth, ensureDevice = deviceProfile::ensure,
        )
    }
    private val updater by lazy { GitHubAppUpdater(context, appScope) }
    val settingsDeps: SettingsDeps by lazy {
        SettingsDeps(
            settings = settings, prompts = prompts, renderer = renderer, catalog = catalog, history = history,
            serviceConnected = serviceConnected,
            tryCleanup = { sample, override, prefs -> orchestrator.tryCleanup(sample, override, prefs) },
            retention = retentionNow, apps = apps, permissions = permissions,
            // Only release builds share the persistent release signing key required for in-place APK upgrades.
            updater = if (fake != null || BuildConfig.DEBUG) null else updater,
        )
    }
    val onboardingDeps: OnboardingDeps by lazy {
        OnboardingDeps(
            settings = settings, catalog = catalog, installed = installed, serviceConnected = serviceConnected,
            permissions = permissions, device = deviceProfile,
            practice = { orchestrator.startInApp(it) },
            bubble = orchestrator.bubble, levels = orchestrator.levels, bubbleInput = { orchestrator.onInput(it) },
            clock = clock, accessibilitySettingsOpened = { A11yStatus.get(context).onSentToAccessibilitySettings() },
        )
    }
    val shellDeps: ShellDeps by lazy { ShellDeps(dashboardDeps, historyDeps, modelsDeps, settingsDeps, practice = onboardingDeps) }

    /**
     * Called once from MurmurApp.onCreate in :main (WP-20 Deliver 1):
     *  - the settings repository is built now: it starts reading the document at once, so the StateFlow every part
     *    reads without waiting holds the stored values before the first dictation or screen needs them;
     *  - retention: start-up run with pending Undo deletes, the period and orphan audio, the midnight job, the
     *    shortening watcher (ADR-072; real data only);
     *  - the orchestrator is built, so that its settings observer (ARCHITECTURE 13: a prompt-input change re-prepares the
     *    active LLM's prefix) runs from the start and not only from the first dictation;
     *  - the device profile: probed once per app version (ARCHITECTURE 8.4). :main starts after the first unlock (the
     *    app is not direct-boot aware) and again after an update, so the probe runs then and not in the first press;
     *  - the stored LLM speeds for the prefix-prefill timeouts;
     *  - the status notification (ADR-055) and the debug switches ([DebugWiring.start]).
     */
    fun start() {
        settings
        if (fake == null) retentionScheduler.start()
        orchestrator
        appScope.launch { deviceProfile.ensure() }
        if (replacements.engines == null) {
            appScope.launch {
                catalog.loaded().models.filter { it.kind == ModelKind.LLM }.flatMap { it.artifacts }.forEach { prefillSpeeds.watch(it.id) }
            }
        }
        StatusNotificationController(statusNotifier, serviceConnected, permissions, settings, installed, models, catalog).start(appScope)
        debug.start(settings, replacements)
    }
}
