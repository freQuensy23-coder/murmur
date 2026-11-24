package app.murmur.feature.settings

import androidx.compose.ui.graphics.painter.Painter
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import app.murmur.core.catalog.Catalog
import app.murmur.core.catalog.GgufTemplates
import app.murmur.core.context.AppCategories
import app.murmur.core.dictation.PermissionState
import app.murmur.core.model.MurmurError
import app.murmur.core.prompt.PromptEditRules
import app.murmur.core.prompt.PromptEditState
import app.murmur.core.prompt.PromptProfile
import app.murmur.core.store.BubbleSize
import app.murmur.core.store.BubbleVisibility
import app.murmur.core.store.CleanupChoice
import app.murmur.core.store.Feedback
import app.murmur.core.store.RetentionPolicy
import app.murmur.core.store.Settings
import app.murmur.core.store.Tone
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.time.LocalDate

/** A shortening of "Keep history" waiting for the confirm dialog ("Delete 12 dictations older than 3 days?"). */
data class RetentionConfirm(val days: Int, val count: Int)

/** The cleanup prompt as the Settings row and the inner preview show it. */
data class PromptSummary(val profile: PromptProfile, val state: PromptEditState, val editable: Boolean, val previewText: String)

/**
 * Launcher apps and the labels of every package Settings shows (launcher apps, Tone per app, Hidden in), read on the
 * io dispatcher: PackageManager lookups are binder calls and stay off the main thread. A package in [labels] has been
 * looked up (null = not visible or uninstalled). [loaded] is false until the first read returned.
 */
data class InstalledApps(
    val launcher: List<String> = emptyList(),
    val labels: Map<String, String?> = emptyMap(),
    val loaded: Boolean = false,
) {
    /** Launcher label, else the package name (DESIGN: screens show the package when the app is not visible). */
    fun label(pkg: String): String = labels[pkg] ?: pkg
}

/** Everything the Settings list renders. */
data class SettingsUi(
    val settings: Settings,
    /** null only for a catalog that lists no cleanup model at all (the prompt rows are then disabled). */
    val prompt: PromptSummary?,
    val tones: List<ToneRow>,
    /** "All other apps": the stored choice, else Neutral (apps without a category use the default style line). */
    val defaultTone: Tone,
    val defaultTonePreview: Boolean,
    val permissions: PermissionState,
    val serviceConnected: Boolean,
    val apps: InstalledApps,
)

/**
 * Settings (DESIGN 6.5) over [SettingsDeps]. Every setting is written only through `SettingsRepository.update`
 * (WP-19 brief: the orchestrator observes prompt, preferences, unit and model changes and rebuilds the prefix
 * itself). [today] is the local calendar day for the retention count; [io] runs PackageManager and repository reads.
 * A failing write, count or deletion never escapes: [log] gets the error code (D9: codes only) and [events] a message.
 */
class SettingsViewModel(
    private val deps: SettingsDeps,
    scope: CoroutineScope? = null,
    private val today: () -> LocalDate = { LocalDate.now() },
    private val io: CoroutineDispatcher = Dispatchers.IO,
    /** Package -> style category (`prompts/tones/app_categories.v1.json`, bundled with :core:context). */
    private val categories: Map<String, String> = AppCategories.bundled(),
    private val log: (code: String, where: String) -> Unit = SettingsLog::code,
) : ViewModel() {
    private val scope: CoroutineScope = scope ?: viewModelScope

    private val installed = MutableStateFlow(InstalledApps())
    /** Icons already looked up (main-thread map; the lookup itself runs on [io]). */
    private val icons = HashMap<String, Painter?>()
    private val tonesWithText: Set<Tone> = runCatching { ToneRules.tonesWithText(deps.prompts.text("tones/tones.v1.json")) }.getOrDefault(emptySet())

    private val _retentionConfirm = MutableStateFlow<RetentionConfirm?>(null)
    /** Set while the retention confirm dialog is open. */
    val retentionConfirm: StateFlow<RetentionConfirm?> = _retentionConfirm.asStateFlow()

    private val _deleteAllCount = MutableStateFlow<Int?>(null)
    /** Set while the "Delete all history" dialog is open: the number of dictations it deletes. */
    val deleteAllCount: StateFlow<Int?> = _deleteAllCount.asStateFlow()

    private val _events = MutableStateFlow<SettingsEvent?>(null)
    /** One-shot messages for the snackbar; the UI calls [eventShown]. */
    val events: StateFlow<SettingsEvent?> = _events.asStateFlow()

    /** Manual GitHub Releases update state; null when no updater is wired. */
    val updateState: StateFlow<AppUpdateState?> = deps.updater?.state ?: MutableStateFlow<AppUpdateState?>(null).asStateFlow()

    fun checkForUpdates() {
        deps.updater?.checkAndUpdate()
    }

    private val profile: StateFlow<PromptProfile?> = combine(deps.settings.settings, deps.catalog.catalog) { s, c -> activeProfileName(s, c) }
        .map { name -> runCatching { deps.prompts.profile(name) }.getOrNull() }
        .stateIn(this.scope, SharingStarted.Eagerly, runCatching { deps.prompts.profile(activeProfileName(deps.settings.settings.value, deps.catalog.catalog.value)) }.getOrNull())

    val ui: StateFlow<SettingsUi> = combine(
        deps.settings.settings, profile, installed, deps.permissions.state, deps.serviceConnected,
    ) { s, p, apps, perms, connected -> build(s, p, apps, perms, connected) }
        .stateIn(
            this.scope, SharingStarted.Eagerly,
            build(deps.settings.settings.value, profile.value, installed.value, deps.permissions.state.value, deps.serviceConnected.value),
        )

    init {
        // Launcher apps once, then the label of every package that Tone per app or Hidden in stores and that has not
        // been looked up yet (stored packages may be apps without a launcher entry).
        this.scope.launch {
            deps.settings.settings.map { it.tonePerApp.keys + it.hiddenInPackages }.distinctUntilChanged().collect { stored ->
                val cur = installed.value
                val launcher = if (cur.loaded) {
                    cur.launcher
                } else {
                    withContext(io) {
                        try {
                            deps.apps.launcherApps()
                        } catch (e: CancellationException) {
                            throw e
                        } catch (e: Exception) {
                            log(MurmurError.Internal(null).code, "launcher_apps")
                            emptyList()
                        }
                    }
                }
                val missing = (launcher + stored).distinct().filter { it !in cur.labels }
                if (cur.loaded && missing.isEmpty()) return@collect
                val found = withContext(io) { missing.associateWith { pkg -> runCatching { deps.apps.label(pkg) }.getOrNull() } }
                installed.value = InstalledApps(launcher, installed.value.labels + found, loaded = true)
            }
        }
    }

    private fun build(s: Settings, p: PromptProfile?, apps: InstalledApps, perms: PermissionState, connected: Boolean): SettingsUi {
        val prompt = p?.let { prof ->
            val editable = PromptEditRules.editable(prof)
            val state = PromptEditRules.editState(prof, s.promptOverride?.baseProfile)
            val text = if (state == PromptEditState.EDITED) s.promptOverride!!.systemText else runCatching { PromptEditor.shippedText(deps.prompts, prof) }.getOrDefault("")
            PromptSummary(prof, state, editable, text)
        }
        val rows = ToneRules.rows(apps.launcher, categories, s.tonePerApp, tonesWithText) { apps.labels[it] }
        val defaultTone = s.defaultTone ?: Tone.NEUTRAL
        return SettingsUi(
            settings = s,
            prompt = prompt,
            tones = rows,
            defaultTone = defaultTone,
            defaultTonePreview = s.defaultTone != null && ToneRules.isPreview(defaultTone, Tone.NEUTRAL, tonesWithText),
            permissions = perms,
            serviceConnected = connected,
            apps = apps,
        )
    }

    /**
     * Runs [block]; an exception (DataStore or Room I/O, disk full) is logged as a code and shown as [onError] instead
     * of escaping the scope (it would end the process). Cancellation passes through.
     */
    private suspend fun guarded(where: String, onError: SettingsEvent, block: suspend () -> Unit) {
        try {
            block()
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            log(MurmurError.StorageFailed(null).code, where)
            _events.value = onError
        }
    }

    private fun write(transform: (Settings) -> Settings) {
        scope.launch { guarded("settings_write", SettingsEvent.NOT_SAVED) { deps.settings.update(transform) } }
    }

    // ------------------------------------------------------------ Dictation

    /** Languages sheet: at least one language stays selected (the resolver keys the speech model on them). */
    fun setLanguage(code: String, on: Boolean) = write { s ->
        val next = if (on) (s.languages + code).distinct() else s.languages - code
        if (next.isEmpty()) s else s.copy(languages = LANGUAGES.filter { it in next } + next.filter { it !in LANGUAGES })
    }

    fun setAutoDetect(on: Boolean) = write { it.copy(autoDetectLanguage = on) }
    fun setHoldToTalk(on: Boolean) = write { it.copy(holdToTalk = on) }
    fun setFeedback(f: Feedback) = write { it.copy(feedback = f) }

    fun setTypingWpm(wpm: Int) {
        if (wpm in MIN_WPM..MAX_WPM) write { it.copy(typingWpm = wpm) }
    }

    // ------------------------------------------------------------ Bubble

    fun setBubbleVisibility(v: BubbleVisibility) = write { it.copy(bubbleVisibility = v) }
    fun setTuckIn(on: Boolean) = write { it.copy(tuckIn = on) }
    fun setBubbleSize(size: BubbleSize) = write { it.copy(bubbleSize = size) }

    fun setHiddenIn(pkg: String, hidden: Boolean) = write { s ->
        s.copy(hiddenInPackages = if (hidden) s.hiddenInPackages + pkg else s.hiddenInPackages - pkg)
    }

    /** "Reset position": the stored per-display positions are dropped (the bubble goes back to the right edge). */
    fun resetPosition() {
        scope.launch {
            guarded("position_reset", SettingsEvent.NOT_SAVED) {
                deps.settings.update { it.copy(bubblePositions = emptyMap()) }
                _events.value = SettingsEvent.POSITION_RESET
            }
        }
    }

    /** An icon looked up before (null also while it was never looked up); the composition starts from it. */
    fun cachedIcon(pkg: String): Painter? = icons[pkg]

    /** The launcher icon of [pkg], looked up on [io] once (null = letter tile). Call from the main thread. */
    suspend fun appIcon(pkg: String): Painter? {
        if (pkg in icons) return icons[pkg]
        val icon = withContext(io) { runCatching { deps.apps.icon(pkg) }.getOrNull() }
        icons[pkg] = icon
        return icon
    }

    /** Launcher apps for the "Hidden in" picker, without [ownPackage]. */
    fun hiddenInCandidates(ownPackage: String?, ui: SettingsUi = this.ui.value): List<String> =
        ui.apps.launcher.filter { it != ownPackage }

    // ------------------------------------------------------------ Cleanup

    fun setUseScreen(on: Boolean) = write { it.copy(useScreen = on) }

    /** Tone dropdown of a listed or added app. */
    fun setTone(pkg: String, tone: Tone) = write { s ->
        val row = ui.value.tones.firstOrNull { it.pkg == pkg }
        val listed = row != null && !row.removable
        val cat = row?.categoryTone ?: ToneRules.categoryTone(categories[pkg])
        s.copy(tonePerApp = ToneRules.pick(s.tonePerApp, pkg, tone, listed, cat))
    }

    /** "Add an app": stored with its category's tone until the user picks another. */
    fun addToneApp(pkg: String) = write { s ->
        if (pkg in s.tonePerApp) s else s.copy(tonePerApp = s.tonePerApp + (pkg to ToneRules.categoryTone(categories[pkg])))
    }

    fun removeToneApp(pkg: String) = write { it.copy(tonePerApp = it.tonePerApp - pkg) }

    /** "All other apps": Neutral is the category default, so picking it stores no choice. */
    fun setDefaultTone(tone: Tone) = write { it.copy(defaultTone = tone.takeIf { t -> t != Tone.NEUTRAL }) }

    /** Apps that "Add an app" offers: launcher apps not in the list yet, without [ownPackage]. */
    fun toneCandidates(ownPackage: String?, ui: SettingsUi = this.ui.value): List<String> {
        val listed = ui.tones.map { it.pkg }.toSet()
        return ui.apps.launcher.filter { it !in listed && it != ownPackage }
    }

    // ------------------------------------------------------------ Navigation inside Settings

    private val _category = MutableStateFlow(SettingsCategory.DICTATION)
    /** Inner display: the category in the detail pane. */
    val category: StateFlow<SettingsCategory> = _category.asStateFlow()

    private val _sub = MutableStateFlow<SubScreen?>(null)
    /** The open sub-screen (pushed on the cover display, the detail pane on the inner display). */
    val sub: StateFlow<SubScreen?> = _sub.asStateFlow()

    private val _promptEditor = MutableStateFlow<PromptEditor?>(null)
    /** The prompt editor while [sub] is [SubScreen.PROMPT]; kept here so that folding keeps unsaved text. */
    val promptEditor: StateFlow<PromptEditor?> = _promptEditor.asStateFlow()

    fun selectCategory(c: SettingsCategory) {
        closeSub()
        _category.value = c
    }

    fun openPrompt(): Boolean {
        val p = profile.value ?: return false
        _promptEditor.value?.close()
        _promptEditor.value = PromptEditor(p, deps.prompts, deps.settings, deps.tryCleanup, scope, deps.settings.settings.value)
        _category.value = SettingsCategory.CLEANUP
        _sub.value = SubScreen.PROMPT
        return true
    }

    fun openTones() {
        closeSub()
        _category.value = SettingsCategory.CLEANUP
        _sub.value = SubScreen.TONE
    }

    fun closeSub() {
        _promptEditor.value?.close()
        _promptEditor.value = null
        _sub.value = null
    }

    override fun onCleared() {
        _promptEditor.value?.close()
    }

    // ------------------------------------------------------------ History and privacy

    fun setKeepRecordings(on: Boolean) = write { it.copy(keepRecordings = on) }

    /**
     * Retention sheet choice. Lengthening is written at once. Shortening (to fewer days, or to "Don't keep history" =
     * 0) first counts the dictations it deletes (DESIGN 6.5; with 0 every entry, today's included,
     * `RetentionPolicy.firstKeptDay`): none -> applied at once; some -> [retentionConfirm] opens the dialog.
     */
    fun chooseRetention(days: Int) {
        scope.launch {
            guarded("retention_choose", SettingsEvent.NOT_SAVED) {
                val current = deps.settings.current().retentionDays
                if (days == current) return@guarded
                if (!shortens(current, days)) {
                    deps.settings.update { it.copy(retentionDays = days) }
                    return@guarded
                }
                val firstKept = RetentionPolicy.firstKeptDay(today(), days)
                val count = withContext(io) { deps.history.entries().first().count { it.localDay.isBefore(firstKept) } }
                if (count == 0) applyRetention(days) else _retentionConfirm.value = RetentionConfirm(days, count)
            }
        }
    }

    /**
     * The confirm dialog's Delete: the period is stored, then WP-12's immediate retention run deletes the entries.
     * The user confirmed the period, so it stays stored when the run fails; WP-12's start-up and midnight runs delete
     * by the stored period, and the snackbar says so ([SettingsEvent.DELETION_DEFERRED]).
     */
    fun confirmRetention() {
        val c = _retentionConfirm.value ?: return
        _retentionConfirm.value = null
        scope.launch { applyRetention(c.days) }
    }

    fun cancelRetention() {
        _retentionConfirm.value = null
    }

    private suspend fun applyRetention(days: Int) {
        guarded("retention_store", SettingsEvent.NOT_SAVED) {
            deps.settings.update { it.copy(retentionDays = days) }
            guarded("retention_run", SettingsEvent.DELETION_DEFERRED) { withContext(io) { deps.retention(days) } }
        }
    }

    /** "Delete all history": counts first; nothing to delete -> a snackbar, else the confirm dialog. */
    fun askDeleteAll() {
        scope.launch {
            guarded("history_count", SettingsEvent.DELETE_FAILED) {
                val n = withContext(io) { deps.history.entries().first().size }
                if (n == 0) _events.value = SettingsEvent.NOTHING_TO_DELETE else _deleteAllCount.value = n
            }
        }
    }

    fun confirmDeleteAll() {
        if (_deleteAllCount.value == null) return
        _deleteAllCount.value = null
        scope.launch {
            guarded("history_delete_all", SettingsEvent.DELETE_FAILED) {
                withContext(io) { deps.history.deleteAll() }
                _events.value = SettingsEvent.HISTORY_DELETED
            }
        }
    }

    fun cancelDeleteAll() {
        _deleteAllCount.value = null
    }

    // ------------------------------------------------------------ Permissions and debug

    fun refreshPermissions() = deps.permissions.refresh()

    fun setDebugFakeEngines(on: Boolean) = write { it.copy(debugFakeEngines = on) }
    fun setDebugTimeoutScale(scale: Float) = write { it.copy(debugTimeoutScale = scale) }
    fun setDebugIgnoreMemoryGate(on: Boolean) = write { it.copy(debugIgnoreMemoryGate = on) }

    fun eventShown() {
        _events.value = null
    }

    companion object {
        /** The dictation languages Settings offers (ru + en are first-class, BRIEF). */
        val LANGUAGES: List<String> = listOf("ru", "en")
        val WPM_PRESETS: List<Int> = listOf(30, 40, 50, 60)
        const val MIN_WPM = 10
        const val MAX_WPM = 200
        val TIMEOUT_SCALES: List<Float> = listOf(1f, 2f, 4f, 8f, 16f)

        /** 0 ("Don't keep history") is the shortest period. */
        fun shortens(from: Int, to: Int): Boolean = rank(to) < rank(from)
        private fun rank(days: Int): Int = if (days <= 0) 0 else days

        fun activeProfileName(s: Settings, c: Catalog): String {
            val chosen = when (val choice = s.cleanup) {
                is CleanupChoice.Model -> c.model(choice.id)?.llm?.promptProfile
                CleanupChoice.Default, CleanupChoice.None -> null
            }
            return chosen ?: c.defaults.llm?.let { c.model(it)?.llm?.promptProfile } ?: GgufTemplates.STOCK_PROFILE
        }
    }
}

/** Snackbar messages of the Settings screen. */
enum class SettingsEvent {
    POSITION_RESET,
    HISTORY_DELETED,
    NOTHING_TO_DELETE,
    /** A settings write failed (storage); the setting keeps its old value. */
    NOT_SAVED,
    /** "Delete all history" (or its count) failed; nothing was deleted. */
    DELETE_FAILED,
    /** A shorter period is stored but the immediate deletion failed; the start-up and midnight runs delete by it. */
    DELETION_DEFERRED,
}

/** D9: codes and the place only, never text, packages or settings values. */
internal object SettingsLog {
    private const val TAG = "feature.settings"

    fun code(code: String, where: String) {
        android.util.Log.w(TAG, "$code at=$where")
    }
}

/** DESIGN 6.5 inner display: the 260 dp category list. [APP] exists when an updater is wired; [DEBUG] in debug builds only. */
enum class SettingsCategory { DICTATION, BUBBLE, CLEANUP, PRIVACY, PERMISSIONS, APP, DEBUG }

/** Screens below Settings: the prompt editor and Tone per app. */
enum class SubScreen { PROMPT, TONE }
