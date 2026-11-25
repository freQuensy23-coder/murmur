package app.murmur.feature.onboarding

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import app.murmur.core.catalog.Catalog
import app.murmur.core.catalog.DeviceProfile
import app.murmur.core.dictation.DictationBubble
import app.murmur.core.dictation.InAppTarget
import app.murmur.core.model.ArtifactId
import app.murmur.core.model.Language
import app.murmur.core.store.InstallState
import app.murmur.core.store.RetentionPolicy
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.updateAndGet
import kotlinx.coroutines.launch

/** The models part as the screens need it: the pack choice, its sizes, and the downloads of the chosen pack. */
data class ModelsUi(
    /** null while DeviceProfileRepository.ensure() and the catalog load run (sizes depend on the device). */
    val options: List<PackOption>?,
    val selectedId: String?,
    val wifiOnly: Boolean,
    val freeBytes: Long?,
    /** The pack whose downloads are shown (the chosen one, or one started in an earlier run). */
    val active: PackOption?,
    val summary: DownloadSummary?,
    val npu: Boolean,
    val languages: List<String> = listOf("ru", "en"),
) {
    val selected: PackOption? get() = options?.firstOrNull { it.pack.id == selectedId }
}

/** Result line under the practice field after a dictation (DESIGN 6.1 ob-ready). */
data class PracticeResult(val seconds: Double, val asr: String, val llm: String?)

/**
 * Onboarding logic around [OnboardingMachine] (survives fold/unfold as a ViewModel; [scope] is injectable for tests).
 * Feeds the machine from [OnboardingDeps.permissions] and from the install states, and performs the models step:
 * pack sizes only after `device.ensure()`, the pack's settings, the downloads, pause and retry.
 *
 * The progress the user made ([OnboardingSaved]: step, skipped steps, microphone denials, the pending trip to the
 * accessibility settings, the chosen pack) is kept in [saved], the ViewModel's SavedStateHandle: Android may kill the
 * process while the user is on a system screen, and setup then continues where it was instead of at Welcome.
 * Everything else is read again (permissions, install states).
 */
class OnboardingController(
    private val deps: OnboardingDeps,
    private val scope: CoroutineScope,
    private val saved: SavedStateHandle = SavedStateHandle(),
) {
    private val _state = MutableStateFlow(OnboardingSaved.restore(saved))
    val state: StateFlow<OnboardingState> = _state.asStateFlow()

    private val device = MutableStateFlow<DeviceProfile?>(null)
    private val catalog = MutableStateFlow<Catalog?>(null)
    private val chosen = MutableStateFlow(saved.get<String>(OnboardingSaved.CHOSEN))
    private val freeBytes = MutableStateFlow<Long?>(null)
    /** Pack id whose download was started in this run. */
    private val started = MutableStateFlow<String?>(null)

    private val options: StateFlow<List<PackOption>?> = combine(device, catalog, deps.settings.settings, deps.installed.states) { d, c, s, st ->
        if (d == null || c == null) null else PackPlanner.options(c, d, s, st)
    }.stateIn(scope, SharingStarted.Eagerly, null)

    val models: StateFlow<ModelsUi> = combine(
        combine(options, chosen, started) { o, ch, st -> Triple(o, ch, st) },
        deps.installed.states, deps.settings.settings, freeBytes, device,
    ) { (opts, ch, st), states, settings, free, dev ->
        val active = activeOption(opts, st, states)
        val c = catalog.value
        val validChosen = ch?.takeIf { id -> opts?.any { it.pack.id == id } == true }
        ModelsUi(
            options = opts,
            selectedId = validChosen ?: opts?.let { PackPlanner.defaultChoice(it)?.pack?.id },
            wifiOnly = settings.wifiOnlyDownloads,
            freeBytes = free,
            active = active,
            summary = if (active != null && c != null) DownloadSummary.of(c, active, states) else null,
            npu = dev?.hasNpu() == true,
            languages = settings.languages,
        )
    }.stateIn(scope, SharingStarted.Eagerly, ModelsUi(null, null, true, null, null, null, false))

    /** Days history is kept (welcome and microphone copy follow the setting; 0 = nothing is kept). */
    val retentionDays: StateFlow<Int> = deps.settings.settings.map { it.retentionDays }.distinctUntilChanged()
        .stateIn(scope, SharingStarted.Eagerly, RetentionPolicy.DEFAULT_DAYS)

    private val _practice = MutableStateFlow<PracticeResult?>(null)
    val practice: StateFlow<PracticeResult?> = _practice.asStateFlow()

    init {
        scope.launch { deps.permissions.state.collect { dispatch(OnboardingEvent.Permissions(it)) } }
        scope.launch {
            // ARCHITECTURE 8.5: pack sizes are the artifacts this phone resolves to, so wait for the probe.
            catalog.value = deps.catalog.loaded()
            device.value = deps.device.ensure()
            freeBytes.value = deps.installed.storage().freeBytes
        }
        scope.launch { deps.catalog.catalog.collect { if (catalog.value != null) catalog.value = it } }
        scope.launch {
            models.map { it.summary?.let { s -> s.phase to s.speechReady } }.distinctUntilChanged().collect { p ->
                if (p != null) dispatch(OnboardingEvent.Downloads(p.first, p.second))
            }
        }
        scope.launch { watchPractice() }
    }

    fun dispatch(e: OnboardingEvent) {
        var becameFinished = false
        val next = _state.updateAndGet { prev ->
            val reduced = OnboardingMachine.reduce(prev, e)
            becameFinished = !prev.finished && reduced.finished
            reduced
        }
        OnboardingSaved.save(saved, next)
        if (becameFinished) {
            scope.launch { deps.settings.update { it.copy(onboardingDone = true) } }
        }
    }

    fun select(packId: String) {
        chosen.value = packId
        saved[OnboardingSaved.CHOSEN] = packId
    }

    fun setLanguage(code: String, on: Boolean) {
        scope.launch {
            deps.settings.update { s ->
                val next = if (on) (s.languages + code).distinct() else s.languages - code
                if (next.isEmpty()) s else s.copy(languages = LANGUAGES.filter { it in next } + next.filter { it !in LANGUAGES })
            }
        }
    }

    fun setWifiOnly(on: Boolean) { scope.launch { deps.settings.update { it.copy(wifiOnlyDownloads = on) } } }

    /** "Download 1.7 GB": the pack's models become the active ones, then every artifact not installed is queued. */
    fun download() {
        val m = models.value
        val option = m.selected ?: return
        scope.launch {
            deps.settings.update { PackPlanner.applyChoice(it, option) }
            started.value = option.pack.id
            val states = deps.installed.states.value
            option.artifacts.filter { states[it] !is InstallState.Installed }.forEach { deps.installed.download(it, m.wifiOnly) }
            dispatch(OnboardingEvent.DownloadStarted)
            freeBytes.value = deps.installed.storage().freeBytes
        }
    }

    /** "Pause": every artifact of the pack that is queued or downloading. */
    fun pause() {
        val rows = models.value.summary?.rows ?: return
        scope.launch { rows.filter { it.active }.forEach { deps.installed.pause(it.artifact) } }
    }

    /** "Try again": re-queue what stopped (the job resumes from the saved part, ARCHITECTURE 8.5). */
    fun retry() {
        val m = models.value
        val rows = m.summary?.rows ?: return
        scope.launch {
            rows.filter { it.state is InstallState.Paused || it.state is InstallState.Failed || it.state is InstallState.NotInstalled }
                .forEach { deps.installed.download(it.artifact, m.wifiOnly) }
            dispatch(OnboardingEvent.RetryDownload)
            freeBytes.value = deps.installed.storage().freeBytes
        }
    }

    /** Finishes onboarding and persists Settings.onboardingDone = true. */
    fun finish(onDone: () -> Unit) {
        _state.updateAndGet { it.copy(finished = true) }
        scope.launch {
            deps.settings.update { it.copy(onboardingDone = true) }
            onDone()
        }
    }

    /**
     * True from [startPractice] until that dictation ends. The bubble flow is the orchestrator's, shared by every
     * dictation (the overlay in other apps included), so only a dictation this screen started reports a result.
     */
    @Volatile private var armed = false

    /** Forgets the last practice result (the practice screen was left); a dictation still running reports nothing. */
    fun clearPractice() {
        armed = false
        _practice.value = null
    }

    fun startPractice(target: InAppTarget) {
        _practice.value = null
        armed = true
        deps.practice(target)
    }

    /**
     * The "Inserted in 1.3 s" line: from the stop (the end of Listening) to Inserted, with the models of the settings
     * (catalog display names; "→ cleanup" only when cleanup ran). Only for the dictation [startPractice] armed: one
     * that ends any other way (error, copied, cancelled back to Idle) disarms it.
     */
    private suspend fun watchPractice() {
        var stoppedAt: Long? = null
        var listening = false
        deps.bubble.collect { b ->
            if (!armed) {
                stoppedAt = null
                listening = false
                return@collect
            }
            when (b) {
                is DictationBubble.Listening -> {
                    listening = b.frozenAtMs == null
                    stoppedAt = b.frozenAtMs
                }
                DictationBubble.Transcribing, DictationBubble.Polishing -> {
                    if (listening || stoppedAt == null) stoppedAt = deps.clock.elapsedMs()
                    listening = false
                }
                is DictationBubble.Inserted -> {
                    val from = stoppedAt ?: deps.clock.elapsedMs()
                    val secs = (deps.clock.elapsedMs() - from).coerceAtLeast(0) / 1000.0
                    _practice.value = PracticeResult(secs, asrName(), if (b.noCleanup) null else llmName())
                    armed = false
                    stoppedAt = null
                    listening = false
                }
                is DictationBubble.Error, is DictationBubble.Copied -> {
                    armed = false
                    stoppedAt = null
                    listening = false
                }
                // Idle right after the start (the bubble flow still shows the previous state) keeps the arm; Idle
                // after the dictation has run means it was cancelled.
                DictationBubble.Idle -> {
                    if (listening || stoppedAt != null) armed = false
                    stoppedAt = null
                    listening = false
                }
            }
        }
    }

    private fun currentCatalog(): Catalog? = catalog.value ?: deps.catalog.catalog.value

    private fun asrName(): String {
        val c = currentCatalog() ?: return ""
        val s = deps.settings.settings.value
        val id = s.asrModel ?: c.defaults.asr[languageProfile(s.languages)] ?: c.defaults.asr.values.firstOrNull()
        return id?.let { c.model(it)?.displayName ?: it.value } ?: ""
    }

    private fun llmName(): String? {
        val c = currentCatalog() ?: return null
        val id = when (val ch = deps.settings.settings.value.cleanup) {
            app.murmur.core.store.CleanupChoice.Default -> c.defaults.llm
            app.murmur.core.store.CleanupChoice.None -> null
            is app.murmur.core.store.CleanupChoice.Model -> ch.id
        } ?: return null
        return c.model(id)?.displayName ?: id.value
    }

    companion object {
        val LANGUAGES: List<String> = listOf("ru", "en", "he")

        /** CatalogDefaults.asr keys: "ru+en", "ru", "en", "he". */
        fun languageProfile(languages: List<String>): String {
            val codes = languages.mapTo(HashSet()) { Language.normalize(it) }
            val ru = "ru" in codes
            val en = "en" in codes
            val he = "he" in codes
            return when {
                he && !ru && !en -> "he"
                ru && !en -> "ru"
                en && !ru -> "en"
                else -> "ru+en"
            }
        }

        /**
         * The pack whose downloads the screens follow: the one started in this run, else one with work on the way or
         * done from an earlier run (so a relaunch continues instead of offering the choice again).
         */
        internal fun activeOption(options: List<PackOption>?, startedId: String?, states: Map<ArtifactId, InstallState>): PackOption? {
            if (options == null) return null
            options.firstOrNull { it.pack.id == startedId }?.let { return it }
            fun touched(o: PackOption): Boolean {
                val sts = o.artifacts.map { states[it] ?: InstallState.NotInstalled }
                val allInstalled = sts.all { it is InstallState.Installed }
                val moving = sts.any { it !is InstallState.Installed && it !is InstallState.NotInstalled }
                return allInstalled || moving
            }
            return options.firstOrNull { o -> o.artifacts.all { states[it] is InstallState.Installed } }
                ?: options.firstOrNull(::touched)
        }
    }
}

/**
 * ViewModel holder so that the flow survives fold/unfold (the Activity is recreated on a display change) and, through
 * [saved], process death while the user is on a system screen.
 */
class OnboardingViewModel(deps: OnboardingDeps, saved: SavedStateHandle) : ViewModel() {
    val controller = OnboardingController(deps, viewModelScope, saved)
}

/**
 * The part of [OnboardingState] that survives process death (plain bundle values in a SavedStateHandle). Permissions
 * and downloads are not kept: they are read again, and the machine moves on from the restored step by itself.
 */
internal object OnboardingSaved {
    const val STEP = "ob.step"
    const val SKIPPED = "ob.skipped"
    const val MIC_DENIALS = "ob.micDenials"
    const val MIC_BLOCKED = "ob.micBlocked"
    const val AWAITING_A11Y = "ob.awaitingA11yReturn"
    const val CHOSEN = "ob.chosenPack"

    fun save(h: SavedStateHandle, s: OnboardingState) {
        h[STEP] = s.step.name
        h[SKIPPED] = ArrayList(s.skipped.map { it.name })
        h[MIC_DENIALS] = s.micDenials
        h[MIC_BLOCKED] = s.micBlocked
        h[AWAITING_A11Y] = s.awaitingA11yReturn
    }

    fun restore(h: SavedStateHandle): OnboardingState {
        fun step(name: String?) = OnboardingStep.entries.firstOrNull { it.name == name }
        return OnboardingState(
            step = step(h[STEP]) ?: OnboardingStep.WELCOME,
            skipped = h.get<ArrayList<String>>(SKIPPED).orEmpty().mapNotNull(::step).toSet(),
            micDenials = h[MIC_DENIALS] ?: 0,
            micBlocked = h[MIC_BLOCKED] ?: false,
            awaitingA11yReturn = h[AWAITING_A11Y] ?: false,
        )
    }
}
