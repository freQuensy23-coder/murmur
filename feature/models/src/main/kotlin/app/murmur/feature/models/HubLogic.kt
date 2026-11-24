package app.murmur.feature.models

import app.murmur.core.benchmark.BenchScores
import app.murmur.core.catalog.Artifact
import app.murmur.core.catalog.ArtifactSelector
import app.murmur.core.catalog.Blocker
import app.murmur.core.catalog.Candidate
import app.murmur.core.catalog.Catalog
import app.murmur.core.catalog.DeviceProfile
import app.murmur.core.catalog.DownloadPlanner
import app.murmur.core.catalog.FileSource
import app.murmur.core.catalog.ModelEntry
import app.murmur.core.catalog.PlannedDownload
import app.murmur.core.catalog.Status
import app.murmur.core.catalog.UnitHealth
import app.murmur.core.model.ArtifactId
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.Language
import app.murmur.core.model.ModelId
import app.murmur.core.model.ModelKind
import app.murmur.core.model.MurmurError
import app.murmur.core.model.UnitPreference
import app.murmur.core.store.BenchmarkRecord
import app.murmur.core.store.CleanupChoice
import app.murmur.core.store.InstallState
import app.murmur.core.store.Settings
import app.murmur.core.store.StorageInfo

/** The two lists of the hub (DESIGN 6.4 segmented control). */
enum class HubTab { SPEECH, CLEANUP }

/**
 * Everything the hub decides with, as one immutable snapshot. [states] holds the install state of every catalog
 * artifact (missing = not installed); [records] the newest benchmark record per unit of every artifact
 * (`BenchmarkRepository.latest`). [device] is null until the device profile is known.
 */
data class HubInput(
    val catalog: Catalog,
    val states: Map<ArtifactId, InstallState>,
    val settings: Settings,
    val device: DeviceProfile?,
    val health: UnitHealth,
    val records: Map<ArtifactId, List<BenchmarkRecord>>,
    val storage: StorageInfo?,
) {
    fun state(id: ArtifactId): InstallState = states[id] ?: InstallState.NotInstalled
    fun installed(id: ArtifactId): Boolean = state(id) is InstallState.Installed
}

/**
 * How one model runs on this phone (D2, ARCHITECTURE 8.2), the same rules as the resolver (WP-13):
 *  - [candidates]: every (artifact x unit) with its blocker ([ArtifactSelector.candidates]);
 *  - [chosen]: among installed artifacts, [ArtifactSelector.ranked] with the model's preference, falling back to Auto
 *    when a pinned unit has nothing installed (what a dictation loads now); null when nothing usable is installed;
 *  - [planned]: what a download would fetch ([DownloadPlanner.artifactFor] rules) when nothing usable is installed;
 *  - [pinnedMissing]: the artifact of a pinned unit that is not installed ("Download (N MB)").
 */
data class UnitView(
    val candidates: List<Candidate>,
    val chosen: Candidate?,
    val planned: Candidate?,
    val preference: UnitPreference,
    val pinnedMissing: Artifact?,
    /** The candidate shown as "this is how it runs": [chosen], else [planned]. */
    val shown: Candidate?,
) {
    /** Usable units in NPU > GPU > CPU order, each once (unit picker rows). */
    val usableUnits: List<ComputeUnit> get() = UNIT_ORDER.filter { u -> candidates.any { it.usable && it.unit == u } }

    /** Units the catalog claims for any artifact of the model, NPU > GPU > CPU. */
    val claimedUnits: List<ComputeUnit> get() = UNIT_ORDER.filter { u -> candidates.any { it.unit == u } }

    companion object {
        val UNIT_ORDER = listOf(ComputeUnit.NPU, ComputeUnit.GPU, ComputeUnit.CPU)
    }
}

/** Right side and extra line of a model row (DESIGN 6.4 state table). */
sealed interface RowState {
    /** Outline pill "⤓ 485 MB". */
    data class NotDownloaded(val bytes: Long) : RowState
    data class Queued(val bytesTotal: Long) : RowState
    /** Tonal pause button, 4 dp progress, "89 MB of 144 MB · 12.3 MB/s". */
    data class Downloading(val bytesDone: Long, val bytesTotal: Long, val bytesPerSec: Long) : RowState
    /** Tonal resume button, ink4 progress, warn line. */
    data class Paused(val bytesDone: Long, val bytesTotal: Long, val reason: MurmurError) : RowState
    /** Download failed (checksum: "Download was damaged…" + Download again). */
    data class Failed(val error: MurmurError, val bytes: Long) : RowState
    /** Verifying / extracting: the ring without a number. */
    data object Checking : RowState
    /** 36 dp ring with %; "Preparing for NPU…". */
    data class Preparing(val progress: Float, val unit: ComputeUnit?) : RowState
    /** Downloaded, not in use: radio. */
    data object Ready : RowState
    /** Accent "✓ In use" pill, row filled accentSoft. */
    data object InUse : RowState
    /** status in_training / planned: "In training" badge, row at 62 % opacity, not tappable for download. */
    data object InTraining : RowState
    /** Nothing in the catalog can run on this phone (every candidate blocked). */
    data class Unavailable(val blocker: Blocker?, val socs: List<String>) : RowState
    /**
     * The device profile is not known yet: which artifact runs here (and so whether the model is downloaded, in use or
     * blocked) cannot be decided. Skeleton instead of a control; nothing can be downloaded, selected or deleted.
     */
    data object Loading : RowState
}

/** Unit badge on a row: [chosen] = the unit the model runs on (filled); others are claimed and usable (outlined). */
data class UnitBadge(val unit: ComputeUnit, val chosen: Boolean)

/** One model row (DESIGN 6.4). Everything comes from the catalog, the install state and the benchmark records. */
data class ModelRow(
    val id: ModelId,
    val kind: ModelKind,
    val name: String,
    val publisher: String,
    /** Quantisation of the artifact the row describes (the "variant" after the publisher), or null. */
    val variant: String?,
    val params: String?,
    /** Download size of the shown artifact (null when the model has no artifact yet). */
    val sizeBytes: Long?,
    val streaming: Boolean,
    val units: List<UnitBadge>,
    /** Language badges: "RU", "EN", "HE", then "+N" for the rest (DESIGN 6.4). */
    val languages: List<String>,
    val noRussian: Boolean,
    /** Selected dictation languages (from Settings.languages) missing from this model's card. */
    val missingLanguages: List<String> = emptyList(),
    val previewOnly: Boolean,
    val state: RowState,
    /** The model is the selected one (speech model of record / cleanup choice) even if it is not downloaded. */
    val selected: Boolean,
    /** Can be made the speech model / cleanup model (installed, available, not preview-only). */
    val selectable: Boolean,
    val custom: Boolean,
    /** Newest valid OCR / VAD figure for support rows: ms per screenshot (OCR), null otherwise. */
    val msPerScreenshot: Long?,
)

/** A fact-table row of the detail (DESIGN 6.4). */
sealed interface Fact {
    data class Parameters(val label: String) : Fact
    data class Languages(val codes: List<String>, val count: Int) : Fact
    data class Download(val bytes: Long, val quant: String?) : Fact
    data class RunsOn(val lines: List<RunsOnLine>) : Fact
    data class Streaming(val streaming: Boolean) : Fact
    data class License(val id: String) : Fact
    /** [label] = HF repo id or "owner/repo · tag" of a GitHub release; [url] opens in the browser. */
    data class Source(val label: String, val url: String, val github: Boolean) : Fact
}

/** One line of "Runs on": a unit and why it cannot be used, if it cannot. */
data class RunsOnLine(
    val unit: ComputeUnit,
    val blocker: Blocker?,
    val chosen: Boolean,
    val socs: List<String> = emptyList(),
    /** The NPU is a Snapdragon's Hexagon (this phone's SoC, or the SoCs the artifact is built for): "NPU (Hexagon)". */
    val hexagon: Boolean = false,
)

/** Warn lines under the facts (ARCHITECTURE 8.2 labels). */
sealed interface DetailNote {
    /** Blocker.UNIT_FAILED on [failed] while [using] runs: "Failed to load on NPU, using CPU". */
    data class UnitFailed(val failed: ComputeUnit, val using: ComputeUnit?) : DetailNote
    /** The newest record of [unit] is invalid: "Results look wrong on NPU". */
    data class ResultsWrong(val unit: ComputeUnit) : DetailNote
    /** followUps npu-port and no NPU candidate: "NPU not supported yet". */
    data object NpuNotYet : DetailNote
    /** ADR-047: "Preview only, not selectable yet". */
    data object PreviewOnly : DetailNote
}

/** Primary action of the detail's action bar. */
sealed interface DetailAction {
    data object InUse : DetailAction
    data object Use : DetailAction
    /** [artifact] = the artifact the detail shows (what a dictation would load once it is downloaded). */
    data class Download(val bytes: Long, val artifact: Artifact? = null) : DetailAction
    /** Download in progress / paused / failed: the row's control lives in the detail too. */
    data class InProgress(val state: RowState) : DetailAction
    /** Nothing to do (in training, unavailable, preview-only installed, support models). */
    data object None : DetailAction
    /** The device profile is not known yet (RowState.Loading): a skeleton in place of the action. */
    data object Loading : DetailAction
}

/**
 * A second artifact of the model being fetched while the row shows the one that runs (DESIGN 6.4 detail): the pinned
 * unit's artifact ([Purpose.PINNED], under the unit picker) or the CPU version after an NPU failure
 * ([Purpose.CPU_VERSION], under the failure line). [state] is its progress (never NotDownloaded, Ready or In use).
 */
data class ExtraDownload(val artifact: Artifact, val unit: ComputeUnit, val state: RowState, val purpose: Purpose) {
    enum class Purpose { PINNED, CPU_VERSION }
}

/** Unit picker row (no frame; radio rows from WP-01). [unit] null = Auto. */
data class UnitOption(val unit: ComputeUnit?, val selected: Boolean, val installed: Boolean, val bytes: Long?)

/** The benchmark block of the detail: the record of the shown (artifact, unit), else the newest one. */
data class BenchBlock(val record: BenchmarkRecord?, val canRun: Boolean)

data class ModelDetailView(
    val row: ModelRow,
    val description: String?,
    val quant: String?,
    val chosenUnit: ComputeUnit?,
    val facts: List<Fact>,
    val notes: List<DetailNote>,
    val unitOptions: List<UnitOption>,
    /** Pinned unit not installed (and not being fetched): "Download (N MB)" under the picker. */
    val pinnedDownload: Artifact?,
    /** An NPU artifact failed and the CPU artifact is not installed nor being fetched: "Download the CPU version (N MB)". */
    val cpuFallbackDownload: Artifact?,
    /** The pinned or CPU-version artifact while it is queued, downloading, paused, failed or being checked. */
    val extraDownload: ExtraDownload?,
    val action: DetailAction,
    val canDelete: Boolean,
    val inUse: Boolean,
    val bench: BenchBlock,
    val sourceUrl: String?,
)

/** Storage banner (`models-storage`): [model] could not be stored, [neededBytes] vs [freeBytes]. */
data class StorageProblem(val model: String, val kind: ModelKind, val neededBytes: Long, val freeBytes: Long)

/** The list of one tab. */
data class HubList(
    val tab: HubTab,
    val rows: List<ModelRow>,
    /** Cleanup only: "None" is selected (CleanupChoice.None). */
    val noneSelected: Boolean,
    /** Speech only: VAD and OCR rows ("Support models", no frame). */
    val support: List<ModelRow>,
    val storage: StorageInfo?,
    val storageProblem: StorageProblem?,
)

/** Pure decisions of the model hub. No model-specific code (D3): every value is read from the catalog entry. */
object HubLogic {

    private val SUPPORT_KINDS = setOf(ModelKind.VAD, ModelKind.OCR_DET, ModelKind.OCR_REC)

    fun preference(settings: Settings, id: ModelId): UnitPreference = settings.unitPreference[id.value] ?: UnitPreference.Auto

    /** BenchLookup over every record of the entry's artifacts (newest per (artifact, unit) decides, ADR-071). */
    private fun lookup(entry: ModelEntry, input: HubInput) =
        BenchScores.lookup(entry.artifacts.flatMap { input.records[it.id].orEmpty() })

    fun units(entry: ModelEntry, input: HubInput): UnitView {
        val pref = preference(input.settings, entry.id)
        val device = input.device ?: return UnitView(emptyList(), null, null, pref, null, null)
        val candidates = ArtifactSelector.candidates(entry, device, input.health)
        val bench = lookup(entry, input)
        val installed = entry.artifacts.map { it.id }.filter(input::installed).toSet()
        val chosen = if (installed.isEmpty()) null else
            ArtifactSelector.choose(entry, device, pref, bench, installed, input.health)
                ?: if (pref is UnitPreference.Pinned) ArtifactSelector.choose(entry, device, UnitPreference.Auto, bench, installed, input.health) else null
        val planned = if (chosen != null) null else
            ArtifactSelector.choose(entry, device, pref, bench, null, input.health)
                ?: ArtifactSelector.choose(entry, device, UnitPreference.Auto, bench, null, input.health)
        val pinnedMissing = (pref as? UnitPreference.Pinned)?.let { p ->
            val usable = candidates.filter { it.usable && it.unit == p.unit }
            if (usable.isEmpty() || usable.any { it.artifact.id in installed }) null
            else ArtifactSelector.ranked(entry, device, p, bench, null, input.health).firstOrNull { it.unit == p.unit }?.artifact
        }
        return UnitView(candidates, chosen, planned, pref, pinnedMissing, chosen ?: planned)
    }

    /** The speech model of record: Settings.asrModel when selectable, else the catalog default of the language profile. */
    fun speechModel(input: HubInput): ModelId? {
        val c = input.catalog
        input.settings.asrModel?.let { id -> c.model(id)?.takeIf { it.selectableAsAsr }?.let { return it.id } }
        val id = c.defaults.asr[languageProfile(input.settings.languages)] ?: c.defaults.asr.values.firstOrNull() ?: return null
        return c.model(id)?.takeIf { it.selectableAsAsr }?.id
    }

    /** The cleanup model of record, or null for None (CleanupChoice.Default = catalog defaults.llm). */
    fun cleanupModel(input: HubInput): ModelId? = when (val c = input.settings.cleanup) {
        CleanupChoice.None -> null
        CleanupChoice.Default -> input.catalog.defaults.llm
        is CleanupChoice.Model -> c.id
    }

    /** Settings.languages -> catalog defaults.asr key (the resolver's rule, ARCHITECTURE 8.1). */
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

    fun isSelected(entry: ModelEntry, input: HubInput): Boolean = when (entry.kind) {
        ModelKind.ASR -> speechModel(input) == entry.id
        ModelKind.LLM -> cleanupModel(input) == entry.id
        else -> false
    }

    private fun available(entry: ModelEntry) = entry.status == Status.AVAILABLE && entry.artifacts.isNotEmpty()

    /** Can be the speech model / cleanup model of record (not counting installation). */
    fun selectableKind(entry: ModelEntry): Boolean = when (entry.kind) {
        ModelKind.ASR -> entry.selectableAsAsr && entry.artifacts.isNotEmpty()
        ModelKind.LLM -> available(entry) && entry.llm != null
        else -> false
    }

    /** Parameter label for the spec line and the fact table: "938M", "1.73B", "~222M" when the card gives an estimate. */
    fun paramsLabel(entry: ModelEntry): String? {
        val p = entry.params ?: return null
        val approx = entry.paramsLabel?.trimStart()?.startsWith("~") == true
        return (if (approx) "~" else "") + Format.params(p)
    }

    /** Language badges: RU, EN, and HE when listed, else the first code; "+N" for the rest of the card's count. */
    fun languageBadges(entry: ModelEntry): List<String> {
        val codes = entry.languages.codes
        if (codes.isEmpty()) return emptyList()
        val firstClass = listOf("ru", "en", "he").filter { it in codes }
        val shown = firstClass.ifEmpty { listOf(codes.first()) }
        val total = maxOf(entry.languages.count, codes.size)
        val rest = total - shown.size
        return shown.map { it.uppercase() } + if (rest > 0) listOf("+$rest") else emptyList()
    }

    private fun rowState(entry: ModelEntry, input: HubInput, units: UnitView, selected: Boolean): RowState {
        if (!available(entry)) return RowState.InTraining
        if (input.device == null) return RowState.Loading
        val shown = units.shown
        if (input.device != null && shown == null) {
            // Every candidate is blocked on this phone: show the most telling blocker (the first artifact's).
            val first = units.candidates.firstOrNull()
            return RowState.Unavailable(first?.blocker, first?.artifact?.socs.orEmpty())
        }
        // An artifact being fetched or prepared wins over "installed": the user sees its progress.
        val active = rowActive(entry, input, units)
        if (active != null) return progressState(active, input.state(active.id), units)
        if (units.chosen != null) {
            return if (selected && selectableKind(entry)) RowState.InUse else RowState.Ready
        }
        val bytes = (shown?.artifact ?: entry.artifacts.first()).downloadBytes
        return RowState.NotDownloaded(bytes)
    }

    private fun inFlight(s: InstallState) = s !is InstallState.NotInstalled && s !is InstallState.Installed

    /**
     * The artifact whose download the row (and the detail's action bar) shows: the first one in flight while nothing
     * usable is installed. Once a model runs, the row shows that ("In use" / radio) and other artifacts in flight are
     * the detail's [ExtraDownload].
     */
    private fun rowActive(entry: ModelEntry, input: HubInput, units: UnitView): Artifact? =
        if (units.chosen != null) null else entry.artifacts.firstOrNull { inFlight(input.state(it.id)) }

    private fun progressState(a: Artifact, s: InstallState, units: UnitView): RowState = when (s) {
        is InstallState.Queued -> RowState.Queued(a.downloadBytes)
        is InstallState.Downloading -> RowState.Downloading(s.bytesDone, s.bytesTotal, s.bytesPerSec)
        is InstallState.Paused -> RowState.Paused(s.bytesDone, s.bytesTotal, s.reason)
        is InstallState.Failed -> RowState.Failed(s.error, a.downloadBytes)
        is InstallState.Verifying, is InstallState.Extracting -> RowState.Checking
        is InstallState.Preparing -> RowState.Preparing(s.progress, units.candidates.firstOrNull { it.artifact.id == a.id && it.usable }?.unit)
        is InstallState.Installed -> RowState.Ready
        InstallState.NotInstalled -> RowState.NotDownloaded(a.downloadBytes)
    }

    fun row(entry: ModelEntry, input: HubInput): ModelRow {
        val units = units(entry, input)
        val selected = isSelected(entry, input)
        val state = rowState(entry, input, units, selected)
        val shown = units.shown
        // DESIGN 6.4 frames: one runtime badge, the unit the model runs on (filled). The other claimed units are
        // listed in the detail ("Runs on", unit picker). Nothing usable here: the claimed units, outlined.
        val badges = when {
            shown != null -> listOf(UnitBadge(shown.unit, true))
            else -> units.claimedUnits.ifEmpty { entry.artifacts.flatMap { it.units }.distinct() }.map { UnitBadge(it, false) }
        }
        val ocrMs = if (entry.kind == ModelKind.OCR_DET || entry.kind == ModelKind.OCR_REC) {
            entry.artifacts.flatMap { input.records[it.id].orEmpty() }
                .filter { it.result.valid }.maxByOrNull { it.atEpochMs }
                ?.let { (it.result as? app.murmur.core.store.BenchmarkResultData.Ocr)?.msPerScreenshot }
        } else null
        val missingLangs = if (entry.kind == ModelKind.ASR || entry.kind == ModelKind.LLM) {
            input.settings.languages.map(Language::normalize).distinct().filter { it !in entry.languages.codes }
        } else emptyList()
        return ModelRow(
            id = entry.id,
            kind = entry.kind,
            name = entry.displayName,
            publisher = entry.publisher,
            variant = (shown?.artifact ?: entry.artifacts.firstOrNull())?.quant,
            params = paramsLabel(entry),
            sizeBytes = (shown?.artifact ?: entry.artifacts.firstOrNull())?.downloadBytes,
            streaming = entry.asr?.streaming == true,
            units = badges,
            languages = languageBadges(entry),
            noRussian = entry.noRussian,
            missingLanguages = missingLangs,
            previewOnly = entry.asr?.previewOnly == true,
            state = state,
            selected = selected,
            selectable = selectableKind(entry) && units.chosen != null,
            custom = entry.origin == app.murmur.core.catalog.Origin.CUSTOM,
            msPerScreenshot = ocrMs,
        )
    }

    fun list(tab: HubTab, input: HubInput): HubList {
        val kind = if (tab == HubTab.SPEECH) ModelKind.ASR else ModelKind.LLM
        val entries = input.catalog.models.filter { it.kind == kind }
        // Catalog order, except: unavailable (in training) models go last, so the list starts with what can be used.
        val rows = entries.map { row(it, input) }.sortedBy { if (it.state is RowState.InTraining) 1 else 0 }
        val support = if (tab == HubTab.SPEECH) input.catalog.models.filter { it.kind in SUPPORT_KINDS }.map { row(it, input) } else emptyList()
        return HubList(
            tab = tab,
            rows = rows,
            noneSelected = tab == HubTab.CLEANUP && cleanupModel(input) == null,
            support = support,
            storage = input.storage,
            storageProblem = storageProblem(tab, input),
        )
    }

    /** The first artifact paused for lack of space whose model belongs to [tab] (support models count as Speech). */
    fun storageProblem(tab: HubTab, input: HubInput): StorageProblem? {
        for (m in input.catalog.models) {
            val onTab = when (m.kind) {
                ModelKind.LLM -> tab == HubTab.CLEANUP
                else -> tab == HubTab.SPEECH
            }
            if (!onTab) continue
            for (a in m.artifacts) {
                val s = input.state(a.id)
                if (s is InstallState.Paused && s.reason == MurmurError.LowStorage) {
                    return StorageProblem(m.displayName, m.kind, a.downloadBytes, input.storage?.freeBytes ?: 0)
                }
            }
        }
        return null
    }

    fun detail(entry: ModelEntry, input: HubInput): ModelDetailView {
        val units = units(entry, input)
        val row = row(entry, input)
        val shown = units.shown
        val facts = buildList {
            paramsLabel(entry)?.let { add(Fact.Parameters(it)) }
            if (entry.languages.codes.isNotEmpty()) add(Fact.Languages(entry.languages.codes, maxOf(entry.languages.count, entry.languages.codes.size)))
            (shown?.artifact ?: entry.artifacts.firstOrNull())?.let { add(Fact.Download(it.downloadBytes, it.quant)) }
            val lines = runsOn(entry, units, input.device)
            if (lines.isNotEmpty()) add(Fact.RunsOn(lines))
            entry.asr?.let { add(Fact.Streaming(it.streaming)) }
            add(Fact.License(entry.license.id))
            source(entry, shown?.artifact)?.let { add(it) }
        }
        val notes = buildList {
            if (entry.asr?.previewOnly == true) add(DetailNote.PreviewOnly)
            val failed = units.candidates.filter { it.blocker == Blocker.UNIT_FAILED }.map { it.unit }.distinct()
            // "using CPU" only when CPU really runs; with nothing installed the line names the failure alone.
            for (u in failed) add(DetailNote.UnitFailed(u, units.chosen?.unit))
            if (entry.followUps.any { it.id == NPU_PORT } && units.candidates.none { it.unit == ComputeUnit.NPU }) add(DetailNote.NpuNotYet)
            // ADR-071: the newest record of a unit decides; an invalid one shows "Results look wrong on <unit>".
            val newestPerUnit = entry.artifacts.flatMap { input.records[it.id].orEmpty() }
                .groupBy { it.artifactId to it.unit }.values.map { l -> l.maxBy { it.atEpochMs } }
            newestPerUnit.filter { !it.result.valid }.map { it.unit }.distinct()
                .sortedBy { VIEW_ORDER.indexOf(it) }.forEach { add(DetailNote.ResultsWrong(it)) }
        }
        val usable = units.usableUnits
        val options = if (usable.size < 2) emptyList() else buildList {
            add(UnitOption(null, units.preference is UnitPreference.Auto, units.chosen != null, null))
            for (u in usable) {
                val cands = units.candidates.filter { it.usable && it.unit == u }
                val inst = cands.firstOrNull { input.installed(it.artifact.id) }
                add(UnitOption(u, (units.preference as? UnitPreference.Pinned)?.unit == u, inst != null, (inst ?: cands.first()).artifact.downloadBytes))
            }
        }
        // The row / action bar already shows this artifact's download: no second control for it.
        val rowActive = rowActive(entry, input, units)?.id
        val npuFailed = units.candidates.any { it.unit == ComputeUnit.NPU && it.blocker == Blocker.UNIT_FAILED }
        val cpuArtifact = units.candidates.firstOrNull { it.unit == ComputeUnit.CPU && it.usable }?.artifact
            ?.takeIf { npuFailed && !input.installed(it.id) && it.id != rowActive }
        // While nothing runs, the action bar downloads what the model needs (the pinned unit included).
        val pinned = units.pinnedMissing?.takeIf { units.chosen != null && it.id != rowActive && it.id != cpuArtifact?.id }
        fun extra(a: Artifact?, unit: ComputeUnit, purpose: ExtraDownload.Purpose): ExtraDownload? =
            a?.takeIf { inFlight(input.state(it.id)) }?.let { ExtraDownload(it, unit, progressState(it, input.state(it.id), units), purpose) }
        val extraDownload = extra(cpuArtifact, ComputeUnit.CPU, ExtraDownload.Purpose.CPU_VERSION)
            ?: (units.preference as? UnitPreference.Pinned)?.let { extra(pinned, it.unit, ExtraDownload.Purpose.PINNED) }
        val cpuFallback = cpuArtifact?.takeIf { input.state(it.id) is InstallState.NotInstalled }
        val inUse = row.state == RowState.InUse
        val action = when (val s = row.state) {
            RowState.InUse -> DetailAction.InUse
            RowState.Ready -> if (selectableKind(entry)) DetailAction.Use else DetailAction.None
            // "Download the CPU version" right under the failure line is that download: no second button for it.
            is RowState.NotDownloaded -> if (cpuFallback != null && shown?.artifact?.id == cpuFallback.id) DetailAction.None
                else DetailAction.Download(s.bytes, shown?.artifact)
            is RowState.Queued, is RowState.Downloading, is RowState.Paused, is RowState.Failed,
            RowState.Checking, is RowState.Preparing -> DetailAction.InProgress(s)
            RowState.InTraining, is RowState.Unavailable -> DetailAction.None
            RowState.Loading -> DetailAction.Loading
        }
        val installedAny = entry.artifacts.any { input.installed(it.id) }
        val benchRecords = entry.artifacts.flatMap { input.records[it.id].orEmpty() }
        val benchRecord = shown?.let { s -> benchRecords.filter { it.artifactId == s.artifact.id && it.unit == s.unit }.maxByOrNull { it.atEpochMs } }
            ?: benchRecords.maxByOrNull { it.atEpochMs }
        val src = facts.filterIsInstance<Fact.Source>().firstOrNull()
        return ModelDetailView(
            row = row,
            description = entry.description,
            quant = shown?.artifact?.quant,
            chosenUnit = shown?.unit,
            facts = facts,
            notes = notes,
            unitOptions = options,
            pinnedDownload = pinned?.takeIf { input.state(it.id) is InstallState.NotInstalled },
            cpuFallbackDownload = cpuFallback,
            extraDownload = extraDownload,
            action = action,
            // Unknown device: whether the model is in use (delete asks first) cannot be decided yet.
            canDelete = input.device != null &&
                (installedAny || entry.artifacts.any { val s = input.state(it.id); s is InstallState.Paused || s is InstallState.Failed }),
            inUse = inUse,
            bench = BenchBlock(benchRecord, canRun = installedAny && benchCandidates(entry, input).any { it.usable || it.blocker == Blocker.UNIT_FAILED }),
            sourceUrl = src?.url,
        )
    }

    /** `BenchmarkRunner.run` input: `ArtifactSelector.candidates` of the installed artifacts. */
    fun benchCandidates(entry: ModelEntry, input: HubInput): List<Candidate> {
        val device = input.device ?: return emptyList()
        return ArtifactSelector.candidates(entry, device, input.health).filter { input.installed(it.artifact.id) }
    }

    private val VIEW_ORDER = UnitView.UNIT_ORDER
    const val NPU_PORT = "npu-port"

    /** One line per claimed unit: the best candidate of that unit (usable first) with its blocker. */
    fun runsOn(entry: ModelEntry, units: UnitView, device: DeviceProfile? = null): List<RunsOnLine> =
        units.claimedUnits.map { u ->
            val cands = units.candidates.filter { it.unit == u }
            val best = cands.firstOrNull { it.usable } ?: cands.first()
            val hexagon = u == ComputeUnit.NPU && (device?.soc?.let(::snapdragon) == true || best.artifact.socs.any(::snapdragon))
            RunsOnLine(u, best.blocker, units.shown?.unit == u && best.usable, best.artifact.socs, hexagon)
        }

    /** Qualcomm Snapdragon part numbers (SM8750, ...): their NPU is the Hexagon processor. A device fact. */
    private fun snapdragon(soc: String) = soc.startsWith("SM") && soc.drop(2).all { it.isDigit() }

    /**
     * Source row: the model card's HF repo ([ModelEntry.baseRepo], text before any note in parentheses) or, without one,
     * where the files come from (HF repo, or the GitHub release of the shown artifact, ADR-016).
     */
    fun source(entry: ModelEntry, artifact: Artifact?): Fact.Source? {
        val repo = entry.baseRepo?.substringBefore(' ')?.trim()?.takeIf { it.contains('/') }
        if (repo != null) return Fact.Source(repo, "https://huggingface.co/$repo", github = false)
        val f = (artifact ?: entry.artifacts.firstOrNull())?.files?.firstOrNull() ?: return null
        return when (f.source) {
            FileSource.HUGGINGFACE -> Fact.Source(f.repo, "https://huggingface.co/${f.repo}", github = false)
            FileSource.GITHUB -> Fact.Source("${f.repo} · ${f.revision}", "https://github.com/${f.repo}/releases/tag/${f.revision}", github = true)
        }
    }

    /**
     * What "Download" queues for a model (ADR-050 pack semantics, the hub's single-model case): its artifact with the
     * model's unit preference, the VAD for a batch ASR runtime, the OCR pair when "Use what's on screen" is on; only
     * artifacts that are not installed yet. [artifact] forces a specific artifact (pinned unit, CPU version).
     */
    fun downloadPlan(entry: ModelEntry, input: HubInput, artifact: Artifact? = null): List<ArtifactId> {
        // Without the device profile nothing is guessed (an NPU build for another SoC, a skipped VAD / OCR pair).
        val device = input.device ?: return emptyList()
        val plan: List<PlannedDownload> = DownloadPlanner.forModels(
            input.catalog, listOf(entry.id), device, input.settings.useScreen,
        ) { id -> preference(input.settings, id) }
        val ids = plan.map { it.artifact.id }.toMutableList()
        if (artifact != null) {
            val requested = plan.firstOrNull { it.model == entry.id && it.reason == PlannedDownload.Reason.REQUESTED }
            if (requested != null) ids[ids.indexOf(requested.artifact.id)] = artifact.id else ids.add(0, artifact.id)
        }
        // A model whose artifacts are all blocked here contributes nothing to the planner; still fetch what was asked.
        if (ids.isEmpty()) entry.artifacts.firstOrNull()?.let { ids += it.id }
        return ids.distinct().filter { input.state(it) !is InstallState.Installed }
    }

    /**
     * After deleting [deleted]: the next downloaded model of the same kind in catalog order (selectable, installed and
     * usable), or null (Speech: no model of record; Cleanup: None).
     */
    fun nextAfterDelete(deleted: ModelEntry, input: HubInput): ModelId? =
        input.catalog.models.asSequence()
            .filter { it.kind == deleted.kind && it.id != deleted.id && selectableKind(it) }
            .firstOrNull { units(it, input).chosen != null }?.id
}
