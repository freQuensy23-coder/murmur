package app.murmur.feature.onboarding

import app.murmur.core.catalog.Catalog
import app.murmur.core.catalog.DeviceProfile
import app.murmur.core.catalog.DownloadPlanner
import app.murmur.core.catalog.ModelEntry
import app.murmur.core.catalog.Pack
import app.murmur.core.catalog.PlannedDownload
import app.murmur.core.model.ArtifactId
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.ModelId
import app.murmur.core.model.ModelKind
import app.murmur.core.model.MurmurError
import app.murmur.core.model.UnitPreference
import app.murmur.core.store.CleanupChoice
import app.murmur.core.store.InstallState
import app.murmur.core.store.Settings

/**
 * One "Download the models" radio row (DESIGN 6.1 step 5): the catalog pack, what it queues on this phone and its size.
 * [plan] follows ARCHITECTURE 8.5 exactly ([DownloadPlanner.forPack]: one artifact per model for this device, the VAD
 * for a batch speech model, the OCR pair when "Use what's on screen" is on). [totalBytes] is the whole download;
 * [remainingBytes] leaves out what is already installed.
 */
data class PackOption(
    val pack: Pack,
    val plan: List<PlannedDownload>,
    val asr: ModelEntry,
    val llm: ModelEntry?,
    val totalBytes: Long,
    val remainingBytes: Long,
) {
    val artifacts: List<ArtifactId> get() = plan.map { it.artifact.id }
}

object PackPlanner {
    /**
     * The packs this phone can use, in catalog order (the recommended one first when the catalog marks it). A pack
     * without a speech model this device can run is left out. Sizes are those of the artifact each item resolves to
     * with the device profile (await DeviceProfileRepository.ensure() first: the emulator and the Fold7 differ).
     */
    fun options(
        catalog: Catalog,
        device: DeviceProfile,
        settings: Settings,
        installed: Map<ArtifactId, InstallState> = emptyMap(),
    ): List<PackOption> {
        val pref: (ModelId) -> UnitPreference = { settings.unitPreference[it.value] ?: UnitPreference.Auto }
        val packs = catalog.packs.sortedByDescending { it.recommended }
        return packs.mapNotNull { pack ->
            val plan = DownloadPlanner.forPack(catalog, pack, device, settings.useScreen, pref)
            val asr = plan.firstOrNull { it.reason == PlannedDownload.Reason.REQUESTED && catalog.model(it.model)?.kind == ModelKind.ASR }
                ?.let { catalog.model(it.model) } ?: return@mapNotNull null
            val llm = plan.firstOrNull { catalog.model(it.model)?.kind == ModelKind.LLM }?.let { catalog.model(it.model) }
            PackOption(
                pack = pack, plan = plan, asr = asr, llm = llm,
                totalBytes = DownloadPlanner.totalBytes(plan),
                remainingBytes = plan.filter { installed[it.artifact.id] !is InstallState.Installed }.sumOf { it.artifact.downloadBytes },
            )
        }
    }

    fun defaultChoice(options: List<PackOption>): PackOption? =
        options.firstOrNull { it.pack.recommended } ?: options.firstOrNull()

    /**
     * The settings a chosen pack implies: its speech model, and its cleanup model (None for a speech-only pack), so
     * that the first dictation uses what was downloaded.
     */
    fun applyChoice(settings: Settings, option: PackOption): Settings = settings.copy(
        asrModel = option.asr.id,
        cleanup = option.llm?.let { CleanupChoice.Model(it.id) } ?: CleanupChoice.None,
    )
}

/** One row of the downloading checklist (prototype .step). */
data class DownloadRow(
    val artifact: ArtifactId,
    val name: String,
    val bytesTotal: Long,
    val bytesDone: Long,
    val state: InstallState,
) {
    val done: Boolean get() = state is InstallState.Installed
    /** Downloading, or waiting for its turn after the active one. */
    val active: Boolean get() = state is InstallState.Downloading || state is InstallState.Queued
}

/**
 * What the downloading / paused screens show for the chosen pack (DESIGN 6.1 5a/5b), computed from the install states.
 * [rows] = the pack's artifacts plus any artifact queued on the way for one of its models (the CPU build after a failed
 * NPU proof, ARCHITECTURE 8.5); [preparing] = the "Check files and prepare" row: null = not reached yet, false = in
 * progress, true = done.
 */
data class DownloadSummary(
    val rows: List<DownloadRow>,
    val bytesDone: Long,
    val bytesTotal: Long,
    val bytesPerSec: Long,
    val phase: ModelsPhase,
    val preparing: Boolean?,
    val preparingProgress: Float,
    /** Why the download stopped (NoNetwork, LowStorage, DownloadPaused, a failure), when [phase] is STOPPED. */
    val stopReason: MurmurError?,
    val speechReady: Boolean,
    /** The pack's cleanup model is not installed yet (DESIGN 5b "Cleanup turns on when <LLM> finishes"). */
    val cleanupPending: String?,
) {
    val fraction: Float get() = if (bytesTotal <= 0) 0f else (bytesDone.toDouble() / bytesTotal).toFloat().coerceIn(0f, 1f)
    /** Seconds left at the current speed; null when unknown (no speed yet). */
    val secondsLeft: Long? get() = if (bytesPerSec <= 0 || phase != ModelsPhase.ACTIVE) null else (bytesTotal - bytesDone).coerceAtLeast(0) / bytesPerSec

    companion object {
        fun of(catalog: Catalog, option: PackOption, states: Map<ArtifactId, InstallState>): DownloadSummary {
            val planned = option.artifacts.toSet()
            val models = option.plan.map { it.model }.toSet()
            // An artifact of one of the pack's models that is on its way but not in the plan (CPU build queued by Preparing).
            val extras = states.filter { (id, st) ->
                id !in planned && st.inFlight() && catalog.artifact(id)?.first?.id in models
            }.keys
            val rows = (option.artifacts + extras).mapNotNull { id ->
                val (entry, artifact) = catalog.artifact(id) ?: return@mapNotNull null
                val st = states[id] ?: InstallState.NotInstalled
                val total = artifact.downloadBytes
                val done = when (st) {
                    is InstallState.Downloading -> st.bytesDone
                    is InstallState.Paused -> st.bytesDone
                    is InstallState.Installed, InstallState.Verifying, InstallState.Extracting, is InstallState.Preparing -> total
                    else -> 0L
                }
                DownloadRow(id, entry.displayName, total, done.coerceAtMost(total), st)
            }
            val stopped = rows.firstOrNull { it.state is InstallState.Paused || it.state is InstallState.Failed }
            val anyFlight = rows.any { it.state.inFlight() }
            val allInstalled = rows.isNotEmpty() && rows.all { it.done }
            val phase = when {
                allInstalled -> ModelsPhase.DONE
                // Another artifact may still be moving, but a stop needs the user ("Try again").
                stopped != null -> ModelsPhase.STOPPED
                anyFlight -> ModelsPhase.ACTIVE
                // Nothing queued (never started, or deleted meanwhile): the choice is open again.
                else -> ModelsPhase.NONE
            }
            val reason = when (val st = stopped?.state) {
                is InstallState.Paused -> st.reason
                is InstallState.Failed -> st.error
                else -> null
            }
            val post = rows.filter { it.state.postDownload() }
            val preparing = when {
                allInstalled -> true
                post.isNotEmpty() -> false
                else -> null
            }
            val prepProgress = post.map { (it.state as? InstallState.Preparing)?.progress ?: 0f }.average().takeIf { !it.isNaN() }?.toFloat() ?: 0f
            val speech = option.plan.firstOrNull { it.model == option.asr.id }?.artifact?.id
            val speechInstalled = speech != null && states[speech] is InstallState.Installed
            val llmArtifact = option.llm?.let { llm -> option.plan.firstOrNull { it.model == llm.id }?.artifact?.id }
            val cleanupPending = if (llmArtifact != null && states[llmArtifact] !is InstallState.Installed) option.llm.displayName else null
            return DownloadSummary(
                rows = rows,
                bytesDone = rows.sumOf { it.bytesDone },
                bytesTotal = rows.sumOf { it.bytesTotal },
                bytesPerSec = rows.sumOf { (it.state as? InstallState.Downloading)?.bytesPerSec ?: 0L },
                phase = phase,
                preparing = preparing,
                preparingProgress = prepProgress,
                stopReason = reason,
                speechReady = speechInstalled,
                cleanupPending = cleanupPending,
            )
        }

        private fun InstallState.inFlight() = this is InstallState.Queued || this is InstallState.Downloading || postDownload()
        private fun InstallState.postDownload() = this == InstallState.Verifying || this == InstallState.Extracting || this is InstallState.Preparing
    }
}

/** Whether "prepare" means preparing for the NPU on this phone (DESIGN copy "Check files and prepare for the NPU"). */
fun DeviceProfile.hasNpu(): Boolean = ComputeUnit.NPU in availableUnits
