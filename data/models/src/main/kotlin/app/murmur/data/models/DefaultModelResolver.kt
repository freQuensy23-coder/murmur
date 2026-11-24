package app.murmur.data.models

import app.murmur.core.benchmark.BenchScores
import app.murmur.core.benchmark.BenchSpecProvider
import app.murmur.core.benchmark.BenchSpecs
import app.murmur.core.catalog.Artifact
import app.murmur.core.catalog.ArtifactSelector
import app.murmur.core.catalog.Candidate
import app.murmur.core.catalog.Catalog
import app.murmur.core.catalog.CatalogRepository
import app.murmur.core.catalog.CatalogValidator
import app.murmur.core.catalog.DeviceProfile
import app.murmur.core.catalog.DeviceProfileRepository
import app.murmur.core.catalog.MemoryEstimate
import app.murmur.core.catalog.ModelEntry
import app.murmur.core.catalog.RuntimeKeys
import app.murmur.core.catalog.Status
import app.murmur.core.catalog.UnitHealth
import app.murmur.core.dictation.ActiveLlm
import app.murmur.core.dictation.ActiveModels
import app.murmur.core.dictation.LlmSpeed
import app.murmur.core.dictation.LlmPrepare
import app.murmur.core.dictation.ModelResolver
import app.murmur.core.dictation.PrepareTargets
import app.murmur.core.engine.LoadOptions
import app.murmur.core.engine.LoadSpec
import app.murmur.core.model.ArtifactId
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.Language
import app.murmur.core.model.ModelId
import app.murmur.core.model.ModelKind
import app.murmur.core.model.MurmurError
import app.murmur.core.model.Outcome
import app.murmur.core.model.UnitPreference
import app.murmur.core.store.BenchmarkRecord
import app.murmur.core.store.BenchmarkRepository
import app.murmur.core.store.CleanupChoice
import app.murmur.core.store.InstallState
import app.murmur.core.store.InstalledModelsRepository
import app.murmur.core.store.Settings
import java.io.File
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withTimeoutOrNull

/**
 * Settings + catalog + install state + device profile + unit health + valid benchmarks -> LoadSpecs
 * (ARCHITECTURE 8.2, 8.4, ADR-050). Primary = first of [ArtifactSelector.ranked] over the installed artifacts, fallback
 * = the next one (the LLM fallback is the same GGUF on the next unit). Never throws.
 */
class DefaultModelResolver(
    private val catalog: CatalogRepository,
    private val installed: InstalledModelsRepository,
    private val device: DeviceProfileRepository,
    private val health: UnitHealth,
    private val benchmarks: BenchmarkRepository,
    /**
     * Folder of an artifact that is being installed (InstallState.Preparing carries no dir): the storage layout's
     * `models/<artifactId>/`. Used only by [prepareTargets]; null = only Installed artifacts can be prepared.
     */
    private val artifactDir: ((ArtifactId) -> String)? = null,
) : ModelResolver {

    override suspend fun resolve(settings: Settings): Outcome<ActiveModels> = try {
        resolveOrErr(settings)
    } catch (e: CancellationException) {
        throw e
    } catch (e: Exception) {
        Outcome.Err(MurmurError.Internal("resolver: ${e.javaClass.simpleName}"))
    }

    private suspend fun resolveOrErr(settings: Settings): Outcome<ActiveModels> {
        // A custom model selected by the user exists only once the stored entries are read (first resolve after start).
        val c = withTimeoutOrNull(CATALOG_WAIT_MS) { catalog.loaded() } ?: catalog.catalog.value
        val dev = device.ensure()

        // Speech model: the user's choice when it may be the speech model (ADR-047), else the language default.
        val asrEntry = speechModel(c, settings) ?: return Outcome.Err(MurmurError.NoAsrModel)
        val asr = rank(asrEntry, dev, settings)
        if (asr.ranked.isEmpty()) return Outcome.Err(MurmurError.NoAsrModel)
        val primary = asr.ranked.first()

        val vad = c.model(c.defaults.vad)?.let { first(it, dev, settings) }
        if (RuntimeKeys.isBatchAsr(primary.artifact.runtime) && vad == null) return Outcome.Err(MurmurError.NoVadModel)
        val asrFallback = ArtifactSelector.fallbackAfter(asr.ranked, primary)
            ?.takeIf { !RuntimeKeys.isBatchAsr(it.artifact.runtime) || vad != null }

        val (ocrDet, ocrRec) = if (!settings.useScreen) null to null else {
            val det = c.model(c.defaults.ocrDet)?.let { first(it, dev, settings) }
            val rec = c.model(c.defaults.ocrRec)?.let { first(it, dev, settings) }
            if (det != null && rec != null) det to rec else null to null   // OCR needs both stages
        }

        val (llm, llmUnavailable) = cleanupModel(c, dev, settings)

        val hebrewEnabled = settings.languages.any { Language.normalize(it) == Language.HE.code }
        val hebrewEntry = if (hebrewEnabled) hebrewSpeechModel(c, settings) else null
        val hebrewRanked = hebrewEntry?.let { rank(it, dev, settings) }
        val hebrewPrimary = hebrewRanked?.ranked?.firstOrNull()
            ?.takeIf { !RuntimeKeys.isBatchAsr(it.artifact.runtime) || vad != null }
        val hebrewFallback = if (hebrewRanked != null && hebrewPrimary != null) {
            ArtifactSelector.fallbackAfter(hebrewRanked.ranked, hebrewPrimary)
                ?.takeIf { !RuntimeKeys.isBatchAsr(it.artifact.runtime) || vad != null }
        } else null

        return Outcome.Ok(
            ActiveModels(
                asr = asr.spec(primary),
                asrFallback = asrFallback?.let { asr.spec(it) },
                asrTraits = asrEntry.asr!!,
                vad = vad,
                ocrDet = ocrDet,
                ocrRec = ocrRec,
                llm = llm,
                llmUnavailable = llmUnavailable,
                hebrewAsr = hebrewPrimary?.let { hebrewRanked!!.spec(it) },
                hebrewAsrFallback = hebrewFallback?.let { hebrewRanked!!.spec(it) },
                hebrewAsrTraits = hebrewPrimary?.let { hebrewEntry!!.asr },
            ),
        )
    }

    override suspend fun prepareTargets(artifact: ArtifactId, settings: Settings): Outcome<PrepareTargets> = try {
        prepareOrErr(artifact, settings)
    } catch (e: CancellationException) {
        throw e
    } catch (e: Exception) {
        Outcome.Err(MurmurError.Internal("resolver: ${e.javaClass.simpleName}"))
    }

    /**
     * EnginePreparer input (ModelResolver.prepareTargets): [id] on every usable unit (ranked, UnitHealth applied),
     * specs built by the same [loadSpec] as [resolve]; the VAD / OCR partner / CPU artifact as resolve would pick them.
     */
    private suspend fun prepareOrErr(id: ArtifactId, settings: Settings): Outcome<PrepareTargets> {
        val c = withTimeoutOrNull(CATALOG_WAIT_MS) { catalog.loaded() } ?: catalog.catalog.value
        val entry = c.models.firstOrNull { m -> m.artifacts.any { it.id == id } }
            ?: return Outcome.Err(MurmurError.CatalogInvalid("unknown artifact"))
        val art = entry.artifacts.first { it.id == id }
        val dir = when (val st = installed.state(id).first()) {
            is InstallState.Installed -> st.dir
            is InstallState.Preparing -> artifactDir?.invoke(id)
            else -> null
        } ?: return Outcome.Err(MurmurError.CatalogInvalid("not installed"))
        val dev = device.ensure()
        val bench = BenchScores.lookup(benchmarks.latest(id).first())
        val ranked = ArtifactSelector.ranked(entry, dev, UnitPreference.Auto, bench, setOf(id), health)
            .filter { it.artifact.id == id }
        val specs = ranked.map { loadSpec(entry, art, it.unit, dir) }
        return Outcome.Ok(
            when (entry.kind) {
                ModelKind.ASR -> {
                    val cpu = entry.artifacts.firstOrNull { it.id != id && it.units == listOf(ComputeUnit.CPU) }
                        ?: entry.artifacts.firstOrNull { it.id != id && ComputeUnit.CPU in it.units }
                    PrepareTargets(
                        kind = ModelKind.ASR,
                        specs = specs,
                        vad = c.model(c.defaults.vad)?.let { first(it, dev, settings) },
                        needsVad = RuntimeKeys.isBatchAsr(art.runtime),
                        cpuArtifact = cpu?.id,
                        cpuArtifactInstalled = cpu != null && installed.state(cpu.id).first() is InstallState.Installed,
                    )
                }
                ModelKind.LLM -> {
                    val traits = entry.llm ?: return Outcome.Err(MurmurError.CatalogInvalid("llm traits"))
                    val gguf = ggufFile(art) ?: return Outcome.Err(MurmurError.CatalogInvalid("no gguf"))
                    // The gate uses the largest per-unit estimate of the units being proven.
                    val mem = ranked.maxOfOrNull { MemoryEstimate.mb(entry, art, it.unit) } ?: MemoryEstimate.mb(entry, art, ComputeUnit.CPU)
                    PrepareTargets(
                        kind = ModelKind.LLM,
                        specs = specs,
                        llm = LlmPrepare(traits.template, traits.promptProfile, traits.nCtx, gguf.sha256, mem),
                    )
                }
                ModelKind.OCR_DET, ModelKind.OCR_REC -> {
                    val partnerId = if (entry.kind == ModelKind.OCR_DET) c.defaults.ocrRec else c.defaults.ocrDet
                    PrepareTargets(entry.kind, specs, ocrPartner = c.model(partnerId)?.let { first(it, dev, settings) })
                }
                ModelKind.VAD -> PrepareTargets(ModelKind.VAD, specs)
            },
        )
    }

    /**
     * The benchmark's load specs (WP-14 `BenchSpecProvider`, ADR-080): [loadSpec] on the candidate's own unit, as a
     * dictation would load it, with UnitHealth not applied (a benchmark re-tests a failed unit; a valid record clears
     * it). Batch ASR gets the installed VAD (required), a streaming model the VAD when installed (ADR-065); OCR the
     * partner stage; an LLM the GGUF's sha256. VAD and partner are ranked with default settings (Auto unit).
     */
    val benchSpecs: BenchSpecProvider = BenchSpecProvider { entry, candidate ->
        try {
            benchSpecsOrErr(entry, candidate)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Outcome.Err(MurmurError.Internal("resolver: ${e.javaClass.simpleName}"))
        }
    }

    private suspend fun benchSpecsOrErr(entry: ModelEntry, candidate: Candidate): Outcome<BenchSpecs> {
        val c = withTimeoutOrNull(CATALOG_WAIT_MS) { catalog.loaded() } ?: catalog.catalog.value
        val art = candidate.artifact
        val dir = (installed.state(art.id).first() as? InstallState.Installed)?.dir
            ?: return Outcome.Err(MurmurError.CatalogInvalid("not installed"))
        val main = loadSpec(entry, art, candidate.unit, dir)
        val dev = device.ensure()
        val defaults = Settings()
        return when (entry.kind) {
            ModelKind.ASR -> {
                val vad = c.model(c.defaults.vad)?.let { first(it, dev, defaults) }
                if (RuntimeKeys.isBatchAsr(art.runtime) && vad == null) Outcome.Err(MurmurError.NoVadModel)
                else Outcome.Ok(BenchSpecs(main, vad = vad))
            }
            ModelKind.LLM -> ggufFile(art)?.let { Outcome.Ok(BenchSpecs(main, modelSha256 = it.sha256)) }
                ?: Outcome.Err(MurmurError.CatalogInvalid("no gguf"))
            ModelKind.OCR_DET, ModelKind.OCR_REC -> {
                val partnerId = if (entry.kind == ModelKind.OCR_DET) c.defaults.ocrRec else c.defaults.ocrDet
                c.model(partnerId)?.let { first(it, dev, defaults) }?.let { Outcome.Ok(BenchSpecs(main, ocrPartner = it)) }
                    ?: Outcome.Err(MurmurError.CatalogInvalid("ocr partner not installed"))
            }
            ModelKind.VAD -> Outcome.Ok(BenchSpecs(main))
        }
    }

    private fun speechModel(c: Catalog, settings: Settings): ModelEntry? {
        settings.asrModel?.let { id -> c.model(id)?.takeIf { it.selectableAsAsr }?.let { return it } }
        val key = languageProfile(settings.languages)
        val id = c.defaults.asr[key] ?: c.defaults.asr.values.firstOrNull() ?: return null
        return c.model(id)?.takeIf { it.selectableAsAsr }
    }

    private fun hebrewSpeechModel(c: Catalog, settings: Settings): ModelEntry? {
        settings.hebrewAsrModel?.let { id -> c.model(id)?.takeIf { it.selectableAsAsr }?.let { return it } }
        val id = c.defaults.asr[Language.HE.code] ?: return null
        return c.model(id)?.takeIf { it.selectableAsAsr }
    }

    private suspend fun cleanupModel(c: Catalog, dev: DeviceProfile, settings: Settings): Pair<ActiveLlm?, String?> {
        val id: ModelId = when (val choice = settings.cleanup) {
            CleanupChoice.None -> return null to LLM_NONE
            CleanupChoice.Default -> c.defaults.llm ?: return null to LLM_NONE
            is CleanupChoice.Model -> choice.id
        }
        val entry = c.model(id)?.takeIf { it.kind == ModelKind.LLM && it.status == Status.AVAILABLE && it.llm != null }
            ?: return null to LLM_NO_MODEL
        val r = rank(entry, dev, settings)
        val primary = r.ranked.firstOrNull() ?: return null to LLM_NO_MODEL
        val fallback = ArtifactSelector.fallbackAfter(r.ranked, primary)
        val traits = entry.llm!!
        val gguf = ggufFile(primary.artifact) ?: return null to LLM_NO_MODEL
        // ADR-071 / ADR-080: the newest record of the chosen (artifact, unit), the same one the selection scored.
        val speed = BenchScores.llmSpeed(r.records[primary.artifact.id].orEmpty(), primary.artifact.id, primary.unit)
            ?.let { LlmSpeed(prefillTps = it.prefillTps, decodeTps = it.decodeTps) }
        return ActiveLlm(
            spec = r.spec(primary),
            fallback = fallback?.let { r.spec(it) },
            template = traits.template,
            profile = traits.promptProfile,
            memoryMb = MemoryEstimate.mb(entry, primary.artifact, primary.unit),
            nCtx = traits.nCtx,
            prefixTokens = traits.prefixTokens,
            modelSha256 = gguf.sha256,
            speed = speed,
        ) to null
    }

    /** Ranked installed candidates of one model; a pinned unit nothing installed serves falls back to Auto. */
    private suspend fun rank(entry: ModelEntry, dev: DeviceProfile, settings: Settings): Ranked {
        val dirs = mutableMapOf<ArtifactId, String>()
        for (a in entry.artifacts) (installed.state(a.id).first() as? InstallState.Installed)?.let { dirs[a.id] = it.dir }
        val records = entry.artifacts.associate { a -> a.id to benchmarks.latest(a.id).first() }
        val bench = BenchScores.lookup(records.values.flatten())
        val pref = settings.unitPreference[entry.id.value] ?: UnitPreference.Auto
        var ranked = ArtifactSelector.ranked(entry, dev, pref, bench, dirs.keys, health)
        if (ranked.isEmpty() && pref is UnitPreference.Pinned)
            ranked = ArtifactSelector.ranked(entry, dev, UnitPreference.Auto, bench, dirs.keys, health)
        return Ranked(entry, ranked, dirs, records)
    }

    private suspend fun first(entry: ModelEntry, dev: DeviceProfile, settings: Settings): LoadSpec? =
        rank(entry, dev, settings).let { r -> r.ranked.firstOrNull()?.let { r.spec(it) } }

    private class Ranked(
        val entry: ModelEntry,
        val ranked: List<Candidate>,
        val dirs: Map<ArtifactId, String>,
        val records: Map<ArtifactId, List<BenchmarkRecord>>,
    ) {
        fun spec(c: Candidate): LoadSpec = loadSpec(entry, c.artifact, c.unit, dirs.getValue(c.artifact.id))
    }

    companion object {
        const val LLM_NONE = "none"
        const val LLM_NO_MODEL = "no_model"
        const val LLM_THREADS = 4
        /** Upper bound on waiting for the stored custom entries (Room's first read takes milliseconds). */
        const val CATALOG_WAIT_MS = 2_000L

        /** Settings.languages -> catalog defaults.asr key. */
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
         * The spec an inference process loads. For archives that extract into one folder (layout role "dir") the
         * spec's dir is that folder and the role is dropped, so the layout paths are relative to [LoadSpec.dir].
         */
        fun loadSpec(entry: ModelEntry, a: Artifact, unit: ComputeUnit, installedDir: String): LoadSpec {
            val sub = a.layout[CatalogValidator.LAYOUT_DIR]
            val options = buildMap {
                putAll(a.options)
                entry.llm?.let { l ->
                    put(LoadOptions.N_CTX, l.nCtx.toString())
                    put(LoadOptions.THREADS, LLM_THREADS.toString())
                    if (unit == ComputeUnit.CPU) put(LoadOptions.USE_MMAP, "0")   // ADR-031
                }
            }
            return LoadSpec(
                kind = entry.kind,
                modelId = entry.id,
                artifactId = a.id,
                runtime = a.runtime,
                unit = unit,
                dir = if (sub != null) File(installedDir, sub).path else installedDir,
                layout = a.layout - CatalogValidator.LAYOUT_DIR,
                maxWindowSec = a.maxWindowSec,
                options = options,
            )
        }
    }
}
