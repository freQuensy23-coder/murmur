package app.murmur.testing

import app.murmur.core.benchmark.BenchProgress
import app.murmur.core.benchmark.BenchStep
import app.murmur.core.benchmark.BenchmarkRunner
import app.murmur.core.catalog.Candidate
import app.murmur.core.catalog.Catalog
import app.murmur.core.catalog.CatalogParser
import app.murmur.core.catalog.CatalogRepository
import app.murmur.core.catalog.CustomModelRequest
import app.murmur.core.catalog.CustomModelResolver
import app.murmur.core.catalog.DeviceProfile
import app.murmur.core.catalog.DeviceProfileRepository
import app.murmur.core.catalog.ModelEntry
import app.murmur.core.catalog.RuntimeKeys
import app.murmur.core.catalog.UnitHealthRepository
import app.murmur.core.context.A11ySnapshot
import app.murmur.core.context.AppInfo
import app.murmur.core.context.ContextAssembler
import app.murmur.core.context.ContextSettings
import app.murmur.core.context.CtxV2
import app.murmur.core.context.FieldInfo
import app.murmur.core.context.LanguageInfo
import app.murmur.core.context.PrivacyDecision
import app.murmur.core.context.Privacy
import app.murmur.core.context.PrivacyLevel
import app.murmur.core.context.PrivacyReason
import app.murmur.core.context.ScreenOutcome
import app.murmur.core.context.Screen
import app.murmur.core.dictation.ActiveLlm
import app.murmur.core.dictation.ActiveModels
import app.murmur.core.dictation.AppNavigator
import app.murmur.core.dictation.LlmPrepare
import app.murmur.core.dictation.LlmSpeed
import app.murmur.core.dictation.PrepareTargets
import app.murmur.core.dictation.ModelResolver
import app.murmur.core.dictation.OpenTarget
import app.murmur.core.dictation.PermissionMonitor
import app.murmur.core.dictation.PermissionState
import app.murmur.core.dictation.StatusNotifier
import app.murmur.core.engine.LoadOptions
import app.murmur.core.engine.LoadSpec
import app.murmur.core.engine.PrefixCacheKey
import app.murmur.core.model.ArtifactId
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.ModelId
import app.murmur.core.model.ModelKind
import app.murmur.core.model.MurmurError
import app.murmur.core.model.Outcome
import app.murmur.core.ocr.ScreenText
import app.murmur.core.prompt.GuardAction
import app.murmur.core.prompt.GuardVerdict
import app.murmur.core.prompt.OutputGuard
import app.murmur.core.prompt.Precheck
import app.murmur.core.prompt.PromptProfile
import app.murmur.core.prompt.PromptRenderer
import app.murmur.core.prompt.RenderInput
import app.murmur.core.prompt.RenderedPrompt
import app.murmur.core.store.BenchmarkRecord
import app.murmur.core.store.BenchmarkRepository
import app.murmur.core.store.BenchmarkResultData
import app.murmur.core.store.CleanupChoice
import app.murmur.core.store.DeliveryMetrics
import app.murmur.core.store.Settings
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flow
import kotlinx.serialization.json.JsonObject

/** Not byte-exact with render.py: a readable stand-in whose suffix contains the transcript tags the fake LLM reads. */
class FakePromptRenderer(
    /** Named tone texts (`Tone` name lowercase -> style text); empty = like tones.v1.json today (no named texts). */
    var toneTexts: Map<String, String> = emptyMap(),
) : PromptRenderer {
    /** Every render input, in order (tests read the style override and the ctx the cleanup used). */
    val inputs: MutableList<RenderInput> = java.util.Collections.synchronizedList(mutableListOf())
    override fun toneText(tone: String): String? = toneTexts[tone]
    override fun render(input: RenderInput): RenderedPrompt {
        inputs += input
        val prefix = cachePrefix(input.profile, input.template, input.preferences, input.systemOverride)
        val ctx = input.ctx?.let { renderContext(it) } ?: ""
        val suffix = "$ctx\n<transcript>\n${input.transcript}\n</transcript><|im_end|>\n"
        return RenderedPrompt(prefix, suffix, prefixKey = PrefixCacheKey.sha256Hex(prefix))
    }
    override fun renderContext(ctx: CtxV2, ctx2: JsonObject?): String =
        "<context>\nApp: ${ctx.app.label ?: ctx.app.pkg}\n" + (ctx.screen?.lines?.joinToString("\n") { it.text } ?: "") + "\n</context>"
    override fun cachePrefix(profile: PromptProfile, template: String, preferences: String?, systemOverride: String?) =
        "<|im_start|>system\n${systemOverride ?: profile.system}${preferences?.let { "\n$it" } ?: ""}<|im_end|>\n<|im_start|>user\n"
}

/** Accepts every non-empty output; empty output falls back to the transcript; transcripts of fillers only are EMPTY. */
class PassThroughGuard(override val fillers: Set<String> = setOf("э", "ээ", "ну", "эм", "um", "uh")) : OutputGuard {
    override fun precheck(transcript: String) = Precheck(skip = transcript.isBlank(), transcript = transcript.trim(), reason = if (transcript.isBlank()) "empty" else "ok")
    override fun guard(transcript: String, output: String, ctx: CtxV2?) = when {
        transcript.split(' ').filter { it.isNotBlank() }.all { it.lowercase() in fillers } -> GuardVerdict("", GuardAction.EMPTY, "fillers_only")
        output.isBlank() -> GuardVerdict(transcript, GuardAction.FALLBACK, "R1_empty")
        else -> GuardVerdict(output.trim(), GuardAction.ACCEPT, "ok")
    }
}

/**
 * Minimal rules with the real codes (PrivacyReason, ARCHITECTURE 9.2): password -> none + skip LLM, dictation refused;
 * excluded app -> app_only, refused; own package -> allowed only for the practice; screen setting off -> app_only.
 * A missing screen keeps its cause through [ScreenOutcome] (secure window -> app_only, as render.py NO_SCREEN_LEVEL).
 */
class SimpleContextAssembler(
    private val ownPackage: String = "app.murmur",
    /** package -> style category (app_categories); empty = every app without a category. */
    var categories: Map<String, String> = emptyMap(),
) : ContextAssembler {
    override fun decide(snapshot: A11ySnapshot, settings: ContextSettings): PrivacyDecision = when {
        snapshot.field?.isPassword == true ->
            PrivacyDecision(PrivacyLevel.NONE, false, PrivacyReason.PASSWORD_FIELD, skipLlm = true, dictationAllowed = false)
        snapshot.pkg in settings.excludedPackages ->
            PrivacyDecision(PrivacyLevel.APP_ONLY, false, PrivacyReason.DENYLISTED_APP, skipLlm = false, dictationAllowed = false)
        snapshot.pkg == ownPackage ->
            PrivacyDecision(PrivacyLevel.APP_ONLY, false, PrivacyReason.DENYLISTED_APP, skipLlm = false, dictationAllowed = settings.inAppPractice)
        !settings.useScreen -> PrivacyDecision(PrivacyLevel.APP_ONLY, false, PrivacyReason.USER_SETTING, skipLlm = false,
            dictationAllowed = bubbleAllowed(snapshot))
        else -> PrivacyDecision(PrivacyLevel.FULL, true, null, skipLlm = false, dictationAllowed = bubbleAllowed(snapshot))
    }

    /** Number / phone / date fields: no dictation (the real FieldKinds rule). */
    private fun bubbleAllowed(snapshot: A11ySnapshot) = snapshot.field?.let { app.murmur.core.context.FieldKinds.bubbleAllowed(it.inputType) } ?: true

    @Deprecated("Pass a ScreenOutcome.", ReplaceWith("assemble(snapshot, decision, screen?.let { ScreenOutcome.Text(it) } ?: ScreenOutcome.Failed, asrLanguage)"))
    override fun assemble(snapshot: A11ySnapshot, decision: PrivacyDecision, screen: ScreenText?, asrLanguage: String?) =
        assemble(snapshot, decision, screen?.let { ScreenOutcome.Text(it) } ?: if (decision.takeScreenshot) ScreenOutcome.Failed else ScreenOutcome.NotTaken, asrLanguage)

    /** The last screen outcome the orchestrator passed (capture kind, cause of a missing screen). */
    @Volatile var lastScreen: ScreenOutcome? = null
        private set

    override fun assemble(snapshot: A11ySnapshot, decision: PrivacyDecision, screen: ScreenOutcome, asrLanguage: String?): CtxV2 {
        lastScreen = screen
        val text = (screen as? ScreenOutcome.Text)?.text?.takeIf { decision.level == PrivacyLevel.FULL && it.lines.isNotEmpty() }
        val privacy = when {
            decision.level != PrivacyLevel.FULL -> Privacy(decision.level, decision.reason)
            text != null -> Privacy(PrivacyLevel.FULL, null)
            else -> {
                val reason = when (screen) {
                    ScreenOutcome.SecureWindow -> PrivacyReason.SECURE_WINDOW
                    is ScreenOutcome.Text -> PrivacyReason.OCR_EMPTY
                    else -> PrivacyReason.SCREENSHOT_ERROR
                }
                Privacy(PrivacyReason.noScreenLevel(reason), reason)
            }
        }
        return CtxV2(
            app = AppInfo(snapshot.pkg, snapshot.label, categories[snapshot.pkg]),
            field = snapshot.field?.takeIf { privacy.level == PrivacyLevel.FULL }
                ?.let { FieldInfo(kind = "message", hint = it.hint, beforeCursor = it.beforeCursor) },
            screen = text?.let { Screen(it.size, it.lines) },
            language = LanguageInfo(asrLanguage),
            privacy = privacy,
        )
    }
}

class FakeCatalogRepository(initial: Catalog = CatalogParser.bundled()) : CatalogRepository {
    private val _catalog = MutableStateFlow(initial)
    override val catalog: StateFlow<Catalog> = _catalog.asStateFlow()
    override suspend fun addCustom(entry: ModelEntry) { _catalog.value = _catalog.value.copy(models = _catalog.value.models + entry) }
    override suspend fun removeCustom(id: ModelId) { _catalog.value = _catalog.value.copy(models = _catalog.value.models.filterNot { it.id == id }) }
}

/**
 * Resolves to specs under /fake/models. [npuFirst] = the Fold7 shape: NPU primary + CPU fallback for ASR (two
 * artifacts) and LLM (same artifact, other unit); [cpuFallback] = false drops the fallbacks (no CPU artifact
 * installed). [noAsr] / [noVad] simulate the resolver errors; [cleanup] = false -> llm null with llmUnavailable.
 * Like the real resolver it skips units that [health] marks failed (ArtifactSelector.ranked): after an NPU load
 * failure the next dictation resolves straight to CPU. [delayMs] slows resolve() (stale-session tests).
 * [prepareTargets] covers the same artifacts (plus the VAD and the OCR pair).
 */
class FakeModelResolver(
    var noAsr: Boolean = false,
    var noVad: Boolean = false,
    var cleanup: Boolean = true,
    var npuFirst: Boolean = false,
    var cpuFallback: Boolean = true,
    var llmSpeed: LlmSpeed? = LlmSpeed(prefillTps = 569.0, decodeTps = 83.8),
    // Stock Qwen3.5 with default.v1 (catalog 2026-11-20.2, ADR-085).
    var nCtx: Int = 7_168,
    var prefixTokens: Int? = 1_047,
    var health: app.murmur.core.catalog.UnitHealth = app.murmur.core.catalog.UnitHealth.NONE,
    var delayMs: Long = 0,
    var llmMemoryMb: Int = 770,
    var hebrewAsrInstalled: Boolean = false,
) : ModelResolver {
    var calls = 0; private set

    private fun spec(kind: ModelKind, id: String, rt: String, unit: ComputeUnit = ComputeUnit.CPU, opts: Map<String, String> = emptyMap()) =
        LoadSpec(kind, ModelId(id.substringBefore('@')), ArtifactId(id), rt, unit, "/fake/models/$id", emptyMap(), options = opts)

    val asrCpu = spec(ModelKind.ASR, "parakeet-tdt-0.6b-v3@onnx-int8", RuntimeKeys.SHERPA_OFFLINE_TRANSDUCER)
    val asrNpu = spec(ModelKind.ASR, "parakeet-tdt-0.6b-v3@qnn-sm8750-20s", RuntimeKeys.SHERPA_OFFLINE_TRANSDUCER, ComputeUnit.NPU)
    val hebrewAsrSpec = spec(ModelKind.ASR, "zipformer2-66m-streaming-he@onnx-int8", RuntimeKeys.SHERPA_ONLINE_TRANSDUCER,
        opts = mapOf("model_type" to "zipformer2"))
    val vad = spec(ModelKind.VAD, "silero-vad-v5@onnx", "ort/silero-vad-v5")
    val ocrDet = spec(ModelKind.OCR_DET, "pp-ocrv6-tiny-det@onnx", "ort/ppocr-det")
    val ocrRec = spec(ModelKind.OCR_REC, "pp-ocrv5-cyrillic-mobile-rec@onnx", "ort/ppocr-rec")
    private val llmOpts get() = mapOf(LoadOptions.N_CTX to nCtx.toString())
    val llmCpu get() = spec(ModelKind.LLM, "qwen3.5-0.8b@q4_0", "llama/gguf", ComputeUnit.CPU, llmOpts + (LoadOptions.USE_MMAP to "0"))
    val llmNpu get() = spec(ModelKind.LLM, "qwen3.5-0.8b@q4_0", "llama/gguf", ComputeUnit.NPU, llmOpts)

    private fun usable(s: LoadSpec) = !health.failed(s.artifactId, s.unit)

    override suspend fun resolve(settings: Settings): Outcome<ActiveModels> {
        calls++
        if (delayMs > 0) delay(delayMs)
        if (noAsr) return Outcome.Err(MurmurError.NoAsrModel)
        if (noVad) return Outcome.Err(MurmurError.NoVadModel)
        val asrRanked = (if (npuFirst) listOfNotNull(asrNpu, asrCpu.takeIf { cpuFallback }) else listOf(asrCpu)).filter(::usable)
        if (asrRanked.isEmpty()) return Outcome.Err(MurmurError.NoAsrModel)
        val llmRanked = (if (npuFirst) listOfNotNull(llmNpu, llmCpu.takeIf { cpuFallback }) else listOf(llmCpu)).filter(::usable)
        val llmOn = cleanup && settings.cleanup != CleanupChoice.None && llmRanked.isNotEmpty()
        val heOn = hebrewAsrInstalled && settings.languages.any { app.murmur.core.model.Language.normalize(it) == "he" }
        return Outcome.Ok(ActiveModels(
            asr = asrRanked[0],
            asrFallback = asrRanked.getOrNull(1),
            asrTraits = app.murmur.core.catalog.AsrTraits(streaming = false, punctuation = true, casing = true),
            hebrewAsr = hebrewAsrSpec.takeIf { heOn },
            hebrewAsrTraits = app.murmur.core.catalog.AsrTraits(streaming = true, punctuation = false, casing = false).takeIf { heOn },
            vad = vad,
            ocrDet = ocrDet.takeIf { settings.useScreen },
            ocrRec = ocrRec.takeIf { settings.useScreen },
            llm = if (!llmOn) null else ActiveLlm(
                spec = llmRanked[0],
                fallback = llmRanked.getOrNull(1),
                template = "qwen3.5", profile = app.murmur.core.catalog.GgufTemplates.STOCK_PROFILE, memoryMb = llmMemoryMb, nCtx = nCtx, prefixTokens = prefixTokens,
                modelSha256 = "0".repeat(64), speed = llmSpeed,
            ),
            llmUnavailable = if (llmOn) null else if (settings.cleanup == CleanupChoice.None) "none" else "no_model",
        ))
    }

    override suspend fun prepareTargets(artifact: ArtifactId, settings: Settings): Outcome<PrepareTargets> {
        return when (artifact) {
            asrNpu.artifactId, asrCpu.artifactId -> Outcome.Ok(PrepareTargets(
                kind = ModelKind.ASR,
                specs = listOf(if (artifact == asrNpu.artifactId) asrNpu else asrCpu).filter(::usable),
                vad = vad, needsVad = true,
                cpuArtifact = asrCpu.artifactId, cpuArtifactInstalled = cpuFallback,
            ))
            llmCpu.artifactId -> Outcome.Ok(PrepareTargets(
                kind = ModelKind.LLM,
                specs = (if (npuFirst) listOf(llmNpu, llmCpu) else listOf(llmCpu)).filter(::usable),
                llm = LlmPrepare("qwen3.5", app.murmur.core.catalog.GgufTemplates.STOCK_PROFILE, nCtx, "0".repeat(64), memoryMb = llmMemoryMb),
            ))
            vad.artifactId -> Outcome.Ok(PrepareTargets(ModelKind.VAD, listOf(vad)))
            ocrDet.artifactId -> Outcome.Ok(PrepareTargets(ModelKind.OCR_DET, listOf(ocrDet), ocrPartner = ocrRec))
            ocrRec.artifactId -> Outcome.Ok(PrepareTargets(ModelKind.OCR_REC, listOf(ocrRec), ocrPartner = ocrDet))
            else -> Outcome.Err(MurmurError.CatalogInvalid("unknown artifact"))
        }
    }
}

class FakeDeviceProfileRepository(
    initial: DeviceProfile = EMULATOR,
    var failProbe: Boolean = false,
) : DeviceProfileRepository {
    private val _profile = MutableStateFlow<DeviceProfile?>(null)
    private val real = initial
    override val profile: StateFlow<DeviceProfile?> = _profile.asStateFlow()
    var probes = 0; private set
    override suspend fun ensure(): DeviceProfile = _profile.value ?: reprobe()
    override suspend fun reprobe(): DeviceProfile {
        probes++
        val p = if (failProbe) DeviceProfile.cpuOnly(real.abi, real.totalRamMb, real.supportedRuntimes) else real
        _profile.value = p
        return p
    }

    companion object {
        val EMULATOR = DeviceProfile(null, "x86_64", setOf(ComputeUnit.CPU), RuntimeKeys.known, 3_072, mapOf("llm" to "fake-1"))
        val FOLD7 = DeviceProfile("SM8750", "arm64-v8a", setOf(ComputeUnit.NPU, ComputeUnit.GPU, ComputeUnit.CPU), RuntimeKeys.known, 11_500, mapOf("llm" to "fake-1"))
    }
}

class FakeUnitHealth : UnitHealthRepository {
    val failedUnits = mutableMapOf<Pair<ArtifactId, ComputeUnit>, String>()
    val provenUnits = mutableSetOf<Pair<ArtifactId, ComputeUnit>>()
    override fun failed(artifact: ArtifactId, unit: ComputeUnit) = (artifact to unit) in failedUnits
    override fun proven(artifact: ArtifactId, unit: ComputeUnit) = (artifact to unit) in provenUnits
    override suspend fun markFailed(artifact: ArtifactId, unit: ComputeUnit, code: String) { failedUnits[artifact to unit] = code }
    override suspend fun markProven(artifact: ArtifactId, unit: ComputeUnit) { provenUnits += artifact to unit }
    override suspend fun clear(artifact: ArtifactId, unit: ComputeUnit) { failedUnits -= artifact to unit }
}

/** Returns [result] (or a CUSTOM copy of the first catalog entry of the requested kind). */
class FakeCustomModelResolver(var result: Outcome<ModelEntry>? = null, private val catalog: Catalog = CatalogParser.bundled()) : CustomModelResolver {
    val requests = mutableListOf<CustomModelRequest>()
    override suspend fun resolve(request: CustomModelRequest): Outcome<ModelEntry> {
        requests += request
        result?.let { return it }
        val base = catalog.models.first { it.kind == request.kind && it.artifacts.isNotEmpty() }
        return Outcome.Ok(base.copy(id = ModelId("custom-" + request.repo.substringAfter('/').lowercase()), displayName = request.displayName ?: request.repo,
            origin = app.murmur.core.catalog.Origin.CUSTOM))
    }
}

/**
 * Scripted benchmark (WP-18 sheets, WP-14 consumers): emits DESIGN bench-run steps per usable candidate with
 * [stepMs] of virtual time each, then Finished with sample records ([valid] false simulates "Results look wrong").
 */
class FakeBenchmarkRunner(
    var stepMs: Long = 300,
    var valid: Boolean = true,
    private val repo: BenchmarkRepository? = null,
) : BenchmarkRunner {
    var stopped = false; private set
    override fun run(entry: ModelEntry, candidates: List<Candidate>): Flow<BenchProgress> = flow {
        stopped = false
        val records = mutableListOf<BenchmarkRecord>()
        val steps = when (entry.kind) {
            ModelKind.ASR -> listOf(BenchStep.LOAD, BenchStep.SAMPLE_RU, BenchStep.SAMPLE_EN, BenchStep.UNLOAD_MEMORY)
            ModelKind.LLM -> listOf(BenchStep.LOAD, BenchStep.CLEANUP_CASE, BenchStep.UNLOAD_MEMORY)
            else -> listOf(BenchStep.LOAD, BenchStep.OCR_PAGE, BenchStep.UNLOAD_MEMORY)
        }
        for (c in candidates.filter { it.usable }) {
            var t = 0L
            for (s in steps) {
                if (stopped) return@flow
                emit(BenchProgress.Step(s, c.unit, t, done = false)); delay(stepMs); t += stepMs
                emit(BenchProgress.Step(s, c.unit, t, done = true))
            }
            val reason = if (valid) null else "wer_high"
            val data = when (entry.kind) {
                ModelKind.ASR -> BenchmarkResultData.Asr(14.2, 180, 800, 120, 1_100, c.unit.name, emptyList(), valid, reason)
                ModelKind.LLM -> BenchmarkResultData.Llm(40.9, 2_597.0, 310.0, 1_300.0, 2_100, 1_160, c.unit.name, 0.97, valid = valid, invalidReason = reason)
                else -> BenchmarkResultData.Ocr(450, 21, 90, 140, c.unit.name, valid, reason)
            }
            records += BenchmarkRecord(c.artifact.id, c.unit, "0.1.0", "fake/fingerprint", 0, data).also { repo?.save(it) }
        }
        emit(BenchProgress.Finished(records))
    }
    override fun stop() { stopped = true }
}

/** Plain reference implementation for orchestrator tests (WP-12 owns the real one in core:store). */
class SimpleDeliveryMetrics : DeliveryMetrics {
    private fun words(s: String) = s.lowercase().split(Regex("[^\\p{L}\\p{N}]+")).filter { it.isNotEmpty() }
    override fun wordCount(text: String) = words(text).size
    override fun editDistanceWords(raw: String, final: String): Int {
        val a = words(raw); val b = words(final)
        val d = IntArray(b.size + 1) { it }
        for (i in 1..a.size) { var prev = d[0]; d[0] = i
            for (j in 1..b.size) { val t = d[j]; d[j] = minOf(d[j] + 1, d[j - 1] + 1, prev + if (a[i - 1] == b[j - 1]) 0 else 1); prev = t } }
        return d[b.size]
    }
    override fun fillersRemoved(raw: String, final: String, fillers: Set<String>) =
        (words(raw).count { it in fillers } - words(final).count { it in fillers }).coerceAtLeast(0)
    override fun fixes(raw: String, final: String, fillers: Set<String>) =
        (editDistanceWords(raw, final) - fillersRemoved(raw, final, fillers)).coerceAtLeast(0)
    override fun mixedLanguage(text: String): Boolean {
        val w = words(text)
        val cyr = w.count { t -> t.any { it in 'Ѐ'..'ӿ' } }
        val lat = w.count { t -> t.any { it in 'a'..'z' } }
        return minOf(cyr, lat) >= 2
    }
}

class RecordingNavigator : AppNavigator {
    val opened = mutableListOf<OpenTarget>()
    override fun open(target: OpenTarget) { opened += target }
}

class FakePermissionMonitor(
    initial: PermissionState = PermissionState(micGranted = true, notificationsGranted = true, a11yEnabled = true,
        a11yConnected = true, a11yEverConnected = true, batteryUnrestricted = true, restrictedLikely = false),
) : PermissionMonitor {
    private val _state = MutableStateFlow(initial)
    override val state: StateFlow<PermissionState> = _state.asStateFlow()
    var refreshes = 0; private set
    /** What a re-read finds (a grant made outside the app); identity by default, so [refresh] only counts. */
    var onRefresh: (PermissionState) -> PermissionState = { it }
    fun set(transform: (PermissionState) -> PermissionState) { _state.value = transform(_state.value) }
    override fun refresh() { refreshes++; _state.value = onRefresh(_state.value) }
}

class RecordingStatusNotifier : StatusNotifier {
    var shown: Pair<String, Long?>? = null; private set
    override fun show(modelsLine: String, hiddenUntilEpochMs: Long?) { shown = modelsLine to hiddenUntilEpochMs }
    override fun hide() { shown = null }
}
