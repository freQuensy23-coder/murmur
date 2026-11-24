package app.murmur.core.benchmark

import app.murmur.core.audio.PcmChunk
import app.murmur.core.catalog.Blocker
import app.murmur.core.catalog.Candidate
import app.murmur.core.catalog.LanguageMode
import app.murmur.core.catalog.ModelEntry
import app.murmur.core.catalog.UnitHealthRepository
import app.murmur.core.engine.AsrEvent
import app.murmur.core.engine.AsrSessionOptions
import app.murmur.core.engine.Engines
import app.murmur.core.engine.HostState
import app.murmur.core.engine.InferenceHost
import app.murmur.core.engine.LlmEvent
import app.murmur.core.engine.LlmRequest
import app.murmur.core.engine.LoadFailures
import app.murmur.core.engine.LoadInfo
import app.murmur.core.engine.LoadOptions
import app.murmur.core.engine.PrefixCacheKey
import app.murmur.core.engine.PrefixSpec
import app.murmur.core.engine.ProcessExits
import app.murmur.core.engine.ProcessMemory
import app.murmur.core.engine.ProcessProbe
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.ModelKind
import app.murmur.core.model.MurmurClock
import app.murmur.core.model.MurmurError
import app.murmur.core.model.Outcome
import app.murmur.core.model.Stage
import app.murmur.core.ocr.ScreenText
import app.murmur.core.prompt.GuardAction
import app.murmur.core.prompt.OutputGuard
import app.murmur.core.prompt.PromptAssets
import app.murmur.core.prompt.PromptRenderer
import app.murmur.core.prompt.RenderInput
import app.murmur.core.store.BenchmarkRecord
import app.murmur.core.store.BenchmarkRepository
import app.murmur.core.store.BenchmarkResultData
import app.murmur.core.store.ClipResult
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.cancel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.FlowCollector
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.ceil

/**
 * The Benchmark button (ARCHITECTURE 10, ADR-051) over the production engines: per unit
 * `host.shutdown()` -> connect + load (cold, timed) -> warm-up -> measured pass -> unload -> memory -> validity gate
 * -> record (saved; a valid one clears UnitHealth). Timings come from [clock] (elapsedRealtime on the phone, virtual
 * time in tests). ASR clips are pushed through `AsrSession.accept` as fast as it takes them, so RTFx is processing
 * speed, not real time.
 */
class DefaultBenchmarkRunner(
    private val engines: Engines,
    private val specs: BenchSpecProvider,
    private val records: BenchmarkRepository,
    private val health: UnitHealthRepository,
    private val clock: MurmurClock,
    private val environment: BenchEnvironment,
    private val renderer: PromptRenderer,
    private val guard: OutputGuard,
    private val assets: PromptAssets,
    private val inputs: () -> BenchInputs = BenchInputs::bundled,
    private val config: BenchConfig = BenchConfig(),
    /** ADR-083: a death during a load that the system made for memory is OutOfMemory (no UnitHealth mark). */
    private val exits: ProcessExits = ProcessExits.NONE,
) : BenchmarkRunner {
    private val mutex = Mutex()
    /** Every run() gets the next id; stop() stops every run with an id up to [stoppedThrough]. */
    private val runsCreated = AtomicLong(0)
    private val stoppedThrough = MutableStateFlow(0L)
    /** Id of the run holding [mutex] (runs never overlap). */
    @Volatile private var activeRun = 0L

    override fun stop() {
        stoppedThrough.update { maxOf(it, runsCreated.get()) }
    }

    private fun stopped(run: Long) = stoppedThrough.value >= run

    override fun run(entry: ModelEntry, candidates: List<Candidate>): Flow<BenchProgress> {
        val id = runsCreated.incrementAndGet()
        return flow {
            mutex.withLock {
                activeRun = id
                // Stopped before it started (before collection, or while an earlier run held the lock): nothing runs.
                if (stopped(id)) return@withLock
                runAsActive(id, entry, candidates)
            }
        }
    }

    private suspend fun FlowCollector<BenchProgress>.runAsActive(id: Long, entry: ModelEntry, candidates: List<Candidate>) {
        val used = mutableSetOf<InferenceHost>()
        try {
            runLocked(entry, candidates, used)
        } catch (e: CancellationException) {
            // Stopped by the user: no Finished, nothing saved for the unit in flight. A cancelled collector
            // (sheet closed) is not swallowed.
            if (!(stopped(id) && currentCoroutineContext().isActive)) throw e
        } finally {
            // Whatever happened (finished, stopped, failed, collector cancelled): no process keeps a model loaded.
            withContext(NonCancellable) { used.forEach { runCatching { it.shutdown() } } }
        }
    }

    private suspend fun FlowCollector<BenchProgress>.runLocked(
        entry: ModelEntry,
        candidates: List<Candidate>,
        used: MutableSet<InferenceHost>,
    ) {
        val units = candidates
            .filter { it.usable || it.blocker == Blocker.UNIT_FAILED }
            .sortedBy { UNIT_ORDER.indexOf(it.unit) }
        if (units.isEmpty()) return emit(BenchProgress.Failed(MurmurError.Internal("bench.no_unit")))
        val data = try {
            inputs()
        } catch (e: Exception) {
            return emit(BenchProgress.Failed(MurmurError.Internal("bench.inputs:${e.javaClass.simpleName}")))
        }
        val plan = when (entry.kind) {
            ModelKind.ASR -> data.clips.filter { it.runsOn(entry.languages.codes) }
                .map { PlannedStep(ComputeUnit.CPU, it.step, it.id, it.audioMs) }
            ModelKind.LLM -> listOf(PlannedStep(ComputeUnit.CPU, BenchStep.CLEANUP_CASE))
            ModelKind.OCR_DET, ModelKind.OCR_REC -> data.pages.map { PlannedStep(ComputeUnit.CPU, BenchStep.OCR_PAGE, it.id) }
            ModelKind.VAD -> emptyList()
        }
        if (plan.isEmpty()) return emit(BenchProgress.Failed(MurmurError.Internal("bench.nothing_to_measure")))
        emit(BenchProgress.Planned(units.flatMap { c ->
            listOf(PlannedStep(c.unit, BenchStep.LOAD)) + plan.map { it.copy(unit = c.unit) } +
                PlannedStep(c.unit, BenchStep.UNLOAD_MEMORY)
        }))

        val saved = mutableListOf<BenchmarkRecord>()
        var firstError: MurmurError? = null
        for (candidate in units) {
            val host = if (entry.kind == ModelKind.LLM) engines.llmHost else engines.asrHost
            used += host
            val error = try {
                val record = stoppable {
                    withTimeoutOrNull(config.unitTimeoutMs) { measureUnit(entry, candidate, data, host) }
                } ?: throw UnitFailure(MurmurError.Timeout(Stage.BENCH))
                guarded(onError = { MurmurError.StorageFailed("benchmark") }) {
                    records.save(record)
                    if (record.result.valid) health.clear(record.artifactId, record.unit)
                }
                saved += record
                null
            } catch (f: UnitFailure) {
                if (f.markFailed) health.markFailed(candidate.artifact.id, candidate.unit, f.error.code)
                f.error
            }
            if (error != null) {
                // Leave the process empty before the next unit (its cold start shuts it down again anyway).
                withContext(NonCancellable) { runCatching { host.shutdown() } }
                emit(BenchProgress.UnitFailed(candidate.artifact.id, candidate.unit, error))
                if (firstError == null) firstError = error
            }
        }
        emit(if (saved.isEmpty()) BenchProgress.Failed(firstError ?: MurmurError.Internal("bench.no_record")) else BenchProgress.Finished(saved))
    }

    /** Runs [block]; [stop] cancels it at its next suspension point and surfaces as [StopRequested]. */
    private suspend fun <T> stoppable(block: suspend () -> T): T = coroutineScope {
        val scope = this
        val watcher = launch(start = CoroutineStart.UNDISPATCHED) {
            stoppedThrough.first { it >= activeRun }
            scope.cancel(StopRequested())
        }
        try {
            block()
        } catch (e: CancellationException) {
            if (stopped(activeRun)) throw StopRequested() else throw e
        } finally {
            watcher.cancel()
        }
    }

    private suspend fun FlowCollector<BenchProgress>.measureUnit(
        entry: ModelEntry, candidate: Candidate, data: BenchInputs, host: InferenceHost,
    ): BenchmarkRecord {
        val unitStart = clock.elapsedMs()
        val notes = linkedMapOf(BenchNotes.THERMAL_BEFORE to environment.thermalStatus())
        val spec = guarded { specs.specs(entry, candidate) }.orFail(markFailed = false)
        val result = when (entry.kind) {
            ModelKind.ASR -> measureAsr(entry, candidate, data.clips.filter { it.runsOn(entry.languages.codes) }, spec, host, notes)
            ModelKind.LLM -> measureLlm(entry, candidate, data.cleanupCase, spec, host, notes)
            ModelKind.OCR_DET, ModelKind.OCR_REC -> measureOcr(entry, candidate, data.pages, spec, host, notes)
            ModelKind.VAD -> throw UnitFailure(MurmurError.Internal("bench.vad"))
        }
        notes[BenchNotes.THERMAL_AFTER] = environment.thermalStatus()
        notes[BenchNotes.WALL_MS] = (clock.elapsedMs() - unitStart).toString()
        return BenchmarkRecord(
            artifactId = candidate.artifact.id,
            unit = candidate.unit,
            appVersion = environment.appVersion,
            buildFingerprint = environment.buildFingerprint,
            atEpochMs = clock.now().toEpochMilli(),
            result = result,
            notes = notes,
        )
    }

    // ---------------------------------------------------------------------------------------------------- ASR

    private suspend fun FlowCollector<BenchProgress>.measureAsr(
        entry: ModelEntry, candidate: Candidate, clips: List<BenchClip>, spec: BenchSpecs, host: InferenceHost,
        notes: MutableMap<String, String>,
    ): BenchmarkResultData.Asr {
        val asr = engines.asr
        val unit = candidate.unit
        lateinit var cold: ColdLoad
        var warmMs = 0L
        step(unit, BenchStep.LOAD) {
            cold = coldStart(host, spec.main.kind) { asr.load(spec.main, spec.vad) }
            // Warm load: same process, page cache warm (npu-runtime 8); the clips then run on this load.
            guarded { asr.unload() }
            val pid = (host.state.value as? HostState.Up)?.pid
            val t = clock.elapsedMs()
            guarded { asr.load(spec.main, spec.vad) }.let { if (it is Outcome.Err) throw loadFailure(it.error, spec.main.kind, pid) }
            warmMs = clock.elapsedMs() - t
        }
        val measured = clips.map { clip ->
            step(unit, clip.step, clip.id, clip.audioMs) {
                pass(entry, clip) // warm-up
                pass(entry, clip)
            }
        }
        val memory = unloadAndMeasure(unit, host, cold.baseline) { asr.unload() }
        val audioMs = measured.sumOf { it.audioMs }
        val processMs = measured.sumOf { it.processMs }.coerceAtLeast(1)
        val wers = measured.map { it.wer }
        val medianWer = Stats.median(wers)
        val invalid = BenchValidity.asr(medianWer, if (unit == ComputeUnit.CPU) null else cpuWer(entry))
        notes.putAll(memory.notes); notes[BenchNotes.LOAD_ENGINE_MS] = cold.info.loadMs.toString()
        measured.filter { !it.tailAnchored }.takeIf { it.isNotEmpty() }?.let { whole ->
            notes[BenchNotes.ASR_TAIL_WHOLE_DECODE] = whole.joinToString(",") { it.clipId }
        }
        return BenchmarkResultData.Asr(
            rtfx = audioMs.toDouble() / processMs,
            tailMs = Stats.medianLong(measured.map { it.tailMs }),
            loadMsCold = cold.ms,
            loadMsWarm = warmMs,
            peakMemMb = memory.peakMb,
            ranOn = cold.info.deviceName,
            perClip = measured.map { ClipResult(it.clipId, it.audioMs, it.processMs, it.wer) },
            valid = invalid == null,
            invalidReason = invalid,
        )
    }

    /** [tailAnchored] = false when the session reported no progress, so [tailMs] is the whole decode (upper bound). */
    private class Pass(
        val clipId: String, val audioMs: Long, val processMs: Long, val tailMs: Long, val tailAnchored: Boolean, val wer: Double,
    )

    /**
     * One clip through a fresh session as fast as accept() takes it. Decode time = first sample -> finish() returned.
     *
     * Tail = what a dictation waits for after the stop tap: the decode of the last segment only. Spoken in real time,
     * every earlier segment is decoded while the user is still talking; pushed at once, they are all still queued when
     * finish() is called, so the time from the push to finish() is the whole backlog, not the tail. The session's own
     * progress events mark where the backlog ends: the last Speech edge (a VAD segment closes and goes to the decoder),
     * Partial (streaming hypothesis) or Committed of an earlier segment that arrived before finish() returned. Tail =
     * finish() returned - that event. The final segment's Committed (index = segments - 1) comes after its own decode and
     * is not a marker. Without any marker the tail falls back to the whole decode (never under-reported).
     */
    private suspend fun pass(entry: ModelEntry, clip: BenchClip): Pass = coroutineScope {
        val hint = if (entry.languages.mode == LanguageMode.HINT && clip.languages.size == 1) clip.languages.single() else null
        // Partials are the only progress a streaming model without a VAD reports; batch models never send them.
        val options = AsrSessionOptions(language = hint, emitPartials = entry.asr?.streaming == true)
        val session = guarded { engines.asr.startSession(options) }
        val progress = ConcurrentLinkedQueue<Pair<AsrEvent, Long>>()
        val listener = launch(start = CoroutineStart.UNDISPATCHED) {
            try {
                session.events.collect { progress += it to clock.elapsedMs() }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                // An event stream that breaks only costs the tail its marker (the whole decode is reported instead).
            }
        }
        try {
            val pcm = clip.pcm16k
            val start = clock.elapsedMs()
            guarded {
                var i = 0
                while (i < pcm.size) {
                    val n = minOf(config.chunkSamples, pcm.size - i)
                    session.accept(PcmChunk(pcm.copyOfRange(i, i + n), i.toLong()))
                    i += n
                }
            }
            val finished = try {
                guarded { session.finish() }
            } catch (e: CancellationException) {
                session.cancel() // stopped mid-decode: do not leave the session decoding
                throw e
            }
            val end = clock.elapsedMs()
            val (text, segments) = when (finished) {
                is Outcome.Ok -> finished.value.text to finished.value.segments
                // A model that hears nothing in a clip full of speech is a wrong result, not a crash: WER 100 %.
                is Outcome.Err -> if (finished.error == MurmurError.NoSpeech) "" to 0 else throw UnitFailure(finished.error)
            }
            val marker = progress.filter { (e, at) -> at <= end && e.precedesFinalDecode(segments) }.maxOfOrNull { it.second }
            val tailFrom = maxOf(start, marker ?: start)
            Pass(clip.id, clip.audioMs, end - start, end - tailFrom, marker != null, WordErrorRate.wer(clip.reference, text))
        } finally {
            listener.cancel()
        }
    }

    /** Progress the session reports before it starts on the final segment (see [pass]). */
    private fun AsrEvent.precedesFinalDecode(segments: Int): Boolean = when (this) {
        is AsrEvent.Speech, is AsrEvent.Partial -> true
        is AsrEvent.Committed -> index < segments - 1
        is AsrEvent.Failure -> false
    }

    /** Median WER of the latest CPU record of the same model (any of its artifacts), for the "CPU + 5 pp" rule. */
    private suspend fun cpuWer(entry: ModelEntry): Double? =
        entry.artifacts.flatMap { guarded { records.latest(it.id).first() } }
            .filter { it.unit == ComputeUnit.CPU && it.result is BenchmarkResultData.Asr }
            .maxByOrNull { it.atEpochMs }
            ?.let { r -> (r.result as BenchmarkResultData.Asr).perClip.mapNotNull { it.wer } }
            ?.takeIf { it.isNotEmpty() }
            ?.let(Stats::median)

    // ---------------------------------------------------------------------------------------------------- LLM

    private suspend fun FlowCollector<BenchProgress>.measureLlm(
        entry: ModelEntry, candidate: Candidate, case: CleanupCase, spec: BenchSpecs, host: InferenceHost,
        notes: MutableMap<String, String>,
    ): BenchmarkResultData.Llm {
        val llm = engines.llm
        val unit = candidate.unit
        val traits = entry.llm ?: throw UnitFailure(MurmurError.Internal("bench.llm_traits"))
        val modelSha = spec.modelSha256 ?: throw UnitFailure(MurmurError.Internal("bench.model_sha"))
        val pre = guarded { guard.precheck(case.transcript) }
        if (pre.skip) throw UnitFailure(MurmurError.Internal("bench.case_precheck:${pre.reason}"))
        val rendered = guarded {
            renderer.render(RenderInput(assets.profile(traits.promptProfile), traits.template, case.context, pre.transcript))
        }
        lateinit var cold: ColdLoad
        step(unit, BenchStep.LOAD) {
            cold = coldStart(host, spec.main.kind) { llm.load(spec.main) }
            val nativeBuild = guarded { host.probe() }.facts[ProcessProbe.FACT_NATIVE_BUILD] ?: "unknown"
            val key = PrefixCacheKey.of(
                modelSha, rendered.prefixKey, unit, nativeBuild, traits.nCtx,
                kvType = spec.main.options[LoadOptions.KV_TYPE] ?: "f16",
                nBatch = spec.main.options[LoadOptions.N_BATCH]?.toIntOrNull() ?: 512,
            )
            val prefix = guarded { llm.setPrefix(PrefixSpec(rendered.prefix, key)) }.orFail(markFailed = false)
            notes[BenchNotes.LLM_PREFIX_TOKENS] = prefix.tokens.toString()
            notes[BenchNotes.LLM_PREFIX_MS] = prefix.prefillMs.toString()
            notes[BenchNotes.LLM_PREFIX_RESTORED] = prefix.restoredFromDisk.toString()
        }
        val chars = pre.transcript.length
        val request = LlmRequest(
            suffix = rendered.suffix,
            // CleanupBudget (ADR-045): output cap 1.5 x and floor 1.1 x the transcript's estimated tokens.
            maxTokens = ceil(chars * TOKENS_PER_CHAR * 1.5).toInt() + 64,
            minTokens = ceil(chars * TOKENS_PER_CHAR * 1.1).toInt() + 16,
        )
        lateinit var done: LlmEvent.Done
        lateinit var synthetic: app.murmur.core.engine.SyntheticBench
        step(unit, BenchStep.CLEANUP_CASE) {
            // Wall time of each part, so the step's duration can be accounted for (ADR-051 budget: < 60 s per unit).
            fun timed(key: String, t: Long) { notes[key] = (clock.elapsedMs() - t).toString() }
            var t = clock.elapsedMs()
            generate(request) // warm-up
            timed(BenchNotes.LLM_WARMUP_MS, t)
            t = clock.elapsedMs()
            done = generate(request)
            timed(BenchNotes.LLM_CASE_MS, t)
            t = clock.elapsedMs()
            synthetic = guarded { llm.bench(config.syntheticPrompt, config.syntheticGen, config.syntheticReps) }.orFail(markFailed = false)
            timed(BenchNotes.LLM_SYNTHETIC_MS, t)
        }
        val memory = unloadAndMeasure(unit, host, cold.baseline) { llm.unload() }
        val stats = done.stats
        val verdict = guarded { guard.guard(pre.transcript, done.text, case.context) }
        val invalid = BenchValidity.llm(
            output = done.text,
            guardAccepted = verdict.action == GuardAction.ACCEPT,
            stopReason = stats.stopReason,
            expectCyrillic = true,
            guardReason = verdict.reason,
            cpuGuardAccepted = if (unit == ComputeUnit.CPU) null else cpuGuardAccepted(entry),
        )
        notes.putAll(memory.notes)
        notes[BenchNotes.LOAD_ENGINE_MS] = cold.info.loadMs.toString()
        notes[BenchNotes.LLM_PROMPT_TOKENS] = stats.promptTokens.toString()
        notes[BenchNotes.LLM_GENERATED_TOKENS] = stats.generatedTokens.toString()
        notes[BenchNotes.LLM_PP_TPS] = "%.1f".format(java.util.Locale.ROOT, synthetic.ppTps)
        notes[BenchNotes.LLM_TG_TPS] = "%.1f".format(java.util.Locale.ROOT, synthetic.tgTps)
        notes[BenchNotes.LLM_GUARD_REASON] = verdict.reason
        return BenchmarkResultData.Llm(
            // LlmStats.decodeMs runs from the first token to the last (murmur_llm.h): n - 1 decode steps.
            decodeTps = perSecond(stats.generatedTokens - 1, stats.decodeMs),
            prefillTps = perSecond(stats.promptTokens, stats.prefillMs),
            ttftMs = stats.ttftMs,
            e2eMs = stats.totalMs,
            loadMsCold = cold.ms,
            peakMemMb = memory.peakMb,
            ranOn = cold.info.deviceName,
            unitShare = cold.info.unitShare,
            guardAction = verdict.action.name.lowercase(),
            stopReason = stats.stopReason,
            valid = invalid == null,
            invalidReason = invalid,
        )
    }

    private suspend fun generate(request: LlmRequest): LlmEvent.Done {
        var done: LlmEvent.Done? = null
        try {
            guarded {
                engines.llm.generate(request).collect { e ->
                    when (e) {
                        is LlmEvent.Piece -> Unit
                        is LlmEvent.Done -> done = e
                        is LlmEvent.Failed -> throw UnitFailure(e.error)
                    }
                }
            }
        } catch (e: CancellationException) {
            engines.llm.cancel() // stopped mid-generation: the engine thread stops at its next token
            throw e
        }
        return done ?: throw UnitFailure(MurmurError.LlmFailed("no_done"))
    }

    /** Whether the latest CPU LLM record of the same model passed the guard (accelerators must not regress it). */
    private suspend fun cpuGuardAccepted(entry: ModelEntry): Boolean? =
        entry.artifacts.flatMap { guarded { records.latest(it.id).first() } }
            .filter { it.unit == ComputeUnit.CPU && it.result is BenchmarkResultData.Llm }
            .maxByOrNull { it.atEpochMs }
            ?.let { (it.result as BenchmarkResultData.Llm).guardAction == "accept" }

    private fun perSecond(count: Int, ms: Double): Double = if (ms > 0 && count > 0) count * 1000.0 / ms else 0.0

    // ---------------------------------------------------------------------------------------------------- OCR

    private suspend fun FlowCollector<BenchProgress>.measureOcr(
        entry: ModelEntry, candidate: Candidate, pages: List<BenchPage>, spec: BenchSpecs, host: InferenceHost,
        notes: MutableMap<String, String>,
    ): BenchmarkResultData.Ocr {
        val ocr = engines.ocr
        val unit = candidate.unit
        val partner = spec.ocrPartner ?: throw UnitFailure(MurmurError.Internal("bench.ocr_pair"))
        val (det, rec) = if (entry.kind == ModelKind.OCR_DET) spec.main to partner else partner to spec.main
        lateinit var cold: ColdLoad
        step(unit, BenchStep.LOAD) { cold = coldStart(host, det.kind) { ocr.loadOcr(det, rec) } }
        val compared = mutableListOf<Pair<ScreenText, ScreenText>>()
        val times = pages.map { page ->
            step(unit, BenchStep.OCR_PAGE, page.id) {
                val image = try { page.image() } catch (e: java.io.IOException) {
                    throw UnitFailure(MurmurError.Internal("bench.page:${page.id}"))
                }
                guarded { ocr.recognize(image) }.orFail(markFailed = false) // warm-up
                val t = clock.elapsedMs()
                val text = guarded { ocr.recognize(image) }.orFail(markFailed = false)
                val ms = clock.elapsedMs() - t
                compared += page.golden to text
                notes[BenchNotes.OCR_MS_PREFIX + page.id] = ms.toString()
                ms
            }
        }
        val memory = unloadAndMeasure(unit, host, cold.baseline) { ocr.unloadOcr() }
        // One gate over all pages (the README tolerance is a suite rule, see OcrGoldenMatch).
        val invalid = OcrGoldenMatch.check(compared)
        notes.putAll(memory.notes)
        notes[BenchNotes.LOAD_ENGINE_MS] = cold.info.loadMs.toString()
        return BenchmarkResultData.Ocr(
            msPerScreenshot = Stats.medianLong(times),
            boxes = compared.sumOf { it.second.lines.size },
            loadMsCold = cold.ms,
            peakMemMb = memory.peakMb,
            ranOn = cold.info.deviceName,
            valid = invalid == null,
            invalidReason = invalid,
        )
    }

    // ---------------------------------------------------------------------------------------------------- shared

    private class ColdLoad(val info: LoadInfo, val ms: Long, val baseline: ProcessMemory)

    /**
     * Kill the process, start it again and load: the cold path a dictation after the idle exit takes. The time covers
     * process start (bind) and load; the baseline memory reading in between is not counted.
     */
    private suspend fun coldStart(host: InferenceHost, kind: ModelKind, load: suspend () -> Outcome<LoadInfo>): ColdLoad {
        guarded { host.shutdown() }
        val t0 = clock.elapsedMs()
        if (!guarded { host.connect() }) throw UnitFailure(MurmurError.ProcessDied(host.process))
        val connectMs = clock.elapsedMs() - t0
        val baseline = guarded { host.memory() }
        val pid = (host.state.value as? HostState.Up)?.pid
        val t1 = clock.elapsedMs()
        val info = guarded { load() }.let {
            when (it) {
                is Outcome.Ok -> it.value
                is Outcome.Err -> throw loadFailure(it.error, kind, pid)
            }
        }
        return ColdLoad(info, connectMs + clock.elapsedMs() - t1, baseline)
    }

    private class Memory(val peakMb: Long, val notes: Map<String, String>)

    /**
     * Reads PSS and availMem while loaded, unloads, then reads VmHWM (the process's peak RSS, which survives the
     * unload). Peak memory = max(VmHWM, PSS); the availMem drop (DSP buffers RSS may miss) is kept in the notes.
     */
    private suspend fun FlowCollector<BenchProgress>.unloadAndMeasure(
        unit: ComputeUnit, host: InferenceHost, baseline: ProcessMemory, unload: suspend () -> Unit,
    ): Memory = step(unit, BenchStep.UNLOAD_MEMORY) {
        val loaded = guarded { host.memory() }
        guarded { unload() }
        val after = guarded { host.memory() }
        val hwmKb = maxOf(after.vmHwmKb, loaded.vmHwmKb)
        val drop = (baseline.systemAvailMb - loaded.systemAvailMb).coerceAtLeast(0)
        Memory(
            peakMb = maxOf(hwmKb, loaded.pssKb) / 1024,
            notes = mapOf(
                BenchNotes.MEM_VM_HWM_KB to hwmKb.toString(),
                BenchNotes.MEM_PSS_KB to loaded.pssKb.toString(),
                BenchNotes.MEM_AVAIL_DROP_MB to drop.toString(),
            ),
        )
    }

    private suspend fun <T> FlowCollector<BenchProgress>.step(
        unit: ComputeUnit, step: BenchStep, sampleId: String? = null, audioMs: Long? = null, block: suspend () -> T,
    ): T {
        emit(BenchProgress.Step(step, unit, 0, done = false, sampleId = sampleId, audioMs = audioMs))
        checkStop()
        val t = clock.elapsedMs()
        val r = block()
        emit(BenchProgress.Step(step, unit, clock.elapsedMs() - t, done = true, sampleId = sampleId, audioMs = audioMs))
        checkStop()
        return r
    }

    /** Between steps: a stop requested while the step ran without suspending (or by the collector) ends the run here. */
    private fun checkStop() {
        if (stopped(activeRun)) throw StopRequested()
    }

    /** Engine calls must not throw (Outcome everywhere); if one does, it fails this unit instead of the whole run. */
    private suspend fun <T> guarded(
        onError: (Exception) -> MurmurError = { MurmurError.Internal("bench.exception:${it.javaClass.simpleName}") },
        block: suspend () -> T,
    ): T = try {
        block()
    } catch (e: CancellationException) {
        throw e
    } catch (e: UnitFailure) {
        throw e
    } catch (e: Exception) {
        throw UnitFailure(onError(e))
    }

    private fun <T> Outcome<T>.orFail(markFailed: Boolean): T = when (this) {
        is Outcome.Ok -> value
        is Outcome.Err -> throw UnitFailure(error, markFailed)
    }

    /** ADR-066 / ADR-083: a load error marks the unit failed unless the device ran out of memory. */
    private suspend fun loadFailure(error: MurmurError, kind: ModelKind, pid: Int?): UnitFailure {
        val e = LoadFailures.classify(error, kind, pid, exits)
        return UnitFailure(e, markFailed = LoadFailures.judgesUnit(e))
    }

    private class UnitFailure(val error: MurmurError, val markFailed: Boolean = false) : Exception(error.code)

    private class StopRequested : CancellationException("benchmark stopped")

    private companion object {
        /** CPU first: an accelerator's WER is judged against a CPU record of the same run ("CPU + 5 pp"). */
        val UNIT_ORDER = listOf(ComputeUnit.CPU, ComputeUnit.GPU, ComputeUnit.NPU)
        /** CleanupBudget.TOKENS_PER_CHAR (core:dictation sits above this module). */
        const val TOKENS_PER_CHAR = 0.40
    }
}
