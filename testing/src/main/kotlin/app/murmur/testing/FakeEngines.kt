package app.murmur.testing

import app.murmur.core.audio.PcmChunk
import app.murmur.core.audio.PcmFormat
import app.murmur.core.engine.AsrEngine
import app.murmur.core.engine.AsrEvent
import app.murmur.core.engine.AsrSession
import app.murmur.core.engine.AsrSessionOptions
import app.murmur.core.engine.Engines
import app.murmur.core.engine.HostState
import app.murmur.core.engine.InferenceHost
import app.murmur.core.engine.LlmEngine
import app.murmur.core.engine.LlmEvent
import app.murmur.core.engine.LlmRequest
import app.murmur.core.engine.LlmStats
import app.murmur.core.engine.LoadInfo
import app.murmur.core.engine.LoadSpec
import app.murmur.core.engine.OcrService
import app.murmur.core.engine.PrefixInfo
import app.murmur.core.engine.PrefixSpec
import app.murmur.core.engine.ProcessMemory
import app.murmur.core.engine.ProcessProbe
import app.murmur.core.engine.SyntheticBench
import app.murmur.core.engine.Transcript
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.InferProcess
import app.murmur.core.model.MurmurError
import app.murmur.core.model.Outcome
import app.murmur.core.ocr.OcrLine
import app.murmur.core.ocr.ScreenImage
import app.murmur.core.ocr.ScreenText
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withTimeoutOrNull

/**
 * A fake inference process. [crash] behaves like a native crash: state Crashed(n), every in-flight session /
 * generation of the engines attached to this host fails with ProcessDied, loaded models are gone, and [deaths]
 * emits so the orchestrator's death watcher sees it (ARCHITECTURE 4.4). [connect] after a crash brings it back up.
 */
class FakeInferenceHost(
    override val process: InferProcess,
    var units: Set<ComputeUnit> = setOf(ComputeUnit.CPU),
    var facts: Map<String, String> = mapOf(ProcessProbe.FACT_NATIVE_BUILD to "fake-1"),
) : InferenceHost {
    private val _state = MutableStateFlow<HostState>(HostState.Down)
    override val state: StateFlow<HostState> = _state.asStateFlow()
    private val _deaths = MutableSharedFlow<Int>(extraBufferCapacity = 8)
    val deaths: Flow<Int> = _deaths.asSharedFlow()
    var idleExitMs: Long? = null; private set
    /** Every scheduleIdleExit(ms) in call order (the last one is [idleExitMs]). */
    val idleExitHistory = mutableListOf<Long>()
    var shutdowns = 0; private set
    var connects = 0; private set
    var crashes = 0; private set
    private val listeners = mutableListOf<() -> Unit>()
    /** The process "dies" on the next connect() while > 0 (simulates a crash during bind/load). */
    var crashOnConnect = 0

    internal fun onDeath(l: () -> Unit) { listeners += l }

    override suspend fun connect(): Boolean {
        connects++
        if (crashOnConnect > 0) { crashOnConnect--; crash(); return false }
        _state.value = HostState.Up(4242 + connects)
        return true
    }
    override suspend fun probe() = ProcessProbe(process, app.murmur.core.catalog.RuntimeKeys.known, units, facts)
    override suspend fun memory() = ProcessMemory(100_000, 90_000, 80_000, 4_000)
    override fun scheduleIdleExit(ms: Long) { idleExitMs = ms; idleExitHistory += ms }
    /** How long shutdown() takes (unbind + process kill), in coroutine time. */
    var shutdownDelayMs: Long = 0

    override suspend fun shutdown() {
        shutdowns++
        if (shutdownDelayMs > 0) delay(shutdownDelayMs)
        _state.value = HostState.Down
        listeners.forEach { it() }
    }

    /**
     * Simulate a native crash of the process. [publish] = false fails the in-flight calls (EPIPE / DeadObject) without
     * the state notice yet; [publishCrash] delivers it later (binderDied / onServiceDisconnected arriving after the
     * failing call, an order the binder does not fix).
     */
    fun crash(publish: Boolean = true) {
        crashes++
        if (publish) _state.value = HostState.Crashed(crashes)
        listeners.forEach { it() }
        _deaths.tryEmit(crashes)
    }

    /** The late state notice of the last [crash] (publish = false). */
    fun publishCrash() { _state.value = HostState.Crashed(crashes) }
}

/**
 * Scripted ASR: the transcript is [script] (or the next item of [scripts] per session). Speech is "present"
 * when any sample has |x| >= [speechAbs]; with no speech, finish() returns Err(NoSpeech). [loadErrors] fails loads
 * per unit (NPU -> CPU fallback tests); [loadDelayMs] / [finishDelayMs] use coroutine time (virtual in runTest).
 */
class FakeAsrEngine(
    var script: String = "привет это тестовая диктовка",
    val scripts: ArrayDeque<String> = ArrayDeque(),
    var loadError: MurmurError? = null,
    val loadErrors: MutableMap<ComputeUnit, MurmurError> = mutableMapOf(),
    var finishError: MurmurError? = null,
    var loadDelayMs: Long = 0,
    var finishDelayMs: Long = 0,
    /** A Committed event every N accepted 20 ms chunks (25 = every 0.5 s), as VAD segments would. */
    var committedEveryChunks: Int = 25,
    private val speechAbs: Int = 500,
    val host: FakeInferenceHost? = null,
) : AsrEngine {
    var loads = 0; private set
    /** Every load call's unit, failed ones included (loadedUnits holds the successful ones). */
    val loadAttempts = mutableListOf<ComputeUnit>()
    val loadedUnits = mutableListOf<ComputeUnit>()
    val loadedSpecs = mutableListOf<LoadSpec>()
    var loaded: LoadSpec? = null; private set
    val sessions = mutableListOf<FakeAsrSession>()

    init { host?.onDeath { loaded = null; sessions.forEach { it.die() } } }

    override suspend fun load(asr: LoadSpec, vad: LoadSpec?): Outcome<LoadInfo> {
        loads++
        loadAttempts += asr.unit
        if (loadDelayMs > 0) delay(loadDelayMs)
        (loadErrors[asr.unit] ?: loadError)?.let { return Outcome.Err(it) }
        loaded = asr
        loadedUnits += asr.unit
        loadedSpecs += asr
        return Outcome.Ok(LoadInfo(asr.unit, loadMs = 5, deviceName = "fake-${asr.unit.name.lowercase()}"))
    }

    override suspend fun startSession(options: AsrSessionOptions): AsrSession =
        FakeAsrSession(scripts.removeFirstOrNull() ?: script, finishError, speechAbs, loaded?.unit ?: ComputeUnit.CPU,
            committedEveryChunks, finishDelayMs,
            streaming = loaded?.runtime == app.murmur.core.catalog.RuntimeKeys.SHERPA_ONLINE_TRANSDUCER).also { sessions += it }

    override suspend fun unload() { loaded = null }
}

class FakeAsrSession(
    private val text: String,
    private val finishError: MurmurError?,
    private val speechAbs: Int,
    private val unit: ComputeUnit,
    private val committedEveryChunks: Int = 25,
    private val finishDelayMs: Long = 0,
    private val streaming: Boolean = false,
) : AsrSession {
    private val _events = MutableSharedFlow<AsrEvent>(replay = 64, extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST)
    override val events: Flow<AsrEvent> = _events.asSharedFlow()
    var samples = 0L; private set
    var speechSamples = 0L; private set
    var cancelled = false; private set
    var dead = false; private set
    private var chunks = 0
    private var committed = 0
    private val words = text.split(' ')
    private val deathSignal = CompletableDeferred<Unit>()

    override fun accept(chunk: PcmChunk) {
        if (dead) return
        samples += chunk.samples.size
        if (chunk.samples.any { kotlin.math.abs(it.toInt()) >= speechAbs }) speechSamples += chunk.samples.size
        chunks++
        if (committedEveryChunks > 0 && chunks % committedEveryChunks == 0 && committed < words.size - 1) {
            val endMs = PcmFormat.samplesToMs(samples)
            _events.tryEmit(AsrEvent.Committed(committed, words[committed], endMs - 500, endMs, "ru"))
            committed++
        }
    }

    internal fun die() {
        if (dead) return
        dead = true
        _events.tryEmit(AsrEvent.Failure(MurmurError.ProcessDied(InferProcess.ASR)))
        deathSignal.complete(Unit)
    }

    override suspend fun finish(): Outcome<Transcript> {
        if (dead) return Outcome.Err(MurmurError.ProcessDied(InferProcess.ASR))
        if (finishDelayMs > 0) {
            val diedFirst = withTimeoutOrNull(finishDelayMs) { deathSignal.await(); true } ?: false
            if (diedFirst) return Outcome.Err(MurmurError.ProcessDied(InferProcess.ASR))
        }
        finishError?.let { return Outcome.Err(it) }
        val audioMs = PcmFormat.samplesToMs(samples)
        val speechMs = if (streaming) 0L else PcmFormat.samplesToMs(speechSamples)
        if (!streaming && speechMs < 500) return Outcome.Err(MurmurError.NoSpeech)
        val lang = if (streaming) null else "ru"
        _events.tryEmit(AsrEvent.Committed(committed, words.drop(committed).joinToString(" "), 0, audioMs, lang))
        return Outcome.Ok(Transcript(text, lang, committed + 1, audioMs, speechMs, tailDecodeMs = 10, unit = unit))
    }

    override fun cancel() { cancelled = true }
}

/**
 * Deterministic "cleanup": collapse spaces, capitalise the first letter, end with a period. Streams one piece per
 * word with [delayPerPieceMs]; [hang] never finishes (LLM timeout tests); [loadErrors] per unit (NPU -> CPU
 * fallback); [stopReason] overrides the final LlmStats (truncation tests); [contextLimit] = n_ctx of the fake: a
 * request whose suffix (chars/2.5) + minTokens exceeds it fails with LlmContextOverflow.
 */
class FakeLlmEngine(
    var loadError: MurmurError? = null,
    val loadErrors: MutableMap<ComputeUnit, MurmurError> = mutableMapOf(),
    var output: ((String) -> String)? = null,
    var delayPerPieceMs: Long = 0,
    var loadDelayMs: Long = 0,
    var prefixDelayMs: Long = 0,
    var hang: Boolean = false,
    var stopReason: Int = LlmStats.STOP_EOG,
    var contextLimit: Int = Int.MAX_VALUE,
    val host: FakeInferenceHost? = null,
) : LlmEngine {
    /** setPrefix fails with this (the model stays loaded). */
    var prefixError: MurmurError? = null
    var prefix: PrefixSpec? = null; private set
    var loadedUnit: ComputeUnit? = null; private set
    /** Every load call's unit, failed ones included (loadedUnits holds the successful ones). */
    val loadAttempts = mutableListOf<ComputeUnit>()
    val loadedUnits = mutableListOf<ComputeUnit>()
    /** Every setPrefix call (snapshot builds and restores). */
    val prefixCalls = mutableListOf<PrefixSpec>()
    var requests = mutableListOf<LlmRequest>()
    var cancelled = 0; private set
    var benches = 0; private set
    private var died = CompletableDeferred<Unit>()

    init { host?.onDeath { loadedUnit = null; prefix = null; died.complete(Unit); died = CompletableDeferred() } }

    override suspend fun load(spec: LoadSpec): Outcome<LoadInfo> {
        loadAttempts += spec.unit
        if (loadDelayMs > 0) delay(loadDelayMs)
        (loadErrors[spec.unit] ?: loadError)?.let { return Outcome.Err(it) }
        loadedUnit = spec.unit
        loadedUnits += spec.unit
        return Outcome.Ok(LoadInfo(spec.unit, 5, "fake-llm-${spec.unit.name.lowercase()}"))
    }

    override suspend fun setPrefix(prefix: PrefixSpec): Outcome<PrefixInfo> {
        prefixCalls += prefix
        if (prefixDelayMs > 0) delay(prefixDelayMs)
        prefixError?.let { return Outcome.Err(it) }
        this.prefix = prefix
        return Outcome.Ok(PrefixInfo((prefix.text.length * 0.27).toInt(), 1, restoredFromDisk = false, bytes = 0))
    }

    override fun generate(request: LlmRequest): Flow<LlmEvent> = flow {
        requests += request
        val deathNow = died
        val needed = (request.suffix.length * 0.4).toInt() + request.minTokens
        if (needed > contextLimit) { emit(LlmEvent.Failed(MurmurError.LlmContextOverflow(needed, contextLimit))); return@flow }
        if (hang) awaitCancellation()
        val transcript = request.suffix.substringAfter("<transcript>\n", request.suffix).substringBefore("\n</transcript>")
        val out = output?.invoke(transcript) ?: cleanup(transcript)
        val unit = loadedUnit ?: ComputeUnit.CPU
        for (piece in out.split(' ')) {
            if (delayPerPieceMs > 0) delay(delayPerPieceMs)
            if (deathNow.isCompleted) { emit(LlmEvent.Failed(MurmurError.ProcessDied(InferProcess.LLM))); return@flow }
            emit(LlmEvent.Piece("$piece "))
        }
        emit(LlmEvent.Done(out, LlmStats(request.suffix.length / 4, out.length / 4, 0.1, 1.0, 2.0, 3.0, 5.0, stopReason, unit)))
    }

    override fun cancel() { cancelled++ }
    override suspend fun unload() { prefix = null; loadedUnit = null }
    override suspend fun bench(nPrompt: Int, nGen: Int, reps: Int): Outcome<SyntheticBench> {
        benches++
        return Outcome.Ok(SyntheticBench(ppTps = 500.0, tgTps = 50.0, ppMs = nPrompt * 2.0, tgMs = nGen * 20.0))
    }

    companion object {
        fun cleanup(t: String): String {
            val s = t.trim().replace(Regex("\\s+"), " ")
            if (s.isEmpty()) return s
            val c = s.replaceFirstChar { it.uppercase() }
            return if (c.last() in ".!?") c else "$c."
        }
    }
}

/** [delayMs] simulates a late OCR (ocrBudgetMs tests); [error] a failing one. */
class FakeOcrService(
    var lines: List<OcrLine> = listOf(OcrLine("Лёша Козлов", 0.97, listOf(156, 212, 498, 268))),
    var delayMs: Long = 0,
    var error: MurmurError? = null,
) : OcrService {
    var calls = 0; private set
    override suspend fun loadOcr(det: LoadSpec, rec: LoadSpec): Outcome<LoadInfo> = Outcome.Ok(LoadInfo(ComputeUnit.CPU, 1, "fake-ocr"))
    override suspend fun recognize(image: ScreenImage): Outcome<ScreenText> {
        calls++
        if (delayMs > 0) delay(delayMs)
        error?.let { return Outcome.Err(it) }
        return Outcome.Ok(ScreenText(listOf(image.width, image.height), lines))
    }
    override suspend fun unloadOcr() {}
}

/** Engines wired to their hosts, so that host.crash() fails in-flight work of that process. */
class FakeEngines(
    override val asrHost: FakeInferenceHost = FakeInferenceHost(InferProcess.ASR),
    override val llmHost: FakeInferenceHost = FakeInferenceHost(InferProcess.LLM),
    override val asr: FakeAsrEngine = FakeAsrEngine(host = asrHost),
    override val llm: FakeLlmEngine = FakeLlmEngine(host = llmHost),
    override val ocr: FakeOcrService = FakeOcrService(),
) : Engines

/**
 * Exit records of the fake processes (ADR-083): [lowMemory] = every death the code asks about was an lmkd kill.
 * [asked] lists the pids it was asked about.
 */
class FakeProcessExits(var lowMemory: Boolean = false) : app.murmur.core.engine.ProcessExits {
    val asked = mutableListOf<Int>()
    override suspend fun killedForMemory(pid: Int): Boolean { asked += pid; return lowMemory }
}
