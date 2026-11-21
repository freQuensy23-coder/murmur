package app.murmur.core.dictation

import app.murmur.core.catalog.UnitHealthRepository
import app.murmur.core.engine.Engines
import app.murmur.core.engine.HostState
import app.murmur.core.engine.InferenceHost
import app.murmur.core.engine.LoadFailures
import app.murmur.core.engine.LoadInfo
import app.murmur.core.engine.LoadOptions
import app.murmur.core.engine.LoadSpec
import app.murmur.core.engine.PrefixCacheKey
import app.murmur.core.engine.PrefixSpec
import app.murmur.core.engine.ProcessExits
import app.murmur.core.engine.ProcessProbe
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.InferProcess
import app.murmur.core.model.MurmurError
import app.murmur.core.model.Outcome

/**
 * What is loaded in which inference process, and the load-with-fallback procedure of ARCHITECTURE 4.4 / ADR-050.
 * A load is skipped when the same spec is already loaded in the same process instance (warm path, D5); a death or a
 * shutdown forgets it. [asrLoading] / [llmLoading] tell the death watcher that a load is in flight: a death during
 * a load is that load's error (it falls back), not a ProcessDied event; one the system made for memory ([exits],
 * ADR-083) is OutOfMemory. Used only on the serial dispatcher.
 */
internal class EngineKeeper(
    private val engines: Engines,
    private val health: UnitHealthRepository,
    private val exits: ProcessExits = ProcessExits.NONE,
) {
    var asrLoading = 0; private set
    var llmLoading = 0; private set

    private data class WarmAsr(val pid: Int, val spec: LoadSpec, val vad: LoadSpec?, val info: LoadInfo)
    private data class WarmOcr(val pid: Int, val det: LoadSpec, val rec: LoadSpec)
    private data class WarmLlm(val pid: Int, val spec: LoadSpec, val info: LoadInfo, val prefixKey: String? = null, val prefixTokens: Int = 0)

    private var warmAsr: WarmAsr? = null
    private var warmOcr: WarmOcr? = null
    private var warmLlm: WarmLlm? = null
    private var nativeBuild: Pair<Int, String>? = null

    data class AsrLoaded(val spec: LoadSpec, val info: LoadInfo, val fellBack: Boolean)
    data class LlmLoaded(val spec: LoadSpec, val info: LoadInfo, val prefixTokens: Int, val fellBack: Boolean)

    fun forget(process: InferProcess) {
        when (process) {
            InferProcess.ASR -> { warmAsr = null; warmOcr = null }
            InferProcess.LLM -> { warmLlm = null; nativeBuild = null }
        }
    }

    suspend fun shutdown(process: InferProcess) {
        forget(process)
        host(process).shutdown()
    }

    private fun host(p: InferProcess): InferenceHost = if (p == InferProcess.ASR) engines.asrHost else engines.llmHost

    /** Bound process id, binding (once more after a failed bind) when it is not up; null = it cannot be bound. */
    private suspend fun up(host: InferenceHost): Int? {
        (host.state.value as? HostState.Up)?.let { return it.pid }
        if (!host.connect() && !host.connect()) return null
        return (host.state.value as? HostState.Up)?.pid ?: UNKNOWN_PID
    }

    /** ARCHITECTURE 4.4 row 3: [primary]; on a load error UnitHealth.markFailed and [fallback] (replay is the caller's). */
    suspend fun loadAsr(primary: LoadSpec, fallback: LoadSpec?, vad: LoadSpec?): Outcome<AsrLoaded> {
        asrLoading++
        try {
            up(engines.asrHost) ?: return Outcome.Err(MurmurError.ProcessDied(InferProcess.ASR))
            val first = loadAsrOn(primary, vad)
            if (first is Outcome.Ok) return Outcome.Ok(AsrLoaded(primary, first.value, fellBack = false))
            val error = (first as Outcome.Err).error
            mark(primary, error)
            if (fallback == null || fallback == primary) {
                // ARCHITECTURE 4.4: an NPU unit error without a CPU artifact is UnitUnavailable ("Download the CPU
                // version"); running out of memory is not about the unit and stays OutOfMemory (ADR-083).
                return Outcome.Err(if (primary.unit == ComputeUnit.NPU && judgesUnit(error)) MurmurError.UnitUnavailable(ComputeUnit.NPU) else error)
            }
            up(engines.asrHost) ?: return Outcome.Err(MurmurError.ProcessDied(InferProcess.ASR))
            return when (val second = loadAsrOn(fallback, vad)) {
                is Outcome.Ok -> Outcome.Ok(AsrLoaded(fallback, second.value, fellBack = true))
                is Outcome.Err -> { mark(fallback, second.error); second }
            }
        } finally {
            asrLoading--
        }
    }

    /**
     * One Preparing trial on one unit (ADR-066, ADR-067). [error] null = it ran. [unitFailed] = the error says the
     * unit cannot run the artifact (the same rule as [mark]: a load error other than OutOfMemory); the caller marks
     * UnitHealth. Any other error says nothing about the unit: the process could not be bound, OutOfMemory, a
     * prefix error after a good load, a timeout of the proof decode.
     */
    data class Trial(val error: MurmurError?, val unitFailed: Boolean) {
        companion object {
            val OK = Trial(null, unitFailed = false)
            fun load(error: MurmurError) = Trial(error, judgesUnit(error))
            fun untested(error: MurmurError) = Trial(error, unitFailed = false)
        }
    }

    /** One spec, no fallback and no UnitHealth mark (Preparing marks per unit itself from the [Trial]). */
    suspend fun trialAsr(spec: LoadSpec, vad: LoadSpec?): Trial {
        asrLoading++
        try {
            up(engines.asrHost) ?: return Trial.untested(MurmurError.ProcessDied(InferProcess.ASR))
            return when (val r = loadAsrOn(spec, vad)) {
                is Outcome.Ok -> Trial.OK
                is Outcome.Err -> Trial.load(r.error)
            }
        } finally {
            asrLoading--
        }
    }

    /** Preparing: the OCR pair on the det/rec units, no UnitHealth mark. */
    suspend fun trialOcr(det: LoadSpec, rec: LoadSpec): Trial {
        asrLoading++
        try {
            val pid = up(engines.asrHost) ?: return Trial.untested(MurmurError.ProcessDied(InferProcess.ASR))
            if (warmOcr == WarmOcr(pid, det, rec) && pid != UNKNOWN_PID) return Trial.OK
            return when (val r = engines.ocr.loadOcr(det, rec)) {
                is Outcome.Ok -> { warmOcr = WarmOcr(pid, det, rec); Trial.OK }
                is Outcome.Err -> { warmOcr = null; Trial.load(LoadFailures.classify(r.error, det.kind, pid, exits)) }
            }
        } finally {
            asrLoading--
        }
    }

    /** Preparing: load [spec] and build the snapshot for [prefix], no fallback and no UnitHealth mark. */
    suspend fun trialLlm(llm: ActiveLlm, spec: LoadSpec, prefix: String): Trial {
        llmLoading++
        try {
            up(engines.llmHost) ?: return Trial.untested(MurmurError.ProcessDied(InferProcess.LLM))
            return when (val r = loadLlmOn(llm, spec, prefix)) {
                is Step.Loaded -> Trial.OK
                is Step.PrefixFailed -> Trial.untested(r.error)
                is Step.LoadFailed -> Trial.load(r.error)
            }
        } finally {
            llmLoading--
        }
    }

    /**
     * ARCHITECTURE 4.4: a load error, a runtime/op rejection or a death during the load means the unit cannot run the
     * artifact, and the selector skips it until the app version changes (ADR-050). Running out of memory says nothing
     * about the unit (other apps hold the memory now), so it is not marked: the next press tries it again.
     */
    private suspend fun mark(spec: LoadSpec, error: MurmurError) {
        if (judgesUnit(error)) health.markFailed(spec.artifactId, spec.unit, error.code)
    }

    /** The resolved LLM is already loaded in the running :llm (no extra memory needed for it). */
    fun llmWarm(llm: ActiveLlm): Boolean {
        val pid = (engines.llmHost.state.value as? HostState.Up)?.pid ?: return false
        val w = warmLlm ?: return false
        return w.pid == pid && w.spec.artifactId == llm.spec.artifactId && w.spec.unit == llm.spec.unit
    }

    private suspend fun loadAsrOn(spec: LoadSpec, vad: LoadSpec?): Outcome<LoadInfo> {
        val pid = (engines.asrHost.state.value as? HostState.Up)?.pid ?: UNKNOWN_PID
        warmAsr?.let { if (it.pid == pid && pid != UNKNOWN_PID && it.spec == spec && it.vad == vad) return Outcome.Ok(it.info.copy(loadMs = 0)) }
        warmAsr = null
        return when (val r = engines.asr.load(spec, vad)) {
            is Outcome.Ok -> r.also { warmAsr = WarmAsr(pid, spec, vad, it.value) }
            is Outcome.Err -> Outcome.Err(LoadFailures.classify(r.error, spec.kind, pid, exits))
        }
    }

    /** OCR shares :asr; loaded right after the ASR (ARCHITECTURE 4.2). */
    suspend fun loadOcr(det: LoadSpec, rec: LoadSpec): Boolean {
        asrLoading++
        try {
            val pid = up(engines.asrHost) ?: return false
            if (warmOcr == WarmOcr(pid, det, rec) && pid != UNKNOWN_PID) return true
            val ok = engines.ocr.loadOcr(det, rec) is Outcome.Ok
            warmOcr = if (ok) WarmOcr(pid, det, rec) else null
            return ok
        } finally {
            asrLoading--
        }
    }

    /**
     * Load [first] (n_ctx = [ActiveLlm.nCtx]) and restore or build the prefix snapshot for [prefix]; on a load error
     * markFailed and the same GGUF on [ActiveLlm.fallback]. A setPrefix error is returned as is (the model loaded:
     * it is not a unit failure).
     */
    suspend fun loadLlm(llm: ActiveLlm, first: LoadSpec, prefix: String): Outcome<LlmLoaded> {
        llmLoading++
        try {
            up(engines.llmHost) ?: return Outcome.Err(MurmurError.ProcessDied(InferProcess.LLM))
            when (val r = loadLlmOn(llm, first, prefix)) {
                is Step.Loaded -> return Outcome.Ok(r.value)
                is Step.PrefixFailed -> return Outcome.Err(r.error)
                is Step.LoadFailed -> {
                    mark(first, r.error)
                    val fb = llm.fallback?.takeIf { it != first } ?: return Outcome.Err(r.error)
                    up(engines.llmHost) ?: return Outcome.Err(MurmurError.ProcessDied(InferProcess.LLM))
                    return when (val r2 = loadLlmOn(llm, fb, prefix)) {
                        is Step.Loaded -> Outcome.Ok(r2.value.copy(fellBack = true))
                        is Step.PrefixFailed -> Outcome.Err(r2.error)
                        is Step.LoadFailed -> { mark(fb, r2.error); Outcome.Err(r2.error) }
                    }
                }
            }
        } finally {
            llmLoading--
        }
    }

    private sealed interface Step {
        data class Loaded(val value: LlmLoaded) : Step
        data class LoadFailed(val error: MurmurError) : Step
        data class PrefixFailed(val error: MurmurError) : Step
    }

    private suspend fun loadLlmOn(llm: ActiveLlm, spec0: LoadSpec, prefix: String): Step {
        val spec = spec0.copy(options = spec0.options + (LoadOptions.N_CTX to llm.nCtx.toString()))
        val pid = (engines.llmHost.state.value as? HostState.Up)?.pid ?: UNKNOWN_PID
        var warm = warmLlm?.takeIf { it.pid == pid && pid != UNKNOWN_PID && it.spec == spec }
        val loadedNow = warm == null
        if (warm == null) {
            warmLlm = null
            when (val r = engines.llm.load(spec)) {
                is Outcome.Err -> return Step.LoadFailed(LoadFailures.classify(r.error, spec.kind, pid, exits))
                is Outcome.Ok -> warm = WarmLlm(pid, spec, r.value).also { warmLlm = it }
            }
        }
        val key = PrefixCacheKey.of(
            llm.modelSha256, PrefixCacheKey.sha256Hex(prefix), spec.unit, nativeBuild(pid), llm.nCtx,
            kvType = spec.options[LoadOptions.KV_TYPE] ?: "f16", nBatch = spec.options[LoadOptions.N_BATCH]?.toIntOrNull() ?: 512,
        )
        val info = if (loadedNow) warm.info else warm.info.copy(loadMs = 0)
        if (warm.prefixKey == key) return Step.Loaded(LlmLoaded(spec, info, warm.prefixTokens, fellBack = false))
        warmLlm = warm.copy(prefixKey = null, prefixTokens = 0)
        return when (val p = engines.llm.setPrefix(PrefixSpec(prefix, key))) {
            is Outcome.Err -> Step.PrefixFailed(p.error)
            is Outcome.Ok -> {
                if (warmLlm?.pid == pid) warmLlm = warm.copy(prefixKey = key, prefixTokens = p.value.tokens)
                Step.Loaded(LlmLoaded(spec, info, p.value.tokens, fellBack = false))
            }
        }
    }

    /** `native.build` of the running :llm (prefix cache key, ADR-011), probed once per process instance. */
    private suspend fun nativeBuild(pid: Int): String {
        nativeBuild?.let { if (it.first == pid && pid != UNKNOWN_PID) return it.second }
        val build = engines.llmHost.probe().facts[ProcessProbe.FACT_NATIVE_BUILD] ?: "unknown"
        nativeBuild = pid to build
        return build
    }

    private companion object {
        /** ADR-066: a load error says the unit cannot run the artifact unless the device ran out of memory. */
        fun judgesUnit(error: MurmurError): Boolean = LoadFailures.judgesUnit(error)
        const val UNKNOWN_PID = -1
    }
}
