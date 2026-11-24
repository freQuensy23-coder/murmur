package app.murmur.feature.models

import androidx.compose.runtime.Composable
import androidx.compose.runtime.ReadOnlyComposable
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import app.murmur.core.benchmark.BenchStep
import app.murmur.core.catalog.Blocker
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.MurmurError
import app.murmur.core.model.ProductName
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale

/** Copy of the hub (strings.xml) for logic values. Composable so that it follows the locale. */
internal object HubTexts {

    @Composable
    @ReadOnlyComposable
    fun locale(): Locale = LocalConfiguration.current.locales[0] ?: Locale.getDefault()

    @Composable
    @ReadOnlyComposable
    fun unit(u: ComputeUnit): String = stringResource(
        when (u) {
            ComputeUnit.NPU -> R.string.unit_npu
            ComputeUnit.GPU -> R.string.unit_gpu
            ComputeUnit.CPU -> R.string.unit_cpu
        },
    )

    /** Marketing names of the SoCs QNN context binaries are built for (device facts, not model facts). */
    private val SOC_NAMES = mapOf("SM8750" to "Snapdragon 8 Elite", "SM8850" to "Snapdragon 8 Elite Gen 5", "SM8650" to "Snapdragon 8 Gen 3")

    fun socName(socs: List<String>): String = socs.joinToString(" / ") { SOC_NAMES[it] ?: it }

    /** Why a unit cannot be used (ARCHITECTURE 8.2 labels); null = usable. */
    @Composable
    @ReadOnlyComposable
    fun blocker(b: Blocker?, socs: List<String>, unit: ComputeUnit): String? = when (b) {
        null -> null
        Blocker.WRONG_SOC -> stringResource(R.string.models_needs_soc, socName(socs))
        Blocker.RUNTIME_NOT_SHIPPED -> stringResource(R.string.models_not_supported)
        Blocker.UNIT_NOT_AVAILABLE -> stringResource(R.string.models_unit_unavailable)
        Blocker.UNIT_FAILED -> stringResource(R.string.detail_unit_failed_alone, unit(unit))
    }

    /** Warn line of a paused or failed download (DESIGN 6.4 errors, ADR-073). */
    @Composable
    @ReadOnlyComposable
    fun downloadProblem(e: MurmurError): String = when (e) {
        MurmurError.NoNetwork -> stringResource(R.string.models_paused_network)
        MurmurError.LowStorage -> stringResource(R.string.models_paused_storage)
        MurmurError.DownloadPaused -> stringResource(R.string.models_paused)
        MurmurError.ChecksumMismatch -> stringResource(R.string.models_failed_damaged, ProductName.NAME)
        is MurmurError.StorageFailed -> stringResource(R.string.models_failed_storage)
        is MurmurError.DownloadFailed -> e.httpStatus?.let { stringResource(R.string.models_failed_http, it) } ?: stringResource(R.string.models_failed)
        else -> stringResource(R.string.models_failed)
    }

    /**
     * Languages in long form (DESIGN 6.4 fact table): one language by name; two as a list; more as a count, "including
     * Russian and English" when listed, or "; Russian not listed" when the card does not list Russian.
     */
    @Composable
    @ReadOnlyComposable
    fun languagesLong(codes: List<String>, count: Int, noRussianFlag: Boolean): String {
        val loc = locale()
        fun name(code: String) = Locale.forLanguageTag(code).getDisplayLanguage(loc).replaceFirstChar { it.titlecase(loc) }
        if (count <= 1 && codes.size == 1) return name(codes[0])
        if (count == 2 && codes.size == 2) return stringResource(R.string.lang_list, name(codes[0]), name(codes[1]))
        val n = pluralStringResource(R.plurals.lang_count, count, count)
        val ru = "ru" in codes
        val en = "en" in codes
        val he = "he" in codes
        val lower = { code: String -> Locale.forLanguageTag(code).getDisplayLanguage(loc).let { if (loc.language == "ru") it else it.replaceFirstChar { c -> c.titlecase(loc) } } }
        return when {
            ru && en && he -> stringResource(R.string.lang_including_three, n, lower("ru"), lower("en"), lower("he"))
            ru && en -> stringResource(R.string.lang_including_two, n, lower("ru"), lower("en"))
            ru && he -> stringResource(R.string.lang_including_two, n, lower("ru"), lower("he"))
            en && he -> stringResource(R.string.lang_including_two, n, lower("en"), lower("he"))
            ru -> stringResource(R.string.lang_including_one, n, lower("ru"))
            noRussianFlag -> stringResource(R.string.lang_not_listed, n, name("ru"))
            he -> stringResource(R.string.lang_including_one, n, lower("he"))
            en -> stringResource(R.string.lang_including_one, n, lower("en"))
            else -> n
        }
    }

    /** Checklist label of a benchmark step (DESIGN bench-run). */
    @Composable
    @ReadOnlyComposable
    fun step(step: BenchStep, sampleId: String?, audioMs: Long?): String {
        val secs = audioMs?.let { Format.sig2(it / 1000.0, locale()) }
        return when (step) {
            BenchStep.LOAD -> stringResource(R.string.bench_step_load)
            BenchStep.SAMPLE_RU -> secs?.let { stringResource(R.string.bench_step_ru, it) } ?: stringResource(R.string.bench_step_ru_plain)
            BenchStep.SAMPLE_EN -> secs?.let { stringResource(R.string.bench_step_en, it) } ?: stringResource(R.string.bench_step_en_plain)
            BenchStep.SAMPLE_MIXED -> secs?.let { stringResource(R.string.bench_step_mixed, it) } ?: stringResource(R.string.bench_step_mixed_plain)
            BenchStep.CLEANUP_CASE -> stringResource(R.string.bench_step_cleanup)
            BenchStep.OCR_PAGE -> when (sampleId) {
                "cover" -> stringResource(R.string.bench_step_ocr_cover)
                "inner" -> stringResource(R.string.bench_step_ocr_inner)
                else -> stringResource(R.string.bench_step_ocr)
            }
            BenchStep.UNLOAD_MEMORY -> stringResource(R.string.bench_step_unload)
        }
    }

    /** Language of a benchmark clip id ("ru", "en", "mixed"). */
    @Composable
    @ReadOnlyComposable
    fun clip(id: String): String = when (id) {
        "ru" -> stringResource(R.string.bench_sample_ru)
        "en" -> stringResource(R.string.bench_sample_en)
        "mixed" -> stringResource(R.string.bench_sample_mixed)
        else -> id
    }

    /** Why a benchmark could not run (BenchProgress.Failed / UnitFailed). */
    @Composable
    @ReadOnlyComposable
    fun benchError(e: MurmurError): String = when (e) {
        is MurmurError.EngineLoadFailed, is MurmurError.EngineRuntimeMissing, is MurmurError.UnitUnavailable, is MurmurError.ProcessDied ->
            stringResource(R.string.bench_failed_load)
        is MurmurError.OutOfMemory -> stringResource(R.string.bench_failed_memory)
        is MurmurError.Timeout -> stringResource(R.string.bench_failed_timeout)
        else -> stringResource(R.string.bench_failed_other, e.code)
    }

    /** "today, 14:05" / "yesterday, 14:05" / "12 Nov, 14:05". */
    @Composable
    fun lastRun(epochMs: Long, nowMs: Long): String {
        val loc = locale()
        val zone = ZoneId.systemDefault()
        val at = Instant.ofEpochMilli(epochMs).atZone(zone)
        val today = Instant.ofEpochMilli(nowMs).atZone(zone).toLocalDate()
        val time = at.format(DateTimeFormatter.ofPattern("HH:mm", loc))
        return when (at.toLocalDate()) {
            today -> stringResource(R.string.bench_today, time)
            today.minusDays(1) -> stringResource(R.string.bench_yesterday, time)
            else -> "${at.format(DateTimeFormatter.ofPattern(if (at.year == LocalDate.now(zone).year) "d MMM" else "d MMM yyyy", loc))}, $time"
        }
    }

    /**
     * The resolver's reason for refusing a repo (WP-13 `HfCustomModelResolver` writes it in CatalogInvalid.detail) in
     * plain words; anything not recognised is shown as written.
     */
    @Composable
    @ReadOnlyComposable
    fun addError(e: MurmurError): String = when (e) {
        MurmurError.NoNetwork -> stringResource(R.string.add_err_network)
        is MurmurError.DownloadFailed -> e.httpStatus?.let { stringResource(R.string.add_err_http, it) } ?: stringResource(R.string.add_err_network)
        is MurmurError.CatalogInvalid -> {
            val d = e.detail.orEmpty()
            when {
                d.startsWith("unsupported GGUF architecture") ->
                    stringResource(R.string.add_err_arch, d.substringAfter('\'', "?").substringBefore('\''))
                d.startsWith("no single-file GGUF") || d.endsWith("is not a GGUF file") || d.startsWith("not a GGUF file") ->
                    stringResource(R.string.add_err_no_gguf)
                d.startsWith("no sherpa-onnx layout") -> stringResource(R.string.add_err_layout)
                d.startsWith("repo or revision not found") || d.startsWith("revision did not resolve") -> stringResource(R.string.add_err_not_found)
                d.startsWith("not a Hugging Face repo id") -> stringResource(R.string.add_repo_invalid)
                else -> stringResource(R.string.add_err_other, d.ifEmpty { e.code })
            }
        }
        else -> stringResource(R.string.add_err_other, e.detail ?: e.code)
    }
}
