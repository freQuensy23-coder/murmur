package app.murmur.feature.models

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.requiredSize
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.dp
import app.murmur.core.benchmark.BenchStep
import app.murmur.core.catalog.Catalog
import app.murmur.core.catalog.CatalogParser
import app.murmur.core.catalog.ModelEntry
import app.murmur.core.catalog.UnitHealth
import app.murmur.core.model.ArtifactId
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.ModelKind
import app.murmur.core.model.MurmurError
import app.murmur.core.store.BenchmarkRecord
import app.murmur.core.store.BenchmarkResultData
import app.murmur.core.store.ClipResult
import app.murmur.core.store.InstallState
import app.murmur.core.store.Settings
import app.murmur.core.store.StorageInfo
import app.murmur.testing.FakeDeviceProfileRepository
import app.murmur.ui.design.Layout
import app.murmur.ui.design.MurmurIcons
import app.murmur.ui.design.MurmurShapes
import app.murmur.ui.design.MurmurTheme
import app.murmur.ui.design.components.MurmurNavigationBar
import app.murmur.ui.design.components.MurmurNavigationRail
import app.murmur.ui.design.components.NavItem
import app.murmur.ui.design.components.SheetSurface
import java.time.LocalDate
import java.time.ZoneId

const val FRAME_TAG = "models-frame"

/** False = render the frame at the device's real density (touch-target checks); true = 1 dp = 2 px like the PNGs. */
val LocalUseFrameDensity = staticCompositionLocalOf { true }

/** Prototype device sizes (DESIGN 2), 1 dp = 2 px like the PNGs. [Tall*] = no-frame shots that show a whole page. */
enum class Device(val width: Int, val height: Int, val wide: Boolean) {
    Cover(412, 960, false), Inner(750, 832, true), TallCover(412, 2400, false)
}

/**
 * The prototype frame at the PNGs' scale (density 2): 36 dp status bar, the screen (with the bottom bar on the cover
 * display when [nav], the rail on the inner one), the 20 dp gesture bar, and an optional [overlay] over everything
 * (scrim + sheet). Window insets are consumed, so the capture contains exactly what the PNG shows.
 */
@Composable
fun HubFrame(device: Device, nav: Boolean = true, overlay: (@Composable BoxScope.() -> Unit)? = null, content: @Composable () -> Unit) {
    val density = if (LocalUseFrameDensity.current) Density(2f, 1f) else LocalDensity.current
    CompositionLocalProvider(LocalDensity provides density) {
        MurmurTheme(darkTheme = false) {
            val c = MurmurTheme.colors
            val items = listOf(
                NavItem("Dashboard", MurmurIcons.dash),
                NavItem("History", MurmurIcons.hist),
                NavItem("Models", MurmurIcons.models),
                NavItem("Settings", MurmurIcons.sliders),
            )
            Box(Modifier.fillMaxWidth().wrapContentSize(Alignment.TopStart)) {
                Box(
                    Modifier
                        .requiredSize(device.width.dp, device.height.dp)
                        .consumeWindowInsets(WindowInsets.safeDrawing)
                        .background(c.bg)
                        .testTag(FRAME_TAG),
                ) {
                    Column(Modifier.fillMaxSize()) {
                        Box(Modifier.fillMaxWidth().height(36.dp))
                        if (!device.wide) {
                            Column(Modifier.weight(1f)) {
                                Box(Modifier.weight(1f)) { content() }
                                if (nav) MurmurNavigationBar(items, 2, {})
                            }
                        } else {
                            Row(Modifier.weight(1f)) {
                                MurmurNavigationRail(items, 2, {}, Modifier.fillMaxHeight())
                                Box(Modifier.weight(1f)) { content() }
                            }
                        }
                        Box(Modifier.fillMaxWidth().height(20.dp), contentAlignment = Alignment.Center) {
                            Box(Modifier.size(108.dp, 4.dp).alpha(0.85f).clip(MurmurShapes.full).background(c.ink))
                        }
                    }
                    overlay?.invoke(this)
                }
            }
        }
    }
}

/** List pane + detail pane as the shell lays them out on the inner display (316 dp list on bg, detail on surface). */
@Composable
fun InnerPanes(list: @Composable () -> Unit, detail: @Composable () -> Unit) {
    val c = MurmurTheme.colors
    Row(Modifier.fillMaxSize()) {
        Box(Modifier.width(Layout.modelsListPane).fillMaxHeight().background(c.bg)) { list() }
        Box(
            Modifier
                .weight(1f)
                .fillMaxHeight()
                .background(c.surface)
                .drawWithContent {
                    drawContent()
                    drawLine(c.line, Offset(0.5.dp.toPx(), 0f), Offset(0.5.dp.toPx(), size.height), 1.dp.toPx())
                },
        ) { detail() }
    }
}

/** Scrim + the bottom sheet (cover) or the 420 dp side sheet between the bars (inner), as the prototype draws them. */
@Composable
fun BoxScope.SheetOverlay(device: Device, content: @Composable androidx.compose.foundation.layout.ColumnScope.() -> Unit) {
    val c = MurmurTheme.colors
    Box(Modifier.matchParentSize().background(c.scrim))
    if (device.wide) {
        Box(Modifier.matchParentSize().padding(top = 36.dp, bottom = 20.dp), contentAlignment = Alignment.TopEnd) {
            SheetSurface(side = true, content = content)
        }
    } else {
        Box(Modifier.matchParentSize(), contentAlignment = Alignment.BottomCenter) {
            SheetSurface(side = false, content = content)
        }
    }
}

/**
 * The data behind the frames: the bundled catalog on the Fold7 profile, a few artifacts installed or downloading, and
 * benchmark records with the prototype's sample numbers (DESIGN 6.4 "Sample-data warning": sample values, not
 * measurements). Models are picked by facts, never by id.
 */
internal object FrameData {
    val catalog: Catalog = CatalogParser.bundled()
    val fold7 = FakeDeviceProfileRepository.FOLD7
    val zone: ZoneId = ZoneId.systemDefault()
    /** "now" = today 14:32 (the frames' status bar); the last run is today 14:05. */
    val nowMs: Long = LocalDate.now(zone).atTime(14, 32).atZone(zone).toInstant().toEpochMilli()
    private val runMs: Long = LocalDate.now(zone).atTime(14, 5).atZone(zone).toInstant().toEpochMilli()

    val asrInUse: ModelEntry = catalog.model(catalog.defaults.asr.getValue("ru+en"))!!
    private val asrList = catalog.models.filter { it.kind == ModelKind.ASR }
    /** Second and third speech rows: one downloaded (radio), one downloading (as the frame's GigaAM / T-one rows). */
    val asrReady: ModelEntry = asrList.first { it.id != asrInUse.id && it.selectableAsAsr && it.artifacts.size == 1 }
    val asrDownloading: ModelEntry = asrList.first { it.id != asrInUse.id && it.id != asrReady.id && it.selectableAsAsr && it.artifacts.size == 1 }

    val llmInUse: ModelEntry = catalog.models.first { it.llm != null && it.artifacts.any { a -> ComputeUnit.NPU in a.units } }
    val llmReady: ModelEntry = catalog.models.first { it.kind == ModelKind.LLM && it.id != llmInUse.id && !it.noRussian && it.artifacts.size > 1 && it.languages.count > 100 }
    /** The bigger sibling that does not fit (models-storage). */
    val llmTooBig: ModelEntry = catalog.models.first { it.kind == ModelKind.LLM && it.family == llmReady.family && it.id != llmReady.id }

    private fun installedAll(e: ModelEntry): Map<ArtifactId, InstallState> = e.artifacts.associate { it.id to InstallState.Installed("/m/${it.id}", it.downloadBytes) }
    private fun installedFirst(e: ModelEntry): Map<ArtifactId, InstallState> = mapOf(e.artifacts.first().id to InstallState.Installed("/m", e.artifacts.first().downloadBytes))

    val asrRecord = BenchmarkRecord(
        asrInUse.artifacts.first { ComputeUnit.NPU in it.units }.id, ComputeUnit.NPU, "0.1.0", "samsung/fold7", runMs,
        BenchmarkResultData.Asr(
            rtfx = 14.0, tailMs = 400, loadMsCold = 800, loadMsWarm = 120, peakMemMb = 720, ranOn = "NPU",
            perClip = listOf(ClipResult("ru", 30_000, 2_100, 0.05), ClipResult("en", 30_000, 1_900, 0.06)),
        ),
    )
    val llmRecord = BenchmarkRecord(
        llmInUse.artifacts.first().id, ComputeUnit.NPU, "0.1.0", "samsung/fold7", runMs,
        BenchmarkResultData.Llm(decodeTps = 31.0, prefillTps = 520.0, ttftMs = 210.0, e2eMs = 1_300.0, loadMsCold = 1_400, peakMemMb = 1_536, ranOn = "NPU", unitShare = 0.97),
    )

    fun input(states: Map<ArtifactId, InstallState>, storage: StorageInfo = StorageInfo(2_400_000_000, 181_000_000_000), records: List<BenchmarkRecord> = listOf(asrRecord, llmRecord)) =
        HubInput(catalog, states, Settings(cleanup = app.murmur.core.store.CleanupChoice.Model(llmInUse.id)), fold7, UnitHealth.NONE, records.groupBy { it.artifactId }, storage)

    /** `models`: speech model in use (NPU), one downloaded, one downloading 89 of 144 MB at 12.3 MB/s. */
    val speech = input(
        installedAll(asrInUse) + installedFirst(asrReady) + installedAll(llmInUse) + installedFirst(llmReady) +
            mapOf(asrDownloading.artifacts.first().id to InstallState.Downloading(89_000_000, 144_000_000, 12_300_000)),
    )

    /** `models-storage`: the bigger cleanup model paused for lack of space, 0.9 GB free. */
    val storage = input(
        installedAll(asrInUse) + installedAll(llmInUse) + installedFirst(llmReady) +
            mapOf(llmTooBig.artifacts.first().id to InstallState.Paused(0, llmTooBig.artifacts.first().downloadBytes, MurmurError.LowStorage)),
        storage = StorageInfo(2_400_000_000, 900_000_000),
    )

    val vad: ModelEntry = catalog.model(catalog.defaults.vad)!!
    val ocrDet: ModelEntry = catalog.model(catalog.defaults.ocrDet)!!
    val ocrRec: ModelEntry = catalog.model(catalog.defaults.ocrRec)!!
    val ocrRecord = BenchmarkRecord(
        ocrDet.artifacts.first().id, ComputeUnit.CPU, "0.1.0", "samsung/fold7", runMs,
        BenchmarkResultData.Ocr(msPerScreenshot = 450, boxes = 42, loadMsCold = 210, peakMemMb = 140, ranOn = "CPU"),
    )

    /** "Support models" (no frame): the VAD and the OCR detector downloaded (the detector benchmarked), the recogniser not. */
    val support = input(
        speech.states + installedAll(vad) + installedAll(ocrDet),
        records = listOf(asrRecord, llmRecord, ocrRecord),
    )

    val deviceLabel = DeviceLabel("Galaxy Z Fold7", Format.ramGb(fold7.totalRamMb))

    fun benchRun(e: ModelEntry): BenchUi {
        val steps = listOf(
            BenchStepUi(ComputeUnit.NPU, BenchStep.LOAD, null, null, StepStatus.DONE, 800),
            BenchStepUi(ComputeUnit.NPU, BenchStep.SAMPLE_RU, "ru", 30_000, StepStatus.DONE, 2_100),
            BenchStepUi(ComputeUnit.NPU, BenchStep.SAMPLE_EN, "en", 30_000, StepStatus.NOW, 0),
            BenchStepUi(ComputeUnit.NPU, BenchStep.UNLOAD_MEMORY, null, null, StepStatus.TODO, 0),
        )
        return BenchUi(e.id, e.displayName, e.kind, Surface.DETAIL, BenchUi.Phase.RUNNING, steps, planned = true)
    }

    fun benchResult(e: ModelEntry, rec: BenchmarkRecord) =
        BenchUi(e.id, e.displayName, e.kind, Surface.DETAIL, BenchUi.Phase.RESULT, emptyList(), true, listOf(rec), shownUnit = rec.unit)
}
