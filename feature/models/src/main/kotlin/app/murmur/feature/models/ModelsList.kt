package app.murmur.feature.models

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import app.murmur.core.model.ModelId
import app.murmur.core.model.ModelKind
import app.murmur.core.model.MurmurError
import app.murmur.ui.design.MurmurIcons
import app.murmur.ui.design.MurmurShapes
import app.murmur.ui.design.MurmurTheme
import app.murmur.ui.design.components.Banner
import app.murmur.ui.design.components.BannerKind
import app.murmur.ui.design.components.ButtonStyle
import app.murmur.ui.design.components.InTrainingBadge
import app.murmur.ui.design.components.InUsePill
import app.murmur.ui.design.components.InsetDivider
import app.murmur.ui.design.components.LinearProgress
import app.murmur.ui.design.components.MurmurBadge
import app.murmur.ui.design.components.BadgeKind
import app.murmur.ui.design.components.MurmurButton
import app.murmur.ui.design.components.MurmurIconButton
import app.murmur.ui.design.components.MurmurRadio
import app.murmur.ui.design.components.NoRussianBadge
import app.murmur.ui.design.components.RingProgress
import app.murmur.ui.design.components.ScreenHeader
import app.murmur.ui.design.components.SectionLabel
import app.murmur.ui.design.components.SegmentedControl
import app.murmur.ui.design.components.Skeleton

/** Test tags of the hub (instrumented tests, the shell's tests). */
object HubTags {
    const val LIST = "screen-models"
    const val DETAIL = "screen-model-detail"
    const val NONE_ROW = "models-none"
    const val SUPPORT = "models-support"
    const val STORAGE_LINE = "models-storage-line"
    const val STORAGE_BANNER = "models-storage-banner"
    const val ADD = "models-add"
    const val BENCH_SHEET = "models-bench-sheet"
    const val ADD_SHEET = "models-add-sheet"
    const val UNIT_PICKER = "models-unit-picker"
    const val ACTION_BAR = "models-action-bar"
    const val EXTRA_DOWNLOAD = "models-extra-download"
    const val LOADING = "models-loading"
    fun row(id: ModelId) = "model-row-${id.value}"
    fun support(id: ModelId) = "support-row-${id.value}"
    fun unitOption(name: String) = "unit-option-$name"
}

/** The list screen: header, Speech | Cleanup, purpose line, storage banner, rows, support models, storage line. */
@Composable
internal fun ModelsListScreen(controller: ModelsController, deps: ModelsDeps, onOpenModel: (ModelId) -> Unit, selected: ModelId?, modifier: Modifier) {
    val input by controller.input.collectAsStateWithLifecycle()
    val tab by controller.tab.collectAsStateWithLifecycle()
    Column(modifier.fillMaxSize().testTag(HubTags.LIST)) {
        ScreenHeader(stringResource(R.string.models_title))
        val inp = input
        if (inp == null) return@Column
        val list = HubLogic.list(tab, inp)
        ModelsListContent(
            list = list,
            selected = selected,
            onTab = controller::selectTab,
            onOpen = onOpenModel,
            onSelect = controller::select,
            onSelectNone = controller::selectNone,
            onDownload = { controller.download(it) },
            onPause = controller::pause,
            onResume = controller::resume,
            onBenchmark = { controller.runBenchmark(it, Surface.LIST) },
            onDelete = { controller.requestDelete(it, Surface.LIST) },
            onAdd = { controller.openAdd(if (tab == HubTab.SPEECH) ModelKind.ASR else ModelKind.LLM) },
        )
    }
    HubOverlays(controller, deps, Surface.LIST)
}

/** Stateless list body (screenshot tests render it with fixed input). */
@Composable
internal fun ModelsListContent(
    list: HubList,
    selected: ModelId?,
    onTab: (HubTab) -> Unit,
    onOpen: (ModelId) -> Unit,
    onSelect: (ModelId) -> Unit,
    onSelectNone: () -> Unit,
    onDownload: (ModelId) -> Unit,
    onPause: (ModelId) -> Unit,
    onResume: (ModelId) -> Unit,
    onBenchmark: (ModelId) -> Unit,
    onDelete: (ModelId) -> Unit,
    onAdd: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    val loc = HubTexts.locale()
    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        SegmentedControl(
            options = listOf(stringResource(R.string.models_tab_speech), stringResource(R.string.models_tab_cleanup)),
            selectedIndex = list.tab.ordinal,
            onSelect = { onTab(HubTab.entries[it]) },
            modifier = Modifier.padding(start = 16.dp, end = 16.dp, top = 0.dp),
        )
        Text(
            stringResource(if (list.tab == HubTab.SPEECH) R.string.models_purpose_speech else R.string.models_purpose_cleanup),
            style = t.caption, color = c.ink3,
            modifier = Modifier.padding(start = 20.dp, end = 20.dp, top = 8.dp),
        )
        list.storageProblem?.let { p ->
            Banner(
                kind = BannerKind.Warn,
                title = stringResource(R.string.models_storage_err_title, p.model),
                body = stringResource(R.string.models_storage_err_body, Format.sizeText(p.neededBytes, loc), Format.storageText(p.freeBytes, loc)),
                icon = MurmurIcons.storage,
                modifier = Modifier.padding(start = 16.dp, end = 16.dp, top = 12.dp).testTag(HubTags.STORAGE_BANNER),
            )
        }
        Column(Modifier.padding(start = 16.dp, end = 16.dp, top = 12.dp)) {
            Group {
                var prevSelected = false
                var first = true
                if (list.tab == HubTab.CLEANUP) {
                    NoneRow(list.noneSelected, onSelectNone)
                    prevSelected = list.noneSelected
                    first = false
                }
                for (row in list.rows) {
                    val filled = row.state == RowState.InUse
                    if (!first && !filled && !prevSelected) InsetDivider(start = 16.dp, end = 16.dp)
                    ModelRowView(
                        row = row,
                        highlighted = selected == row.id,
                        onOpen = { onOpen(row.id) },
                        onSelect = { onSelect(row.id) },
                        onDownload = { onDownload(row.id) },
                        onPause = { onPause(row.id) },
                        onResume = { onResume(row.id) },
                    )
                    prevSelected = filled
                    first = false
                }
            }
            if (list.support.isNotEmpty()) {
                SectionLabel(
                    stringResource(R.string.models_support),
                    contentPadding = androidx.compose.foundation.layout.PaddingValues(start = 4.dp, top = 24.dp, end = 4.dp, bottom = 8.dp),
                )
                Group(Modifier.testTag(HubTags.SUPPORT)) {
                    list.support.forEachIndexed { i, row ->
                        if (i > 0) InsetDivider(start = 16.dp)
                        SupportRow(row, highlighted = selected == row.id, onOpen = { onOpen(row.id) }, onDownload = { onDownload(row.id) }, onPause = { onPause(row.id) },
                            onResume = { onResume(row.id) }, onBenchmark = { onBenchmark(row.id) }, onDelete = { onDelete(row.id) })
                    }
                }
            }
            list.storage?.let { s ->
                Row(
                    Modifier.padding(start = 4.dp, end = 4.dp, top = 14.dp).testTag(HubTags.STORAGE_LINE),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(MurmurIcons.storage, null, tint = c.ink3, modifier = Modifier.size(14.dp))
                    Text(
                        stringResource(R.string.models_storage_line, Format.storageText(s.modelsBytes, loc), Format.storageText(s.freeBytes, loc)),
                        style = t.caption.copy(fontFeatureSettings = "tnum"), color = c.ink3,
                    )
                }
            }
            MurmurButton(
                text = stringResource(R.string.models_add),
                onClick = onAdd,
                style = ButtonStyle.Text,
                leadingIcon = MurmurIcons.plus,
                modifier = Modifier.padding(top = 8.dp).testTag(HubTags.ADD),
            )
            Box(Modifier.height(16.dp))
        }
    }
}

/** One grouped surface (DESIGN 4: surface, lg radius, no cards inside cards). */
@Composable
private fun Group(modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Column(modifier.fillMaxWidth().clip(MurmurShapes.lg).background(MurmurTheme.colors.surface)) { content() }
}

/** "None" (ADR-033): first row of Cleanup, a radio, the copy-deck description. */
@Composable
private fun NoneRow(selected: Boolean, onSelect: () -> Unit) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    Row(
        Modifier
            .fillMaxWidth()
            .background(if (selected) c.accentSoft else c.surface)
            .clickable(role = Role.RadioButton, onClick = onSelect)
            .semantics(mergeDescendants = true) { }
            .padding(16.dp)
            .testTag(HubTags.NONE_ROW),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Column(Modifier.weight(1f)) {
            Text(stringResource(R.string.models_none), style = t.titleSm, color = c.ink)
            Text(stringResource(R.string.models_none_desc), style = t.rowSub, color = if (selected) c.ink2 else c.ink3, modifier = Modifier.padding(top = 4.dp))
        }
        MurmurRadio(selected = selected, onClick = null)
    }
}

/** A model row (DESIGN 6.4, prototype .mrow). The whole row opens the detail; the right column is the state control. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun ModelRowView(
    row: ModelRow,
    highlighted: Boolean,
    onOpen: () -> Unit,
    onSelect: () -> Unit,
    onDownload: () -> Unit,
    onPause: () -> Unit,
    onResume: () -> Unit,
) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    val loc = HubTexts.locale()
    val inUse = row.state == RowState.InUse
    val dim = row.state == RowState.InTraining
    val bg = when {
        inUse -> c.accentSoft
        highlighted -> c.surface3
        else -> c.surface
    }
    Row(
        Modifier
            .fillMaxWidth()
            .background(bg)
            .clickable(role = Role.Button, onClick = onOpen)
            .alpha(if (dim) 0.62f else 1f)
            .padding(16.dp)
            .testTag(HubTags.row(row.id)),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Column(Modifier.weight(1f)) {
            Text(row.name, style = t.titleSm, color = c.ink)
            val by = row.variant?.let { stringResource(R.string.models_byline, row.publisher, it) } ?: row.publisher
            Text(by, style = t.caption, color = if (inUse) c.ink2 else c.ink3, modifier = Modifier.padding(top = 1.dp))
            val spec = buildList {
                row.params?.let { add(stringResource(R.string.models_params, it)) }
                row.sizeBytes?.let { add(Format.sizeText(it, loc)) }
                if (row.streaming) add(stringResource(R.string.models_streaming))
            }
            if (spec.isNotEmpty()) {
                FlowRow(Modifier.padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    spec.forEach { Text(it, style = t.mono, color = c.ink2) }
                }
            }
            FlowRow(Modifier.padding(top = 10.dp), horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                row.units.forEach { UnitBadgeView(it.unit, it.chosen) }
                row.languages.forEach { MurmurBadge(it, BadgeKind.Language) }
                if (row.noRussian && (row.missingLanguages.isEmpty() || "ru" in row.missingLanguages)) NoRussianBadge()
                if ("he" in row.missingLanguages) MurmurBadge(stringResource(R.string.models_no_hebrew), BadgeKind.NoRussian)
            }
            RowExtra(row)
        }
        Column(horizontalAlignment = Alignment.End, verticalArrangement = Arrangement.spacedBy(6.dp)) {
            RowControl(row, onSelect, onDownload, onPause, onResume)
        }
    }
}

/** Progress bar + meta, warn lines, "Preview only", blocker text (left column, under the tags). */
@Composable
private fun RowExtra(row: ModelRow) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    when (val s = row.state) {
        is RowState.Unavailable -> HubTexts.blocker(s.blocker, s.socs, row.units.firstOrNull()?.unit ?: app.murmur.core.model.ComputeUnit.NPU)?.let {
            Text(it, style = t.caption, color = c.ink3, modifier = Modifier.padding(top = 10.dp))
        }
        is RowState.NotDownloaded -> if (row.selected) {
            Text(stringResource(R.string.models_selected_download), style = t.caption, color = c.accentInk, modifier = Modifier.padding(top = 10.dp))
        }
        else -> DownloadProgress(s)
    }
    if (row.previewOnly) Text(stringResource(R.string.models_preview_only), style = t.caption, color = c.ink3, modifier = Modifier.padding(top = 10.dp))
}

/**
 * The extra line of a download in flight (DESIGN 6.4 state table): 4 dp progress + mono "89 MB of 144 MB · 12.3 MB/s",
 * the ink4 bar + warn line when paused, the error line when failed, "Checking files…" / "Preparing for NPU…".
 * [onPaddingTop] = the gap above a progress bar (text lines sit 2 dp closer). Nothing for other states.
 */
@Composable
internal fun DownloadProgress(state: RowState, onPaddingTop: androidx.compose.ui.unit.Dp = 12.dp) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    val loc = HubTexts.locale()
    val textTop = onPaddingTop - 2.dp
    when (val s = state) {
        is RowState.Downloading -> {
            val p = if (s.bytesTotal > 0) s.bytesDone.toFloat() / s.bytesTotal else 0f
            LinearProgress(p, Modifier.padding(top = onPaddingTop))
            Row(Modifier.fillMaxWidth().padding(top = 6.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                val (d, tt) = Format.progressPair(s.bytesDone, s.bytesTotal, loc)
                Text(stringResource(R.string.models_progress, d, tt), style = t.mono, color = c.ink3)
                if (s.bytesPerSec > 0) Text(Format.speed(s.bytesPerSec, loc), style = t.mono, color = c.ink3)
            }
        }
        is RowState.Queued -> {
            LinearProgress(0f, Modifier.padding(top = onPaddingTop))
            Text(stringResource(R.string.models_queued), style = t.caption, color = c.ink3, modifier = Modifier.padding(top = 6.dp))
        }
        is RowState.Paused -> {
            val p = if (s.bytesTotal > 0) s.bytesDone.toFloat() / s.bytesTotal else 0f
            LinearProgress(p, Modifier.padding(top = onPaddingTop), paused = true)
            Row(Modifier.fillMaxWidth().padding(top = 6.dp), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                Row(Modifier.weight(1f, fill = false), horizontalArrangement = Arrangement.spacedBy(4.dp), verticalAlignment = Alignment.CenterVertically) {
                    if (s.reason == MurmurError.NoNetwork) Icon(MurmurIcons.wifiOff, null, tint = c.warn, modifier = Modifier.size(14.dp))
                    Text(HubTexts.downloadProblem(s.reason), style = t.caption, color = c.warn)
                }
                val (d, tt) = Format.progressPair(s.bytesDone, s.bytesTotal, loc)
                Text(stringResource(R.string.models_progress, d, tt), style = t.mono, color = c.warn, modifier = Modifier.padding(start = 8.dp))
            }
        }
        is RowState.Failed -> Row(Modifier.padding(top = textTop), horizontalArrangement = Arrangement.spacedBy(4.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(MurmurIcons.alert, null, tint = c.err, modifier = Modifier.size(14.dp))
            Text(HubTexts.downloadProblem(s.error), style = t.caption, color = c.err)
        }
        RowState.Checking -> Text(stringResource(R.string.models_checking), style = t.caption, color = c.ink3, modifier = Modifier.padding(top = textTop))
        is RowState.Preparing -> Text(
            s.unit?.let { stringResource(R.string.models_preparing_unit, HubTexts.unit(it)) } ?: stringResource(R.string.models_preparing),
            style = t.caption, color = c.ink3, modifier = Modifier.padding(top = textTop),
        )
        else -> Unit
    }
}

/** Right column: pill, pause/resume, ring, radio, in-use pill or the In training badge (DESIGN 6.4 state table). */
@Composable
private fun RowControl(row: ModelRow, onSelect: () -> Unit, onDownload: () -> Unit, onPause: () -> Unit, onResume: () -> Unit) {
    val loc = HubTexts.locale()
    when (val s = row.state) {
        is RowState.NotDownloaded -> {
            val size = Format.sizeText(s.bytes, loc)
            DownloadPill(size, stringResource(R.string.models_cd_download, row.name, size), onDownload, Modifier.alignVisualTop(PILL_PAD, 0.dp))
        }
        is RowState.Downloading, is RowState.Queued ->
            MurmurIconButton(MurmurIcons.pause, stringResource(R.string.models_cd_pause), onPause, Modifier.alignVisualTop(ICON_PAD, ICON_PAD), tonal = true, small = true, iconSize = 18.dp)
        is RowState.Paused ->
            MurmurIconButton(MurmurIcons.refresh, stringResource(R.string.models_cd_resume), onResume, Modifier.alignVisualTop(ICON_PAD, ICON_PAD), tonal = true, small = true, iconSize = 18.dp)
        is RowState.Failed ->
            MurmurIconButton(MurmurIcons.refresh, stringResource(R.string.models_cd_retry), onResume, Modifier.alignVisualTop(ICON_PAD, ICON_PAD), tonal = true, small = true, iconSize = 18.dp)
        RowState.Checking -> RingProgress(1f, showPercent = false)
        is RowState.Preparing -> RingProgress(s.progress)
        RowState.Ready -> if (row.selectable) {
            val cd = stringResource(R.string.models_cd_select, row.name)
            MurmurRadio(selected = false, onClick = onSelect, modifier = Modifier.alignVisualTop(RADIO_PAD, RADIO_PAD).semantics { contentDescription = cd })
        }
        RowState.InUse -> InUsePill()
        RowState.InTraining -> InTrainingBadge()
        is RowState.Unavailable -> Unit
        // Same place and size as the download pill's visual (36 dp tall), no hit target: nothing can be done yet.
        RowState.Loading -> Skeleton(Modifier.size(LOADING_PILL_W, 36.dp).testTag(HubTags.LOADING), MurmurShapes.full)
    }
}

/**
 * Support model row (no frame; grouped-list row from WP-01): name, "size · Downloaded / 450 ms per screenshot", and the
 * install control or Benchmark (OCR) and Delete buttons. The row opens the detail.
 */
@Composable
private fun SupportRow(row: ModelRow, highlighted: Boolean, onOpen: () -> Unit, onDownload: () -> Unit, onPause: () -> Unit, onResume: () -> Unit, onBenchmark: () -> Unit, onDelete: () -> Unit) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    val loc = HubTexts.locale()
    val size = row.sizeBytes?.let { Format.sizeText(it, loc) }
    val status = when (val s = row.state) {
        RowState.Ready, RowState.InUse -> row.msPerScreenshot?.let { stringResource(R.string.models_ocr_ms, it.toString()) }
            ?: stringResource(R.string.models_support_downloaded)
        is RowState.Downloading -> stringResource(R.string.models_progress, Format.progressPair(s.bytesDone, s.bytesTotal, loc).first, Format.progressPair(s.bytesDone, s.bytesTotal, loc).second)
        is RowState.Queued -> stringResource(R.string.models_queued)
        is RowState.Paused -> HubTexts.downloadProblem(s.reason)
        is RowState.Failed -> HubTexts.downloadProblem(s.error)
        RowState.Checking -> stringResource(R.string.models_checking)
        is RowState.Preparing -> stringResource(R.string.models_preparing)
        else -> null
    }
    val warn = row.state is RowState.Paused || row.state is RowState.Failed
    Row(
        Modifier
            .fillMaxWidth()
            // The model shown in the inner display's detail pane, as for the model rows.
            .background(if (highlighted) c.surface3 else c.surface)
            .clickable(role = Role.Button, onClick = onOpen)
            .padding(start = 16.dp, end = 8.dp, top = 10.dp, bottom = 10.dp)
            .testTag(HubTags.support(row.id)),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f).padding(vertical = 2.dp)) {
            Text(row.name, style = t.rowTitle, color = c.ink)
            Text(
                listOfNotNull(size, status).joinToString(" · "),
                style = t.rowSub, color = if (warn) c.warn else c.ink3, modifier = Modifier.padding(top = 2.dp),
            )
        }
        when (val st = row.state) {
            RowState.Ready, RowState.InUse -> {
                if (row.kind == ModelKind.OCR_DET || row.kind == ModelKind.OCR_REC) {
                    MurmurIconButton(MurmurIcons.gauge, stringResource(R.string.models_cd_benchmark, row.name), onBenchmark, small = true, tint = c.ink2)
                }
                MurmurIconButton(MurmurIcons.trash, stringResource(R.string.models_cd_delete, row.name), onDelete, small = true, tint = c.ink2)
            }
            is RowState.NotDownloaded -> Box(Modifier.padding(end = 8.dp)) {
                DownloadPill(size.orEmpty(), stringResource(R.string.models_cd_download, row.name, size.orEmpty()), onDownload)
            }
            is RowState.Downloading, is RowState.Queued ->
                MurmurIconButton(MurmurIcons.pause, stringResource(R.string.models_cd_pause), onPause, tonal = true, small = true, iconSize = 18.dp)
            is RowState.Paused, is RowState.Failed ->
                MurmurIconButton(MurmurIcons.refresh, stringResource(R.string.models_cd_resume), onResume, tonal = true, small = true, iconSize = 18.dp)
            RowState.Checking -> RingProgress(1f, showPercent = false, modifier = Modifier.padding(end = 8.dp))
            is RowState.Preparing -> RingProgress(st.progress, modifier = Modifier.padding(end = 8.dp))
            RowState.Loading -> Skeleton(Modifier.padding(end = 8.dp).size(LOADING_PILL_W, 36.dp).testTag(HubTags.LOADING), MurmurShapes.full)
            else -> Unit
        }
    }
}

/** Hit padding around the row controls' visuals (48 dp targets): radio 22, small icon button 40, pill 36 dp. */
private val RADIO_PAD = 13.dp
private val ICON_PAD = 4.dp
private val PILL_PAD = 6.dp
private val LOADING_PILL_W = 88.dp

/**
 * The prototype puts a control's *visual* at the row's 16 dp padding (top and end). Our controls carry an invisible
 * hit margin around the visual, so shift them by that margin; the 48 dp target then overlaps the row padding.
 */
private fun Modifier.alignVisualTop(vertical: androidx.compose.ui.unit.Dp, horizontal: androidx.compose.ui.unit.Dp): Modifier =
    this.offset(x = horizontal, y = -vertical)
