package app.murmur.feature.models

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import app.murmur.core.catalog.Blocker
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.ModelId
import app.murmur.core.model.ModelKind
import app.murmur.core.store.BenchmarkRecord
import app.murmur.core.store.BenchmarkResultData
import app.murmur.ui.design.MurmurIcons
import app.murmur.ui.design.MurmurShapes
import app.murmur.ui.design.MurmurTheme
import app.murmur.ui.design.components.BadgeKind
import app.murmur.ui.design.components.ButtonStyle
import app.murmur.ui.design.components.InTrainingBadge
import app.murmur.ui.design.components.InUsePill
import app.murmur.ui.design.components.InsetDivider
import app.murmur.ui.design.components.ListRow
import app.murmur.ui.design.components.MurmurBadge
import app.murmur.ui.design.components.MurmurButton
import app.murmur.ui.design.components.MurmurIconButton
import app.murmur.ui.design.components.NoRussianBadge
import app.murmur.ui.design.components.RowTrailing
import app.murmur.ui.design.components.Skeleton
import app.murmur.ui.design.components.StatusKind
import app.murmur.ui.design.components.StatusLabel
import app.murmur.ui.design.components.TopBar
import app.murmur.ui.design.rememberIsWide

/** The detail screen: data from the controller, plus its sheets and dialogs. */
@Composable
internal fun ModelDetailScreen(controller: ModelsController, deps: ModelsDeps, id: ModelId, onClose: () -> Unit, pane: Boolean, modifier: Modifier) {
    val input by controller.input.collectAsStateWithLifecycle()
    val inp = input
    val entry = inp?.catalog?.model(id)
    val onBack: (() -> Unit)? = if (pane) null else onClose
    // The catalog is loaded and the model is not in it (a custom model deleted here or elsewhere): leave the page.
    val gone = inp != null && entry == null
    val close by rememberUpdatedState(onClose)
    LaunchedEffect(gone) { if (gone) close() }
    Box(modifier.fillMaxSize().testTag(HubTags.DETAIL)) {
        if (inp != null && entry != null) {
            ModelDetailContent(
                view = HubLogic.detail(entry, inp),
                kind = entry.kind,
                onBack = onBack,
                nowMs = deps.now(),
                onUse = { controller.select(id) },
                onDownload = { controller.download(id, it) },
                onPause = { controller.pause(id) },
                onResume = { controller.resume(id) },
                onDelete = { controller.requestDelete(id, Surface.DETAIL) },
                onPin = { controller.pinUnit(id, it) },
                onBenchmark = { controller.runBenchmark(id, Surface.DETAIL) },
            )
        } else if (onBack != null) {
            TopBar(title = "", onBack = onBack)
        }
    }
    HubOverlays(controller, deps, Surface.DETAIL)
}

/**
 * Stateless detail (DESIGN 6.4 model-detail). Cover display: top bar with back and "open model page"; inner display
 * ([onBack] null): a 16 dp top inset, no bar. Content scrolls above a fixed action bar.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun ModelDetailContent(
    view: ModelDetailView,
    kind: ModelKind,
    onBack: (() -> Unit)?,
    nowMs: Long,
    onUse: () -> Unit,
    onDownload: (app.murmur.core.catalog.Artifact?) -> Unit,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onDelete: () -> Unit,
    onPin: (ComputeUnit?) -> Unit,
    onBenchmark: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    val uri = LocalUriHandler.current
    val open: (() -> Unit)? = view.sourceUrl?.let { url -> { runCatching { uri.openUri(url) } } }
    val loc = HubTexts.locale()
    Column(modifier.fillMaxSize()) {
        if (onBack != null) {
            TopBar(title = "", onBack = onBack, actions = {
                if (open != null) MurmurIconButton(MurmurIcons.ext, stringResource(R.string.models_cd_open_page), open)
            })
        } else {
            Box(Modifier.height(16.dp))
        }
        Column(Modifier.weight(1f).verticalScroll(rememberScrollState())) {
            Column(Modifier.padding(start = 20.dp, end = 20.dp, top = 4.dp)) {
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp), itemVerticalAlignment = Alignment.CenterVertically) {
                    if (view.inUse) InUsePill()
                    view.chosenUnit?.let { UnitBadgeView(it, chosen = true) }
                    view.quant?.let { MurmurBadge(it, BadgeKind.Neutral) }
                    if (view.row.noRussian && (view.row.missingLanguages.isEmpty() || "ru" in view.row.missingLanguages)) NoRussianBadge()
                    if ("he" in view.row.missingLanguages) MurmurBadge(stringResource(R.string.models_no_hebrew), BadgeKind.NoRussian)
                    if (view.row.state == RowState.InTraining) InTrainingBadge()
                }
                Text(
                    view.row.name, style = t.headline, color = c.ink,
                    modifier = Modifier.padding(top = 12.dp).semantics { heading() },
                )
                view.description?.let { Text(it, style = t.bodyLg, color = c.ink2, modifier = Modifier.padding(top = 8.dp)) }
                KvTable(factRows(view, open), Modifier.padding(top = 20.dp))
                if (view.notes.isNotEmpty()) {
                    Column(Modifier.padding(top = 12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        view.notes.forEach { NoteLine(it) }
                    }
                }
                if (view.cpuFallbackDownload != null) {
                    MurmurButton(
                        text = stringResource(R.string.detail_download_cpu, Format.sizeText(view.cpuFallbackDownload.downloadBytes, loc)),
                        onClick = { onDownload(view.cpuFallbackDownload) },
                        style = ButtonStyle.Outline, leadingIcon = MurmurIcons.dl,
                        modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
                    )
                }
                view.extraDownload?.takeIf { it.purpose == ExtraDownload.Purpose.CPU_VERSION }?.let {
                    ExtraDownloadBlock(it, onPause, onResume, Modifier.padding(top = 12.dp))
                }
            }
            if (kind != ModelKind.VAD && view.row.state != RowState.InTraining) {
                BenchSection(view, kind, nowMs, onBenchmark)
            } else {
                Box(Modifier.height(24.dp))
            }
            // Below the benchmark (the frame's order: facts, then "Benchmark on this phone"); no frame of its own.
            if (view.unitOptions.isNotEmpty()) {
                Box(Modifier.padding(start = 20.dp, end = 20.dp, bottom = 24.dp)) { UnitPicker(view, onPin, onDownload, onPause, onResume) }
            }
        }
        ActionBar(view, onUse, onDownload = { onDownload((view.action as? DetailAction.Download)?.artifact) }, onPause, onResume, onDelete)
    }
}

@Composable
private fun factRows(view: ModelDetailView, open: (() -> Unit)?): List<KvRow> {
    val loc = HubTexts.locale()
    return view.facts.map { f ->
        when (f) {
            is Fact.Parameters -> KvRow(stringResource(R.string.detail_parameters), f.label)
            is Fact.Languages -> KvRow(stringResource(R.string.detail_languages), HubTexts.languagesLong(f.codes, f.count, view.row.noRussian))
            is Fact.Download -> KvRow(
                stringResource(R.string.detail_download),
                f.quant?.let { stringResource(R.string.detail_value_pair, Format.sizeText(f.bytes, loc), it) } ?: Format.sizeText(f.bytes, loc),
            )
            is Fact.RunsOn -> KvRow(stringResource(R.string.detail_runs_on), f.lines.map { l ->
                val u = if (l.hexagon) stringResource(R.string.unit_npu_hexagon) else HubTexts.unit(l.unit)
                when (val why = HubTexts.blocker(l.blocker, l.socs, l.unit)) {
                    null -> u
                    // "Failed to load on NPU" already names the unit.
                    else -> if (l.blocker == Blocker.UNIT_FAILED) why else stringResource(R.string.detail_value_pair, u, why)
                }
            }.joinToString("\n"))
            is Fact.Streaming -> KvRow(stringResource(R.string.detail_streaming), stringResource(if (f.streaming) R.string.detail_streaming_yes else R.string.detail_streaming_no))
            is Fact.License -> KvRow(stringResource(R.string.detail_license), f.id)
            is Fact.Source -> KvRow(stringResource(R.string.detail_source), f.label, onClick = open, valueTag = "model-source", small = true)
        }
    }
}

@Composable
private fun NoteLine(n: DetailNote) {
    val text = when (n) {
        is DetailNote.UnitFailed -> n.using?.let { stringResource(R.string.detail_unit_failed, HubTexts.unit(n.failed), HubTexts.unit(it)) }
            ?: stringResource(R.string.detail_unit_failed_alone, HubTexts.unit(n.failed))
        is DetailNote.ResultsWrong -> stringResource(R.string.detail_results_wrong, HubTexts.unit(n.unit))
        DetailNote.NpuNotYet -> stringResource(R.string.detail_npu_not_yet)
        DetailNote.PreviewOnly -> stringResource(R.string.models_preview_only)
    }
    val kind = when (n) {
        is DetailNote.UnitFailed, is DetailNote.ResultsWrong -> StatusKind.Warn
        else -> null
    }
    if (kind != null) StatusLabel(text, kind, icon = MurmurIcons.alert)
    else Row(horizontalArrangement = Arrangement.spacedBy(4.dp), verticalAlignment = Alignment.CenterVertically) {
        Icon(MurmurIcons.info, null, tint = MurmurTheme.colors.ink3, modifier = Modifier.size(14.dp))
        Text(text, style = MurmurTheme.type.labelSm, color = MurmurTheme.colors.ink3)
    }
}

/** Unit picker (no frame; radio rows from WP-01): Auto and each usable unit -> Settings.unitPreference[model]. */
@Composable
private fun UnitPicker(
    view: ModelDetailView,
    onPin: (ComputeUnit?) -> Unit,
    onDownload: (app.murmur.core.catalog.Artifact?) -> Unit,
    onPause: () -> Unit,
    onResume: () -> Unit,
) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    val loc = HubTexts.locale()
    Column(Modifier.testTag(HubTags.UNIT_PICKER)) {
        Text(stringResource(R.string.detail_unit_title), style = t.titleSm, color = c.ink, modifier = Modifier.semantics { heading() })
        Text(stringResource(R.string.detail_unit_hint), style = t.caption, color = c.ink3, modifier = Modifier.padding(top = 2.dp))
        Column(
            Modifier
                .padding(top = 12.dp)
                .fillMaxWidth()
                .clip(MurmurShapes.lg)
                // A filled group on either pane: surface on the cover's bg, bg on the inner pane's surface.
                .background(if (rememberIsWide()) c.bg else c.surface, MurmurShapes.lg),
        ) {
            view.unitOptions.forEachIndexed { i, o ->
                if (i > 0) InsetDivider(start = 16.dp)
                val title = o.unit?.let { HubTexts.unit(it) } ?: stringResource(R.string.detail_unit_auto)
                val sub = when {
                    o.unit == null -> view.chosenUnit?.let { stringResource(R.string.detail_unit_auto_now, HubTexts.unit(it)) }
                    o.installed -> stringResource(R.string.detail_unit_installed)
                    else -> stringResource(R.string.detail_unit_not_installed, Format.sizeText(o.bytes ?: 0, loc))
                }
                ListRow(
                    title = title, subtitle = sub, trailing = RowTrailing.Radio(o.selected),
                    onClick = { onPin(o.unit) },
                    modifier = Modifier.testTag(HubTags.unitOption(o.unit?.name ?: "AUTO")),
                )
            }
        }
        view.pinnedDownload?.let { a ->
            MurmurButton(
                text = stringResource(R.string.detail_download_pinned, Format.sizeText(a.downloadBytes, loc)),
                onClick = { onDownload(a) }, style = ButtonStyle.Outline, leadingIcon = MurmurIcons.dl,
                modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
            )
        }
        view.extraDownload?.takeIf { it.purpose == ExtraDownload.Purpose.PINNED }?.let {
            ExtraDownloadBlock(it, onPause, onResume, Modifier.padding(top = 12.dp))
        }
    }
}

/**
 * The pinned unit's or the CPU version's download while the model keeps running on what is installed: "CPU version ·
 * 446 MB", the row's progress and meta line, and its control (Pause / Resume / Download again; none while checking).
 */
@Composable
private fun ExtraDownloadBlock(extra: ExtraDownload, onPause: () -> Unit, onResume: () -> Unit, modifier: Modifier = Modifier) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    val loc = HubTexts.locale()
    Column(modifier.fillMaxWidth().testTag(HubTags.EXTRA_DOWNLOAD)) {
        Text(
            stringResource(R.string.detail_extra_title, HubTexts.unit(extra.unit), Format.sizeText(extra.artifact.downloadBytes, loc)),
            style = t.label, color = c.ink2,
        )
        DownloadProgress(extra.state, onPaddingTop = 8.dp)
        when (extra.state) {
            is RowState.Downloading, is RowState.Queued ->
                MurmurButton(stringResource(R.string.detail_pause), onPause, Modifier.fillMaxWidth().padding(top = 12.dp), style = ButtonStyle.Secondary, leadingIcon = MurmurIcons.pause)
            is RowState.Paused ->
                MurmurButton(stringResource(R.string.detail_resume), onResume, Modifier.fillMaxWidth().padding(top = 12.dp), style = ButtonStyle.Outline, leadingIcon = MurmurIcons.refresh)
            is RowState.Failed ->
                MurmurButton(stringResource(R.string.detail_download_again), onResume, Modifier.fillMaxWidth().padding(top = 12.dp), style = ButtonStyle.Outline, leadingIcon = MurmurIcons.refresh)
            else -> Unit
        }
    }
}

/** "Benchmark on this phone": last result grid or the explanation, then Run benchmark / Run again. */
@Composable
private fun BenchSection(view: ModelDetailView, kind: ModelKind, nowMs: Long, onBenchmark: () -> Unit) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    val rec = view.bench.record
    Column(Modifier.padding(start = 20.dp, end = 20.dp, top = 24.dp, bottom = 24.dp)) {
        Text(stringResource(R.string.bench_title), style = t.titleSm, color = c.ink, modifier = Modifier.semantics { heading() })
        if (rec != null) {
            Text(stringResource(R.string.bench_last_run, HubTexts.lastRun(rec.atEpochMs, nowMs)), style = t.caption, color = c.ink3, modifier = Modifier.padding(top = 2.dp))
            Big4(detailCells(rec), Modifier.padding(top = 12.dp))
        } else {
            val takes = when (kind) {
                ModelKind.LLM -> R.string.bench_takes_llm
                ModelKind.ASR -> R.string.bench_takes_asr
                else -> R.string.bench_takes_ocr
            }
            Text(
                stringResource(if (view.bench.canRun) takes else R.string.bench_download_first),
                style = t.body, color = c.ink2, modifier = Modifier.padding(top = 6.dp),
            )
        }
        if (view.bench.canRun) {
            MurmurButton(
                text = stringResource(if (rec != null) R.string.bench_run_again else R.string.bench_run),
                onClick = onBenchmark, style = ButtonStyle.Outline, leadingIcon = MurmurIcons.gauge,
                modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
            )
        }
    }
}

/** Detail grid (prototype modelDetailBody): ASR = ×, load, memory, ready after stop; LLM = writing, reading, load, memory. */
@Composable
internal fun detailCells(rec: BenchmarkRecord): List<Big4Cell> {
    val loc = HubTexts.locale()
    val s = stringResource(R.string.bench_unit_s)
    val tps = stringResource(R.string.bench_unit_tps)
    return when (val r = rec.result) {
        is BenchmarkResultData.Asr -> listOf(
            // "14×": the sign belongs to the number (prototype: no <small> unit).
            Big4Cell(Format.rtf(r.rtfx, loc) + stringResource(R.string.bench_unit_x), "", stringResource(R.string.bench_rtf)),
            Big4Cell(Format.seconds(r.loadMsCold.toDouble(), loc), s, stringResource(R.string.bench_load)),
            memCell(r.peakMemMb),
            Big4Cell(Format.seconds(r.tailMs.toDouble(), loc), s, stringResource(R.string.bench_ready_after)),
        )
        is BenchmarkResultData.Llm -> listOf(
            Big4Cell(Format.tps(r.decodeTps, loc), tps, stringResource(R.string.bench_decode)),
            Big4Cell(Format.tps(r.prefillTps, loc), tps, stringResource(R.string.bench_prefill)),
            Big4Cell(Format.seconds(r.loadMsCold.toDouble(), loc), s, stringResource(R.string.bench_load)),
            memCell(r.peakMemMb),
        )
        is BenchmarkResultData.Ocr -> listOf(
            Big4Cell(r.msPerScreenshot.toString(), stringResource(R.string.bench_unit_ms), stringResource(R.string.bench_ocr_ms)),
            Big4Cell(Format.seconds(r.loadMsCold.toDouble(), loc), s, stringResource(R.string.bench_load)),
            memCell(r.peakMemMb),
            Big4Cell(r.ranOn.uppercase(loc), "", stringResource(R.string.bench_ran_on)),
        )
    }
}

@Composable
internal fun memCell(mb: Long): Big4Cell {
    val (n, u) = Format.memory(mb, HubTexts.locale())
    return Big4Cell(n, stringResource(if (u == "GB") R.string.bench_unit_gb else R.string.bench_unit_mb), stringResource(R.string.bench_mem))
}

/** Fixed action bar (prototype detailActions): "✓ In use" or the primary action, plus Delete. 1 dp line on top. */
@Composable
private fun ActionBar(view: ModelDetailView, onUse: () -> Unit, onDownload: () -> Unit, onPause: () -> Unit, onResume: () -> Unit, onDelete: () -> Unit) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    val loc = HubTexts.locale()
    Row(
        Modifier
            .fillMaxWidth()
            .background(c.surface)
            .drawBehind { drawRect(c.line, Offset.Zero, androidx.compose.ui.geometry.Size(size.width, 1.dp.toPx())) }
            .padding(start = 20.dp, end = 20.dp, top = 12.dp, bottom = 14.dp)
            .testTag(HubTags.ACTION_BAR),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.weight(1f)) {
            when (val a = view.action) {
                DetailAction.InUse -> Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(MurmurIcons.bold("check"), null, tint = c.accentInk, modifier = Modifier.size(20.dp))
                    Text(stringResource(app.murmur.ui.design.R.string.design_in_use), style = t.label, color = c.ink2)
                }
                DetailAction.Use -> MurmurButton(stringResource(R.string.detail_use), onUse, Modifier.fillMaxWidth())
                is DetailAction.Download -> MurmurButton(
                    stringResource(R.string.detail_download_action, Format.sizeText(a.bytes, loc)), onDownload, Modifier.fillMaxWidth(),
                    leadingIcon = MurmurIcons.dl,
                )
                is DetailAction.InProgress -> when (val s = a.state) {
                    is RowState.Downloading, is RowState.Queued ->
                        MurmurButton(stringResource(R.string.detail_pause), onPause, Modifier.fillMaxWidth(), style = ButtonStyle.Secondary, leadingIcon = MurmurIcons.pause)
                    is RowState.Paused ->
                        MurmurButton(stringResource(R.string.detail_resume), onResume, Modifier.fillMaxWidth(), leadingIcon = MurmurIcons.refresh)
                    is RowState.Failed ->
                        MurmurButton(stringResource(R.string.detail_download_again), onResume, Modifier.fillMaxWidth(), leadingIcon = MurmurIcons.refresh)
                    is RowState.Preparing -> MurmurButton(
                        s.unit?.let { stringResource(R.string.models_preparing_unit, HubTexts.unit(it)) } ?: stringResource(R.string.models_preparing),
                        {}, Modifier.fillMaxWidth(), style = ButtonStyle.Secondary, loading = true,
                    )
                    else -> MurmurButton(stringResource(R.string.models_checking), {}, Modifier.fillMaxWidth(), style = ButtonStyle.Secondary, loading = true)
                }
                DetailAction.Loading -> Skeleton(Modifier.fillMaxWidth().height(52.dp), MurmurShapes.full)
                DetailAction.None -> {
                    val text = when (val s = view.row.state) {
                        RowState.InTraining -> stringResource(R.string.detail_not_available_yet)
                        is RowState.Unavailable -> HubTexts.blocker(s.blocker, s.socs, view.row.units.firstOrNull()?.unit ?: ComputeUnit.NPU)
                        else -> if (view.row.previewOnly) stringResource(R.string.models_preview_only) else null
                    }
                    if (text != null) Text(text, style = t.label, color = c.ink2)
                }
            }
        }
        if (view.canDelete) {
            MurmurButton(
                stringResource(R.string.detail_delete), onDelete,
                style = ButtonStyle.Outline, leadingIcon = MurmurIcons.trash,
            )
        }
    }
}
