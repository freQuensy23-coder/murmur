package app.murmur.feature.history.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.WindowInsetsSides
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.only
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.ScrollState
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.CompositingStrategy
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.layout.Layout
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import app.murmur.core.model.EntryId
import app.murmur.core.model.MurmurError
import app.murmur.feature.history.AudioUi
import app.murmur.feature.history.Busy
import app.murmur.feature.history.DayKind
import app.murmur.feature.history.DetailUiState
import app.murmur.feature.history.HistoryEvent
import app.murmur.feature.history.HistoryViewModel
import app.murmur.feature.history.R
import app.murmur.feature.history.WordDiff
import app.murmur.ui.design.MurmurIcons
import app.murmur.ui.design.MurmurTheme
import app.murmur.ui.design.components.AppTile
import app.murmur.ui.design.components.AppTileSize
import app.murmur.ui.design.components.AudioPlayer
import app.murmur.ui.design.components.ButtonStyle
import app.murmur.ui.design.components.MurmurButton
import app.murmur.ui.design.components.MurmurDialog
import app.murmur.ui.design.components.MurmurIconButton
import app.murmur.ui.design.components.MurmurSnackbarHost
import app.murmur.ui.design.components.MurmurSnackbarState
import app.murmur.ui.design.components.Overline
import app.murmur.ui.design.components.SegmentedControl
import app.murmur.ui.design.components.TopBar
import app.murmur.ui.design.components.formatClock
import kotlinx.coroutines.launch

/** Which text the segmented control shows. */
private enum class TextMode { FINAL, ORIGINAL, CHANGES }

/** One entry (DESIGN 6.3 detail). [pane]: inner display's detail pane (no back arrow, surface background). */
@Composable
internal fun EntryDetail(vm: HistoryViewModel, id: EntryId, onClose: () -> Unit, modifier: Modifier = Modifier, pane: Boolean = false) {
    DisposableEffect(vm, id) {
        vm.open(id)
        onDispose { vm.closeDetail(id) }
    }
    val detail by vm.detail.collectAsStateWithLifecycle()
    val c = MurmurTheme.colors
    val s = detail?.takeIf { it.id == id }
    val snack = remember { MurmurSnackbarState() }
    val undoWindow = rememberUndoWindowMs()
    var confirm by rememberSaveable { mutableStateOf(false) }
    val context = LocalContext.current
    val messages by rememberUpdatedState(eventMessages())

    // Back, a delete made here and the entry disappearing all close the detail: the host hears it once. (A navigation
    // host keeps an exiting destination composed during its transition; a second call would pop another entry.)
    val latestOnClose by rememberUpdatedState(onClose)
    var closed by remember(id) { mutableStateOf(false) }
    val close: () -> Unit = remember(id) { { if (!closed) { closed = true; latestOnClose() } } }

    LaunchedEffect(s?.missing) { if (s?.missing == true) close() }
    // This entry's Retry / Re-polish failures; those of other entries are left for the list pane.
    LaunchedEffect(vm, id) {
        vm.failures.collect { list ->
            for (e in list) if (e.id == id && vm.consume(e)) launch { snack.show(messages(e)) }
        }
    }

    Column(
        modifier
            .fillMaxSize()
            .background(if (pane) c.surface else c.bg)
            // Above the gesture bar (cover: the pushed screen has no bottom navigation to take the inset).
            .windowInsetsPadding(WindowInsets.navigationBars.only(WindowInsetsSides.Bottom))
            .testTag(HistoryTags.DETAIL),
    ) {
        if (s == null || s.loading || s.entry == null) {
            if (!pane) TopBar(title = "", onBack = close)
            return@Column
        }
        val entry = s.entry
        var mode by rememberSaveable(id) { mutableIntStateOf(TextMode.FINAL.ordinal) }
        val textMode = TextMode.entries[mode]
        val shownText = when (textMode) {
            TextMode.ORIGINAL -> entry.rawText
            else -> entry.finalText
        }.orEmpty()
        DetailTopBar(s, pane, vm, close, onShare = { shareText(context, shownText.ifEmpty { entry.finalText.orEmpty() }) }, onDelete = { confirm = true })
        Box(Modifier.weight(1f).fillMaxWidth()) {
            Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
                DetailBody(s, textMode, onMode = { mode = it.ordinal })
                ContextSection(s)
                Spacer(Modifier.height(24.dp))
            }
            MurmurSnackbarHost(snack, Modifier.align(Alignment.BottomCenter))
        }
        ActionBar(
            s,
            onCopy = { copyText(context, shownText.ifEmpty { entry.finalText.orEmpty() }) },
            onRepolish = { vm.repolish(id) },
            onRetry = { vm.retry(id) },
        )
    }

    if (confirm && s?.entry != null) {
        MurmurDialog(
            title = stringResource(R.string.hist_delete_title),
            body = stringResource(R.string.hist_delete_body, s.appName),
            confirmLabel = stringResource(R.string.hist_delete),
            onConfirm = {
                confirm = false
                vm.delete(setOf(id), undoWindow)
                close()
            },
            dismissLabel = stringResource(R.string.hist_cancel),
            onDismiss = { confirm = false },
            destructive = true,
        )
    }
}

@Composable
internal fun eventMessages(): (HistoryEvent) -> String {
    val noSpeech = stringResource(R.string.hist_retry_nospeech)
    val noRecording = stringResource(R.string.hist_recording_unavailable)
    val retryFailed = stringResource(R.string.hist_retry_failed)
    val repolishFailed = stringResource(R.string.hist_repolish_failed)
    val noModel = stringResource(R.string.hist_repolish_no_model)
    return { e ->
        when (e) {
            is HistoryEvent.RetryFailed -> when (val err = e.error) {
                MurmurError.NoSpeech -> noSpeech
                is MurmurError.StorageFailed -> if (err.detail == "audio" || err.detail == "audio_read") noRecording else retryFailed
                else -> retryFailed
            }
            is HistoryEvent.RepolishFailed -> when (val err = e.error) {
                is MurmurError.LlmFailed -> if (err.detail == "no_model" || err.detail == "none") noModel else repolishFailed
                else -> repolishFailed
            }
        }
    }
}

/** Cover: back, 36 dp tile, title + "Telegram · Today, 14:32 · 0:12", Share, Delete. Pane: no back, 20 dp start. */
@Composable
private fun DetailTopBar(s: DetailUiState, pane: Boolean, vm: HistoryViewModel, onClose: () -> Unit, onShare: () -> Unit, onDelete: () -> Unit) {
    val formats = rememberHistoryFormats()
    val entry = s.entry ?: return
    val created = s.created ?: return
    val day = when (s.dayKind) {
        DayKind.TODAY -> stringResource(R.string.hist_today)
        DayKind.YESTERDAY -> stringResource(R.string.hist_yesterday)
        DayKind.DATE -> formats.dayShort(created.toLocalDate())
    }
    // "Telegram · Today, 14:32 · 0:12"; without a conversation the title is already the app name.
    val subtitle = listOfNotNull(
        s.appName.takeIf { s.title != it },
        stringResource(R.string.hist_day_time, day, formats.time(created)),
        s.durationMs?.let { formatClock(it) },
    ).joinToString(" · ")
    TopBar(
        title = s.title,
        onBack = if (pane) null else onClose,
        subtitle = subtitle,
        modifier = if (pane) Modifier.padding(top = 4.dp) else Modifier,
        leading = {
            AppTile(s.appName, vm.apps.icon(entry.app.pkg), size = AppTileSize.Large, packageName = entry.app.pkg,
                modifier = if (pane) Modifier.padding(end = 2.dp) else Modifier)
        },
    ) {
        MurmurIconButton(MurmurIcons.share, stringResource(R.string.hist_share), onShare)
        MurmurIconButton(MurmurIcons.trash, stringResource(R.string.hist_delete), onDelete)
    }
}

/** Segmented Final / Original / Changes, the text (reading 18/28), the legend for Changes and the player. */
@Composable
private fun DetailBody(s: DetailUiState, mode: TextMode, onMode: (TextMode) -> Unit) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    val entry = s.entry ?: return
    val readingStyle = t.reading
    Column(Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, top = if (hasTextsOf(entry)) 0.dp else 4.dp)) {
        val hasTexts = !entry.finalText.isNullOrBlank() && !entry.rawText.isNullOrBlank()
        if (hasTexts) {
            // The 48 dp control holds the 40 dp track with 4 dp above and below: that margin is the prototype's
            // 4 dp top padding, and its 18 dp gap to the text is 14 dp below the hit area.
            TextModeControl(
                options = listOf(stringResource(R.string.hist_seg_final), stringResource(R.string.hist_seg_original), stringResource(R.string.hist_seg_changes)),
                selected = mode.ordinal,
                onSelect = { onMode(TextMode.entries[it]) },
            )
        }
        val textTop = if (hasTexts) 14.dp else 18.dp
        when {
            s.failed && entry.rawText.isNullOrBlank() -> Text(
                failureCopy(s.failure!!, s.audio !is AudioUi.Unavailable), style = readingStyle, color = c.ink3,
                modifier = Modifier.padding(top = textTop).testTag(HistoryTags.READING),
            )
            mode == TextMode.CHANGES && hasTexts -> {
                val (annotated, marks) = diffText(s.diff)
                val added = stringResource(R.string.hist_diff_added)
                val removed = stringResource(R.string.hist_diff_removed)
                val none = stringResource(R.string.hist_diff_none)
                val description = WordDiff.description(s.diff, added, removed, none)
                MarkedText(
                    annotated, marks, style = readingStyle, color = c.ink,
                    modifier = Modifier.fillMaxWidth().padding(top = textTop).testTag(HistoryTags.DIFF).clearAndSetSemantics { contentDescription = description },
                )
                if (WordDiff.hasChanges(s.diff)) DiffLegend(added, removed) else {
                    Text(none, style = t.rowSub, color = c.ink3, modifier = Modifier.padding(top = 12.dp))
                }
            }
            else -> {
                val text = if (mode == TextMode.ORIGINAL || entry.finalText.isNullOrBlank()) entry.rawText.orEmpty() else entry.finalText.orEmpty()
                MarkedText(
                    AnnotatedString(text), emptyList(), style = readingStyle, color = c.ink, selectable = true,
                    modifier = Modifier.fillMaxWidth().padding(top = textTop).testTag(HistoryTags.READING),
                )
            }
        }
        Recording(s.audio, Modifier.padding(top = 20.dp))
    }
}

/**
 * The Final / Original / Changes control. When large text (DESIGN 8) leaves a segment narrower than its label on one
 * line, the segmented control would break a word in the middle ("Origin|al"); then it keeps the width its labels need
 * and scrolls sideways inside the pane instead (the selected segment is scrolled into view).
 */
@Composable
private fun TextModeControl(options: List<String>, selected: Int, onSelect: (Int) -> Unit) {
    val measurer = rememberTextMeasurer()
    val style = MurmurTheme.type.segment
    val density = LocalDensity.current
    BoxWithConstraints(Modifier.fillMaxWidth()) {
        val labels = remember(options, style, density) {
            options.map { measurer.measure(it, style, maxLines = 1, softWrap = false, density = density).size.width }
        }
        // SegmentedControl: 3 dp track padding at both ends, 2 dp between segments, 12 dp label padding per side;
        // 2 dp of slack for rounding.
        val need = with(density) { labels.sum().toDp() } + SEG_LABEL_PAD * 2 * options.size +
            SEG_GAP * (options.size - 1) + SEG_TRACK_PAD * 2 + 2.dp
        if (need <= maxWidth) {
            SegmentedControl(options, selected, onSelect)
        } else {
            val scroll = rememberScrollState()
            LaunchedEffect(selected, labels) {
                val pad = with(density) { (SEG_LABEL_PAD * 2).roundToPx() }
                val gap = with(density) { SEG_GAP.roundToPx() }
                val start = with(density) { SEG_TRACK_PAD.roundToPx() } + (0 until selected).sumOf { labels[it] + pad + gap }
                val end = start + labels[selected] + pad
                val viewport = scroll.viewportSize
                when {
                    start < scroll.value -> scroll.animateScrollTo(start)
                    end > scroll.value + viewport -> scroll.animateScrollTo((end - viewport).coerceAtLeast(0))
                }
            }
            Box(
                Modifier
                    .fillMaxWidth()
                    .scrollEdgeFade(scroll)
                    .horizontalScroll(scroll)
                    .testTag(HistoryTags.SEGMENTS_SCROLL),
            ) {
                SegmentedControl(options, selected, onSelect, Modifier.width(need))
            }
        }
    }
}

/** Fades the side that can still scroll (24 dp), so a cut segment reads as "more this way", not as clipped. */
private fun Modifier.scrollEdgeFade(scroll: ScrollState): Modifier = this
    .graphicsLayer(compositingStrategy = CompositingStrategy.Offscreen)
    .drawWithContent {
        drawContent()
        val edge = 24.dp.toPx()
        if (scroll.value < scroll.maxValue) {
            drawRect(
                Brush.horizontalGradient(listOf(Color.Black, Color.Transparent), startX = size.width - edge, endX = size.width),
                topLeft = Offset(size.width - edge, 0f), size = Size(edge, size.height), blendMode = BlendMode.DstIn,
            )
        }
        if (scroll.value > 0) {
            drawRect(
                Brush.horizontalGradient(listOf(Color.Transparent, Color.Black), startX = 0f, endX = edge),
                size = Size(edge, size.height), blendMode = BlendMode.DstIn,
            )
        }
    }

private val SEG_LABEL_PAD = 12.dp
private val SEG_GAP = 2.dp
private val SEG_TRACK_PAD = 3.dp

private fun hasTextsOf(e: app.murmur.core.store.HistoryEntry) = !e.finalText.isNullOrBlank() && !e.rawText.isNullOrBlank()

/** Prototype .legend: "abc added" (accentSoft mark) and "abc removed" (err line), 13/18 ink2, 16 dp apart. */
@Composable
private fun DiffLegend(added: String, removed: String) {
    val c = MurmurTheme.colors
    val style = MurmurTheme.type.rowSub
    val sample = "abc"
    Row(Modifier.padding(top = 12.dp), horizontalArrangement = Arrangement.spacedBy(16.dp), verticalAlignment = Alignment.CenterVertically) {
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            val (a, m) = diffText(listOf(app.murmur.feature.history.DiffSpan(sample, app.murmur.feature.history.DiffSpan.Kind.ADDED)))
            MarkedText(a, m, style = style, color = c.ink2, modifier = Modifier.padding(start = 1.dp).clearAndSetSemantics {})
            Text(added, style = style, color = c.ink2)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            val (a, m) = diffText(listOf(app.murmur.feature.history.DiffSpan(sample, app.murmur.feature.history.DiffSpan.Kind.REMOVED)))
            MarkedText(a, m, style = style, color = c.ink2, modifier = Modifier.clearAndSetSemantics {})
            Text(removed, style = style, color = c.ink2)
        }
    }
}

@Composable
private fun Recording(audio: AudioUi, modifier: Modifier) {
    val c = MurmurTheme.colors
    when (audio) {
        is AudioUi.Ready -> {
            val player = rememberPcmPlayer(audio.pcm)
            AudioPlayer(
                peaks = audio.peaks,
                positionMs = player.positionMs,
                durationMs = audio.durationMs,
                playing = player.playing,
                onPlayPause = player::toggle,
                onSeek = player::seek,
                modifier = modifier.testTag(HistoryTags.PLAYER),
            )
        }
        AudioUi.Unavailable -> Row(
            modifier.fillMaxWidth().testTag(HistoryTags.NO_RECORDING),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(MurmurIcons.mic, null, tint = c.ink3, modifier = Modifier.size(16.dp))
            Text(stringResource(R.string.hist_recording_unavailable), style = MurmurTheme.type.body, color = c.ink3)
        }
        AudioUi.Loading -> Box(modifier.fillMaxWidth().height(64.dp))
    }
}

/** "Context used" (prototype .kv) and the retention footnote. */
@Composable
private fun ContextSection(s: DetailUiState) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    val entry = s.entry ?: return
    val formats = rememberHistoryFormats()
    val rows = ArrayList<Pair<String, @Composable () -> Unit>>()
    val value = t.body
    val small = t.body.copy(fontSize = 12.sp)
    val appValue = if (entry.app.conversation.isNullOrBlank()) s.appName else stringResource(R.string.hist_ctx_app_chat, s.appName, entry.app.conversation!!)
    rows += stringResource(R.string.hist_ctx_app) to { Text(appValue, style = value, color = c.ink, textAlign = TextAlign.End) }
    val words = entry.contextWords
    val screen = if (words != null && words > 0) pluralStringResource(R.plurals.hist_ctx_screen_words, words, words) else stringResource(R.string.hist_ctx_screen_unused)
    rows += stringResource(R.string.hist_ctx_screen) to { Text(screen, style = value, color = c.ink, textAlign = TextAlign.End) }
    val cleanup = entry.cleanup
    if (cleanup?.model != null) {
        val tone = when (cleanup.tone) {
            "casual" -> stringResource(R.string.hist_tone_casual)
            "neutral" -> stringResource(R.string.hist_tone_neutral)
            "formal" -> stringResource(R.string.hist_tone_formal)
            "verbatim" -> stringResource(R.string.hist_tone_verbatim)
            else -> stringResource(R.string.hist_tone_default)
        }
        rows += stringResource(R.string.hist_ctx_tone) to { Text(tone, style = value, color = c.ink, textAlign = TextAlign.End) }
    }
    if (s.models.isNotEmpty()) {
        val models = s.models.joinToString(" → ") { "${it.name} ${it.unit.name}" }
        rows += stringResource(R.string.hist_ctx_models) to { Text(models, style = small, color = c.ink, textAlign = TextAlign.End) }
    }
    val tail = entry.timings.asrTailMs
    val llm = entry.timings.llmMs
    val timing = when {
        tail != null && llm != null -> stringResource(R.string.hist_timing_two, sec(formats, tail), sec(formats, llm))
        tail != null -> stringResource(R.string.hist_timing_one, sec(formats, tail))
        entry.timings.stopToDoneMs != null -> stringResource(R.string.hist_timing_one, sec(formats, entry.timings.stopToDoneMs!!))
        else -> null
    }
    if (timing != null) rows += stringResource(R.string.hist_ctx_timing) to { Text(timing, style = small, color = c.ink, textAlign = TextAlign.End) }

    Column(Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, top = 24.dp)) {
        Overline(stringResource(R.string.hist_context), Modifier.padding(bottom = 4.dp))
        KeyValueTable(rows)
        s.deletesAt?.let { at ->
            val whenText = stringResource(R.string.hist_date_at_time, formats.dayShort(at.toLocalDate()), formats.time(at))
            Row(Modifier.padding(top = 14.dp), horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
                Icon(MurmurIcons.clock, null, tint = c.ink3, modifier = Modifier.size(14.dp))
                Text(
                    stringResource(if (s.audio is AudioUi.Ready) R.string.hist_deletes_on else R.string.hist_deletes_on_noaudio, whenText),
                    style = t.caption, color = c.ink3,
                )
            }
        }
    }
}

@Composable
private fun sec(formats: HistoryFormats, ms: Long): String = stringResource(R.string.hist_seconds, formats.seconds(ms))

/**
 * Prototype .kv: a two-column grid (labels auto width, values right-aligned in the rest, 16 dp apart), cells padded
 * 12 dp vertically, a 1 dp line above every row but the first drawn under each column separately (so it breaks in the
 * 16 dp gap). A label column never takes more than 45 % of the width.
 */
@Composable
private fun KeyValueTable(rows: List<Pair<String, @Composable () -> Unit>>) {
    val c = MurmurTheme.colors
    val line = c.line
    val keyStyle = MurmurTheme.type.body
    var keyWidth by remember { mutableIntStateOf(0) }
    var rowTops by remember { mutableStateOf(IntArray(0)) }
    Layout(
        content = {
            rows.forEach { (k, v) ->
                Text(k, style = keyStyle, color = c.ink3)
                Box(contentAlignment = Alignment.TopEnd) { v() }
            }
        },
        modifier = Modifier.fillMaxWidth().drawBehind {
            val gap = 16.dp.toPx()
            val w = 1.dp.toPx()
            for (i in 1 until rowTops.size) {
                val y = rowTops[i] + w / 2
                drawLine(line, Offset(0f, y), Offset(keyWidth.toFloat(), y), w)
                drawLine(line, Offset(keyWidth + gap, y), Offset(size.width, y), w)
            }
        },
    ) { measurables, constraints ->
        val gap = 16.dp.roundToPx()
        val pad = 12.dp.roundToPx()
        val width = constraints.maxWidth
        val keys = measurables.filterIndexed { i, _ -> i % 2 == 0 }
        val values = measurables.filterIndexed { i, _ -> i % 2 == 1 }
        val kw = minOf(keys.maxOfOrNull { it.maxIntrinsicWidth(Constraints.Infinity) } ?: 0, (width * 0.45f).toInt())
        val vw = (width - kw - gap).coerceAtLeast(0)
        val kp = keys.map { it.measure(Constraints(maxWidth = kw)) }
        val vp = values.map { it.measure(Constraints(minWidth = vw, maxWidth = vw)) }
        val tops = IntArray(kp.size)
        var y = 0
        for (i in kp.indices) { tops[i] = y; y += maxOf(kp[i].height, vp[i].height) + 2 * pad }
        keyWidth = kw
        rowTops = tops
        layout(width, y) {
            for (i in kp.indices) {
                kp[i].place(0, tops[i] + pad)
                vp[i].place(kw + gap, tops[i] + pad)
            }
        }
    }
}

/**
 * Bottom action bar: 12/20/14 padding, line above, surface. Copy (primary, takes the rest of the row) + Re-polish;
 * Failed: Try again (+ Re-polish when there is an original transcript). When large text (DESIGN 8) leaves the primary
 * button less than its label needs, the buttons stack, each the full width.
 */
@Composable
private fun ActionBar(s: DetailUiState, onCopy: () -> Unit, onRepolish: () -> Unit, onRetry: () -> Unit) {
    val c = MurmurTheme.colors
    val line = c.line
    val repolishBusy = s.busy == Busy.REPOLISH
    val showRepolish = !s.failed || !s.entry?.rawText.isNullOrBlank()
    Layout(
        content = {
            if (s.failed) {
                MurmurButton(
                    stringResource(R.string.hist_try_again), onRetry, leadingIcon = MurmurIcons.refresh,
                    enabled = s.canRetry || s.busy == Busy.RETRY, loading = s.busy == Busy.RETRY,
                )
            } else {
                MurmurButton(stringResource(R.string.hist_copy), onCopy, leadingIcon = MurmurIcons.copy)
            }
            if (showRepolish) {
                MurmurButton(
                    stringResource(R.string.hist_repolish), onRepolish, style = ButtonStyle.Secondary,
                    leadingIcon = MurmurIcons.refresh, enabled = s.canRepolish || repolishBusy, loading = repolishBusy,
                )
            }
        },
        modifier = Modifier
            .fillMaxWidth()
            .background(c.surface)
            .drawBehind { drawLine(line, Offset(0f, 0f), Offset(size.width, 0f), 1.dp.toPx()) }
            .padding(start = 20.dp, end = 20.dp, top = 12.dp, bottom = 14.dp),
    ) { ms, constraints ->
        val gap = 8.dp.roundToPx()
        val width = constraints.maxWidth
        val primary = ms[0]
        val secondary = ms.getOrNull(1)
        if (secondary == null) {
            val p = primary.measure(Constraints.fixedWidth(width))
            return@Layout layout(width, p.height) { p.place(0, 0) }
        }
        val secondW = secondary.maxIntrinsicWidth(Constraints.Infinity)
        val primaryNeed = primary.maxIntrinsicWidth(Constraints.Infinity)
        if (secondW + gap + primaryNeed <= width) {
            val sp = secondary.measure(Constraints.fixedWidth(secondW))
            val pp = primary.measure(Constraints.fixedWidth(width - secondW - gap))
            val h = maxOf(sp.height, pp.height)
            layout(width, h) {
                pp.place(0, (h - pp.height) / 2)
                sp.place(width - secondW, (h - sp.height) / 2)
            }
        } else {
            val pp = primary.measure(Constraints.fixedWidth(width))
            val sp = secondary.measure(Constraints.fixedWidth(width))
            layout(width, pp.height + gap + sp.height) {
                pp.place(0, 0)
                sp.place(0, pp.height + gap)
            }
        }
    }
}
