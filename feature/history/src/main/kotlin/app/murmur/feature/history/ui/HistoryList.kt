package app.murmur.feature.history.ui

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import androidx.activity.compose.BackHandler
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.platform.LocalAccessibilityManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import app.murmur.core.model.EntryId
import app.murmur.core.store.EntryStatus
import app.murmur.feature.history.DayGroup
import app.murmur.feature.history.FailureKind
import app.murmur.feature.history.HistoryRow
import app.murmur.feature.history.HistoryUiState
import app.murmur.feature.history.HistoryViewModel
import app.murmur.feature.history.ListMode
import app.murmur.feature.history.MatchIn
import app.murmur.feature.history.R
import app.murmur.feature.history.SearchUiState
import app.murmur.ui.design.Motion
import app.murmur.ui.design.MurmurIcons
import app.murmur.ui.design.MurmurTheme
import app.murmur.ui.design.REDUCED_MOTION_FADE_MS
import app.murmur.ui.design.components.AppTile
import app.murmur.ui.design.components.AppTileSize
import app.murmur.ui.design.components.ButtonStyle
import app.murmur.ui.design.components.EmptyState
import app.murmur.ui.design.components.MurmurButton
import app.murmur.ui.design.components.MurmurChip
import app.murmur.ui.design.components.MurmurDialog
import app.murmur.ui.design.components.MurmurIconButton
import app.murmur.ui.design.components.MurmurSnackbar
import app.murmur.ui.design.components.MurmurSnackbarHost
import app.murmur.ui.design.components.MurmurSnackbarState
import app.murmur.ui.design.components.SNACKBAR_ACTION_MS
import app.murmur.ui.design.components.ScreenHeader
import app.murmur.ui.design.components.SearchField
import app.murmur.ui.design.components.Skeleton
import app.murmur.ui.design.components.StatusKind
import app.murmur.ui.design.components.StatusLabel
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.launch

/** Test tags (instrumented tests and the shell's tests). */
object HistoryTags {
    const val LIST = "history-list"
    const val SCREEN = "screen-history"
    const val SEARCH = "history-search"
    const val UNDO = "history-undo"
    const val EMPTY = "history-empty"
    const val NO_RESULTS = "history-noresults"
    const val DETAIL = "history-detail"
    const val DIFF = "history-diff"
    const val READING = "history-reading"
    const val PLAYER = "history-player"
    const val NO_RECORDING = "history-no-recording"
    /** The Final / Original / Changes control when large text makes it scroll sideways. */
    const val SEGMENTS_SCROLL = "history-segments-scroll"
    /** Snackbars of the list pane (Retry / Re-polish failures of an entry that is not open). */
    const val LIST_SNACK = "history-list-snack"
    fun row(id: EntryId) = "history-row-${id.value}"
}

/** The list pane (DESIGN 6.3 list, search, multi-select, delete Undo). */
@Composable
internal fun HistoryListPane(
    vm: HistoryViewModel,
    onOpenEntry: (EntryId) -> Unit,
    modifier: Modifier = Modifier,
    selected: EntryId? = null,
    onTryDictation: (() -> Unit)? = null,
) {
    val state by vm.state.collectAsStateWithLifecycle()
    val c = MurmurTheme.colors
    val context = LocalContext.current
    val undoWindow = rememberUndoWindowMs()
    var confirmMany by rememberSaveable { mutableStateOf(false) }
    // Hoisted, so the lists keep their scroll position across state changes (search keystrokes, loading).
    val listState = rememberLazyListState()
    val searchListState = rememberLazyListState()
    val snack = remember { MurmurSnackbarState() }
    val messages by rememberUpdatedState(eventMessages())
    val withApp = stringResource(R.string.hist_event_app)

    // Retry / Re-polish failures of entries whose detail is not on screen (the user moved on, or the inner display's
    // detail pane shows another entry): shown here with the app's name, so none is lost.
    LaunchedEffect(vm) {
        combine(vm.failures, vm.openEntry) { list, open -> list.filter { it.id != open } }.collect { list ->
            for (e in list) {
                if (!vm.consume(e)) continue
                val text = vm.appNameOf(e.id)?.let { withApp.format(it, messages(e)) } ?: messages(e)
                launch { snack.show(text) }
            }
        }
    }

    BackHandler(enabled = state.selecting) { vm.clearSelection() }
    BackHandler(enabled = !state.selecting && state.search != null) { vm.closeSearch() }

    Box(modifier.fillMaxSize().background(c.bg).testTag(HistoryTags.SCREEN)) {
        Column(Modifier.fillMaxSize()) {
            val search = state.search
            when {
                state.selecting -> SelectionHeader(
                    count = state.selection.size,
                    onClose = vm::clearSelection,
                    onCopy = { copyText(context, vm.textsOf(state.selection)); vm.clearSelection() },
                    onShare = { shareText(context, vm.textsOf(state.selection)) },
                    onDelete = { confirmMany = true },
                )
                search != null -> SearchHeader(search, state.retentionDays, vm)
                else -> ScreenHeader(stringResource(R.string.hist_title)) {
                    if (state.mode == ListMode.LIST) {
                        MurmurIconButton(MurmurIcons.search, stringResource(R.string.hist_search_cd), vm::openSearch)
                    }
                }
            }
            Box(Modifier.weight(1f).fillMaxWidth()) {
                if (search != null) {
                    SearchBody(search, state, selected, vm, onOpenEntry, searchListState)
                } else {
                    when (state.mode) {
                        ListMode.LOADING -> LoadingRows()
                        ListMode.ERROR -> ErrorState(vm::retryLoad)
                        ListMode.OFF -> EmptyBlock(
                            title = stringResource(R.string.hist_off_title),
                            body = stringResource(R.string.hist_off_body),
                        )
                        ListMode.EMPTY -> EmptyBlock(
                            title = stringResource(R.string.hist_empty_title),
                            body = pluralStringResource(R.plurals.hist_empty_body, state.retentionDays, state.retentionDays),
                            actionLabel = onTryDictation?.let { stringResource(R.string.hist_try_dictation) },
                            onAction = onTryDictation,
                        )
                        ListMode.LIST -> DayList(
                            listState = listState,
                            days = state.days,
                            state = state,
                            selected = selected,
                            vm = vm,
                            onOpenEntry = onOpenEntry,
                            header = { RetentionNote(state.retentionDays) },
                            countOf = { g -> pluralStringResource(R.plurals.hist_count, g.rows.size, g.rows.size) },
                        )
                    }
                }
            }
        }
        Column(Modifier.align(Alignment.BottomCenter).fillMaxWidth()) {
            MurmurSnackbarHost(snack, Modifier.testTag(HistoryTags.LIST_SNACK))
            UndoSnackbar(state, vm, Modifier)
        }
    }

    if (confirmMany && state.selecting) {
        val n = state.selection.size
        MurmurDialog(
            title = pluralStringResource(R.plurals.hist_delete_title_many, n, n),
            body = stringResource(R.string.hist_delete_body_many),
            confirmLabel = stringResource(R.string.hist_delete),
            onConfirm = { confirmMany = false; vm.delete(state.selection, undoWindow) },
            dismissLabel = stringResource(R.string.hist_cancel),
            onDismiss = { confirmMany = false },
            destructive = true,
        )
    } else if (confirmMany) {
        confirmMany = false
    }
}

/** The Undo window: 6 s (DESIGN 4), longer when the accessibility manager recommends it (TalkBack users). */
@Composable
internal fun rememberUndoWindowMs(): Long {
    val a11y = LocalAccessibilityManager.current
    return remember(a11y) {
        a11y?.calculateRecommendedTimeoutMillis(SNACKBAR_ACTION_MS, containsIcons = false, containsText = true, containsControls = true)
            ?: SNACKBAR_ACTION_MS
    }
}

@Composable
private fun UndoSnackbar(state: HistoryUiState, vm: HistoryViewModel, modifier: Modifier) {
    val pending = state.pendingDelete
    var shown by remember { mutableStateOf(pending) }
    if (pending != null) shown = pending
    val reduced = MurmurTheme.reducedMotion
    AnimatedVisibility(
        visible = pending != null,
        modifier = modifier,
        enter = if (reduced) fadeIn(tween(REDUCED_MOTION_FADE_MS)) else fadeIn(tween(Motion.shortMs)) + slideInVertically(tween(Motion.mediumMs, easing = Motion.decelerateEasing)) { it / 2 },
        exit = fadeOut(tween(Motion.shortMs, easing = Motion.accelerateEasing)),
    ) {
        val p = shown ?: return@AnimatedVisibility
        val message = if (p.ids.size == 1) stringResource(R.string.hist_deleted_snack)
        else pluralStringResource(R.plurals.hist_deleted_snack_many, p.ids.size, p.ids.size)
        MurmurSnackbar(
            message, stringResource(R.string.hist_undo), onAction = { vm.undo() },
            modifier = Modifier.padding(start = 12.dp, end = 12.dp, bottom = 12.dp).testTag(HistoryTags.UNDO),
        )
    }
}

// ---------------------------------------------------------------------------------------------------- list

@Composable
private fun DayList(
    listState: LazyListState,
    days: List<DayGroup>,
    state: HistoryUiState,
    selected: EntryId?,
    vm: HistoryViewModel,
    onOpenEntry: (EntryId) -> Unit,
    header: (@Composable () -> Unit)?,
    countOf: @Composable (DayGroup) -> String,
) {
    val formats = rememberHistoryFormats()
    val today = stringResource(R.string.hist_today)
    val yesterday = stringResource(R.string.hist_yesterday)
    LazyColumn(Modifier.fillMaxSize().testTag(HistoryTags.LIST), state = listState, contentPadding = PaddingValues(bottom = 24.dp)) {
        if (header != null) item(key = "note") { header() }
        for (g in days) {
            item(key = "day-${g.day}") {
                DayHeader(formats.dayLabel(g.day, g.kind, today, yesterday), if (g.deletesTonight) null else countOf(g))
            }
            itemsIndexed(g.rows, key = { _, r -> r.id.value }) { i, r ->
                val prev = g.rows.getOrNull(i - 1)
                val highlighted = r.id == selected && !state.selecting
                val prevHighlighted = prev != null && ((prev.id == selected && !state.selecting) || prev.id in state.selection)
                HistoryRowItem(
                    row = r,
                    time = formats.time(r.time),
                    divider = i > 0 && !highlighted && r.id !in state.selection && !prevHighlighted,
                    highlighted = highlighted,
                    selecting = state.selecting,
                    checked = r.id in state.selection,
                    onClick = { if (state.selecting) vm.toggleSelection(r.id) else onOpenEntry(r.id) },
                    onLongClick = { vm.toggleSelection(r.id) },
                    vm = vm,
                )
            }
        }
    }
}

/** Prototype .hist-note: 16 dp clock, 13/18 ink3, 4/20/12 padding, 10 dp gap. */
@Composable
private fun RetentionNote(days: Int) {
    val c = MurmurTheme.colors
    Row(Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, top = 4.dp, bottom = 12.dp), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        Icon(MurmurIcons.clock, null, tint = c.ink3, modifier = Modifier.padding(top = 1.dp).size(16.dp))
        Text(pluralStringResource(R.plurals.hist_retention_line, days, days), style = MurmurTheme.type.rowSub, color = c.ink3)
    }
}

/** Prototype .day: 18/20/8 padding, day 15/20 600, count 13/18 ink3, or "Deletes tonight" in warn with a clock. */
@Composable
private fun DayHeader(label: String, count: String?) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    val tonight = stringResource(R.string.hist_deletes_tonight)
    Row(
        Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, top = 18.dp, bottom = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalAlignment = Alignment.Bottom,
    ) {
        Text(
            label, style = t.rowTitle.copy(fontWeight = FontWeight.SemiBold), color = c.ink,
            modifier = Modifier.weight(1f).alignByBaseline().semantics { heading() },
        )
        if (count != null) {
            Text(count, style = t.rowSub, color = c.ink3, maxLines = 1, modifier = Modifier.alignByBaseline())
        } else {
            StatusLabel(tonight, StatusKind.Warn, icon = MurmurIcons.clock, modifier = Modifier.alignByBaseline())
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun HistoryRowItem(
    row: HistoryRow,
    time: String,
    divider: Boolean,
    highlighted: Boolean,
    selecting: Boolean,
    checked: Boolean,
    onClick: () -> Unit,
    onLongClick: () -> Unit,
    vm: HistoryViewModel,
) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    val line = c.line
    val meta = if (row.conversation != null) "${row.appName} · ${row.conversation}" else row.appName
    val status = when (row.status) {
        EntryStatus.COPIED -> stringResource(R.string.hist_status_copied)
        EntryStatus.FAILED -> stringResource(R.string.hist_status_failed)
        EntryStatus.INSERTED -> null
    }
    val failureText = row.failure?.let { failureCopy(it, row.audioKept) }
    val originalPrefix = if (row.matchIn == MatchIn.ORIGINAL) stringResource(R.string.hist_match_original) else null
    val shownText = row.text ?: failureText.orEmpty()
    val selectLabel = stringResource(R.string.hist_select)
    val selectedState = stringResource(R.string.hist_selected_state)
    val description = listOfNotNull(meta, time, status, listOfNotNull(originalPrefix, shownText).joinToString(" ")).joinToString(", ")
    Column(
        Modifier
            .fillMaxWidth()
            .testTag(HistoryTags.row(row.id))
            .background(if (highlighted || checked) c.accentSoft else androidx.compose.ui.graphics.Color.Transparent)
            .combinedClickable(onClick = onClick, onLongClick = onLongClick, onLongClickLabel = selectLabel)
            .semantics(mergeDescendants = true) {
                contentDescription = description
                if (selecting) { selected = checked; if (checked) stateDescription = selectedState }
            }
            .drawBehind { if (divider) drawLine(line, Offset(20.dp.toPx(), 0f), Offset(size.width - 20.dp.toPx(), 0f), 1.dp.toPx()) }
            .padding(start = 20.dp, end = 20.dp, top = 12.dp, bottom = 14.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (selecting && checked) {
                Box(Modifier.size(20.dp).clip(RoundedCornerShape(6.dp)).background(c.accent), contentAlignment = Alignment.Center) {
                    Icon(MurmurIcons.bold("check"), null, tint = c.onAccent, modifier = Modifier.size(14.dp))
                }
            } else {
                AppTile(row.appName, vm.apps.icon(row.entry.app.pkg), size = AppTileSize.Small, packageName = row.entry.app.pkg)
            }
            Text(
                meta, style = t.rowSub.copy(fontWeight = FontWeight.Medium), color = c.ink2, maxLines = 1, overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f),
            )
            when (row.status) {
                EntryStatus.COPIED -> StatusLabel(status!!, StatusKind.Warn, icon = MurmurIcons.clip)
                EntryStatus.FAILED -> StatusLabel(status!!, StatusKind.Err, icon = MurmurIcons.alert)
                EntryStatus.INSERTED -> Unit
            }
            Text(time, style = t.mono, color = c.ink3, maxLines = 1)
        }
        val (annotated, marks) = highlighted(shownText, row.highlights)
        val full = if (originalPrefix != null) {
            // "Original:" in ink3 before a hit that is only in the original transcript.
            val prefix = "$originalPrefix "
            val shifted = marks.map { Mark((it.range.first + prefix.length)..(it.range.last + prefix.length), it.kind) }
            androidx.compose.ui.text.buildAnnotatedString {
                pushStyle(androidx.compose.ui.text.SpanStyle(color = c.ink3)); append(prefix); pop(); append(annotated)
            } to shifted
        } else annotated to marks
        MarkedText(
            full.first, full.second, style = t.entry, color = if (row.text == null) c.ink3 else c.ink,
            maxLines = 3, modifier = Modifier.fillMaxWidth().padding(top = 6.dp),
        )
    }
}

@Composable
internal fun failureCopy(kind: FailureKind, audioKept: Boolean): String = stringResource(
    when (kind) {
        FailureKind.NO_SPEECH -> if (audioKept) R.string.hist_failed_nospeech else R.string.hist_failed_nospeech_noaudio
        FailureKind.OTHER -> if (audioKept) R.string.hist_failed_other else R.string.hist_failed_other_noaudio
    },
)

// ---------------------------------------------------------------------------------------------------- headers

@Composable
private fun SelectionHeader(count: Int, onClose: () -> Unit, onCopy: () -> Unit, onShare: () -> Unit, onDelete: () -> Unit) {
    val c = MurmurTheme.colors
    Row(
        Modifier.fillMaxWidth().heightIn(min = 64.dp).padding(start = 8.dp, end = 8.dp, top = 8.dp, bottom = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        MurmurIconButton(MurmurIcons.x, stringResource(R.string.hist_cancel_selection), onClose)
        Text(
            pluralStringResource(R.plurals.hist_selected, count, count), style = MurmurTheme.type.topBar, color = c.ink,
            modifier = Modifier.weight(1f).padding(start = 4.dp).semantics { heading() },
        )
        MurmurIconButton(MurmurIcons.copy, stringResource(R.string.hist_copy), onCopy)
        MurmurIconButton(MurmurIcons.share, stringResource(R.string.hist_share), onShare)
        MurmurIconButton(MurmurIcons.trash, stringResource(R.string.hist_delete), onDelete)
    }
}

/** hist-search: back + active search field (10/16/6), app chips (8/20/4, 8 dp apart). */
@Composable
private fun SearchHeader(search: SearchUiState, retentionDays: Int, vm: HistoryViewModel) {
    val focus = remember { FocusRequester() }
    LaunchedEffect(Unit) { runCatching { focus.requestFocus() } }
    Column(Modifier.testTag(HistoryTags.SEARCH)) {
        SearchField(
            query = search.query,
            onQueryChange = vm::setQuery,
            hint = pluralStringResource(R.plurals.hist_search_hint, retentionDays.coerceAtLeast(1), retentionDays.coerceAtLeast(1)),
            active = true,
            onBack = vm::closeSearch,
            focusRequester = focus,
            modifier = Modifier.fillMaxWidth().padding(start = 16.dp, end = 16.dp, top = 10.dp, bottom = 6.dp),
        )
        LazyRow(
            contentPadding = PaddingValues(horizontal = 20.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            // Chips are 32 dp inside a 48 dp hit area: the 8/4 padding of the prototype minus the 8 dp hit margins.
            modifier = Modifier.fillMaxWidth().padding(top = 0.dp, bottom = 0.dp),
        ) {
            item(key = "all") { MurmurChip(stringResource(R.string.hist_all_apps), { vm.setAppFilter(null) }, selected = search.pkg == null) }
            items(search.apps, key = { it.pkg }) { a -> MurmurChip(a.label, { vm.setAppFilter(a.pkg) }, selected = search.pkg == a.pkg) }
        }
    }
}

@Composable
private fun SearchBody(
    search: SearchUiState,
    state: HistoryUiState,
    selected: EntryId?,
    vm: HistoryViewModel,
    onOpenEntry: (EntryId) -> Unit,
    listState: LazyListState,
) {
    val c = MurmurTheme.colors
    // null only until the first answer; later queries keep the previous results until theirs arrive.
    val results = search.results ?: return
    if (search.active && results.isEmpty()) {
        Column(Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 40.dp).testTag(HistoryTags.NO_RESULTS)) {
            val d = state.retentionDays.coerceAtLeast(1)
            Text(pluralStringResource(R.plurals.hist_search_empty, d, d), style = MurmurTheme.type.titleSm, color = c.ink)
            Text(stringResource(R.string.hist_search_empty_body), style = MurmurTheme.type.body, color = c.ink2, modifier = Modifier.padding(top = 6.dp))
        }
        return
    }
    DayList(
        listState = listState,
        days = results,
        state = state,
        selected = selected,
        vm = vm,
        onOpenEntry = onOpenEntry,
        header = if (search.active) {
            {
                Text(
                    pluralStringResource(R.plurals.hist_search_caption, search.count, search.count),
                    style = MurmurTheme.type.caption, color = c.ink3,
                    // 12 dp below the chips' 32 dp visual = 8 dp below their 48 dp hit area.
                    modifier = Modifier.padding(start = 20.dp, end = 20.dp, top = 8.dp),
                )
            }
        } else null,
        countOf = { g ->
            if (search.active) pluralStringResource(R.plurals.hist_result_count, g.rows.size, g.rows.size)
            else pluralStringResource(R.plurals.hist_count, g.rows.size, g.rows.size)
        },
    )
}

// ---------------------------------------------------------------------------------------------------- states

@Composable
private fun EmptyBlock(title: String, body: String, actionLabel: String? = null, onAction: (() -> Unit)? = null) {
    EmptyState(
        icon = MurmurIcons.hist, title = title, body = body,
        actionLabel = actionLabel, actionIcon = MurmurIcons.mic, onAction = onAction,
        modifier = Modifier.padding(start = 24.dp, end = 24.dp, top = 48.dp).testTag(HistoryTags.EMPTY),
    )
}

@Composable
private fun ErrorState(onRetry: () -> Unit) {
    val c = MurmurTheme.colors
    Column(Modifier.fillMaxWidth().padding(start = 24.dp, end = 24.dp, top = 48.dp)) {
        Box(Modifier.size(56.dp).clip(RoundedCornerShape(18.dp)).background(c.errSoft), contentAlignment = Alignment.Center) {
            Icon(MurmurIcons.alert, null, tint = c.err, modifier = Modifier.size(26.dp))
        }
        Text(stringResource(R.string.hist_error_title), style = MurmurTheme.type.title, color = c.ink, modifier = Modifier.padding(top = 20.dp))
        Text(stringResource(R.string.hist_error_body), style = MurmurTheme.type.bodyLg, color = c.ink2, modifier = Modifier.padding(top = 8.dp))
        MurmurButton(stringResource(R.string.hist_retry), onRetry, style = ButtonStyle.Secondary, leadingIcon = MurmurIcons.refresh, modifier = Modifier.padding(top = 24.dp))
    }
}

/** Skeleton rows while the first page loads. */
@Composable
private fun LoadingRows() {
    Column(Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 16.dp), verticalArrangement = Arrangement.spacedBy(22.dp)) {
        repeat(4) {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Skeleton(Modifier.width(160.dp).height(16.dp))
                Skeleton(Modifier.fillMaxWidth().height(16.dp))
                Skeleton(Modifier.fillMaxWidth(0.7f).height(16.dp))
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------- system

internal fun copyText(context: Context, text: String) {
    if (text.isEmpty()) return
    val cm = context.getSystemService(ClipboardManager::class.java) ?: return
    // Android 13+ shows its own clipboard confirmation (DESIGN 9: no extra toast).
    cm.setPrimaryClip(ClipData.newPlainText("", text))
}

internal fun shareText(context: Context, text: String) {
    if (text.isEmpty()) return
    val send = Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_TEXT, text)
    val chooser = Intent.createChooser(send, context.getString(R.string.hist_share_title))
    if (context !is android.app.Activity) chooser.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    context.startActivity(chooser)
}
