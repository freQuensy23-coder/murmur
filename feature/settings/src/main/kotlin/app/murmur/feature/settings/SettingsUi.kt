package app.murmur.feature.settings

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.layout.Layout
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.LifecycleResumeEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import app.murmur.core.model.ProductName
import app.murmur.core.prompt.PromptEditState
import app.murmur.core.store.BubbleSize
import app.murmur.core.store.BubbleVisibility
import app.murmur.core.store.Feedback
import app.murmur.core.store.Settings
import app.murmur.core.store.Tone
import app.murmur.ui.design.MurmurIcons
import app.murmur.ui.design.MurmurShapes
import app.murmur.ui.design.MurmurTheme
import app.murmur.ui.design.components.Banner
import app.murmur.ui.design.components.BannerKind
import app.murmur.ui.design.components.ListRow
import app.murmur.ui.design.components.MurmurSnackbarHost
import app.murmur.ui.design.components.MurmurSnackbarState
import app.murmur.ui.design.components.Overline
import app.murmur.ui.design.components.RowTrailing
import app.murmur.ui.design.components.ScreenHeader
import app.murmur.ui.design.components.SectionLabel
import app.murmur.ui.design.components.StatusKind
import app.murmur.ui.design.rememberIsWide
import java.text.NumberFormat

/** Sheets of the Settings screen. */
enum class SheetKind { LANGUAGES, FEEDBACK, TYPING, VISIBILITY, SIZE, HIDDEN_IN, RETENTION, TONE_ADD, TIMEOUT_SCALE }

/** Test tags (instrumented tests and the screenshot harness). */
object SettingsTags {
    const val SCREEN = "screen-settings"
    const val LIST = "settings-list"
    const val DETAIL = "settings-detail"
    const val PROMPT = "settings-prompt-editor"
    const val PROMPT_FIELD = "settings-prompt-field"
    /** The inner Cleanup pane's preview: an excerpt clipped at 236 dp on purpose (prototype .editor max-height). */
    const val PROMPT_PREVIEW = "settings-prompt-preview"
    const val PREFS_FIELD = "settings-prefs-field"
    const val PROMPT_SAVE = "settings-prompt-save"
    const val PROMPT_TEST = "settings-prompt-test"
    const val PROMPT_RESET = "settings-prompt-reset"
    const val TRY_RESULT = "settings-try-result"
    const val TONE = "settings-tone"
    const val OFF_BANNER = "settings-off-banner"
    fun row(key: String) = "settings-row-$key"
    fun category(c: SettingsCategory) = "settings-cat-${c.name.lowercase()}"
    fun sheetOption(key: String) = "settings-opt-$key"
    fun toneChip(pkg: String) = "settings-tone-chip-$pkg"
}

/** One row of a Settings group; [indented] rows (all `srow`s of the prototype) get the 54 dp divider between them. */
internal class GroupItem(val indented: Boolean = true, val content: @Composable () -> Unit)

/**
 * A grouped list (DESIGN 4: surface, lg radius, 1 dp dividers). As in the prototype, the divider between two
 * indented rows starts at 54 dp (after the 24 dp icon column, also for rows without an icon), else at 16 dp.
 */
@Composable
internal fun SettingsGroup(items: List<GroupItem>, modifier: Modifier = Modifier) {
    val line = MurmurTheme.colors.line
    Column(modifier.clip(MurmurShapes.lg).background(MurmurTheme.colors.surface)) {
        items.forEachIndexed { i, item ->
            // Prototype `.group .row + .row::before`: the 1 dp rule is drawn over the top of the row, it takes no height.
            val inset = if (i == 0) null else if (item.indented && items[i - 1].indented) 54.dp else 16.dp
            Box(
                if (inset == null) Modifier else Modifier.drawWithContent {
                    drawContent()
                    drawRect(line, topLeft = Offset(inset.toPx(), 0f), size = Size(size.width - inset.toPx(), 1.dp.toPx()))
                },
            ) { item.content() }
        }
    }
}

/** Everything the list needs besides the view model. */
internal class SettingsActions(
    val vm: SettingsViewModel,
    val openSheet: (SheetKind) -> Unit,
    val openSystem: (SystemScreen) -> Unit,
)

/**
 * The Settings screen (DESIGN 6.5): grouped lists on compact width, the 260 dp category list + detail pane on medium
 * width and up; the prompt editor and Tone per app replace the list (cover) or the detail pane (inner).
 */
@Composable
internal fun SettingsRoot(
    vm: SettingsViewModel,
    showDebug: Boolean,
    versionName: String?,
    onSubScreenChange: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
    initialSheet: SheetKind? = null,
    initialPromptDialog: PromptDialog? = null,
) {
    val ui by vm.ui.collectAsStateWithLifecycle()
    val updateState by vm.updateState.collectAsStateWithLifecycle()
    val sub by vm.sub.collectAsStateWithLifecycle()
    val category by vm.category.collectAsStateWithLifecycle()
    val wide = rememberIsWide()
    val context = LocalContext.current
    val c = MurmurTheme.colors

    LaunchedEffect(sub, wide) { onSubScreenChange(sub != null && !wide) }
    // Back from a system screen (Permissions rows, the banner): read the permissions again.
    LifecycleResumeEffect(vm) {
        vm.refreshPermissions()
        onPauseOrDispose { }
    }
    var sheet by rememberSaveable { mutableStateOf(initialSheet) }
    val actions = remember(vm) {
        SettingsActions(vm, openSheet = { sheet = it }, openSystem = { SystemScreens.open(context, it) })
    }
    val snack = remember { MurmurSnackbarState() }
    val event by vm.events.collectAsStateWithLifecycle()
    val eventText = event?.let {
        when (it) {
            SettingsEvent.POSITION_RESET -> stringResource(R.string.settings_position_reset)
            SettingsEvent.HISTORY_DELETED -> stringResource(R.string.settings_history_deleted)
            SettingsEvent.NOTHING_TO_DELETE -> stringResource(R.string.settings_nothing_to_delete)
            SettingsEvent.NOT_SAVED -> stringResource(R.string.settings_not_saved)
            SettingsEvent.DELETE_FAILED -> stringResource(R.string.settings_delete_failed)
            SettingsEvent.DELETION_DEFERRED -> stringResource(R.string.settings_deletion_deferred)
        }
    }
    LaunchedEffect(event) {
        if (eventText != null) {
            vm.eventShown()
            snack.show(eventText)
        }
    }
    val showApp = updateState != null
    // A category that is not shown in this build (debug in release, app without updater) falls back to the first.
    val shownCategory = when {
        category == SettingsCategory.DEBUG && !showDebug -> SettingsCategory.DICTATION
        category == SettingsCategory.APP && !showApp -> SettingsCategory.DICTATION
        else -> category
    }

    Box(modifier.fillMaxSize().background(c.bg).testTag(SettingsTags.SCREEN)) {
        if (wide) {
            Row(Modifier.fillMaxSize()) {
                CategoryPane(shownCategory, showApp, showDebug, versionName, onSelect = vm::selectCategory)
                Box(Modifier.weight(1f).fillMaxHeight()) {
                    when (sub) {
                        SubScreen.PROMPT -> PromptEditorHost(vm, wide = true, initialPromptDialog)
                        SubScreen.TONE -> ToneScreen(vm, ui, actions, wide = true)
                        null -> DetailPane(shownCategory, ui, updateState, versionName, actions)
                    }
                }
            }
        } else {
            when (sub) {
                SubScreen.PROMPT -> PromptEditorHost(vm, wide = false, initialPromptDialog)
                SubScreen.TONE -> ToneScreen(vm, ui, actions, wide = false)
                null -> CoverList(ui, updateState, versionName, actions, showDebug)
            }
        }
        MurmurSnackbarHost(snack, Modifier.align(Alignment.BottomCenter))
    }

    sheet?.let { kind -> SettingsSheets(kind, ui, vm, onDismiss = { sheet = null }) }

    val retention by vm.retentionConfirm.collectAsStateWithLifecycle()
    retention?.let { r ->
        val title = if (r.days == 0) {
            stringResource(R.string.retention_stop_title)
        } else {
            pluralStringResource(R.plurals.retention_confirm_title, r.count, r.count, pluralStringResource(R.plurals.settings_older_days, r.days, r.days))
        }
        val body = if (r.days == 0) pluralStringResource(R.plurals.retention_stop_body, r.count, r.count) else stringResource(R.string.retention_confirm_body)
        SettingsDialog(
            title, body, stringResource(R.string.settings_delete), vm::confirmRetention,
            stringResource(R.string.settings_cancel), vm::cancelRetention, destructive = true,
        )
    }
    val deleteAll by vm.deleteAllCount.collectAsStateWithLifecycle()
    deleteAll?.let { n ->
        SettingsDialog(
            stringResource(R.string.settings_delete_all_title), pluralStringResource(R.plurals.settings_delete_all_body, n, n),
            stringResource(R.string.settings_delete), vm::confirmDeleteAll,
            stringResource(R.string.settings_cancel), vm::cancelDeleteAll, destructive = true,
        )
    }
}

// ================================================================ cover

@Composable
private fun CoverList(ui: SettingsUi, updateState: AppUpdateState?, versionName: String?, actions: SettingsActions, showDebug: Boolean) {
    Column(Modifier.fillMaxSize()) {
        ScreenHeader(stringResource(R.string.settings_title))
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).testTag(SettingsTags.LIST)) {
            if (!ui.serviceConnected) {
                OffBanner(actions, Modifier.padding(start = 16.dp, end = 16.dp, bottom = 4.dp))
            }
            val sections = buildList {
                add(SettingsCategory.DICTATION); add(SettingsCategory.BUBBLE); add(SettingsCategory.CLEANUP)
                add(SettingsCategory.PRIVACY); add(SettingsCategory.PERMISSIONS)
                if (updateState != null) add(SettingsCategory.APP)
                if (showDebug) add(SettingsCategory.DEBUG)
            }
            sections.forEach { s ->
                SectionLabel(stringResource(s.title()))
                SettingsGroup(groupItems(s, ui, updateState, versionName, actions), Modifier.padding(horizontal = 16.dp))
            }
            Spacer(Modifier.height(24.dp))
        }
    }
}

// ================================================================ inner

/** Inner display: 260 dp category list with the screen title, and the version footer at the bottom. */
@Composable
private fun CategoryPane(selected: SettingsCategory, showApp: Boolean, showDebug: Boolean, versionName: String?, onSelect: (SettingsCategory) -> Unit) {
    val c = MurmurTheme.colors
    val categories = SettingsCategory.entries.filter {
        (it != SettingsCategory.DEBUG || showDebug) && (it != SettingsCategory.APP || showApp)
    }
    BoxWithConstraints(
        Modifier
            .width(260.dp)
            .fillMaxHeight()
            .drawBehind { drawLine(c.line, Offset(size.width - 0.5.dp.toPx(), 0f), Offset(size.width - 0.5.dp.toPx(), size.height), 1.dp.toPx()) },
    ) {
        val minHeight = maxHeight
        Column(
            Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).heightIn(min = minHeight).padding(end = 1.dp),
            verticalArrangement = Arrangement.SpaceBetween,
        ) {
            Column {
                ScreenHeader(stringResource(R.string.settings_title))
                Column(
                    Modifier.padding(horizontal = 12.dp, vertical = 4.dp).selectableGroup(),
                    verticalArrangement = Arrangement.spacedBy(2.dp),
                ) {
                    categories.forEach { cat ->
                        CategoryRow(cat, cat == selected) { onSelect(cat) }
                    }
                }
            }
            val version = versionName?.let { stringResource(R.string.settings_version, ProductName.NAME, it) }
            Text(
                listOfNotNull(version, stringResource(R.string.settings_on_device)).joinToString("\n"),
                style = MurmurTheme.type.caption, color = c.ink3,
                modifier = Modifier.padding(horizontal = 20.dp, vertical = 16.dp),
            )
        }
    }
}

@Composable
private fun CategoryRow(cat: SettingsCategory, selected: Boolean, onClick: () -> Unit) {
    val c = MurmurTheme.colors
    val shape = RoundedCornerShape(14.dp)
    Row(
        Modifier
            .fillMaxWidth()
            .clip(shape)
            .then(if (selected) Modifier.background(c.surface3) else Modifier)
            .selectable(selected, role = Role.Tab, onClick = onClick)
            .heightIn(min = 52.dp)
            .padding(horizontal = 16.dp, vertical = 10.dp)
            .testTag(SettingsTags.category(cat)),
        horizontalArrangement = Arrangement.spacedBy(14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.size(24.dp), contentAlignment = Alignment.Center) {
            Icon(cat.icon(), null, tint = c.ink2, modifier = Modifier.size(22.dp))
        }
        Text(stringResource(cat.title()), style = MurmurTheme.type.rowTitle, color = c.ink, modifier = Modifier.weight(1f))
    }
}

/** Inner detail pane: the category title, the banner, the category's group, and for Cleanup the prompt preview. */
@Composable
private fun DetailPane(category: SettingsCategory, ui: SettingsUi, updateState: AppUpdateState?, versionName: String?, actions: SettingsActions) {
    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(end = 4.dp, bottom = 24.dp)
            .testTag(SettingsTags.DETAIL),
    ) {
        Text(
            stringResource(category.title()), style = MurmurTheme.type.title, color = MurmurTheme.colors.ink,
            modifier = Modifier.padding(start = 24.dp, end = 24.dp, top = 22.dp, bottom = 6.dp).semantics { heading() },
        )
        if (!ui.serviceConnected) {
            OffBanner(actions, Modifier.padding(start = 16.dp, end = 16.dp, top = 8.dp, bottom = 4.dp))
        }
        SettingsGroup(groupItems(category, ui, updateState, versionName, actions), Modifier.padding(horizontal = 16.dp))
        if (category == SettingsCategory.CLEANUP && ui.prompt != null) {
            Column(Modifier.padding(start = 16.dp, end = 16.dp, top = 20.dp)) {
                Overline(stringResource(R.string.settings_prompt_preview), Modifier.padding(start = 4.dp, end = 4.dp, bottom = 8.dp))
                PromptPreview(ui.prompt.previewText)
            }
        }
    }
}

/** The mono editor box (prototype .editor): surface, 1 dp line2 (2 dp err when [error]), md radius, 14/14/10 padding. */
@Composable
internal fun PromptBox(modifier: Modifier = Modifier, error: Boolean = false, bottomPadding: Dp = 10.dp, content: @Composable () -> Unit) {
    val c = MurmurTheme.colors
    Box(
        modifier
            .fillMaxWidth()
            .clip(MurmurShapes.md)
            .background(c.surface)
            .border(if (error) 2.dp else 1.dp, if (error) c.err else c.line2, MurmurShapes.md)
            .padding(start = 14.dp, end = 14.dp, top = 14.dp, bottom = bottomPadding),
    ) { content() }
}

/** dash-off / settings-off: the red "The bubble is off." banner with the Accessibility settings action. */
@Composable
private fun OffBanner(actions: SettingsActions, modifier: Modifier) {
    Banner(
        BannerKind.Err,
        title = stringResource(R.string.settings_off_title),
        body = stringResource(R.string.settings_off_body, ProductName.NAME),
        actionLabel = stringResource(R.string.settings_off_action),
        onAction = { actions.openSystem(SystemScreen.ACCESSIBILITY) },
        modifier = modifier.testTag(SettingsTags.OFF_BANNER),
    )
}

// ================================================================ groups

private fun SettingsCategory.title(): Int = when (this) {
    SettingsCategory.DICTATION -> R.string.settings_section_dictation
    SettingsCategory.BUBBLE -> R.string.settings_section_bubble
    SettingsCategory.CLEANUP -> R.string.settings_section_cleanup
    SettingsCategory.PRIVACY -> R.string.settings_section_privacy
    SettingsCategory.PERMISSIONS -> R.string.settings_section_permissions
    SettingsCategory.APP -> R.string.settings_section_app
    SettingsCategory.DEBUG -> R.string.settings_section_debug
}

private fun SettingsCategory.icon(): ImageVector = when (this) {
    SettingsCategory.DICTATION -> MurmurIcons.mic
    SettingsCategory.BUBBLE -> MurmurIcons.bubble
    SettingsCategory.CLEANUP -> MurmurIcons.polish
    SettingsCategory.PRIVACY -> MurmurIcons.shield
    SettingsCategory.PERMISSIONS -> MurmurIcons.lock
    SettingsCategory.APP -> MurmurIcons.dl
    SettingsCategory.DEBUG -> MurmurIcons.chip
}

@Composable
private fun groupItems(
    category: SettingsCategory,
    ui: SettingsUi,
    updateState: AppUpdateState?,
    versionName: String?,
    a: SettingsActions,
): List<GroupItem> = when (category) {
    SettingsCategory.DICTATION -> dictationItems(ui.settings, a)
    SettingsCategory.BUBBLE -> bubbleItems(ui.settings, a)
    SettingsCategory.CLEANUP -> cleanupItems(ui, a)
    SettingsCategory.PRIVACY -> privacyItems(ui.settings, a)
    SettingsCategory.PERMISSIONS -> permissionItems(ui, a)
    SettingsCategory.APP -> appItems(updateState ?: AppUpdateState.Idle, versionName, a)
    SettingsCategory.DEBUG -> debugItems(ui.settings, a)
}

@Composable
private fun SRow(key: String, title: String, subtitle: String? = null, icon: ImageVector? = null, trailing: RowTrailing = RowTrailing.None, titleColor: androidx.compose.ui.graphics.Color? = null, enabled: Boolean = true, onClick: (() -> Unit)?) {
    ListRow(
        title, subtitle = subtitle, leadingIcon = icon, trailing = trailing, onClick = onClick, titleColor = titleColor,
        enabled = enabled, modifier = Modifier.testTag(SettingsTags.row(key)),
    )
}

@Composable
internal fun languagesSummary(s: Settings): String {
    val ru = "ru" in s.languages
    val en = "en" in s.languages
    val he = "he" in s.languages
    if (!he) {
        return when {
            ru && en && s.autoDetectLanguage -> stringResource(R.string.settings_languages_both_auto)
            ru && en -> stringResource(R.string.settings_languages_both)
            ru -> stringResource(R.string.settings_languages_ru)
            else -> stringResource(R.string.settings_languages_en)
        }
    }
    if (!ru && !en) return stringResource(R.string.settings_languages_he)
    val names = buildList {
        if (ru) add(stringResource(R.string.settings_language_ru))
        if (en) add(stringResource(R.string.settings_language_en))
        add(stringResource(R.string.settings_language_he))
    }.joinToString(", ")
    return if (s.autoDetectLanguage) stringResource(R.string.settings_languages_list_auto, names) else names
}

@Composable
internal fun feedbackLabel(f: Feedback): String = stringResource(
    when (f) {
        Feedback.VIBRATION -> R.string.settings_feedback_vibration
        Feedback.SOUND -> R.string.settings_feedback_sound
        Feedback.BOTH -> R.string.settings_feedback_both
        Feedback.OFF -> R.string.settings_feedback_off
    },
)

@Composable
internal fun wpmLabel(wpm: Int): String {
    val locale = LocalConfiguration.current.locales[0]
    return stringResource(R.string.settings_typing_value, NumberFormat.getIntegerInstance(locale).format(wpm))
}

@Composable
internal fun visibilityLabel(v: BubbleVisibility): String =
    stringResource(if (v == BubbleVisibility.ALWAYS) R.string.settings_show_always else R.string.settings_show_when_typing)

@Composable
internal fun sizeLabel(s: BubbleSize): String = stringResource(
    when (s) {
        BubbleSize.SMALL -> R.string.settings_size_small
        BubbleSize.STANDARD -> R.string.settings_size_standard
        BubbleSize.LARGE -> R.string.settings_size_large
    },
)

@Composable
internal fun retentionLabel(days: Int): String =
    if (days <= 0) stringResource(R.string.settings_keep_none) else pluralStringResource(R.plurals.settings_days, days, days)

@Composable
internal fun appsLabel(n: Int): String = pluralStringResource(R.plurals.settings_apps, n, n)

@Composable
internal fun toneLabel(t: Tone): String = stringResource(
    when (t) {
        Tone.CASUAL -> R.string.tone_casual
        Tone.NEUTRAL -> R.string.tone_neutral
        Tone.FORMAL -> R.string.tone_formal
        Tone.VERBATIM -> R.string.tone_verbatim
    },
)

@Composable
private fun dictationItems(s: Settings, a: SettingsActions): List<GroupItem> {
    val languages = languagesSummary(s)
    val feedbackSub = stringResource(
        when (s.feedback) {
            Feedback.VIBRATION -> R.string.settings_feedback_sub_vibration
            Feedback.SOUND -> R.string.settings_feedback_sub_sound
            Feedback.BOTH -> R.string.settings_feedback_sub_both
            Feedback.OFF -> R.string.settings_feedback_sub_off
        },
    )
    val feedback = feedbackLabel(s.feedback)
    val wpm = wpmLabel(s.typingWpm)
    return listOf(
        GroupItem { SRow("languages", stringResource(R.string.settings_languages), languages, MurmurIcons.globe, RowTrailing.Chevron(null)) { a.openSheet(SheetKind.LANGUAGES) } },
        GroupItem {
            SRow("hold", stringResource(R.string.settings_hold), stringResource(R.string.settings_hold_sub), MurmurIcons.hand, RowTrailing.Switch(s.holdToTalk)) {
                a.vm.setHoldToTalk(!s.holdToTalk)
            }
        },
        GroupItem { SRow("feedback", stringResource(R.string.settings_feedback), feedbackSub, MurmurIcons.vib, RowTrailing.Chevron(feedback)) { a.openSheet(SheetKind.FEEDBACK) } },
        GroupItem { SRow("typing", stringResource(R.string.settings_typing), stringResource(R.string.settings_typing_sub), MurmurIcons.keyboard, RowTrailing.Chevron(wpm)) { a.openSheet(SheetKind.TYPING) } },
    )
}

@Composable
private fun bubbleItems(s: Settings, a: SettingsActions): List<GroupItem> {
    val visibility = visibilityLabel(s.bubbleVisibility)
    val size = sizeLabel(s.bubbleSize)
    val hidden = if (s.hiddenInPackages.isEmpty()) stringResource(R.string.settings_hidden_none) else appsLabel(s.hiddenInPackages.size)
    return listOf(
        GroupItem { SRow("visibility", stringResource(R.string.settings_show_bubble), null, MurmurIcons.bubble, RowTrailing.Chevron(visibility)) { a.openSheet(SheetKind.VISIBILITY) } },
        GroupItem {
            SRow("tuck", stringResource(R.string.settings_tuck), stringResource(R.string.settings_tuck_sub), MurmurIcons.eyeOff, RowTrailing.Switch(s.tuckIn)) {
                a.vm.setTuckIn(!s.tuckIn)
            }
        },
        GroupItem { SRow("size", stringResource(R.string.settings_size), null, null, RowTrailing.Chevron(size)) { a.openSheet(SheetKind.SIZE) } },
        GroupItem { SRow("hidden", stringResource(R.string.settings_hidden), stringResource(R.string.settings_hidden_sub), null, RowTrailing.Chevron(hidden)) { a.openSheet(SheetKind.HIDDEN_IN) } },
        GroupItem {
            SRow(
                "reset-position", stringResource(R.string.settings_reset_position), stringResource(R.string.settings_reset_position_sub),
                titleColor = MurmurTheme.colors.accentInk,
            ) { a.vm.resetPosition() }
        },
    )
}

@Composable
private fun cleanupItems(ui: SettingsUi, a: SettingsActions): List<GroupItem> {
    val s = ui.settings
    val promptValue = ui.prompt?.let { p ->
        stringResource(
            when {
                !p.editable -> R.string.settings_prompt_built_in
                p.state == PromptEditState.EDITED -> R.string.settings_prompt_edited
                p.state == PromptEditState.MADE_FOR_OLDER_PROMPT -> R.string.settings_prompt_not_applied
                else -> R.string.settings_prompt_default
            },
        )
    }
    val tones = appsLabel(ui.tones.size)
    return listOf(
        GroupItem {
            SRow(
                "prompt", stringResource(R.string.settings_prompt), stringResource(R.string.settings_prompt_sub), MurmurIcons.polish,
                RowTrailing.Chevron(promptValue), enabled = ui.prompt != null,
            ) { a.vm.openPrompt() }
        },
        GroupItem { SRow("tone", stringResource(R.string.settings_tone), stringResource(R.string.settings_tone_sub), MurmurIcons.text, RowTrailing.Chevron(tones)) { a.vm.openTones() } },
        GroupItem {
            SRow("screen", stringResource(R.string.settings_screen), stringResource(R.string.settings_screen_sub), MurmurIcons.search, RowTrailing.Switch(s.useScreen)) {
                a.vm.setUseScreen(!s.useScreen)
            }
        },
    )
}

@Composable
private fun privacyItems(s: Settings, a: SettingsActions): List<GroupItem> {
    val keep = retentionLabel(s.retentionDays)
    val c = MurmurTheme.colors
    return listOf(
        GroupItem { SRow("retention", stringResource(R.string.settings_keep_history), null, MurmurIcons.hist, RowTrailing.Chevron(keep)) { a.openSheet(SheetKind.RETENTION) } },
        GroupItem {
            SRow("recordings", stringResource(R.string.settings_keep_recordings), stringResource(R.string.settings_keep_recordings_sub), MurmurIcons.mic, RowTrailing.Switch(s.keepRecordings)) {
                a.vm.setKeepRecordings(!s.keepRecordings)
            }
        },
        GroupItem {
            SRow("offline", stringResource(R.string.settings_offline), stringResource(R.string.settings_offline_sub), MurmurIcons.shield, RowTrailing.Value(stringResource(R.string.settings_offline_on), c.ok), onClick = null)
        },
        GroupItem(indented = false) {
            SRow("delete-all", stringResource(R.string.settings_delete_all), titleColor = c.err) { a.vm.askDeleteAll() }
        },
    )
}

@Composable
private fun permissionItems(ui: SettingsUi, a: SettingsActions): List<GroupItem> = PermissionRows.of(ui.permissions).map { r ->
    GroupItem {
        val (title, icon) = when (r.kind) {
            PermissionRow.Kind.MIC -> R.string.settings_perm_mic to MurmurIcons.mic
            PermissionRow.Kind.ACCESSIBILITY -> R.string.settings_perm_a11y to MurmurIcons.a11y
            PermissionRow.Kind.NOTIFICATIONS -> R.string.settings_perm_notifications to MurmurIcons.bell
            PermissionRow.Kind.BATTERY -> R.string.settings_perm_battery to MurmurIcons.battO
        }
        val (text, kind, statusIcon) = when (r.status) {
            PermissionStatus.ALLOWED -> Triple(R.string.settings_status_allowed, StatusKind.Ok, MurmurIcons.check)
            PermissionStatus.ON -> Triple(R.string.settings_status_on, StatusKind.Ok, MurmurIcons.check)
            PermissionStatus.RESTRICTED -> Triple(R.string.settings_status_restricted, StatusKind.Warn, null)
            PermissionStatus.OFF -> Triple(R.string.settings_status_off, StatusKind.Err, MurmurIcons.alert)
        }
        SRow("perm-${r.kind.name.lowercase()}", stringResource(title), null, icon, RowTrailing.Status(stringResource(text), kind, statusIcon)) {
            a.openSystem(r.opens)
        }
    }
}

@Composable
private fun appItems(state: AppUpdateState, versionName: String?, a: SettingsActions): List<GroupItem> {
    val c = MurmurTheme.colors
    val busy = state is AppUpdateState.Checking || state is AppUpdateState.Downloading || state is AppUpdateState.Installing
    val subtitle = when (state) {
        AppUpdateState.Idle -> versionName?.let { stringResource(R.string.settings_update_current, it) }
            ?: stringResource(R.string.settings_update_sub)
        AppUpdateState.Checking -> stringResource(R.string.settings_update_checking)
        is AppUpdateState.Downloading -> if (state.progressPercent != null) {
            stringResource(R.string.settings_update_downloading_pct, state.version, state.progressPercent)
        } else {
            stringResource(R.string.settings_update_downloading, state.version)
        }
        is AppUpdateState.Installing -> stringResource(R.string.settings_update_installing, state.version)
        is AppUpdateState.UpToDate -> stringResource(R.string.settings_update_up_to_date, state.currentVersion)
        AppUpdateState.PermissionRequired -> stringResource(R.string.settings_update_permission)
        is AppUpdateState.Failed -> stringResource(
            when (state.reason) {
                AppUpdateState.Failed.Failure.NETWORK -> R.string.settings_update_err_network
                AppUpdateState.Failed.Failure.NO_APK -> R.string.settings_update_err_no_apk
                AppUpdateState.Failed.Failure.VERIFY -> R.string.settings_update_err_verify
                AppUpdateState.Failed.Failure.INSTALL -> R.string.settings_update_err_install
            },
        )
    }
    val trailing = when (state) {
        is AppUpdateState.Downloading -> state.progressPercent?.let { RowTrailing.Value("$it%", c.accentInk) } ?: RowTrailing.None
        AppUpdateState.Checking, is AppUpdateState.Installing -> RowTrailing.None
        is AppUpdateState.UpToDate -> RowTrailing.Status(stringResource(R.string.settings_update_latest_badge), StatusKind.Ok, MurmurIcons.check)
        else -> RowTrailing.Chevron(null)
    }
    return listOf(
        GroupItem {
            SRow(
                key = "app-update",
                title = stringResource(R.string.settings_update),
                subtitle = subtitle,
                icon = MurmurIcons.dl,
                trailing = trailing,
                enabled = !busy,
                onClick = if (busy) null else ({ a.vm.checkForUpdates() }),
            )
        },
    )
}

@Composable
private fun debugItems(s: Settings, a: SettingsActions): List<GroupItem> {
    val scale = stringResource(R.string.settings_debug_timeout_value, formatScale(s.debugTimeoutScale))
    return listOf(
        GroupItem {
            SRow("debug-fakes", stringResource(R.string.settings_debug_fakes), stringResource(R.string.settings_debug_fakes_sub), MurmurIcons.chip, RowTrailing.Switch(s.debugFakeEngines)) {
                a.vm.setDebugFakeEngines(!s.debugFakeEngines)
            }
        },
        GroupItem { SRow("debug-timeout", stringResource(R.string.settings_debug_timeout), stringResource(R.string.settings_debug_timeout_sub), MurmurIcons.clock, RowTrailing.Chevron(scale)) { a.openSheet(SheetKind.TIMEOUT_SCALE) } },
        GroupItem {
            SRow("debug-memory", stringResource(R.string.settings_debug_memory), stringResource(R.string.settings_debug_memory_sub), MurmurIcons.storage, RowTrailing.Switch(s.debugIgnoreMemoryGate)) {
                a.vm.setDebugIgnoreMemoryGate(!s.debugIgnoreMemoryGate)
            }
        },
    )
}

internal fun formatScale(f: Float): String = if (f == f.toInt().toFloat()) f.toInt().toString() else f.toString()

/** Sub-screen top bar (prototype .topbar): 48 dp back (cover) or close (inner) button, `topBar` title, actions. */
@Composable
internal fun SubTopBar(title: String, wide: Boolean, onClose: () -> Unit, actions: @Composable () -> Unit = {}) {
    Row(
        Modifier.fillMaxWidth().heightIn(min = 56.dp).padding(horizontal = 8.dp, vertical = 4.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        app.murmur.ui.design.components.MurmurIconButton(
            if (wide) MurmurIcons.x else MurmurIcons.back,
            stringResource(if (wide) R.string.prompt_close else app.murmur.ui.design.R.string.design_back),
            onClose,
        )
        Text(
            title, style = MurmurTheme.type.topBar, color = MurmurTheme.colors.ink,
            modifier = Modifier.weight(1f).semantics { heading() },
        )
        actions()
    }
}

/** Back closes the sub-screen (the prompt editor asks first when there are unsaved changes). */
@Composable
internal fun SubBackHandler(onBack: () -> Unit) = BackHandler(onBack = onBack)

internal val SheetOptionsPadding: Dp = 8.dp

/**
 * The editor type (13/20 mono) with JetBrains Mono's ligatures off: an editor shows the characters as typed, so
 * `<|im_start|>` must not turn into arrows (contextual alternates) and `->` must stay two characters.
 */
@Composable
internal fun editorStyle(): androidx.compose.ui.text.TextStyle =
    MurmurTheme.type.editor.copy(
        fontFeatureSettings = "liga 0, calt 0, tnum",
        // Greedy breaking like a code editor (and the browser that rendered the frames); the default high-quality
        // strategy wraps mono lines a word early.
        lineBreak = androidx.compose.ui.text.style.LineBreak.Simple,
        // Subpixel advances: hinted mono glyphs round 15.6 px to 16 px at 13 sp x 2, and a 43-column line no longer
        // fits the 336 dp editor that the frames (and the font's 0.6 em advance) give it.
        textMotion = androidx.compose.ui.text.style.TextMotion.Animated,
    )

/**
 * The inner Cleanup pane's read-only excerpt (prototype: `.editor` with max-height 236 px and overflow hidden). As in
 * CSS, a long text runs on into the bottom padding and is cut at the border; a short one keeps the 10 dp padding.
 */
@Composable
private fun PromptPreview(text: String) {
    val c = MurmurTheme.colors
    Layout(
        content = { Text(text, style = editorStyle(), color = c.ink, modifier = Modifier.testTag(SettingsTags.PROMPT_PREVIEW)) },
        modifier = Modifier
            .fillMaxWidth()
            .clip(MurmurShapes.md)
            .background(c.surface)
            .border(1.dp, c.line2, MurmurShapes.md),
    ) { measurables, constraints ->
        val side = 14.dp.roundToPx()
        val top = 14.dp.roundToPx()
        val bottom = 10.dp.roundToPx()
        val w = constraints.maxWidth
        val p = measurables.single().measure(androidx.compose.ui.unit.Constraints(maxWidth = (w - 2 * side).coerceAtLeast(0)))
        val h = minOf(top + p.height + bottom, 236.dp.roundToPx())
        layout(w, h) { p.place(side, top) }
    }
}
