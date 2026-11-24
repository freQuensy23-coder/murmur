package app.murmur.feature.settings

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.painter.Painter
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import app.murmur.core.store.BubbleSize
import app.murmur.core.store.BubbleVisibility
import app.murmur.core.store.Feedback
import app.murmur.core.store.RetentionPolicy
import app.murmur.ui.design.MurmurIcons
import app.murmur.ui.design.MurmurShapes
import app.murmur.ui.design.MurmurTheme
import app.murmur.ui.design.components.AppTile
import app.murmur.ui.design.components.AppTileSize
import app.murmur.ui.design.components.ButtonSize
import app.murmur.ui.design.components.ButtonStyle
import app.murmur.ui.design.components.DISABLED_ALPHA
import app.murmur.ui.design.components.ListRow
import app.murmur.ui.design.components.MurmurButton
import app.murmur.ui.design.components.RowTrailing
import java.text.NumberFormat

/** The sheet of [kind]. Choices apply at once; single-choice sheets close after the choice. */
@Composable
internal fun SettingsSheets(kind: SheetKind, ui: SettingsUi, vm: SettingsViewModel, onDismiss: () -> Unit) {
    val s = ui.settings
    SettingsSheet(onDismiss = onDismiss) {
        when (kind) {
            SheetKind.LANGUAGES -> LanguagesSheet(ui, vm)
            SheetKind.FEEDBACK -> {
                SheetHeader(stringResource(R.string.settings_feedback), null)
                Options {
                    Feedback.entries.forEach { f ->
                        val sub = when (f) {
                            Feedback.SOUND -> stringResource(R.string.settings_feedback_sound_note)
                            Feedback.BOTH -> stringResource(R.string.settings_feedback_both_note)
                            Feedback.VIBRATION -> stringResource(R.string.settings_default)
                            Feedback.OFF -> null
                        }
                        RadioRow(feedbackLabel(f), sub, s.feedback == f, { vm.setFeedback(f); onDismiss() }, Modifier.testTag(SettingsTags.sheetOption("feedback-${f.name.lowercase()}")))
                    }
                }
            }
            SheetKind.TYPING -> TypingSheet(s.typingWpm, onPick = { vm.setTypingWpm(it); onDismiss() })
            SheetKind.VISIBILITY -> {
                SheetHeader(stringResource(R.string.settings_show_bubble), null)
                Options {
                    BubbleVisibility.entries.forEach { v ->
                        val sub = stringResource(if (v == BubbleVisibility.ALWAYS) R.string.settings_show_always_sub else R.string.settings_show_when_typing_sub)
                        RadioRow(visibilityLabel(v), sub, s.bubbleVisibility == v, { vm.setBubbleVisibility(v); onDismiss() }, Modifier.testTag(SettingsTags.sheetOption("visibility-${v.name.lowercase()}")))
                    }
                }
            }
            SheetKind.SIZE -> {
                SheetHeader(stringResource(R.string.settings_size), null)
                Options {
                    BubbleSize.entries.forEach { b ->
                        val sub = if (b == BubbleSize.STANDARD) stringResource(R.string.settings_default) else null
                        RadioRow(sizeLabel(b), sub, s.bubbleSize == b, { vm.setBubbleSize(b); onDismiss() }, Modifier.testTag(SettingsTags.sheetOption("size-${b.name.lowercase()}")))
                    }
                }
            }
            SheetKind.HIDDEN_IN -> HiddenInSheet(ui, vm)
            SheetKind.RETENTION -> {
                SheetHeader(stringResource(R.string.retention_title), stringResource(R.string.retention_body))
                Options {
                    RetentionPolicy.OPTIONS.forEach { d ->
                        val title = if (d == 0) stringResource(R.string.retention_none) else retentionLabel(d)
                        val sub = when (d) {
                            0 -> stringResource(R.string.retention_none_sub)
                            RetentionPolicy.DEFAULT_DAYS -> stringResource(R.string.settings_default)
                            else -> null
                        }
                        RadioRow(title, sub, s.retentionDays == d, { vm.chooseRetention(d); onDismiss() }, Modifier.testTag(SettingsTags.sheetOption("retention-$d")))
                    }
                }
                SheetFootnote(stringResource(R.string.retention_footnote))
            }
            SheetKind.TONE_ADD -> ToneAddSheet(ui, vm, onDismiss)
            SheetKind.TIMEOUT_SCALE -> {
                SheetHeader(stringResource(R.string.settings_debug_timeout), stringResource(R.string.settings_debug_timeout_sub))
                Options {
                    SettingsViewModel.TIMEOUT_SCALES.forEach { f ->
                        RadioRow(
                            stringResource(R.string.settings_debug_timeout_value, formatScale(f)), null, s.debugTimeoutScale == f,
                            { vm.setDebugTimeoutScale(f); onDismiss() },
                        )
                    }
                }
            }
        }
    }
}

/** Option rows of a sheet (prototype: 8/8/0 padding around 60 dp rows). */
@Composable
private fun Options(content: @Composable ColumnScope.() -> Unit) {
    Column(Modifier.padding(start = SheetOptionsPadding, end = SheetOptionsPadding, top = 8.dp).selectableGroup(), content = content)
}

@Composable
private fun LanguagesSheet(ui: SettingsUi, vm: SettingsViewModel) {
    val s = ui.settings
    SheetHeader(stringResource(R.string.settings_languages), stringResource(R.string.settings_languages_sheet_body))
    Column(Modifier.padding(start = SheetOptionsPadding, end = SheetOptionsPadding, top = 8.dp)) {
        SettingsViewModel.LANGUAGES.forEach { code ->
            val on = code in s.languages
            val label = when (code) {
                "ru" -> stringResource(R.string.settings_language_ru)
                "en" -> stringResource(R.string.settings_language_en)
                "he" -> stringResource(R.string.settings_language_he)
                else -> code.uppercase()
            }
            val nativeSub = if (code == "he") "עברית" else null
            CheckRow(
                label, nativeSub, on,
                onCheckedChange = { vm.setLanguage(code, it) },
                // The last language cannot be unchecked: the speech model needs one.
                enabled = !(on && s.languages.size == 1),
                modifier = Modifier.testTag(SettingsTags.sheetOption("lang-$code")),
            )
        }
        ListRow(
            stringResource(R.string.settings_languages_auto), subtitle = stringResource(R.string.settings_languages_auto_sub),
            trailing = RowTrailing.Switch(s.autoDetectLanguage), onClick = { vm.setAutoDetect(!s.autoDetectLanguage) },
            modifier = Modifier.testTag(SettingsTags.sheetOption("lang-auto")),
        )
    }
}

/** Typing speed for "time saved" (DESIGN 6.5: 30/40/50/60/custom), laid out like the retention sheet. */
@Composable
private fun TypingSheet(current: Int, onPick: (Int) -> Unit) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    val presets = SettingsViewModel.WPM_PRESETS
    var custom by rememberSaveable { mutableStateOf(current !in presets) }
    var text by rememberSaveable { mutableStateOf(if (current in presets) "" else current.toString()) }
    val locale = LocalConfiguration.current.locales[0]
    val format = remember(locale) { NumberFormat.getIntegerInstance(locale) }
    SheetHeader(stringResource(R.string.settings_typing_title), stringResource(R.string.settings_typing_body))
    Options {
        presets.forEach { wpm ->
            RadioRow(
                wpmLabel(wpm), if (wpm == 40) stringResource(R.string.settings_default) else null, !custom && current == wpm,
                { onPick(wpm) }, Modifier.testTag(SettingsTags.sheetOption("wpm-$wpm")),
            )
        }
        RadioRow(
            stringResource(R.string.settings_typing_custom), if (current !in presets) wpmLabel(current) else null, custom,
            { custom = true }, Modifier.testTag(SettingsTags.sheetOption("wpm-custom")),
        )
    }
    if (custom) {
        val value = text.toIntOrNull()
        val valid = value != null && value in SettingsViewModel.MIN_WPM..SettingsViewModel.MAX_WPM
        val label = stringResource(R.string.settings_typing_custom_label)
        Row(
            Modifier.fillMaxWidth().padding(start = 24.dp, end = 24.dp, top = 4.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                Modifier
                    .weight(1f)
                    .heightIn(min = 48.dp)
                    .clip(MurmurShapes.full)
                    .background(c.surface)
                    .border(2.dp, c.ink, MurmurShapes.full)
                    .padding(horizontal = 16.dp),
                contentAlignment = Alignment.CenterStart,
            ) {
                BasicTextField(
                    value = text,
                    onValueChange = { v -> text = v.filter { it.isDigit() }.take(3) },
                    singleLine = true,
                    textStyle = t.bodyLg.copy(color = c.ink),
                    cursorBrush = SolidColor(c.accent),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number, imeAction = ImeAction.Done),
                    keyboardActions = KeyboardActions(onDone = { if (valid) onPick(value!!) }),
                    modifier = Modifier.fillMaxWidth().testTag(SettingsTags.sheetOption("wpm-field")).semantics { contentDescription = label },
                    decorationBox = { inner ->
                        if (text.isEmpty()) Text(label, style = t.bodyLg, color = c.ink3)
                        inner()
                    },
                )
            }
            MurmurButton(
                stringResource(R.string.settings_save), { if (valid) onPick(value!!) },
                style = ButtonStyle.Primary, size = ButtonSize.Small, enabled = valid,
                modifier = Modifier.testTag(SettingsTags.sheetOption("wpm-save")),
            )
        }
        Text(
            stringResource(R.string.settings_typing_custom_range, format.format(SettingsViewModel.MIN_WPM), format.format(SettingsViewModel.MAX_WPM)),
            style = t.caption, color = c.ink3, modifier = Modifier.padding(start = 40.dp, end = 24.dp, top = 6.dp),
        )
    }
}

@Composable
private fun ColumnScope.HiddenInSheet(ui: SettingsUi, vm: SettingsViewModel) {
    val own = LocalContext.current.packageName
    val apps = remember(ui.apps, own) { vm.hiddenInCandidates(own, ui) }
    val hidden = ui.settings.hiddenInPackages
    // Apps already hidden but no longer launchable stay listed so that they can be un-hidden.
    val rows = apps + hidden.filter { it !in apps }.sorted()
    SheetHeader(stringResource(R.string.settings_hidden_title), stringResource(R.string.settings_hidden_body))
    LazyColumn(Modifier.weight(1f, fill = false).padding(start = SheetOptionsPadding, end = SheetOptionsPadding, top = 8.dp)) {
        items(rows, key = { it }) { pkg ->
            val label = ui.apps.label(pkg)
            CheckRow(
                label, null, pkg in hidden, onCheckedChange = { vm.setHiddenIn(pkg, it) },
                leading = { AppTile(label, rememberAppIcon(vm, pkg), size = AppTileSize.Small, packageName = pkg) },
                modifier = Modifier.testTag(SettingsTags.sheetOption("hidden-$pkg")),
            )
        }
    }
}

@Composable
private fun ColumnScope.ToneAddSheet(ui: SettingsUi, vm: SettingsViewModel, onDismiss: () -> Unit) {
    val own = LocalContext.current.packageName
    val apps = remember(ui.apps, ui.tones, own) { vm.toneCandidates(own, ui) }
    SheetHeader(stringResource(R.string.tone_add), stringResource(R.string.tone_add_body))
    LazyColumn(Modifier.weight(1f, fill = false).padding(start = SheetOptionsPadding, end = SheetOptionsPadding, top = 8.dp)) {
        items(apps, key = { it }) { pkg ->
            val label = ui.apps.label(pkg)
            ListRow(
                label,
                leading = { AppTile(label, rememberAppIcon(vm, pkg), size = AppTileSize.Large, packageName = pkg) },
                onClick = { vm.addToneApp(pkg); onDismiss() },
                modifier = Modifier.testTag(SettingsTags.sheetOption("tone-add-$pkg")),
            )
        }
    }
}

/** The launcher icon of [pkg], looked up off the main thread; the letter tile shows until it arrives. */
@Composable
internal fun rememberAppIcon(vm: SettingsViewModel, pkg: String): Painter? =
    produceState(vm.cachedIcon(pkg), vm, pkg) { value = vm.appIcon(pkg) }.value

/**
 * Checkbox (no WP-01 component; drawn like the radio): 22 dp, xs radius, 2 dp line2 border; checked = accent fill
 * with an onAccent check. Decorative inside a toggleable row.
 */
@Composable
internal fun Checkbox(checked: Boolean, enabled: Boolean = true) {
    val c = MurmurTheme.colors
    Box(
        Modifier
            .alpha(if (enabled) 1f else DISABLED_ALPHA)
            .size(22.dp)
            .clip(MurmurShapes.xs)
            .background(if (checked) c.accent else c.surface)
            .border(2.dp, if (checked) c.accent else c.line2, MurmurShapes.xs),
        contentAlignment = Alignment.Center,
    ) {
        if (checked) Icon(MurmurIcons.bold("check"), null, tint = c.onAccent, modifier = Modifier.size(16.dp))
    }
}
