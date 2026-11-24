package app.murmur.feature.onboarding

import android.text.InputType
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.style.TextDirection
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.os.ConfigurationCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import app.murmur.core.context.A11ySnapshot
import app.murmur.core.context.FieldSnapshot
import app.murmur.core.dictation.DictationBubble
import app.murmur.core.dictation.InAppTarget
import app.murmur.core.model.MurmurClock
import app.murmur.core.model.MurmurError
import app.murmur.core.model.ProductName
import app.murmur.core.store.InstallState
import app.murmur.ui.design.Elevation
import app.murmur.ui.design.MurmurIcons
import app.murmur.ui.design.MurmurShapes
import app.murmur.ui.design.MurmurTheme
import app.murmur.ui.design.Sizes
import app.murmur.ui.design.components.BannerKind
import app.murmur.ui.design.components.ButtonStyle
import app.murmur.ui.design.components.Banner
import app.murmur.ui.design.components.InsetDivider
import app.murmur.ui.design.components.LinearProgress
import app.murmur.ui.design.components.MurmurChip
import app.murmur.ui.design.components.MurmurRadio
import app.murmur.ui.design.components.MurmurSwitch
import app.murmur.ui.design.components.Skeleton
import app.murmur.ui.design.components.Spinner
import app.murmur.ui.design.elevation
import java.util.Locale

/** Renders the current step; steps cross-fade into each other (150 ms, DESIGN 3.5 cross-fade). */
@Composable
internal fun OnboardingContent(
    state: OnboardingState,
    ctl: OnboardingController,
    a: ObActions,
    wide: Boolean,
    modifier: Modifier = Modifier,
) {
    val models by ctl.models.collectAsStateWithLifecycle()
    val days by ctl.retentionDays.collectAsStateWithLifecycle()
    Box(modifier) {
        AnimatedContent(
            targetState = state.step,
            transitionSpec = { fadeIn(tween(150)) togetherWith fadeOut(tween(150)) },
            label = "onboarding-step",
        ) { step ->
            when (step) {
                OnboardingStep.WELCOME -> Welcome(wide, days, a)
                OnboardingStep.MIC -> Mic(wide, days, state.micBlocked, a)
                OnboardingStep.ACCESSIBILITY -> Access(wide, a)
                OnboardingStep.RESTRICTED -> Restricted(wide, a)
                OnboardingStep.NOTIFICATIONS -> Notifications(wide, models, a)
                OnboardingStep.BATTERY -> Battery(wide, a)
                OnboardingStep.MODELS -> ModelsStep(wide, models, a)
                OnboardingStep.DOWNLOADING -> Downloading(wide, models, a)
                OnboardingStep.DOWNLOAD_PAUSED -> Paused(wide, models, a)
            }
        }
    }
}

private val name get() = ProductName.NAME

@Composable
private fun locale(): Locale = ConfigurationCompat.getLocales(LocalConfiguration.current)[0] ?: Locale.getDefault()

@Composable
private fun kicker(index: Int): String = stringResource(R.string.ob_kicker, index + 1, stringResource(CHECKLIST[index].title))

// ------------------------------------------------------------------ 0 welcome

@Composable
private fun Welcome(wide: Boolean, days: Int, a: ObActions) {
    val c = MurmurTheme.colors
    ObFrame(wide, dotIndex = -1, testTag = "ob-welcome", foot = {
        PrimaryCta(stringResource(R.string.ob_welcome_cta, name), a.start, "ob-cta")
        Fine(stringResource(R.string.ob_welcome_fine))
    }) {
        Icon(MurmurIcons.bold("mark"), null, tint = c.ink, modifier = Modifier.padding(top = if (wide) 48.dp else 72.dp).size(56.dp))
        ObTitle(
            stringResource(R.string.ob_welcome_title, name), Modifier.padding(top = 28.dp),
            size = if (wide) 34 else 36, line = if (wide) 40 else 42,
        )
        Lead(stringResource(R.string.ob_welcome_body))
        DemoStrip()
        Why(
            listOf(
                MurmurIcons.bubble to stringResource(R.string.ob_welcome_point_bubble),
                MurmurIcons.polish to stringResource(R.string.ob_welcome_point_cleanup),
                MurmurIcons.hist to if (days > 0) pluralStringResource(R.plurals.ob_welcome_point_history, days, days)
                else stringResource(R.string.ob_welcome_point_nohistory),
            ),
            top = 12.dp,
        )
    }
}

/** The live demo strip: a field with mixed-language text and the listening pill docked below its right edge. */
@Composable
private fun DemoStrip() {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    Box(Modifier.padding(top = 28.dp).fillMaxWidth().height(112.dp).semantics { }) {
        Row(
            Modifier
                .fillMaxWidth()
                .heightIn(min = 64.dp)
                .clip(RoundedCornerShape(18.dp))
                .background(c.surface)
                .border(1.dp, c.line, RoundedCornerShape(18.dp))
                .padding(horizontal = 16.dp, vertical = 12.dp),
        ) {
            Text(stringResource(R.string.ob_welcome_demo), style = t.entry, color = c.ink)
            Box(Modifier.padding(start = 1.dp, top = 1.dp).size(width = 2.dp, height = 20.dp).background(c.accent))
        }
        Row(
            Modifier
                .align(Alignment.TopEnd)
                .offset(y = 56.dp)
                .width(Sizes.bubbleListeningWidth)
                .height(Sizes.bubbleDiameter)
                .elevation(Elevation.e2, MurmurShapes.full)
                .clip(MurmurShapes.full)
                .background(c.accent)
                .padding(start = 16.dp, end = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Row(Modifier.height(24.dp), horizontalArrangement = Arrangement.spacedBy(3.dp), verticalAlignment = Alignment.CenterVertically) {
                listOf(8, 18, 24, 14, 20, 10, 6).forEach { h ->
                    Box(Modifier.size(width = 3.dp, height = h.dp).clip(RoundedCornerShape(2.dp)).background(c.onAccent))
                }
            }
            Text("0:04", style = t.monoTimer.copy(fontWeight = FontWeight.Medium), color = c.onAccent, modifier = Modifier.weight(1f))
            Box(Modifier.size(40.dp).clip(CircleShape).background(Color.White.copy(alpha = 0.18f)), contentAlignment = Alignment.Center) {
                Icon(MurmurIcons.stop, null, tint = c.onAccent, modifier = Modifier.size(18.dp))
            }
        }
    }
}

// ------------------------------------------------------------------ 1 microphone

@Composable
private fun Mic(wide: Boolean, days: Int, blocked: Boolean, a: ObActions) {
    ObFrame(wide, dotIndex = 0, testTag = "ob-mic", foot = {
        PrimaryCta(stringResource(if (blocked) R.string.ob_mic_cta_settings else R.string.ob_mic_cta), a.requestMic, "ob-cta")
        Fine(stringResource(if (blocked) R.string.ob_mic_fine_settings else R.string.ob_mic_fine))
    }) {
        Ill(MurmurIcons.mic)
        Kicker(kicker(0))
        ObTitle(stringResource(R.string.ob_mic_title, name))
        Lead(stringResource(R.string.ob_mic_lead))
        Why(
            listOf(
                MurmurIcons.check to stringResource(R.string.ob_mic_point_tap),
                MurmurIcons.check to if (days > 0) pluralStringResource(R.plurals.ob_mic_point_kept, days, days)
                else stringResource(R.string.ob_mic_point_notkept),
            ),
        )
    }
}

// ------------------------------------------------------------------ 2 accessibility (+ restricted)

@Composable
private fun Access(wide: Boolean, a: ObActions) {
    ObFrame(wide, dotIndex = 1, testTag = "ob-access", foot = {
        PrimaryCta(stringResource(R.string.ob_access_cta), a.openAccessibility, "ob-cta")
        Fine(stringResource(R.string.ob_access_privacy, name))
    }) {
        Ill(MurmurIcons.a11y)
        Kicker(kicker(1))
        ObTitle(stringResource(R.string.ob_access_title, name))
        Lead(stringResource(R.string.ob_access_lead, name))
        Why(
            listOf(
                MurmurIcons.bubble to stringResource(R.string.ob_access_point_bubble),
                MurmurIcons.keyboard to stringResource(R.string.ob_access_point_type),
                MurmurIcons.search to stringResource(R.string.ob_access_point_screen),
            ),
        )
        SysBox(stringResource(R.string.ob_next_screen)) {
            Text(htmlString(R.string.ob_access_path, name), style = MurmurTheme.type.body, color = MurmurTheme.colors.ink)
        }
    }
}

@Composable
private fun Restricted(wide: Boolean, a: ObActions) {
    val t = MurmurTheme.type
    val c = MurmurTheme.colors
    ObFrame(wide, dotIndex = 1, testTag = "ob-restricted", foot = {
        PrimaryCta(stringResource(R.string.ob_restricted_cta), a.openAppInfo, "ob-cta")
        SecondaryCta(stringResource(R.string.ob_restricted_done), a.restrictedDone, "ob-secondary")
    }) {
        Ill(MurmurIcons.lock)
        Kicker(kicker(1))
        ObTitle(stringResource(R.string.ob_restricted_title))
        Lead(stringResource(R.string.ob_restricted_lead))
        SysBox(stringResource(R.string.ob_restricted_box)) {
            listOf(
                htmlString(R.string.ob_restricted_step1, name),
                htmlString(R.string.ob_restricted_step2),
                htmlString(R.string.ob_restricted_step3, name),
            ).forEachIndexed { i, s ->
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("${i + 1}", style = t.label, color = c.ink, modifier = Modifier.width(16.dp))
                    Text(s, style = t.body, color = c.ink, modifier = Modifier.weight(1f))
                }
            }
        }
    }
}

// ------------------------------------------------------------------ 3 notifications, 4 battery

@Composable
private fun Notifications(wide: Boolean, models: ModelsUi, a: ObActions) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    ObFrame(wide, dotIndex = 2, testTag = "ob-notif", foot = {
        PrimaryCta(stringResource(R.string.ob_notif_cta), a.requestNotifications, "ob-cta")
        SecondaryCta(stringResource(R.string.ob_notif_skip), a.skip, "ob-secondary")
    }) {
        Ill(MurmurIcons.bell)
        Kicker(kicker(2))
        ObTitle(stringResource(R.string.ob_notif_title))
        Lead(stringResource(R.string.ob_notif_lead))
        SysBox(stringResource(R.string.ob_notif_box)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Box(Modifier.size(32.dp).clip(CircleShape).background(c.ink), contentAlignment = Alignment.Center) {
                    Icon(MurmurIcons.bold("mark"), null, tint = c.bg, modifier = Modifier.size(18.dp))
                }
                Column(Modifier.weight(1f)) {
                    Text(stringResource(R.string.ob_notif_preview_title, name), style = t.label, color = c.ink)
                    // The subtitle shows the models of the recommended pack (the real one names the active models).
                    val line = (models.selected ?: models.options?.firstOrNull())?.let { o ->
                        listOfNotNull(o.asr.displayName, o.llm?.displayName).joinToString(" · ")
                    }
                    if (line != null) Text(line, style = t.caption, color = c.ink3)
                }
                Text(stringResource(R.string.ob_notif_preview_hide), style = t.labelSm, color = c.accentInk)
            }
        }
    }
}

@Composable
private fun Battery(wide: Boolean, a: ObActions) {
    ObFrame(wide, dotIndex = 3, testTag = "ob-battery", foot = {
        PrimaryCta(stringResource(R.string.ob_battery_cta), a.requestBattery, "ob-cta")
        SecondaryCta(stringResource(R.string.ob_battery_skip), a.skip, "ob-secondary")
    }) {
        Ill(MurmurIcons.batt)
        Kicker(kicker(3))
        ObTitle(stringResource(R.string.ob_battery_title, name))
        Lead(stringResource(R.string.ob_battery_lead, name))
        Why(listOf(MurmurIcons.check to stringResource(R.string.ob_battery_point)))
        SysBox(stringResource(R.string.ob_next_screen)) {
            Text(htmlString(R.string.ob_battery_path), style = MurmurTheme.type.body, color = MurmurTheme.colors.ink)
        }
    }
}

// ------------------------------------------------------------------ 5 models

private val ONBOARDING_LANGUAGE_CHIPS = listOf(
    "ru" to "Русский",
    "en" to "English",
    "he" to "עברית",
)

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ModelsStep(wide: Boolean, m: ModelsUi, a: ObActions) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    val loc = locale()
    val selected = m.selected
    ObFrame(wide, dotIndex = 4, testTag = "ob-model", foot = {
        when {
            m.options == null -> PrimaryCta(stringResource(R.string.ob_models_checking), {}, "ob-cta", loading = true)
            selected == null -> PrimaryCta(stringResource(R.string.ob_models_cta, ""), {}, "ob-cta", icon = MurmurIcons.dl, enabled = false)
            else -> PrimaryCta(
                stringResource(R.string.ob_models_cta, stringResource(R.string.ob_size_gb, ByteFormat.packGb(selected.remainingBytes, loc))),
                a.download, "ob-cta", icon = MurmurIcons.dl,
            )
        }
        m.freeBytes?.let { Fine(stringResource(R.string.ob_models_fine, ByteFormat.freeGb(it, loc))) }
    }) {
        Kicker(kicker(4), top = 20.dp)
        ObTitle(stringResource(R.string.ob_models_title))
        Lead(stringResource(R.string.ob_models_lead, name))
        FlowRow(
            Modifier.padding(top = 14.dp).fillMaxWidth().testTag("ob-languages"),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            ONBOARDING_LANGUAGE_CHIPS.forEach { (code, label) ->
                val on = code in m.languages
                MurmurChip(
                    label = label,
                    onClick = { a.setLanguage(code, !on) },
                    selected = on,
                    enabled = !(on && m.languages.size == 1),
                    modifier = Modifier.testTag("ob-lang-$code"),
                )
            }
        }
        Column(
            Modifier
                .padding(top = 12.dp)
                .fillMaxWidth()
                .clip(MurmurShapes.lg)
                .background(c.surface)
                .border(1.dp, c.line, MurmurShapes.lg)
                .selectableGroup()
                .testTag("ob-packs"),
        ) {
            val options = m.options
            if (options == null) {
                repeat(3) { i ->
                    if (i > 0) InsetDivider(start = 16.dp)
                    Row(Modifier.padding(horizontal = 16.dp, vertical = 14.dp), horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                        Skeleton(Modifier.padding(top = 2.dp).size(22.dp), CircleShape)
                        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            Skeleton(Modifier.fillMaxWidth(0.6f).height(16.dp))
                            Skeleton(Modifier.fillMaxWidth(0.85f).height(14.dp))
                        }
                    }
                }
            } else if (options.isEmpty()) {
                Text(stringResource(R.string.ob_models_none), style = t.body, color = c.ink2, modifier = Modifier.padding(16.dp))
            } else {
                options.forEachIndexed { i, o ->
                    if (i > 0) InsetDivider(start = 16.dp)
                    val on = o.pack.id == m.selectedId
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .selectable(on, role = Role.RadioButton) { a.select(o.pack.id) }
                            .padding(horizontal = 16.dp, vertical = 14.dp)
                            .testTag("ob-pack-${o.pack.id}"),
                        horizontalArrangement = Arrangement.spacedBy(14.dp),
                    ) {
                        MurmurRadio(on, null, Modifier.padding(top = 2.dp))
                        Column(Modifier.weight(1f)) {
                            Text(o.pack.title, style = t.rowTitle, color = c.ink)
                            val speechNames = if (o.hebrewAsr != null && o.hebrewAsr.id != o.asr.id) {
                                "${o.asr.displayName} + ${o.hebrewAsr.displayName}"
                            } else o.asr.displayName
                            val sub = o.llm?.let { stringResource(R.string.ob_models_pack_sub, speechNames, it.displayName) } ?: speechNames
                            Text(sub, style = t.rowSub, color = c.ink3, modifier = Modifier.padding(top = 2.dp))
                        }
                        Text(
                            stringResource(R.string.ob_size_gb, ByteFormat.packGb(o.totalBytes, loc)),
                            style = t.body.copy(fontFeatureSettings = "tnum"), color = c.ink3,
                        )
                    }
                }
            }
        }
        Row(
            Modifier
                .fillMaxWidth()
                .toggleable(m.wifiOnly, role = Role.Switch) { a.setWifiOnly(it) }
                .padding(horizontal = 4.dp, vertical = 12.dp)
                .testTag("ob-wifi"),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Text(stringResource(R.string.ob_models_wifi), style = t.rowTitle, color = c.ink, modifier = Modifier.weight(1f))
            MurmurSwitch(m.wifiOnly, null)
        }
    }
}

// ------------------------------------------------------------------ 5a downloading, 5b paused

@Composable
private fun Downloading(wide: Boolean, m: ModelsUi, a: ObActions) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    val loc = locale()
    val s = m.summary
    ObFrame(wide, dotIndex = 4, testTag = "ob-downloading", foot = {
        SecondaryCta(stringResource(R.string.ob_dl_continue), a.continueSetup, "ob-cta", style = ButtonStyle.Secondary)
        SecondaryCta(stringResource(R.string.ob_dl_pause), a.pause, "ob-secondary")
    }) {
        Kicker(kicker(4), top = 20.dp)
        ObTitle(stringResource(R.string.ob_dl_title))
        Lead(stringResource(R.string.ob_dl_lead))
        if (s != null) {
            Column(Modifier.padding(top = 28.dp).testTag("ob-dl-progress")) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.Bottom) {
                    Text(
                        stringResource(R.string.ob_dl_percent, ByteFormat.percent(s.fraction)),
                        style = t.display.copy(fontSize = 40.sp, fontFeatureSettings = "tnum"), color = c.ink,
                        modifier = Modifier.alignByBaseline(),
                    )
                    Text(
                        stringResource(R.string.ob_dl_of_gb, ByteFormat.gb2(s.bytesDone, loc), ByteFormat.gb2(s.bytesTotal, loc)),
                        style = t.mono.copy(fontFeatureSettings = "tnum"), color = c.ink3, modifier = Modifier.alignByBaseline(),
                    )
                }
                LinearProgress(s.fraction, Modifier.padding(top = 12.dp), height = 6.dp)
                Text(speedLine(s, m.wifiOnly, loc), style = t.mono.copy(fontFeatureSettings = "tnum"), color = c.ink3, modifier = Modifier.padding(top = 8.dp))
            }
            Steps(s, m.npu, loc, Modifier.padding(top = 24.dp), paused = false)
        }
    }
}

@Composable
private fun speedLine(s: DownloadSummary, wifiOnly: Boolean, loc: Locale): String {
    val left = s.secondsLeft
    return when {
        s.bytesPerSec > 0 && left != null -> when (val tl = TimeLeft.of(left)) {
            is TimeLeft.Seconds -> stringResource(R.string.ob_dl_speed_s, ByteFormat.mbPerSec(s.bytesPerSec, loc), tl.s.toInt())
            is TimeLeft.Minutes -> stringResource(R.string.ob_dl_speed_min, ByteFormat.mbPerSec(s.bytesPerSec, loc), tl.min.toInt())
        }
        s.rows.any { it.state is InstallState.Queued } && wifiOnly -> stringResource(R.string.ob_dl_waiting_wifi)
        else -> stringResource(R.string.ob_dl_waiting)
    }
}

/** The per-artifact checklist (prototype .steps) and the "Check files and prepare" row. */
@Composable
private fun Steps(s: DownloadSummary, npu: Boolean, loc: Locale, modifier: Modifier, paused: Boolean) {
    // The artifact the job is working on now; the rest wait their turn.
    val current = if (paused) null else s.rows.firstOrNull { it.state is InstallState.Downloading } ?: s.rows.firstOrNull { it.active }
    Column(modifier.testTag("ob-dl-steps")) {
        s.rows.forEach { r ->
            val mark = when {
                r.done || r.state.isPost() -> StepState.DONE
                r == current -> StepState.NOW
                else -> StepState.TODO
            }
            val right = when {
                mark == StepState.DONE || r.bytesDone == 0L -> stringResource(R.string.ob_size_mb, ByteFormat.mb(r.bytesTotal, loc))
                else -> stringResource(R.string.ob_dl_of_mb, ByteFormat.mb(r.bytesDone, loc), ByteFormat.mb(r.bytesTotal, loc))
            }
            StepRow(mark, r.name, right)
        }
        if (!paused) {
            val mark = when (s.preparing) {
                true -> StepState.DONE
                false -> StepState.NOW
                null -> StepState.TODO
            }
            val pct = if (s.preparing == false && s.preparingProgress > 0f) stringResource(R.string.ob_dl_percent, ByteFormat.percent(s.preparingProgress)) else ""
            StepRow(mark, stringResource(if (npu) R.string.ob_dl_prepare_npu else R.string.ob_dl_prepare), pct)
        }
    }
}

private fun InstallState.isPost() = this == InstallState.Verifying || this == InstallState.Extracting || this is InstallState.Preparing

private enum class StepState { DONE, NOW, TODO }

@Composable
private fun StepRow(mark: StepState, label: String, right: String) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    Row(
        Modifier.fillMaxWidth().heightIn(min = 44.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        val circle = Modifier.size(24.dp).clip(CircleShape)
        when (mark) {
            StepState.DONE -> Box(circle.background(c.okSoft), contentAlignment = Alignment.Center) {
                Icon(MurmurIcons.bold("check"), null, tint = c.ok, modifier = Modifier.size(14.dp))
            }
            StepState.NOW -> Box(circle.background(c.accentSoft), contentAlignment = Alignment.Center) {
                Spinner(color = c.accentInk, size = 16.dp)
            }
            StepState.TODO -> Box(Modifier.size(24.dp).border(1.5.dp, c.line2, CircleShape))
        }
        Text(label, style = t.body, color = if (mark == StepState.TODO) c.ink3 else c.ink, modifier = Modifier.weight(1f))
        if (right.isNotEmpty()) Text(right, style = t.mono.copy(fontFeatureSettings = "tnum"), color = c.ink3)
    }
}

@Composable
private fun Paused(wide: Boolean, m: ModelsUi, a: ObActions) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    val loc = locale()
    val s = m.summary
    ObFrame(wide, dotIndex = 4, testTag = "ob-dlerror", foot = {
        PrimaryCta(stringResource(R.string.ob_paused_retry), a.retry, "ob-cta", icon = MurmurIcons.refresh)
        if (s?.speechReady == true && s.cleanupPending != null) {
            SecondaryCta(stringResource(R.string.ob_paused_without_cleanup), a.continueSetup, "ob-secondary")
        }
    }) {
        Kicker(kicker(4), top = 20.dp)
        ObTitle(stringResource(R.string.ob_paused_title))
        if (s != null) {
            val done = ByteFormat.gb2(s.bytesDone, loc)
            val total = ByteFormat.gb2(s.bytesTotal, loc)
            val (kind, title, body) = when (s.stopReason) {
                MurmurError.NoNetwork -> Triple(
                    BannerKind.Err, stringResource(R.string.ob_paused_offline),
                    stringResource(if (m.wifiOnly) R.string.ob_paused_offline_body_wifi else R.string.ob_paused_offline_body, done, total),
                )
                MurmurError.LowStorage -> Triple(BannerKind.Warn, stringResource(R.string.ob_paused_storage), stringResource(R.string.ob_paused_storage_body, done, total))
                MurmurError.DownloadPaused -> Triple(BannerKind.Info, stringResource(R.string.ob_paused_user), stringResource(R.string.ob_paused_user_body, done, total))
                MurmurError.ChecksumMismatch -> Triple(BannerKind.Err, stringResource(R.string.ob_paused_damaged), stringResource(R.string.ob_paused_damaged_body, name))
                else -> Triple(BannerKind.Err, stringResource(R.string.ob_paused_failed), stringResource(R.string.ob_paused_failed_body, done, total))
            }
            Banner(
                kind, title, body, Modifier.padding(top = 20.dp).testTag("ob-dl-banner"),
                icon = if (s.stopReason == MurmurError.NoNetwork) MurmurIcons.wifiOff else if (kind == BannerKind.Info) MurmurIcons.pause else MurmurIcons.alert,
            )
            Steps(s, m.npu, loc, Modifier.padding(top = 20.dp), paused = true)
            val note = when {
                s.speechReady && s.cleanupPending != null -> stringResource(R.string.ob_paused_speech_works, s.cleanupPending)
                !s.speechReady -> m.active?.asr?.displayName?.let { stringResource(R.string.ob_paused_speech_waits, it) }
                else -> null
            }
            if (note != null) Text(note, style = t.body, color = c.ink2, modifier = Modifier.padding(top = 16.dp))
        }
    }
}

// ------------------------------------------------------------------ 6 practice

@Composable
internal fun Ready(
    wide: Boolean,
    bubble: DictationBubble,
    levels: FloatArray,
    clock: MurmurClock,
    result: PracticeResult?,
    a: ObActions,
    chrome: Boolean = true,
) {
    val c = MurmurTheme.colors
    val t = MurmurTheme.type
    val ctx = LocalContext.current
    // A plain state object (not a delegate): the orchestrator reads and writes it from its own thread.
    val field = rememberSaveable(saver = textSaver) { mutableStateOf(TextFieldValue()) }
    val label = remember(ctx) { ctx.applicationInfo.loadLabel(ctx.packageManager).toString() }
    val start = {
        a.startPractice(PracticeTarget(ctx.packageName, label, clock, read = { field.value }, write = { field.value = it }))
    }
    ObFrame(wide, dotIndex = 5, testTag = if (chrome) "ob-ready" else "practice", chrome = chrome, foot = {
        PrimaryCta(stringResource(R.string.ob_ready_cta), a.done, "ob-cta")
        Fine(stringResource(R.string.ob_ready_fine))
    }) {
        // Onboarding: the "All set" kicker over the title. The practice screen keeps the same title position.
        if (chrome) Kicker(stringResource(R.string.ob_ready_kicker), top = 20.dp) else Spacer(Modifier.height(20.dp))
        ObTitle(stringResource(R.string.ob_ready_title))
        Lead(stringResource(R.string.ob_ready_lead))
        val fieldCd = stringResource(R.string.ob_ready_field_cd)
        Box(Modifier.padding(top = 24.dp).fillMaxWidth()) {
            BasicTextField(
                value = field.value,
                onValueChange = { field.value = it },
                textStyle = t.entry.copy(color = c.ink, textDirection = TextDirection.Content),
                cursorBrush = SolidColor(c.accent),
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 120.dp)
                    .clip(RoundedCornerShape(18.dp))
                    .background(c.surface)
                    .border(2.dp, c.ink, RoundedCornerShape(18.dp))
                    .padding(start = 16.dp, end = 56.dp, top = 11.dp, bottom = 11.dp)
                    .semantics { contentDescription = fieldCd }
                    .testTag("practice-field"),
            )
            // The bubble docks on the field's right edge, 84 dp down (prototype ob-ready: right -6, top 84); pills grow left.
            PracticeBubble(
                bubble = bubble, levels = levels, clock = clock,
                onStart = start, onTap = a.bubbleTap, onRetry = start,
                modifier = Modifier.align(Alignment.TopEnd).offset(x = if (bubble == DictationBubble.Idle) 8.dp else 6.dp, y = if (bubble == DictationBubble.Idle) 82.dp else 84.dp),
            )
        }
        // Shown once the bubble is back to its circle (an Inserted pill would sit on it); it keeps clear of the
        // bubble's column (the field's 56 dp end padding) so that long model names wrap instead of running under it.
        if (result != null && bubble == DictationBubble.Idle) {
            val models = result.llm?.let { stringResource(R.string.ob_ready_models, result.asr, it) } ?: result.asr
            val secs = java.text.NumberFormat.getNumberInstance(locale()).apply { maximumFractionDigits = 1; minimumFractionDigits = 1 }.format(result.seconds)
            Row(
                Modifier.padding(top = 8.dp, end = 56.dp).testTag("practice-result"),
                verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                Icon(MurmurIcons.check, null, tint = c.ink3, modifier = Modifier.size(14.dp))
                Text(stringResource(R.string.ob_ready_result, secs, models), style = t.caption, color = c.ink3)
            }
        }
    }
}

/**
 * The practice field as the orchestrator's [InAppTarget]: the snapshot is this app's own package with a multi-line
 * text field and the text before the cursor (so the join rules see it); [insert] replaces the selection and moves the
 * cursor after the text. Nothing is stored and no screenshot is taken (ARCHITECTURE 9.2, ADR-070).
 */
private val textSaver = androidx.compose.runtime.saveable.Saver<androidx.compose.runtime.MutableState<TextFieldValue>, Any>(
    save = { with(TextFieldValue.Saver) { save(it.value) } },
    restore = { v -> mutableStateOf(TextFieldValue.Saver.restore(v) ?: TextFieldValue()) },
)

internal class PracticeTarget(
    pkg: String,
    label: String,
    clock: MurmurClock,
    private val read: () -> TextFieldValue,
    private val write: (TextFieldValue) -> Unit,
) : InAppTarget {
    override val snapshot: A11ySnapshot = run {
        val v = read()
        A11ySnapshot(
            pkg = pkg, label = label, windowTitle = null, windowId = -1, displayId = 0,
            field = FieldSnapshot(
                inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES,
                imeOptions = 0, hint = null,
                beforeCursor = v.text.substring(0, v.selection.min.coerceIn(0, v.text.length)),
                boundsInScreen = null, isPassword = false, isMultiline = true,
            ),
            excludeRects = emptyList(),
            capturedAtMs = clock.elapsedMs(),
        )
    }

    override fun insert(text: String) {
        val v = read()
        val start = v.selection.min.coerceIn(0, v.text.length)
        val end = v.selection.max.coerceIn(0, v.text.length)
        val out = v.text.substring(0, start) + text + v.text.substring(end)
        write(TextFieldValue(out, TextRange(start + text.length)))
    }
}
