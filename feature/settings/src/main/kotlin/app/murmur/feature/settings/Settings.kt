package app.murmur.feature.settings

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel
import app.murmur.core.catalog.CatalogRepository
import app.murmur.core.dictation.PermissionMonitor
import app.murmur.core.model.Outcome
import app.murmur.core.prompt.PromptAssets
import app.murmur.core.prompt.PromptRenderer
import app.murmur.core.store.HistoryRepository
import app.murmur.core.store.SettingsRepository
import app.murmur.ui.design.AppInfoProvider
import kotlinx.coroutines.flow.StateFlow

class SettingsDeps(
    val settings: SettingsRepository,
    val prompts: PromptAssets,
    val renderer: PromptRenderer,
    val catalog: CatalogRepository,
    val history: HistoryRepository,
    val serviceConnected: StateFlow<Boolean>,
    /** "Try it" card: DictationOrchestrator.tryCleanup(sample, unsaved override, unsaved preferences) (DESIGN 6.5). */
    val tryCleanup: suspend (sample: String, systemOverride: String?, preferences: String?) -> Outcome<String>,
    /** Retention shortening: run deletion now with the new period, returns the number of entries deleted (WP-12 Retention.runNow). */
    val retention: suspend (days: Int) -> Int,
    /** "Hidden in" and "Tone per app" pickers: labels, icons, launcher apps. */
    val apps: AppInfoProvider,
    /** Settings > Permissions rows and the "bubble is off" banner. */
    val permissions: PermissionMonitor,
    /** Manual GitHub Releases updater ("Update" button in Settings); null in screenshot/unit tests. */
    val updater: AppUpdater? = null,
)

/**
 * WP-19: Settings (DESIGN 6.5). Lays itself out by the window size class: grouped lists on compact width (cover
 * display), a 260 dp category list with a detail pane on medium width and up (inner display). The prompt editor and
 * Tone per app are screens below Settings: on compact width they cover the whole screen, so the shell hides its
 * bottom bar while [onSubScreenChange] reports true; on the inner display they open in the detail pane (false).
 * Back closes them (the prompt editor asks first when there are unsaved changes).
 */
@Composable
fun SettingsScreen(
    deps: SettingsDeps,
    modifier: Modifier = Modifier,
    onSubScreenChange: (open: Boolean) -> Unit = {},
) {
    val context = LocalContext.current
    val showDebug = remember(context) { SystemScreens.isDebuggable(context) }
    val version = remember(context) { SystemScreens.versionName(context) }
    val vm = viewModel { SettingsViewModel(deps) }
    SettingsRoot(vm, showDebug, version, onSubScreenChange, modifier)
}
