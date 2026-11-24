package app.murmur.shell

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.getOrNull
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.UiDevice
import app.murmur.MainActivity
import app.murmur.MurmurApp
import app.murmur.shell.Device.hasTag
import app.murmur.shell.Device.sh
import app.murmur.shell.Device.waitForTag
import kotlinx.coroutines.launch
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The first run on the emulator with fakes (WP-15 acceptance), `-e murmurFake fresh`: onboarding not done, nothing
 * installed, staged downloads, fake engines and capture. Permissions are real: the runner script revokes them before
 * the run (a revoke kills the app), the test grants each one the way the brief says (pm grant, settings put,
 * deviceidle whitelist) while the system screen is in front, comes back, and the step advances on its own.
 * Screenshots of every step go to the app's external files dir (pulled to .cache/spikes/wp-15/).
 */
@RunWith(AndroidJUnit4::class)
class OnboardingFlowTest {
    @get:Rule
    val compose = createEmptyComposeRule()
    private var scenario: ActivityScenario<MainActivity>? = null
    private val ui: UiDevice get() = Device.ui
    private val container get() = (Device.context.applicationContext as MurmurApp).container!!

    @After
    fun tearDown() {
        scenario?.close()
    }

    private fun start() {
        assumeTrue("run with -e murmurFake fresh", container.fakeContainer)
        scenario = ActivityScenario.launch(MainActivity::class.java)
        compose.waitForTag("ob-welcome")
        Thread.sleep(1_500) // first frame after a cold start (fonts, first composition) before the screenshot
    }

    /**
     * Leaves the system screen (dialog or Settings) and waits until the app is in front again. A permission dialog
     * closes by itself once the permission is granted from the shell, so give it time before pressing Back (a Back
     * that reaches the app would leave setup).
     */
    private fun backToApp() {
        repeat(4) {
            val end = android.os.SystemClock.uptimeMillis() + 2_000
            while (android.os.SystemClock.uptimeMillis() < end) {
                if (ui.currentPackageName == Device.pkg) return
                Thread.sleep(100)
            }
            ui.pressBack()
            ui.waitForIdle()
        }
        Device.waitFor(what = "the app in front") { ui.currentPackageName == Device.pkg }
    }

    private fun cta() = compose.onNodeWithTag("ob-cta").performClick()

    @Test
    fun walkThroughEveryStepWithFakes() {
        start()
        Device.shot("ob-welcome")
        cta()

        // 1 Microphone: the runtime dialog opens; grant with pm grant, leave the dialog, the step advances.
        compose.waitForTag("ob-mic")
        Device.shot("ob-mic")
        cta()
        Device.waitFor(what = "permission dialog") { ui.currentPackageName != Device.pkg }
        sh("pm grant ${Device.pkg} android.permission.RECORD_AUDIO")
        backToApp()
        compose.waitForTag("ob-access")

        // 2 Accessibility: the settings screen opens; enable with settings put; on return the step advances.
        Device.shot("ob-access")
        cta()
        Device.waitFor(what = "accessibility settings") { ui.currentPackageName != Device.pkg }
        sh("settings put secure enabled_accessibility_services ${Device.SERVICE}")
        sh("settings put secure accessibility_enabled 1")
        Device.waitFor(what = "service connected") { container.serviceConnected.value }
        backToApp()
        compose.waitForTag("ob-notif")

        // 3 Notifications.
        Device.shot("ob-notif")
        cta()
        Device.waitFor(what = "notification dialog") { ui.currentPackageName != Device.pkg }
        sh("pm grant ${Device.pkg} android.permission.POST_NOTIFICATIONS")
        backToApp()
        compose.waitForTag("ob-battery")

        // 4 Background use: the system asks; exempt with deviceidle; on return the step advances.
        Device.shot("ob-battery")
        cta()
        Device.waitFor(what = "battery dialog") { ui.currentPackageName != Device.pkg }
        sh("dumpsys deviceidle whitelist +${Device.pkg}")
        backToApp()
        compose.waitForTag("ob-model")

        // 5 Models: sizes appear after the device probe; completing the download finishes onboarding directly.
        compose.waitForTag("ob-pack-recommended")
        Device.shot("ob-model")
        cta()
        compose.waitForTag("ob-dl-progress")
        Thread.sleep(2_500)
        Device.shot("ob-downloading")
        compose.waitForTag("shell", timeoutMs = 60_000)
        assertTrue(container.settings.settings.value.onboardingDone)
    }

    /**
     * ob-restricted (ADR-054): needs the app installed from a file session (packageSource 3) and the service off; the
     * runner script installs it with `pm install-create --package-source 3`. Coming back from the settings screen
     * without the service connected shows the recovery screen; "I've done this" returns to the accessibility step.
     */
    @Test
    fun restrictedSettingsBranchForAFileInstall() {
        start()
        val source = Device.context.packageManager.getInstallSourceInfo(Device.pkg).packageSource
        assumeTrue("installed from a file session (packageSource 3), was $source", source == 3)
        cta()
        compose.waitForTag("ob-mic")
        sh("pm grant ${Device.pkg} android.permission.RECORD_AUDIO")
        // Granted outside the dialog: the next resume refresh advances (simulate a return with a monitor refresh).
        container.permissions.refresh()
        compose.waitForTag("ob-access")
        cta()
        Device.waitFor(what = "accessibility settings") { ui.currentPackageName != Device.pkg }
        backToApp()
        compose.waitForTag("ob-restricted")
        Device.shot("ob-restricted")
        compose.onNodeWithTag("ob-secondary").performClick()
        compose.waitForTag("ob-access")
        assertTrue(!compose.hasTag("ob-restricted"))
    }

    /**
     * ob-dlerror: the cleanup model stops half way with "no internet" (staged fake download); the paused screen offers
     * Try again and, because speech already works, Continue without cleanup. Try again resumes and ends in the practice.
     */
    @Test
    fun aDownloadThatStopsShowsThePausedScreenAndResumes() {
        sh("pm grant ${Device.pkg} android.permission.RECORD_AUDIO")
        sh("pm grant ${Device.pkg} android.permission.POST_NOTIFICATIONS")
        sh("dumpsys deviceidle whitelist +${Device.pkg}")
        sh("settings put secure enabled_accessibility_services ${Device.SERVICE}")
        sh("settings put secure accessibility_enabled 1")
        start()
        Device.waitFor(what = "service connected") { container.serviceConnected.value }
        container.permissions.refresh()
        val downloads = container.installed as app.murmur.testing.contrib.Wp15StagedDownloads
        cta()
        compose.waitForTag("ob-pack-recommended")
        val llm = container.catalog.catalog.value.models.first { it.id.value == "lfm3-1.7b" }.artifacts.map { it.id }
        llm.forEach { downloads.stopAt[it] = app.murmur.core.model.MurmurError.NoNetwork }
        cta()
        compose.waitForTag("ob-dlerror", timeoutMs = 60_000)
        compose.onNodeWithTag("ob-dl-banner").assertIsDisplayed()
        compose.onNodeWithTag("ob-secondary").assertIsDisplayed()
        Device.shot("ob-dlerror")
        cta()
        compose.waitForTag("ob-downloading")
        compose.waitForTag("shell", timeoutMs = 60_000)
    }
}
