package app.murmur.feature.onboarding

import app.murmur.core.catalog.CatalogParser
import app.murmur.core.catalog.DeviceProfile
import app.murmur.core.catalog.DeviceProfileRepository
import app.murmur.core.dictation.DictationBubble
import app.murmur.core.dictation.InAppTarget
import app.murmur.core.dictation.PermissionState
import app.murmur.core.dictation.UserInput
import app.murmur.core.model.ArtifactId
import app.murmur.core.model.ModelId
import app.murmur.core.model.MurmurError
import app.murmur.core.store.CleanupChoice
import app.murmur.core.store.InstallState
import app.murmur.testing.FakeCatalogRepository
import app.murmur.testing.FakeClock
import app.murmur.testing.FakeDeviceProfileRepository
import app.murmur.testing.FakePermissionMonitor
import app.murmur.testing.InMemorySettingsRepository
import app.murmur.testing.contrib.Wp15StagedDownloads
import androidx.lifecycle.SavedStateHandle
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class OnboardingControllerTest {
    private val catalog = CatalogParser.bundled()

    /** A device repository whose probe finishes only when the test says so. */
    private class SlowDevice(private val probed: DeviceProfile) : DeviceProfileRepository {
        val gate = CompletableDeferred<Unit>()
        private val _p = MutableStateFlow<DeviceProfile?>(null)
        override val profile: StateFlow<DeviceProfile?> = _p.asStateFlow()
        override suspend fun ensure(): DeviceProfile { gate.await(); _p.value = probed; return probed }
        override suspend fun reprobe(): DeviceProfile = ensure()
    }

    private val allowed = PermissionState(
        micGranted = true, notificationsGranted = true, a11yEnabled = true, a11yConnected = true,
        a11yEverConnected = true, batteryUnrestricted = true, restrictedLikely = false,
    )

    private inner class World(scope: CoroutineScope, device: DeviceProfileRepository = FakeDeviceProfileRepository(FakeDeviceProfileRepository.EMULATOR)) {
        val settings = InMemorySettingsRepository()
        val catalogRepo = FakeCatalogRepository(catalog)
        val downloads = Wp15StagedDownloads(catalog = { catalog }, scope = scope)
        val permissions = FakePermissionMonitor(allowed)
        val bubble = MutableStateFlow<DictationBubble>(DictationBubble.Idle)
        val clock = FakeClock()
        val practised = mutableListOf<InAppTarget>()
        val inputs = mutableListOf<UserInput>()
        var opened = 0
        val deps = OnboardingDeps(
            settings = settings, catalog = catalogRepo, installed = downloads, serviceConnected = MutableStateFlow(true),
            permissions = permissions, device = device, practice = { practised += it },
            bubble = bubble, levels = MutableStateFlow(FloatArray(7)), bubbleInput = { inputs += it }, clock = clock,
            accessibilitySettingsOpened = { opened++ },
        )
        val handle = SavedStateHandle()
        val ctl = OnboardingController(deps, scope, handle)

        /** A new controller over the same repositories, from what [handle] would hand back after process death. */
        fun afterProcessDeath(scope: CoroutineScope): OnboardingController =
            OnboardingController(deps, scope, SavedStateHandle(handle.keys().associateWith { handle.get<Any?>(it) }))
    }

    private val noTarget = object : InAppTarget {
        override val snapshot get() = error("not read here")
        override fun insert(text: String) {}
    }

    private fun TestScope.dictation(w: World, cleanup: Boolean = true) {
        w.clock.elapsed += 1_000; w.bubble.value = DictationBubble.Listening(w.clock.elapsed, hold = false, warn = false); runCurrent()
        w.clock.elapsed += 2_000; w.bubble.value = DictationBubble.Transcribing; runCurrent()
        w.clock.elapsed += 500; w.bubble.value = DictationBubble.Inserted(noCleanup = !cleanup); runCurrent()
        w.bubble.value = DictationBubble.Idle; runCurrent()
    }

    private fun TestScope.toModels(w: World) {
        runCurrent()
        w.ctl.dispatch(OnboardingEvent.Start)
        assertEquals(OnboardingStep.MODELS, w.ctl.state.value.step)
    }

    @Test
    fun packSizesWaitForTheDeviceProbe() = runTest {
        val device = SlowDevice(FakeDeviceProfileRepository.FOLD7)
        val w = World(backgroundScope, device)
        runCurrent()
        assertNull("no sizes before ensure()", w.ctl.models.value.options)
        device.gate.complete(Unit)
        runCurrent()
        val opts = w.ctl.models.value.options
        assertNotNull(opts)
        opts!!
        assertEquals("recommended", w.ctl.models.value.selectedId)
        assertEquals("recommended", opts.first().pack.id)
        assertTrue(ArtifactId("parakeet-tdt-0.6b-v3@qnn-sm8750-20s") in opts.first { it.pack.recommended }.artifacts)
        assertEquals(181_000_000_000L, w.ctl.models.value.freeBytes)
        assertTrue(w.ctl.models.value.npu)
    }

    @Test
    fun downloadSetsThePackModelsQueuesItAndFinishesOnboardingDirectly() = runTest {
        val w = World(backgroundScope)
        toModels(w)
        w.ctl.select("light")
        runCurrent()
        w.ctl.download()
        runCurrent()
        assertEquals(OnboardingStep.DOWNLOADING, w.ctl.state.value.step)
        val light = w.ctl.models.value.options!!.first { it.pack.id == "light" }
        assertEquals(light.artifacts, w.downloads.requests)
        assertEquals(ModelId("parakeet-tdt-0.6b-v3"), w.settings.settings.value.asrModel)
        assertEquals(CleanupChoice.Model(ModelId("murmur-cleanup-qwen3.5-0.8b")), w.settings.settings.value.cleanup)
        advanceTimeBy(500)
        val mid = w.ctl.models.value.summary!!
        assertEquals(ModelsPhase.ACTIVE, mid.phase)
        assertTrue(mid.bytesDone in 1 until mid.bytesTotal)
        advanceTimeBy(30_000); runCurrent()
        assertEquals(ModelsPhase.DONE, w.ctl.models.value.summary!!.phase)
        assertTrue(w.ctl.state.value.finished)
        assertTrue(w.settings.settings.value.onboardingDone)
    }

    @Test
    fun togglingHebrewInOnboardingHidesLightPackAndQueuesHebrewAsrAndQwen2b() = runTest {
        val w = World(backgroundScope)
        toModels(w)
        w.ctl.select("light")
        runCurrent()
        assertEquals("light", w.ctl.models.value.selectedId)
        w.ctl.setLanguage("he", true)
        runCurrent()
        assertEquals(listOf("ru", "en", "he"), w.ctl.models.value.languages)
        assertEquals(listOf("recommended", "speech-only"), w.ctl.models.value.options!!.map { it.pack.id })
        assertEquals("falls back from hidden light pack", "recommended", w.ctl.models.value.selectedId)
        val rec = w.ctl.models.value.selected!!
        assertEquals(ModelId("zipformer2-66m-streaming-he"), rec.hebrewAsr?.id)
        assertEquals(ModelId("girgur-cleanup-qwen3.5-2b"), rec.llm?.id)
        w.ctl.download()
        runCurrent()
        assertEquals(ModelId("zipformer2-66m-streaming-he"), w.settings.settings.value.hebrewAsrModel)
        assertEquals(CleanupChoice.Model(ModelId("girgur-cleanup-qwen3.5-2b")), w.settings.settings.value.cleanup)
        assertTrue(ArtifactId("zipformer2-66m-streaming-he@onnx-int8") in w.downloads.requests)
        assertTrue(ArtifactId("girgur-cleanup-qwen3.5-2b@q4_0") in w.downloads.requests)
    }

    @Test
    fun acceptingThePreselectedPackSelectsTheFineTunedCleanupModel() = runTest {
        val w = World(backgroundScope)
        toModels(w)
        val selected = w.ctl.models.value.selected!!
        assertEquals("recommended", selected.pack.id)
        assertEquals(ModelId("murmur-cleanup-qwen3.5-2b"), selected.llm?.id)
        w.ctl.download()
        runCurrent()
        assertEquals(CleanupChoice.Model(ModelId("murmur-cleanup-qwen3.5-2b")), w.settings.settings.value.cleanup)
        assertEquals(selected.artifacts, w.downloads.requests)
    }

    @Test
    fun aStoppedDownloadShowsThePausedStepAndTryAgainResumes() = runTest {
        val w = World(backgroundScope)
        toModels(w)
        w.ctl.select("recommended")
        runCurrent()
        val rec = w.ctl.models.value.selected!!
        val llm = rec.artifacts.last()
        w.downloads.stopAt[llm] = MurmurError.NoNetwork
        w.ctl.download()
        advanceTimeBy(30_000); runCurrent()
        assertEquals(OnboardingStep.DOWNLOAD_PAUSED, w.ctl.state.value.step)
        val s = w.ctl.models.value.summary!!
        assertEquals(MurmurError.NoNetwork, s.stopReason)
        assertTrue("speech works without cleanup", s.speechReady)
        assertEquals("Murmur Cleanup (Qwen3.5 2B)", s.cleanupPending)
        w.ctl.retry()
        runCurrent()
        assertEquals(OnboardingStep.DOWNLOADING, w.ctl.state.value.step)
        advanceTimeBy(30_000); runCurrent()
        assertTrue(w.ctl.state.value.finished)
        assertTrue(w.settings.settings.value.onboardingDone)
        assertTrue(w.downloads.states.value[llm] is InstallState.Installed)
    }

    @Test
    fun pauseStopsTheRunningArtifact() = runTest {
        val w = World(backgroundScope)
        toModels(w)
        w.ctl.download()
        advanceTimeBy(200)
        w.ctl.pause()
        advanceTimeBy(100)
        assertEquals(OnboardingStep.DOWNLOAD_PAUSED, w.ctl.state.value.step)
        assertEquals(MurmurError.DownloadPaused, w.ctl.models.value.summary!!.stopReason)
    }

    @Test
    fun permissionsFromTheMonitorDriveTheSteps() = runTest {
        val w = World(backgroundScope)
        w.permissions.set { it.copy(micGranted = false, a11yConnected = false, notificationsGranted = false, batteryUnrestricted = false) }
        runCurrent()
        w.ctl.dispatch(OnboardingEvent.Start)
        assertEquals(OnboardingStep.MIC, w.ctl.state.value.step)
        w.permissions.set { it.copy(micGranted = true) }
        runCurrent()
        assertEquals(OnboardingStep.ACCESSIBILITY, w.ctl.state.value.step)
        w.ctl.dispatch(OnboardingEvent.A11ySettingsOpened)
        w.permissions.set { it.copy(restrictedLikely = true) }
        runCurrent()
        assertEquals(OnboardingStep.RESTRICTED, w.ctl.state.value.step)
        w.permissions.set { it.copy(a11yConnected = true, restrictedLikely = false) }
        runCurrent()
        assertEquals(OnboardingStep.NOTIFICATIONS, w.ctl.state.value.step)
    }

    @Test
    fun practiceReportsTheTimeFromStopToInsertAndTheModels() = runTest {
        val w = World(backgroundScope)
        w.settings.update { it.copy(cleanup = CleanupChoice.Model(ModelId("lfm3-1.7b"))) }
        runCurrent()
        w.ctl.startPractice(object : InAppTarget {
            override val snapshot get() = error("not read here")
            override fun insert(text: String) {}
        })
        assertEquals(1, w.practised.size)
        w.clock.elapsed = 1_000; w.bubble.value = DictationBubble.Listening(1_000, hold = false, warn = false); runCurrent()
        w.clock.elapsed = 4_000; w.bubble.value = DictationBubble.Listening(1_000, hold = false, warn = false, frozenAtMs = 4_000); runCurrent()
        w.clock.elapsed = 4_300; w.bubble.value = DictationBubble.Transcribing; runCurrent()
        w.clock.elapsed = 4_700; w.bubble.value = DictationBubble.Polishing; runCurrent()
        w.clock.elapsed = 5_300; w.bubble.value = DictationBubble.Inserted(noCleanup = false); runCurrent()
        val r = w.ctl.practice.value
        assertNotNull(r)
        r!!
        assertEquals(1.3, r.seconds, 1e-9)
        assertEquals("Parakeet TDT 0.6B v3", r.asr)
        assertEquals("LFM3 1.7B", r.llm)
        w.bubble.value = DictationBubble.Idle; runCurrent()
        w.ctl.startPractice(object : InAppTarget {
            override val snapshot get() = error("not read here")
            override fun insert(text: String) {}
        })
        w.clock.elapsed = 6_000; w.bubble.value = DictationBubble.Listening(6_000, hold = false, warn = false); runCurrent()
        w.clock.elapsed = 7_000; w.bubble.value = DictationBubble.Transcribing; runCurrent()
        w.clock.elapsed = 7_500; w.bubble.value = DictationBubble.Inserted(noCleanup = true); runCurrent()
        assertNull("no cleanup ran: no arrow", w.ctl.practice.value!!.llm)
        assertEquals(0.5, w.ctl.practice.value!!.seconds, 1e-9)
    }

    @Test
    fun doneStoresOnboardingDone() = runTest {
        val w = World(backgroundScope)
        runCurrent()
        w.ctl.dispatch(OnboardingEvent.Start)
        w.ctl.dispatch(OnboardingEvent.Downloads(ModelsPhase.DONE, true))
        runCurrent()
        assertTrue(w.ctl.state.value.finished)
        assertTrue(w.settings.settings.value.onboardingDone)
    }

    @Test
    fun aDictationThisScreenDidNotStartReportsNothing() = runTest {
        val w = World(backgroundScope)
        runCurrent()
        // The overlay inserts into another app: the shared bubble flow shows it, the practice line must not.
        dictation(w)
        assertNull(w.ctl.practice.value)
        // Started here: reported, then the arm is used up.
        w.ctl.startPractice(noTarget)
        dictation(w)
        assertNotNull(w.ctl.practice.value)
        w.ctl.clearPractice()
        dictation(w)
        assertNull("a later dictation elsewhere after the result was cleared", w.ctl.practice.value)
        // Started here but failed: disarmed, so the next dictation elsewhere reports nothing.
        w.ctl.startPractice(noTarget)
        w.bubble.value = DictationBubble.Listening(w.clock.elapsed, hold = false, warn = false); runCurrent()
        w.bubble.value = DictationBubble.Error(app.murmur.core.dictation.BubbleError.NO_SPEECH, app.murmur.core.dictation.BubbleAction.RETRY); runCurrent()
        w.bubble.value = DictationBubble.Idle; runCurrent()
        dictation(w)
        assertNull(w.ctl.practice.value)
        // Started here, then the screen was left mid-dictation (clearPractice): its Inserted reports nothing.
        w.ctl.startPractice(noTarget)
        w.bubble.value = DictationBubble.Listening(w.clock.elapsed, hold = false, warn = false); runCurrent()
        w.ctl.clearPractice()
        w.bubble.value = DictationBubble.Inserted(noCleanup = false); runCurrent()
        assertNull(w.ctl.practice.value)
    }

    @Test
    fun theProgressSurvivesProcessDeathOnASystemScreen() = runTest {
        val w = World(backgroundScope)
        w.permissions.set { it.copy(a11yConnected = false, a11yEnabled = false, a11yEverConnected = false) }
        runCurrent()
        w.ctl.dispatch(OnboardingEvent.Start)
        assertEquals(OnboardingStep.ACCESSIBILITY, w.ctl.state.value.step)
        // "Open Accessibility settings"; Android kills the process while the user is there; the toggle was refused.
        w.ctl.dispatch(OnboardingEvent.A11ySettingsOpened)
        w.permissions.set { it.copy(restrictedLikely = true) }
        val restored = w.afterProcessDeath(backgroundScope)
        runCurrent()
        assertEquals("not back at Welcome: the restricted screen", OnboardingStep.RESTRICTED, restored.state.value.step)
    }

    @Test
    fun skippedStepsAndMicDenialsSurviveProcessDeath() = runTest {
        val w = World(backgroundScope)
        w.permissions.set { it.copy(micGranted = false, notificationsGranted = false, batteryUnrestricted = false) }
        runCurrent()
        w.ctl.dispatch(OnboardingEvent.Start)
        w.ctl.dispatch(OnboardingEvent.MicResult(granted = false, canAskAgain = true))
        w.ctl.dispatch(OnboardingEvent.MicResult(granted = false, canAskAgain = true))
        assertTrue(w.ctl.state.value.micBlocked)
        // The user opens App info to allow the microphone; the process dies there.
        val atMic = w.afterProcessDeath(backgroundScope)
        runCurrent()
        assertEquals(OnboardingStep.MIC, atMic.state.value.step)
        assertTrue("the button still opens App info", atMic.state.value.micBlocked)
        assertEquals(2, atMic.state.value.micDenials)
        // Further on: notifications skipped, then the battery screen of the system.
        w.permissions.set { it.copy(micGranted = true) }
        runCurrent()
        w.ctl.dispatch(OnboardingEvent.Skip)
        assertEquals(OnboardingStep.BATTERY, w.ctl.state.value.step)
        w.ctl.select("light")
        val atBattery = w.afterProcessDeath(backgroundScope)
        runCurrent()
        assertEquals(OnboardingStep.BATTERY, atBattery.state.value.step)
        assertEquals(setOf(OnboardingStep.NOTIFICATIONS), atBattery.state.value.skipped)
        assertEquals("the pack choice too", "light", atBattery.models.value.selectedId)
    }
}
