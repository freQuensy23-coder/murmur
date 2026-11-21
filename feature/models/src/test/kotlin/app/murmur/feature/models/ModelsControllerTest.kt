package app.murmur.feature.models

import app.murmur.core.catalog.CustomModelRequest
import app.murmur.core.catalog.CustomModelResolver
import app.murmur.core.catalog.ModelEntry
import app.murmur.core.catalog.Origin
import app.murmur.core.model.ArtifactId
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.ModelId
import app.murmur.core.model.ModelKind
import app.murmur.core.model.MurmurError
import app.murmur.core.model.Outcome
import app.murmur.core.model.UnitPreference
import app.murmur.core.store.CleanupChoice
import app.murmur.core.store.InstallState
import app.murmur.core.store.Settings
import app.murmur.feature.models.HubFixtures.catalog
import app.murmur.feature.models.HubFixtures.cpuArtifact
import app.murmur.testing.FakeBenchmarkRunner
import app.murmur.testing.FakeCatalogRepository
import app.murmur.testing.FakeCustomModelResolver
import app.murmur.testing.FakeDeviceProfileRepository
import app.murmur.testing.FakeInstalledModelsRepository
import app.murmur.testing.FakeUnitHealth
import app.murmur.testing.InMemoryBenchmarkRepository
import app.murmur.testing.InMemorySettingsRepository
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class ModelsControllerTest {

    /** Every artifact missing except [installed] (the :testing fake treats unlisted artifacts as installed). */
    private class World(
        installed: Set<ArtifactId> = emptySet(),
        settings: Settings = Settings(),
        deviceKnown: Boolean = true,
        /** Completed by the test: the probe answers only then (null = answers at once). */
        val probe: CompletableDeferred<Unit>? = null,
    ) {
        val catalog = FakeCatalogRepository()
        val installed = FakeInstalledModelsRepository(HubFixtures.catalog.models.flatMap { m -> m.artifacts.map { it.id } }.filterNot { it in installed }.toMutableSet())
        val benchmarks = InMemoryBenchmarkRepository()
        val runner = FakeBenchmarkRunner(stepMs = 100, repo = benchmarks)
        val settings = InMemorySettingsRepository(settings)
        val custom = FakeCustomModelResolver()
        val profiles = FakeDeviceProfileRepository(FakeDeviceProfileRepository.EMULATOR)
        val device = MutableStateFlow(if (deviceKnown) FakeDeviceProfileRepository.EMULATOR else null)
        val health = FakeUnitHealth()
        var ensured = 0
        val deps = ModelsDeps(
            catalog, this.installed, benchmarks, runner, this.settings, custom, device, health,
            ensureDevice = { ensured++; probe?.await(); profiles.ensure().also { device.value = it } },
            deviceName = "Test phone",
            now = { 0L },
        )
    }

    private fun TestScope.controller(w: World): ModelsController = ModelsController(w.deps, backgroundScope).also { advanceUntilIdle() }

    private val asr = HubFixtures.npuAndCpuAsr
    private val llm = HubFixtures.defaultLlm
    private val other = HubFixtures.otherLlm

    @Test
    fun pinningCpuPersistsUnderTheModelId() = runTest(UnconfinedTestDispatcher()) {
        val w = World(setOf(cpuArtifact(asr).id))
        val c = controller(w)
        c.pinUnit(asr.id, ComputeUnit.CPU)
        advanceUntilIdle()
        assertEquals(mapOf(asr.id.value to UnitPreference.Pinned(ComputeUnit.CPU)), w.settings.settings.value.unitPreference)
        c.pinUnit(asr.id, null)
        advanceUntilIdle()
        assertTrue("Auto removes the key", w.settings.settings.value.unitPreference.isEmpty())
    }

    @Test
    fun deletingTheInUseModelAsksAndSwitchesToTheNextDownloaded() = runTest(UnconfinedTestDispatcher()) {
        val w = World((llm.artifacts + other.artifacts).map { it.id }.toSet(), Settings(cleanup = CleanupChoice.Model(llm.id)))
        val c = controller(w)
        c.requestDelete(llm.id, Surface.DETAIL)
        val ask = c.delete.value!!
        assertTrue(ask.inUse)
        assertEquals(Surface.DETAIL, ask.origin)
        assertTrue("nothing deleted before the confirmation", llm.artifacts.all { w.installed.isInstalled(it.id) })
        c.confirmDelete()
        advanceUntilIdle()
        assertTrue(llm.artifacts.none { w.installed.isInstalled(it.id) })
        assertEquals(CleanupChoice.Model(other.id), w.settings.settings.value.cleanup)
        assertNull(c.delete.value)
    }

    @Test
    fun deletingTheLastCleanupModelInUseTurnsCleanupOff() = runTest(UnconfinedTestDispatcher()) {
        val w = World(llm.artifacts.map { it.id }.toSet(), Settings(cleanup = CleanupChoice.Model(llm.id)))
        val c = controller(w)
        c.requestDelete(llm.id, Surface.LIST)
        c.confirmDelete()
        advanceUntilIdle()
        assertEquals(CleanupChoice.None, w.settings.settings.value.cleanup)
    }

    @Test
    fun deletingASpeechModelInUseSwitchesToTheNextDownloadedSpeechModel() = runTest(UnconfinedTestDispatcher()) {
        val next = catalog.models.first { it.kind == ModelKind.ASR && it.id != asr.id && it.selectableAsAsr && it.artifacts.any { a -> a.units == listOf(ComputeUnit.CPU) } }
        val w = World(setOf(cpuArtifact(asr).id, cpuArtifact(next).id))
        val c = controller(w)
        assertEquals(RowState.InUse, HubLogic.row(asr, c.input.value!!).state)
        c.requestDelete(asr.id, Surface.DETAIL)
        c.confirmDelete()
        advanceUntilIdle()
        assertEquals(next.id, w.settings.settings.value.asrModel)
        assertEquals(RowState.InUse, HubLogic.row(next, c.input.value!!).state)
    }

    @Test
    fun aModelNotInUseIsDeletedWithoutAsking() = runTest(UnconfinedTestDispatcher()) {
        val w = World((llm.artifacts + other.artifacts).map { it.id }.toSet())
        val c = controller(w)
        c.requestDelete(other.id, Surface.DETAIL)
        advanceUntilIdle()
        assertNull(c.delete.value)
        assertTrue(other.artifacts.none { w.installed.isInstalled(it.id) })
        assertEquals(CleanupChoice.Default, w.settings.settings.value.cleanup)
    }

    @Test
    fun noneSelectionSetsCleanupNoneAndSelectingAModelSetsIt() = runTest(UnconfinedTestDispatcher()) {
        val w = World((llm.artifacts + other.artifacts).map { it.id }.toSet())
        val c = controller(w)
        c.selectNone()
        advanceUntilIdle()
        assertEquals(CleanupChoice.None, w.settings.settings.value.cleanup)
        assertTrue(HubLogic.list(HubTab.CLEANUP, c.input.value!!).noneSelected)
        c.select(other.id)
        advanceUntilIdle()
        assertEquals(CleanupChoice.Model(other.id), w.settings.settings.value.cleanup)
    }

    @Test
    fun aPreviewOnlyModelCannotBeSelected() = runTest(UnconfinedTestDispatcher()) {
        val e = HubFixtures.previewOnly
        val w = World(e.artifacts.map { it.id }.toSet() + cpuArtifact(asr).id)
        val c = controller(w)
        c.select(e.id)
        advanceUntilIdle()
        assertNull(w.settings.settings.value.asrModel)
        // An uninstalled model is not selectable either.
        c.select(other.id)
        advanceUntilIdle()
        assertEquals(CleanupChoice.Default, w.settings.settings.value.cleanup)
    }

    @Test
    fun downloadQueuesTheModelWithTheVadAndTheOcrPair() = runTest(UnconfinedTestDispatcher()) {
        val w = World()
        val c = controller(w)
        c.download(asr.id)
        advanceUntilIdle()
        assertTrue(w.installed.isInstalled(cpuArtifact(asr).id))
        assertFalse("the blocked NPU build is not fetched on the emulator", w.installed.isInstalled(HubFixtures.npuArtifact(asr).id))
        for (m in listOf(catalog.defaults.vad, catalog.defaults.ocrDet, catalog.defaults.ocrRec)) {
            assertTrue(m.value, w.installed.isInstalled(catalog.model(m)!!.artifacts.first().id))
        }
        assertEquals(RowState.InUse, HubLogic.row(asr, c.input.value!!).state)
    }

    @Test
    fun theDeviceProfileIsEnsuredWhenNobodyProbedYet() = runTest(UnconfinedTestDispatcher()) {
        val w = World(deviceKnown = false)
        val c = controller(w)
        assertEquals(1, w.ensured)
        assertNotNull(c.input.value!!.device)
    }

    @Test
    fun anUnknownDeviceRefusesDownloadsAndDeletesUntilTheProfileIsKnown() = runTest(UnconfinedTestDispatcher()) {
        // Review F2: first visit before anything probed; the probe is slow.
        val w = World(setOf(cpuArtifact(asr).id), deviceKnown = false, probe = CompletableDeferred())
        val c = controller(w)
        assertEquals(1, w.ensured)
        assertNull(c.input.value!!.device)
        assertEquals(RowState.Loading, HubLogic.row(asr, c.input.value!!).state)
        val npu = HubFixtures.npuArtifact(asr)
        c.download(asr.id)
        c.download(asr.id, npu)
        advanceUntilIdle()
        assertFalse("no NPU build for another SoC queued blind", w.installed.isInstalled(npu.id))
        assertTrue(listOf(catalog.defaults.vad, catalog.defaults.ocrDet).none { w.installed.isInstalled(catalog.model(it)!!.artifacts.first().id) })
        c.requestDelete(asr.id, Surface.DETAIL)
        advanceUntilIdle()
        assertNull(c.delete.value)
        assertTrue("the model in use is not deleted without asking", w.installed.isInstalled(cpuArtifact(asr).id))
        // The profile arrives: the row is decided (the speech model of record, in use) and delete asks first.
        w.probe!!.complete(Unit)
        advanceUntilIdle()
        assertEquals(RowState.InUse, HubLogic.row(asr, c.input.value!!).state)
        c.requestDelete(asr.id, Surface.DETAIL)
        assertTrue(c.delete.value!!.inUse)
    }

    @Test
    fun aRunStoppedElsewhereClosesTheSheet() = runTest(UnconfinedTestDispatcher()) {
        // Review F10: another surface's stop() ends this run's flow without Finished or Failed (ADR-080).
        val w = World(setOf(cpuArtifact(asr).id))
        val c = controller(w)
        c.runBenchmark(asr.id, Surface.DETAIL)
        testScheduler.advanceTimeBy(150)
        assertEquals(BenchUi.Phase.RUNNING, c.bench.value!!.phase)
        w.runner.stop()
        testScheduler.advanceTimeBy(60_000)
        testScheduler.runCurrent()
        assertNull("no sheet left with Stop forever", c.bench.value)
        assertTrue(w.benchmarks.latest(cpuArtifact(asr).id).first().isEmpty())
    }

    /** Resolves through [w]'s fake once [gate] opens (the first request only when [firstOnly]). */
    private class SlowResolver(val w: World, val gate: CompletableDeferred<Unit>, val firstOnly: Boolean) : CustomModelResolver {
        val requests = mutableListOf<CustomModelRequest>()
        override suspend fun resolve(request: CustomModelRequest): Outcome<ModelEntry> {
            requests += request
            if (!firstOnly || requests.size == 1) gate.await()
            return w.custom.resolve(request)
        }
    }

    private fun TestScope.controllerWith(w: World, resolver: CustomModelResolver) = ModelsController(
        ModelsDeps(w.catalog, w.installed, w.benchmarks, w.runner, w.settings, resolver, w.device, w.health, ensureDevice = { FakeDeviceProfileRepository.EMULATOR }),
        backgroundScope,
    ).also { advanceUntilIdle() }

    @Test
    fun aResolveAnswersOnlyTheFormItWasStartedFor() = runTest(UnconfinedTestDispatcher()) {
        // Review F12: the user edits the repo while "Checking…" is shown.
        val w = World()
        val slow = SlowResolver(w, CompletableDeferred(), firstOnly = true)
        val c = controllerWith(w, slow)
        c.openAdd(ModelKind.LLM)
        c.editAdd { it.copy(repo = "first/Old-GGUF") }
        c.resolveAdd()
        assertTrue(c.add.value!!.busy)
        c.editAdd { it.copy(repo = "second/New-GGUF") }
        assertFalse("an edit ends the old check", c.add.value!!.busy)
        slow.gate.complete(Unit)
        advanceUntilIdle()
        assertNull("the old answer is dropped", c.add.value!!.preview)
        assertEquals("second/New-GGUF", c.add.value!!.repo)
        c.saveAdd()
        advanceUntilIdle()
        assertTrue("nothing to save", w.catalog.catalog.value.models.none { it.origin == Origin.CUSTOM })
        // Checking the new repo previews the new repo.
        c.resolveAdd()
        advanceUntilIdle()
        assertEquals("second/New-GGUF", slow.requests.last().repo)
        assertEquals("custom-new-gguf", c.add.value!!.preview!!.id.value)

        // Closed and reopened while a check runs: the old answer does not fill the new form either.
        val again = SlowResolver(w, CompletableDeferred(), firstOnly = false)
        val c2 = controllerWith(w, again)
        c2.openAdd(ModelKind.LLM)
        c2.editAdd { it.copy(repo = "owner/Model-GGUF") }
        c2.resolveAdd()
        c2.closeAdd()
        c2.openAdd(ModelKind.LLM)
        again.gate.complete(Unit)
        advanceUntilIdle()
        assertNull(c2.add.value!!.preview)
        assertFalse(c2.add.value!!.busy)
    }

    @Test
    fun benchmarkRunsThroughTheRunnerAndShowsTheResult() = runTest(UnconfinedTestDispatcher()) {
        val w = World(setOf(cpuArtifact(asr).id))
        val c = controller(w)
        c.runBenchmark(asr.id, Surface.DETAIL)
        val running = c.bench.value!!
        assertEquals(BenchUi.Phase.RUNNING, running.phase)
        assertEquals(Surface.DETAIL, running.origin)
        // The runner's steps take virtual time in the background scope: advanceUntilIdle does not run those.
        testScheduler.advanceTimeBy(60_000)
        testScheduler.runCurrent()
        val done = c.bench.value!!
        assertEquals(BenchUi.Phase.RESULT, done.phase)
        assertEquals(listOf(ComputeUnit.CPU), done.records.map { it.unit })
        assertTrue(done.steps.all { it.status == StepStatus.DONE })
        // The record reaches the detail through the repository.
        assertNotNull(HubLogic.detail(asr, c.input.value!!).bench.record)
        c.closeBenchmark()
        assertNull(c.bench.value)
    }

    @Test
    fun stopCancelsARunningBenchmark() = runTest(UnconfinedTestDispatcher()) {
        val w = World(setOf(cpuArtifact(asr).id))
        val c = controller(w)
        c.runBenchmark(asr.id, Surface.DETAIL)
        testScheduler.advanceTimeBy(150)
        assertEquals(BenchUi.Phase.RUNNING, c.bench.value!!.phase)
        c.stopBenchmark()
        assertTrue(w.runner.stopped)
        assertNull(c.bench.value)
        advanceUntilIdle()
        assertNull(c.bench.value)
        assertTrue(w.benchmarks.latest(cpuArtifact(asr).id).first().isEmpty())
    }

    @Test
    fun nothingInstalledMeansNoBenchmark() = runTest(UnconfinedTestDispatcher()) {
        val c = controller(World())
        c.runBenchmark(asr.id, Surface.DETAIL)
        assertNull(c.bench.value)
        assertFalse(HubLogic.detail(asr, c.input.value!!).bench.canRun)
    }

    @Test
    fun addModelValidatesResolvesPreviewsAndSaves() = runTest(UnconfinedTestDispatcher()) {
        val w = World()
        val c = controller(w)
        c.openAdd(ModelKind.LLM)
        c.editAdd { it.copy(repo = "not a repo") }
        c.resolveAdd()
        assertTrue(c.add.value!!.repoInvalid)
        assertTrue(w.custom.requests.isEmpty())

        c.editAdd { it.copy(repo = "https://huggingface.co/someone/Some-Model-GGUF", format = PromptFormat.FINETUNE) }
        c.resolveAdd()
        advanceUntilIdle()
        val req = w.custom.requests.single()
        assertEquals("someone/Some-Model-GGUF", req.repo)
        assertEquals(app.murmur.core.catalog.GgufTemplates.FINETUNE_PROFILE, req.promptProfile)
        val preview = c.add.value!!.preview!!
        assertEquals(Origin.CUSTOM, preview.origin)

        c.saveAdd()
        advanceUntilIdle()
        assertNull(c.add.value)
        assertEquals(HubTab.CLEANUP, c.tab.value)
        assertNotNull(c.input.value!!.catalog.model(preview.id))
        assertTrue(HubLogic.list(HubTab.CLEANUP, c.input.value!!).rows.any { it.id == preview.id && it.custom })
    }

    @Test
    fun addModelShowsTheResolverError() = runTest(UnconfinedTestDispatcher()) {
        val w = World()
        w.custom.result = Outcome.Err(MurmurError.CatalogInvalid("unsupported GGUF architecture 'gemma3'; supported: qwen35"))
        val c = controller(w)
        c.openAdd(ModelKind.LLM)
        c.editAdd { it.copy(repo = "owner/name") }
        c.resolveAdd()
        advanceUntilIdle()
        val s = c.add.value!!
        assertFalse(s.busy)
        assertNull(s.preview)
        assertTrue(s.error is MurmurError.CatalogInvalid)
        // Speech requests carry no GGUF file or prompt profile.
        c.editAdd { it.copy(kind = ModelKind.ASR, file = "x.gguf", format = PromptFormat.STANDARD) }
        c.resolveAdd()
        advanceUntilIdle()
        val asrReq = w.custom.requests.last()
        assertNull(asrReq.file)
        assertNull(asrReq.promptProfile)
    }

    @Test
    fun deletingACustomModelRemovesItsEntry() = runTest(UnconfinedTestDispatcher()) {
        val w = World()
        val base = other
        val custom = base.copy(id = ModelId("custom-x"), origin = Origin.CUSTOM)
        w.catalog.addCustom(custom)
        val c = controller(w)
        c.requestDelete(custom.id, Surface.LIST)
        assertTrue(c.delete.value!!.custom)
        c.confirmDelete()
        advanceUntilIdle()
        assertNull(c.input.value!!.catalog.model(custom.id))
    }

    @Test
    fun pauseAndResumeReachTheInstaller() = runTest(UnconfinedTestDispatcher()) {
        val paused = mutableListOf<ArtifactId>()
        val downloads = mutableListOf<ArtifactId>()
        val a = cpuArtifact(asr)
        val states = MutableStateFlow<Map<ArtifactId, InstallState>>(mapOf(a.id to InstallState.Downloading(10, 100, 5)))
        val repo = object : app.murmur.core.store.InstalledModelsRepository {
            override val states = states
            override fun state(id: ArtifactId) = states.map { it[id] ?: InstallState.NotInstalled }
            override suspend fun download(id: ArtifactId, wifiOnly: Boolean) { downloads += id }
            override suspend fun pause(id: ArtifactId) { paused += id }
            override suspend fun delete(id: ArtifactId) {}
            override suspend fun storage() = app.murmur.core.store.StorageInfo(0, 0)
        }
        val w = World()
        val deps = ModelsDeps(w.catalog, repo, w.benchmarks, w.runner, w.settings, w.custom, w.device, w.health, ensureDevice = { FakeDeviceProfileRepository.EMULATOR })
        val c = ModelsController(deps, backgroundScope)
        advanceUntilIdle()
        c.pause(asr.id)
        advanceUntilIdle()
        assertEquals(listOf(a.id), paused)
        states.value = mapOf(a.id to InstallState.Failed(MurmurError.ChecksumMismatch))
        advanceUntilIdle()
        c.resume(asr.id)
        advanceUntilIdle()
        assertEquals(listOf(a.id), downloads)
    }
}
