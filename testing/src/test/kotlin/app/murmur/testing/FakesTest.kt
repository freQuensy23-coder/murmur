package app.murmur.testing

import app.murmur.core.audio.PcmChunk
import app.murmur.core.audio.PcmFormat
import app.murmur.core.benchmark.BenchProgress
import app.murmur.core.catalog.ArtifactSelector
import app.murmur.core.engine.AsrEvent
import app.murmur.core.engine.AsrSessionOptions
import app.murmur.core.engine.LlmEvent
import app.murmur.core.engine.LlmRequest
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.InferProcess
import app.murmur.core.model.MurmurError
import app.murmur.core.model.Outcome
import app.murmur.core.model.UnitPreference
import app.murmur.core.store.CleanupChoice
import app.murmur.core.store.Settings
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withTimeoutOrNull
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FakesTest {
    /** Integration pass 6 (WP-16 note): a soft delete / restore re-emits like the Room store; search folds "ё". */
    @Test
    fun inMemoryHistoryReEmitsAfterASoftDeleteAndFoldsYo() = runTest {
        val history = InMemoryHistoryRepository()
        val ids = app.murmur.testing.contrib.Wp16HistoryFixture.seed(history, audio = null)
        val first = ids.first()
        val seen = mutableListOf<Int>()
        val job = launch { history.entries().collect { seen += it.size } }
        testScheduler.advanceUntilIdle()
        history.markDeleted(setOf(first))
        testScheduler.advanceUntilIdle()
        assertEquals(null, history.entry(first).first())
        history.restore(setOf(first))
        testScheduler.advanceUntilIdle()
        history.markDeleted(setOf(first)); history.commitDeletes()
        testScheduler.advanceUntilIdle()
        job.cancel()
        val n = ids.size
        assertEquals(listOf(n, n - 1, n, n - 1), seen.distinctUntilChangedList())
        assertEquals(n - 1, history.snapshot().size)
        assertTrue(history.search("ПРОШЕЛ", null).first().isNotEmpty()) // stored text has "прошёл"
    }

    private fun List<Int>.distinctUntilChangedList() = fold(mutableListOf<Int>()) { acc, v -> if (acc.lastOrNull() != v) acc += v; acc }

    @Test
    fun fakeAsrNeedsSpeech() = runTest {
        val asr = FakeAsrEngine(script = "тест")
        val s = asr.startSession(AsrSessionOptions())
        s.accept(PcmChunk(ShortArray(16_000), 0))
        assertEquals(Outcome.Err(MurmurError.NoSpeech), s.finish())
        val s2 = asr.startSession(AsrSessionOptions())
        s2.accept(PcmChunk(syntheticSpeech(), 0))
        assertEquals("тест", (s2.finish() as Outcome.Ok).value.text)
    }

    @Test
    fun fakeAsrCommitsSegmentsWhileAudioFlowsAndDiesWithItsHost() = runTest {
        val e = FakeEngines()
        val s = e.asr.startSession(AsrSessionOptions())
        val pcm = syntheticSpeech(totalMs = 2_000, speechMs = 1_800)
        for (i in pcm.indices step PcmFormat.FRAME_SAMPLES) s.accept(PcmChunk(pcm.copyOfRange(i, minOf(i + PcmFormat.FRAME_SAMPLES, pcm.size)), i.toLong()))
        assertTrue(s.events.first() is AsrEvent.Committed)
        e.asrHost.crash()
        assertEquals(Outcome.Err(MurmurError.ProcessDied(InferProcess.ASR)), s.finish())
        assertEquals(1, e.asrHost.crashes)
    }

    @Test
    fun fakeLlmIsDeterministicAndCanHangOrFailPerUnit() = runTest {
        val out = FakeLlmEngine().generate(LlmRequest("<transcript>\nпривет мир\n</transcript>", 64)).toList()
        assertEquals("Привет мир.", (out.last() as LlmEvent.Done).text)
        val hanging = FakeLlmEngine(hang = true)
        assertNull(withTimeoutOrNull(5_000) { hanging.generate(LlmRequest("x", 8)).toList() })
        val npuBroken = FakeLlmEngine(loadErrors = mutableMapOf(ComputeUnit.NPU to MurmurError.UnitUnavailable(ComputeUnit.NPU)))
        val r = FakeModelResolver(npuFirst = true).resolve(Settings()) as Outcome.Ok
        assertTrue(npuBroken.load(r.value.llm!!.spec) is Outcome.Err)
        assertTrue(npuBroken.load(r.value.llm!!.fallback!!) is Outcome.Ok)
        assertEquals(ComputeUnit.CPU, r.value.asrFallback!!.unit)
    }

    @Test
    fun resolverReportsWhyThereIsNoCleanup() = runTest {
        val r = FakeModelResolver().resolve(Settings(cleanup = CleanupChoice.None)) as Outcome.Ok
        assertNull(r.value.llm)
        assertEquals("none", r.value.llmUnavailable)
        assertEquals(Outcome.Err(MurmurError.NoVadModel), FakeModelResolver(noVad = true).resolve(Settings()))
    }

    @OptIn(kotlinx.coroutines.ExperimentalCoroutinesApi::class)
    @Test
    fun virtualClockFollowsTheTestScheduler() = runTest {
        val clock = VirtualClock { testScheduler.currentTime }
        kotlinx.coroutines.delay(1_500)
        assertEquals(1_500, clock.elapsedMs())
    }

    @Test
    fun benchmarkFakeEmitsStepsPerUsableUnit() = runTest {
        val catalog = app.murmur.core.catalog.CatalogParser.bundled()
        val entry = catalog.models.first { it.llm != null && it.artifacts.any { a -> ComputeUnit.NPU in a.units && ComputeUnit.GPU in a.units } }
        val cands = ArtifactSelector.ranked(entry, FakeDeviceProfileRepository.FOLD7, UnitPreference.Auto).take(2)
        val events = FakeBenchmarkRunner().run(entry, cands).toList()
        val finished = events.last() as BenchProgress.Finished
        assertEquals(2, finished.records.size)
    }

    @Test
    fun fakeCaptureIsPacedStopsOnStopAndWritesTheHistoryAudio() = runTest {
        val store = InMemoryAudioStore()
        val cap = FakeAudioCapture(pcm = syntheticSpeech(totalMs = 2_000), store = store)
        val h = (cap.start(app.murmur.core.model.SessionId(1), keepRecording = true, audioKey = "k-1") as Outcome.Ok).value
        val got = mutableListOf<PcmChunk>()
        val job = launch { h.chunks.collect { got += it } }
        kotlinx.coroutines.delay(1_010)
        val r = h.stop()
        job.join()
        assertEquals(50, got.size)                                           // 1 s of 20 ms chunks, then the stream ends
        assertEquals(1_000L, r.durationMs)
        assertEquals(16_000, store.files.getValue(r.audio!!.path).size)
        assertEquals(listOf("k-1"), cap.audioKeys)
    }

    @OptIn(kotlinx.coroutines.ExperimentalCoroutinesApi::class)
    @Test
    fun shadeWaitReturnsTheTargetOnceCollapsedOrNullAfterTheTimeout() = runTest {
        val r = FakeTargetReader(shadeDelayMs = 400)
        val t0 = testScheduler.currentTime
        assertEquals("org.telegram.messenger", r.snapshotAfterShade(1_000)?.pkg)
        assertEquals(400, testScheduler.currentTime - t0)
        r.shadeDelayMs = 5_000
        assertNull(r.snapshotAfterShade(1_000))
    }

    @Test
    fun resolverSkipsFailedUnitsLikeTheSelectorAndPreparesEveryUnit() = runTest {
        val health = FakeUnitHealth()
        val m = FakeModelResolver(npuFirst = true, health = health)
        assertEquals(ComputeUnit.NPU, (m.resolve(Settings()) as Outcome.Ok).value.llm!!.spec.unit)
        health.markFailed(m.llmNpu.artifactId, ComputeUnit.NPU, "engine.load.llm")
        health.markFailed(m.asrNpu.artifactId, ComputeUnit.NPU, "engine.unit_unavailable")
        val after = (m.resolve(Settings()) as Outcome.Ok).value
        assertEquals(ComputeUnit.CPU, after.llm!!.spec.unit)
        assertNull(after.llm!!.fallback)
        assertEquals(ComputeUnit.CPU, after.asr.unit)
        val llm = (FakeModelResolver(npuFirst = true).prepareTargets(m.llmCpu.artifactId, Settings()) as Outcome.Ok).value
        assertEquals(listOf(ComputeUnit.NPU, ComputeUnit.CPU), llm.specs.map { it.unit })
    }

    @Test
    fun fakeWorldBuildsDeps() {
        val d = FakeWorld().deps()
        assertTrue(d.prompts.systemPrompt("default.v1").isNotBlank())
        assertTrue(d.prompts.systemPrompt("default.v4").isNotBlank())
    }

    @Test
    fun benchmarkFakeReturnsNewestPerUnitAndValidIgnoresNewerInvalidRuns() = runTest {
        val repo = InMemoryBenchmarkRepository()
        val id = app.murmur.core.model.ArtifactId("a")
        fun rec(unit: ComputeUnit, at: Long, valid: Boolean) = app.murmur.core.store.BenchmarkRecord(
            id, unit, "1", "fp", at,
            app.murmur.core.store.BenchmarkResultData.Asr(1.0, 1, 1, null, 1, unit.name, emptyList(), valid = valid),
        )
        val cpuOld = rec(ComputeUnit.CPU, 1, valid = true)
        val npuValid = rec(ComputeUnit.NPU, 2, valid = true)
        val cpuNew = rec(ComputeUnit.CPU, 3, valid = true)
        val npuInvalid = rec(ComputeUnit.NPU, 4, valid = false)
        listOf(cpuOld, npuValid, cpuNew, npuInvalid).forEach { repo.save(it) }
        assertEquals(listOf(npuInvalid, cpuNew), repo.latest(id).first())
        assertEquals(npuValid, repo.latestValid(id, ComputeUnit.NPU))
        assertEquals(cpuNew, repo.latestValid(id, ComputeUnit.CPU))
    }

    @Test
    fun fakeAssemblerKeepsTheCauseOfAMissingScreen() {
        val a = SimpleContextAssembler()
        val snap = FakeTargetReader.defaultTarget()
        val d = a.decide(snap, app.murmur.core.context.ContextSettings())
        val secure = a.assemble(snap, d, app.murmur.core.context.ScreenOutcome.SecureWindow, null)
        assertEquals(app.murmur.core.context.Privacy("app_only", "secure_window"), secure.privacy)
        assertNull(secure.field)
        val failed = a.assemble(snap, d, app.murmur.core.context.ScreenOutcome.Failed, null)
        assertEquals(app.murmur.core.context.Privacy("full", "screenshot_error"), failed.privacy)
        val own = a.decide(FakeTargetReader.defaultTarget("app.murmur"), app.murmur.core.context.ContextSettings())
        assertTrue(!own.dictationAllowed)
        assertTrue(a.decide(FakeTargetReader.defaultTarget("app.murmur"), app.murmur.core.context.ContextSettings(inAppPractice = true)).dictationAllowed)
    }
}
