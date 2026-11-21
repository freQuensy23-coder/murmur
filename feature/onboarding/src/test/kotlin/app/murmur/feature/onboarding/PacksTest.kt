package app.murmur.feature.onboarding

import app.murmur.core.catalog.CatalogParser
import app.murmur.core.catalog.DownloadPlanner
import app.murmur.core.catalog.PlannedDownload
import app.murmur.core.model.ArtifactId
import app.murmur.core.model.ModelId
import app.murmur.core.model.MurmurError
import app.murmur.core.store.CleanupChoice
import app.murmur.core.store.InstallState
import app.murmur.core.store.Settings
import app.murmur.testing.FakeDeviceProfileRepository
import java.util.Locale
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PacksTest {
    private val catalog = CatalogParser.bundled()
    private val emulator = FakeDeviceProfileRepository.EMULATOR
    private val fold7 = FakeDeviceProfileRepository.FOLD7
    private fun bytes(id: String) = catalog.artifact(ArtifactId(id))!!.second.downloadBytes

    @Test
    fun packSizesFollowTheArtifactsEachDeviceResolvesTo() {
        val onFold = PackPlanner.options(catalog, fold7, Settings())
        val onEmu = PackPlanner.options(catalog, emulator, Settings())
        assertEquals("every bundled pack is offered on both", catalog.packs.map { it.id }.toSet(), onFold.map { it.pack.id }.toSet())
        assertEquals(catalog.packs.map { it.id }.toSet(), onEmu.map { it.pack.id }.toSet())
        assertEquals("recommended first", "speech-only", onFold.first().pack.id)

        val foldRec = onFold.first { it.pack.id == "recommended" }
        val emuRec = onEmu.first { it.pack.id == "recommended" }
        // Fold7: the NPU build of Parakeet (sherpa SM8750 context binaries); emulator: the CPU int8 build.
        assertTrue(ArtifactId("parakeet-tdt-0.6b-v3@qnn-sm8750-20s") in foldRec.artifacts)
        assertFalse(ArtifactId("parakeet-tdt-0.6b-v3@onnx-int8") in foldRec.artifacts)
        assertTrue(ArtifactId("parakeet-tdt-0.6b-v3@onnx-int8") in emuRec.artifacts)
        assertFalse(ArtifactId("parakeet-tdt-0.6b-v3@qnn-sm8750-20s") in emuRec.artifacts)
        // Both queue the VAD (batch ASR) and the OCR pair ("Use what's on screen" is on by default), ARCHITECTURE 8.5.
        for (o in listOf(foldRec, emuRec)) {
            assertTrue(ArtifactId("silero-vad-v5@onnx") in o.artifacts)
            assertTrue(ArtifactId("pp-ocrv6-tiny-det@onnx") in o.artifacts)
            assertTrue(ArtifactId("pp-ocrv5-cyrillic-mobile-rec@onnx") in o.artifacts)
            assertEquals(ModelId("lfm3-1.7b"), o.llm?.id)
            assertEquals(o.plan.sumOf { it.artifact.downloadBytes }, o.totalBytes)
            assertEquals("nothing installed yet", o.totalBytes, o.remainingBytes)
        }
        assertEquals(
            bytes("parakeet-tdt-0.6b-v3@qnn-sm8750-20s") - bytes("parakeet-tdt-0.6b-v3@onnx-int8"),
            foldRec.plan.first { it.model.value == "parakeet-tdt-0.6b-v3" }.artifact.downloadBytes -
                emuRec.plan.first { it.model.value == "parakeet-tdt-0.6b-v3" }.artifact.downloadBytes,
        )
        assertNotEquals("the two phones get different downloads", foldRec.totalBytes, emuRec.totalBytes)
        // The planner is the hub's rule too (DownloadPlanner): onboarding adds nothing of its own.
        assertEquals(DownloadPlanner.forPack(catalog, foldRec.pack, fold7, true), foldRec.plan)
        assertEquals("1.5", ByteFormat.packGb(foldRec.totalBytes, Locale.US))
    }

    @Test
    fun speechOnlyHasNoCleanupAndScreenReadingFollowsTheSetting() {
        val speech = PackPlanner.options(catalog, emulator, Settings()).first { it.pack.id == "speech-only" }
        assertNull(speech.llm)
        assertTrue(speech.plan.any { it.reason == PlannedDownload.Reason.OCR_PAIR })
        val noScreen = PackPlanner.options(catalog, emulator, Settings(useScreen = false)).first { it.pack.id == "recommended" }
        assertFalse("OCR left out with Use what's on screen off", noScreen.plan.any { it.reason == PlannedDownload.Reason.OCR_PAIR })
    }

    @Test
    fun remainingBytesLeaveOutWhatIsInstalled() {
        val rec = PackPlanner.options(catalog, fold7, Settings()).first { it.pack.recommended }
        val vad = ArtifactId("silero-vad-v5@onnx")
        val again = PackPlanner.options(catalog, fold7, Settings(), mapOf(vad to InstallState.Installed("/m", 1))).first { it.pack.recommended }
        assertEquals(rec.totalBytes - bytes("silero-vad-v5@onnx"), again.remainingBytes)
        assertEquals(rec.totalBytes, again.totalBytes)
    }

    @Test
    fun choosingAPackSetsItsModels() {
        val opts = PackPlanner.options(catalog, emulator, Settings())
        val light = PackPlanner.applyChoice(Settings(), opts.first { it.pack.id == "light" })
        assertEquals(ModelId("parakeet-tdt-0.6b-v3"), light.asrModel)
        assertEquals(CleanupChoice.Model(ModelId("qwen3.5-0.8b")), light.cleanup)
        assertEquals(CleanupChoice.None, PackPlanner.applyChoice(Settings(), opts.first { it.pack.id == "speech-only" }).cleanup)
        assertEquals("speech-only", PackPlanner.defaultChoice(opts)?.pack?.id)
        assertNull(PackPlanner.defaultChoice(opts.filter { it.llm != null }))
    }

    @Test
    fun summaryTracksProgressPreparingAndStops() {
        val rec = PackPlanner.options(catalog, emulator, Settings()).first { it.pack.id == "recommended" }
        val (asr, vad, det, rec2, llm) = rec.artifacts
        val total = rec.totalBytes
        val llmBytes = catalog.artifact(llm)!!.second.downloadBytes

        val nothing = DownloadSummary.of(catalog, rec, emptyMap())
        assertEquals(ModelsPhase.NONE, nothing.phase)
        assertNull(nothing.preparing)

        val mid = DownloadSummary.of(catalog, rec, mapOf(
            asr to InstallState.Installed("/a", 1), vad to InstallState.Installed("/v", 1),
            det to InstallState.Preparing(0.5f), rec2 to InstallState.Queued(true),
            llm to InstallState.Downloading(llmBytes / 4, llmBytes, 18_400_000),
        ))
        assertEquals(ModelsPhase.ACTIVE, mid.phase)
        assertEquals(false, mid.preparing)
        assertTrue(mid.speechReady)
        assertEquals("cleanup model still on its way", "LFM3 1.7B", mid.cleanupPending)
        assertEquals(total, mid.bytesTotal)
        val expectedDone = total - catalog.artifact(rec2)!!.second.downloadBytes - llmBytes + llmBytes / 4
        assertEquals(expectedDone, mid.bytesDone)
        assertEquals(18_400_000L, mid.bytesPerSec)
        assertEquals((total - expectedDone) / 18_400_000, mid.secondsLeft)

        val stopped = DownloadSummary.of(catalog, rec, mapOf(
            asr to InstallState.Installed("/a", 1), llm to InstallState.Paused(llmBytes / 2, llmBytes, MurmurError.NoNetwork),
        ))
        assertEquals(ModelsPhase.STOPPED, stopped.phase)
        assertEquals(MurmurError.NoNetwork, stopped.stopReason)
        assertNull(stopped.secondsLeft)

        val done = DownloadSummary.of(catalog, rec, rec.artifacts.associateWith { InstallState.Installed("/x", 1) })
        assertEquals(ModelsPhase.DONE, done.phase)
        assertEquals(true, done.preparing)
        assertEquals(1f, done.fraction)
        assertNull(done.cleanupPending)
    }

    @Test
    fun aCpuBuildQueuedByPreparingJoinsTheChecklist() {
        val rec = PackPlanner.options(catalog, fold7, Settings()).first { it.pack.recommended }
        val npu = ArtifactId("parakeet-tdt-0.6b-v3@qnn-sm8750-20s")
        val cpu = ArtifactId("parakeet-tdt-0.6b-v3@onnx-int8")
        val s = DownloadSummary.of(catalog, rec, rec.artifacts.associateWith<ArtifactId, InstallState> { InstallState.Installed("/x", 1) } +
            (npu to InstallState.Installed("/n", 1)) + (cpu to InstallState.Queued(true)))
        assertTrue(s.rows.any { it.artifact == cpu })
        assertEquals("the extra download keeps the step going", ModelsPhase.ACTIVE, s.phase)
        assertEquals(rec.totalBytes + bytes("parakeet-tdt-0.6b-v3@onnx-int8"), s.bytesTotal)
    }

    @Test
    fun figuresAreFormattedLikeTheFrames() {
        assertEquals("1.7", ByteFormat.packGb(1_690_000_000, Locale.US))
        assertEquals("0.78", ByteFormat.gb2(780_000_000, Locale.US))
        assertEquals("1,100", ByteFormat.mb(1_100_000_000, Locale.US))
        val ru = Locale.forLanguageTag("ru")
        assertEquals("1${java.text.DecimalFormatSymbols.getInstance(ru).groupingSeparator}100", ByteFormat.mb(1_100_000_000, ru))
        assertEquals("18.4", ByteFormat.mbPerSec(18_400_000, Locale.US))
        assertEquals("181", ByteFormat.freeGb(181_000_000_000, Locale.US))
        assertEquals("3.4", ByteFormat.freeGb(3_400_000_000, Locale.US))
        assertEquals(46, ByteFormat.percent(0.4615f))
        assertEquals(TimeLeft.Seconds(50), TimeLeft.of(50))
        assertEquals(TimeLeft.Minutes(3), TimeLeft.of(150))
    }
}
