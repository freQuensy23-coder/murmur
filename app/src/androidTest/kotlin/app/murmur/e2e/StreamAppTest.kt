package app.murmur.e2e

import android.os.SystemClock
import androidx.activity.ComponentActivity
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.getOrNull
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.test.ext.junit.runners.AndroidJUnit4
import app.murmur.core.store.BubbleVisibility
import app.murmur.core.benchmark.WordErrorRate
import app.murmur.core.dictation.AsrReadiness
import app.murmur.core.dictation.ContextReadiness
import app.murmur.core.dictation.DictationOutcome
import app.murmur.core.dictation.DictationState
import app.murmur.core.dictation.UserInput
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.ModelId
import app.murmur.core.model.UnitPreference
import app.murmur.core.store.CleanupChoice
import app.murmur.core.store.EntryStatus
import app.murmur.core.store.InsertMethod
import app.murmur.feature.onboarding.PracticeScreen
import app.murmur.ui.design.MurmurTheme
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeNotNull
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class StreamAppTest {
    @get:Rule
    val compose = createAndroidComposeRule<ComponentActivity>()

    @Test
    fun fixtureToEditors() {
        val requested = E2e.args.getString("stream.mode")
        assumeNotNull(requested)
        val mode = requireNotNull(requested)
        require(mode in setOf("insertion", "ocr", "full"))
        val fake = mode == "insertion"
        val full = mode == "full"
        val emu = E2e.args.getString("stream.emulator") == "1"
        val c = E2e.container
        check(!c.fakeContainer)
        if (fake) Pipeline.requireFakes() else Pipeline.requireReal()
        val asset = E2e.instrumentation.context.assets
        val bytes = asset.open("stream/fleurs-ru.wav").use { it.readBytes() }
        val reference = asset.open("stream/fleurs-ru.txt").bufferedReader().use { it.readText().trim() }
        check(bytes.size > 44 && String(bytes, 0, 4) == "RIFF")
        val clipMs = (bytes.size - 44) / 32L
        val before = runBlocking { c.settings.current() }
        try {
            Fixtures.captureFile.writeBytes(bytes)
            runBlocking {
                c.settings.update {
                    it.copy(
                        asrModel = ModelId(Pipeline.ASR_MODEL),
                        cleanup = if (fake || full) CleanupChoice.Model(ModelId(Pipeline.LLM_MODEL)) else CleanupChoice.None,
                        unitPreference = it.unitPreference + listOf(Pipeline.ASR_MODEL, Pipeline.LLM_MODEL)
                            .associateWith { UnitPreference.Pinned(ComputeUnit.CPU) },
                        useScreen = !fake,
                        bubbleVisibility = BubbleVisibility.ALWAYS,
                        tuckIn = false,
                        hiddenInPackages = emptySet(),
                        hiddenUntilEpochMs = null,
                        keepRecordings = false,
                        retentionDays = 7,
                        onboardingDone = true,
                        debugIgnoreMemoryGate = emu || fake,
                        debugTimeoutScale = if (emu && full) 8f else 1f,
                    )
                }
            }
            c.permissions.refresh()
            practice(clipMs, reference, fake, full)
            Accessibility.enable()
            overlay(clipMs, reference, fake, full)
        } finally {
            Fixtures.captureFile.delete()
            runBlocking { c.settings.update { before } }
        }
        E2e.report("stream.pass", mode)
    }

    private fun practice(clipMs: Long, reference: String, fake: Boolean, full: Boolean) {
        val c = E2e.container
        val historyCount = Pipeline.historyCount()
        compose.setContent { MurmurTheme { PracticeScreen(c.onboardingDeps, onClose = {}) } }
        DictationRecorder().use { rec ->
            compose.onNodeWithTag("practice-bubble-ready").assertIsDisplayed().performClick()
            val listening = rec.awaitListening(12_000)
            check(!listening.plan.screenshot) { "practice must not request OCR" }
            SystemClock.sleep(clipMs + 500)
            compose.onNodeWithTag("practice-bubble-listening").performClick()
            val done = rec.awaitDone(if (full) 65_000 else 25_000)
            val inserted = done.outcome as? DictationOutcome.Inserted ?: error("practice did not insert: ${rec.stateKinds()}")
            check(inserted.entry == null)
            if (full || fake) check(!inserted.noCleanup) { "practice cleanup skipped: ${inserted.skipReason}" }
            val field = compose.onNodeWithTag("practice-field").fetchSemanticsNode().config
                .getOrNull(SemanticsProperties.EditableText)?.text.orEmpty()
            assertTrue("practice field is empty", field.isNotBlank())
            if (fake) assertEquals(Pipeline.fakeCleaned().trim(), field.trim())
            else checkWer("practice", reference, field)
            assertEquals("practice has no history", historyCount, Pipeline.historyCount())
            E2e.report("stream.practice", "inserted=true screenshot=false history=false")
            E2e.waitUntil("idle after practice", 12_000) { rec.state is DictationState.Idle }
        }
    }

    private fun overlay(clipMs: Long, reference: String, fake: Boolean, full: Boolean) {
        val historyCount = Pipeline.historyCount()
        TargetApp.open(TargetApp.Target.PLAIN)
        E2e.waitUntil("target owns the served input connection", 10_000) {
            E2e.shell("dumpsys input_method").lineSequence().any { line ->
                ("mCurAttribute=" in line || "mCurrentEditorInfo=" in line || "packageName=" in line) &&
                    "packageName=${TargetApp.PKG}" in line
            }
        }
        val ime = E2e.shell("settings get secure default_input_method").trim()
        check(ime.isNotEmpty() && ime != "null") { "no active IME" }
        E2e.report("stream.ime", ime)
        check(TargetApp.lastText().isNullOrEmpty()) { "expected a fresh external editor" }
        DictationRecorder().use { rec ->
            val start = SystemClock.uptimeMillis()
            val orchestrator = E2e.container.orchestrator
            orchestrator.onInput(UserInput.Tap)
            val listening = rec.awaitListening(12_000)
            assertEquals("external screenshot plan", !fake, listening.plan.screenshot)
            SystemClock.sleep(clipMs + 500)
            val words = if (fake) null else {
                val ready = E2e.waitFor("OCR to finish", 12_000) {
                    rec.states.mapNotNull { rec.enginesOf(it.value)?.context as? ContextReadiness.Ready }.lastOrNull()
                }
                check(ready.hasScreen && ready.contextWords > 0) { "no real OCR text: $ready" }
                val asr = checkNotNull(rec.states.mapNotNull { rec.enginesOf(it.value)?.asr as? AsrReadiness.Ready }.lastOrNull())
                check(!asr.fellBack && asr.info.deviceName.equals("cpu", ignoreCase = true)) { "unexpected ASR backend" }
                ready.contextWords
            }
            orchestrator.onInput(UserInput.Tap)
            val stop = SystemClock.uptimeMillis()
            val done = rec.awaitDone(if (full) 65_000 else 25_000)
            val inserted = done.outcome as? DictationOutcome.Inserted ?: error("overlay did not insert: ${rec.stateKinds()}")
            if (full || fake) check(!inserted.noCleanup) { "overlay cleanup skipped: ${inserted.skipReason}" }
            val row = Pipeline.awaitNewEntry(historyCount)
            assertEquals(inserted.entry, row.id)
            assertEquals(EntryStatus.INSERTED, row.status)
            assertEquals(TargetApp.PKG, row.app.pkg)
            assertEquals("keyboard InputConnection, no paste/set-text fallback", InsertMethod.COMMIT_TEXT, row.insertMethod)
            val final = checkNotNull(row.finalText).trimEnd()
            check(final.isNotBlank())
            TargetApp.awaitText("matches history exactly") { it.trimEnd() == final }
            if (!fake) {
                checkWer("raw", reference, checkNotNull(row.rawText))
                check((row.timings.ocrMs ?: 0) > 0) { "history lacks OCR timing" }
            }
            if (full) {
                assertEquals("accept", row.cleanup?.outcome)
                check((row.contextWords ?: 0) > 0) { "cleanup did not consume OCR context" }
                check((row.timings.llmMs ?: 0) > 0) { "history lacks cleanup timing" }
            }
            E2e.report("stream.overlay", "trigger=instrumented-tap insert=${row.insertMethod} ocrWords=$words ocrMs=${row.timings.ocrMs} cleanup=${row.cleanup?.outcome} llmMs=${row.timings.llmMs} wallMs=${SystemClock.uptimeMillis() - start} afterStopMs=${SystemClock.uptimeMillis() - stop}")
        }
    }

    private fun checkWer(label: String, reference: String, text: String) {
        val wer = WordErrorRate.wer(reference, text)
        check(wer <= 0.20) { "$label WER $wer > 0.20" }
        E2e.report("stream.$label.wer", wer)
    }
}
