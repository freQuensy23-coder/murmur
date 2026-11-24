package app.murmur.feature.settings

import app.murmur.core.model.MurmurError
import app.murmur.core.model.ModelId
import app.murmur.core.model.ModelKind
import app.murmur.core.model.Outcome
import app.murmur.core.prompt.ClasspathPromptAssets
import app.murmur.core.prompt.PromptEditRules
import app.murmur.core.prompt.PromptEditState
import app.murmur.core.store.CleanupChoice
import app.murmur.core.store.PromptOverride
import app.murmur.core.store.Settings
import app.murmur.core.store.Tone
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class PromptEditorTest {
    private val prompts = ClasspathPromptAssets()
    private val shippedV1 = prompts.systemPrompt("default.v1").trimEnd('\n')
    private val fineTune = CleanupChoice.Model(ModelId("murmur-cleanup-qwen3.5-0.8b"))

    private suspend fun TestScope.open(settings: Settings = Settings()): Pair<SettingsFixtures.World, PromptEditor> {
        val w = SettingsFixtures.world(settings)
        val d = UnconfinedTestDispatcher(testScheduler)
        val scope = CoroutineScope(d + Job())
        backgroundScope.coroutineContext[Job]!!.invokeOnCompletion { scope.cancel() }
        val vm = SettingsViewModel(w.deps, scope, today = { w.today }, io = d)
        assertTrue(vm.openPrompt())
        return w to vm.promptEditor.value!!
    }

    private fun rowState(w: SettingsFixtures.World): PromptEditState {
        val s = w.settings.settings.value
        val profile = w.prompts.profile(SettingsViewModel.activeProfileName(s, w.catalog.catalog.value))
        return PromptEditRules.editState(profile, s.promptOverride?.baseProfile)
    }

    @Test
    fun opensWithTheShippedPromptOfTheActiveProfile() = runTest {
        val (_, e) = open()
        val ui = e.ui.value
        assertEquals("default.v1", ui.profile.name)
        assertEquals(shippedV1, ui.text)
        assertTrue(ui.editable)
        assertEquals(PromptNotice.NONE, ui.notice)
        assertFalse(ui.dirty)
        assertFalse("nothing to reset", ui.canReset)
        assertTrue(ui.canSave)
        assertEquals(TryState.Idle, ui.tryState)
    }

    @Test
    fun validationBlocksSavingEmptyTooLongAndMarkup() = runTest {
        val (w, e) = open()
        fun problems(text: String): List<String> { e.setText(text); return e.ui.value.problems.system }
        assertEquals(listOf("empty"), problems("  \n "))
        assertFalse(e.ui.value.canSave)
        assertEquals(listOf("too_long"), problems("a".repeat(PromptEditRules.MAX_SYSTEM_CHARS + 1)))
        assertEquals(emptyList<String>(), problems("a".repeat(PromptEditRules.MAX_SYSTEM_CHARS)))
        assertEquals(listOf("markup"), problems(SettingsFixtures.PROTOTYPE_PROMPT_MARKUP))
        assertEquals(listOf("markup"), problems("Hello <think>"))
        assertEquals("the shipped text names <context> and is valid", emptyList<String>(), problems(shippedV1))
        e.setText("")
        assertFalse(e.save())
        assertNull("nothing written", w.settings.settings.value.promptOverride)
        e.setText("Fine rules")
        e.setPreferences("p".repeat(PromptEditRules.MAX_PREFERENCES_CHARS + 1))
        assertTrue(e.ui.value.problems.preferencesTooLong)
        assertFalse(e.ui.value.canSave)
        assertFalse(e.save())
        e.setPreferences("  " + "p".repeat(PromptEditRules.MAX_PREFERENCES_CHARS) + "\n")
        assertFalse("surrounding blanks do not count", e.ui.value.problems.preferencesTooLong)
        assertTrue(e.save())
        assertEquals("p".repeat(PromptEditRules.MAX_PREFERENCES_CHARS), w.settings.settings.value.preferences)
    }

    @Test
    fun editedMeansTheStoredTextDiffersFromTheShippedOne() = runTest {
        val (w, e) = open()
        e.setText("$shippedV1\n11. Always write ё.")
        assertTrue(e.ui.value.dirty)
        assertTrue(e.ui.value.canReset)
        assertTrue(e.save())
        assertEquals(PromptOverride("$shippedV1\n11. Always write ё.", "default.v1"), w.settings.settings.value.promptOverride)
        assertEquals(PromptEditState.EDITED, rowState(w))
        assertFalse(e.ui.value.dirty)
        // Typing the shipped text back (CRLF and a trailing newline are the same text to the renderer) is Default.
        e.setText(shippedV1.replace("\n", "\r\n") + "\r\n")
        assertTrue(e.save())
        assertNull(w.settings.settings.value.promptOverride)
        assertEquals(PromptEditState.DEFAULT, rowState(w))
    }

    @Test
    fun resetRemovesTheEditAndKeepsTonesAndPreferences() = runTest {
        val start = Settings(
            promptOverride = PromptOverride("My own rules", "default.v1"),
            preferences = "Write ё.",
            tonePerApp = mapOf(SettingsFixtures.TELEGRAM to Tone.FORMAL),
            defaultTone = Tone.CASUAL,
        )
        val (w, e) = open(start)
        assertEquals("My own rules", e.ui.value.text)
        assertTrue(e.ui.value.canReset)
        e.reset()
        val s = w.settings.settings.value
        assertNull(s.promptOverride)
        assertEquals("Write ё.", s.preferences)
        assertEquals(mapOf(SettingsFixtures.TELEGRAM to Tone.FORMAL), s.tonePerApp)
        assertEquals(Tone.CASUAL, s.defaultTone)
        assertEquals(shippedV1, e.ui.value.text)
        assertFalse(e.ui.value.canReset)
        assertFalse(e.ui.value.dirty)
        assertEquals(PromptEditState.DEFAULT, rowState(w))
    }

    @Test
    fun aFineTuneProfileMakesTheEditorReadOnly() = runTest {
        val (w, e) = open(Settings(cleanup = fineTune, preferences = "Old"))
        val ui = e.ui.value
        assertEquals("finetuned.v2", ui.profile.name)
        assertFalse(ui.editable)
        assertEquals(prompts.systemPrompt("finetuned.v2").trimEnd('\n'), ui.text)
        assertEquals(PromptNotice.FINE_TUNE, ui.notice)
        assertFalse(ui.canReset)
        e.setText("changed")
        assertEquals("the text cannot change", prompts.systemPrompt("finetuned.v2").trimEnd('\n'), e.ui.value.text)
        // My preferences still apply to every model.
        e.setPreferences("New")
        assertTrue(e.save())
        assertEquals("New", w.settings.settings.value.preferences)
        assertNull(w.settings.settings.value.promptOverride)
        e.reset()
        assertEquals("New", w.settings.settings.value.preferences)
    }

    @Test
    fun anEditForAnOlderPromptShowsTheNoticeWithResetAndKeepEditing() = runTest {
        val (w, e) = open(Settings(promptOverride = PromptOverride("Rules for v4", "default.v4")))
        assertEquals(PromptEditState.MADE_FOR_OLDER_PROMPT, rowState(w))
        assertEquals(PromptNotice.OLDER_PROMPT, e.ui.value.notice)
        assertEquals("Keep editing continues the user's text", "Rules for v4", e.ui.value.text)
        assertTrue(e.ui.value.canReset)
        e.keepEditing()
        assertEquals(PromptNotice.NONE, e.ui.value.notice)
        e.setText("Rules for v1")
        assertTrue(e.save())
        assertEquals(PromptOverride("Rules for v1", "default.v1"), w.settings.settings.value.promptOverride)
        assertEquals(PromptEditState.EDITED, rowState(w))
    }

    @Test
    fun resetFromTheOlderPromptNotice() = runTest {
        val (w, e) = open(Settings(promptOverride = PromptOverride("Rules for v4", "default.v4")))
        e.reset()
        assertNull(w.settings.settings.value.promptOverride)
        assertEquals(PromptNotice.NONE, e.ui.value.notice)
        assertEquals(shippedV1, e.ui.value.text)
    }

    @Test
    fun onAFineTuneAStoredEditIsKeptWithoutReset() = runTest {
        val edit = PromptOverride("Rules for stock", "default.v1")
        val (w, e) = open(Settings(cleanup = fineTune, promptOverride = edit))
        assertEquals(PromptNotice.FINE_TUNE_EDIT_KEPT, e.ui.value.notice)
        assertFalse(e.ui.value.editable)
        assertFalse("no Reset: it would delete an edit that applies again", e.ui.value.canReset)
        e.reset()
        assertTrue(e.save())
        assertEquals(edit, w.settings.settings.value.promptOverride)
    }

    @Test
    fun tryItPassesTheUnsavedOverrideAndPreferences() = runTest {
        val (w, e) = open(Settings(preferences = "Saved prefs"))
        e.setText("Unsaved rules")
        e.setPreferences("  Unsaved prefs ")
        e.runTry(PromptEditor.SAMPLE)
        assertEquals(listOf(Triple(PromptEditor.SAMPLE, "Unsaved rules", "Unsaved prefs")), w.tryCalls)
        assertEquals(TryState.Done("Скинь мне отчёт до пятницы, и созвонимся в понедельник в 10:00."), e.ui.value.tryState)
        assertNull("nothing saved by Try it", w.settings.settings.value.promptOverride)
        assertEquals("Saved prefs", w.settings.settings.value.preferences)
        // The shipped text is passed as "no override"; empty preferences as null.
        e.setText(shippedV1)
        e.setPreferences(" ")
        e.runTry(PromptEditor.SAMPLE)
        assertEquals(Triple(PromptEditor.SAMPLE, null, null), w.tryCalls.last())
    }

    @Test
    fun tryItShowsPolishingWhileTheModelRuns() = runTest {
        val (w, e) = open()
        val gate = CompletableDeferred<Outcome<String>>()
        val slow = PromptEditor(e.profile, w.prompts, w.settings, { _, _, _ -> gate.await() }, backgroundScope, w.settings.settings.value)
        slow.runTry(PromptEditor.SAMPLE)
        testScheduler.runCurrent()
        assertEquals(TryState.Polishing, slow.ui.value.tryState)
        gate.complete(Outcome.Err(MurmurError.LlmFailed("no_model")))
        testScheduler.runCurrent()
        assertEquals(TryState.Failed(TryFailure.NO_MODEL), slow.ui.value.tryState)
    }

    @Test
    fun tryItErrorsMapOntoTheCardsLines() {
        assertEquals(TryFailure.NO_MODEL, TryFailure.of(MurmurError.LlmFailed("no_model")))
        assertEquals(TryFailure.CLEANUP_OFF, TryFailure.of(MurmurError.LlmFailed("none")))
        assertEquals(TryFailure.NO_SPEECH_MODEL, TryFailure.of(MurmurError.NoAsrModel))
        assertEquals(TryFailure.LOW_MEMORY, TryFailure.of(MurmurError.OutOfMemory(ModelKind.LLM)))
        assertEquals(TryFailure.REJECTED, TryFailure.of(MurmurError.GuardRejected("R3_preamble")))
        assertEquals(TryFailure.REJECTED, TryFailure.of(MurmurError.NoSpeech))
        assertEquals(TryFailure.FAILED, TryFailure.of(MurmurError.LlmFailed(null)))
        assertEquals(TryFailure.FAILED, TryFailure.of(MurmurError.Timeout(app.murmur.core.model.Stage.LLM)))
    }

    @Test
    fun tryItNeverSendsTextThatFailsValidation() = runTest {
        val (w, e) = open()
        val invalid = listOf(
            "",
            " \n ",
            SettingsFixtures.PROTOTYPE_PROMPT_MARKUP,
            "Rules <think>",
            "a".repeat(PromptEditRules.MAX_SYSTEM_CHARS + 1),
        )
        for (text in invalid) {
            e.setText(text)
            e.runTry(PromptEditor.SAMPLE)
            assertEquals("'${text.take(20)}' is not sent", emptyList<Any>(), w.tryCalls)
            assertEquals(TryState.Failed(TryFailure.INVALID), e.ui.value.tryState)
        }
        e.setText("Fine rules")
        e.setPreferences("p".repeat(PromptEditRules.MAX_PREFERENCES_CHARS + 1))
        e.runTry(PromptEditor.SAMPLE)
        assertEquals("oversized preferences are not sent", emptyList<Any>(), w.tryCalls)
        assertEquals(TryState.Failed(TryFailure.INVALID), e.ui.value.tryState)
        // Fixing the text makes Test run again.
        e.setPreferences("Write ё.")
        e.runTry(PromptEditor.SAMPLE)
        assertEquals(listOf(Triple(PromptEditor.SAMPLE, "Fine rules", "Write ё.")), w.tryCalls)
        assertTrue(e.ui.value.tryState is TryState.Done)
        // Editing into an error keeps the last valid result on the card until Test is pressed.
        e.setText("")
        assertTrue(e.ui.value.tryState is TryState.Done)
    }

    @Test
    fun aFineTuneNeverGetsTheEditorTextInTryIt() = runTest {
        val (w, e) = open(Settings(cleanup = fineTune))
        e.runTry(PromptEditor.SAMPLE)
        assertEquals(Triple(PromptEditor.SAMPLE, null, null), w.tryCalls.single())
    }
}
