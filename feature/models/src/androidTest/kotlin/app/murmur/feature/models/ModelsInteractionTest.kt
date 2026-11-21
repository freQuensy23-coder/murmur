package app.murmur.feature.models

import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.hasAnyAncestor
import androidx.compose.ui.test.isDialog
import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.v2.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTextInput
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import app.murmur.core.catalog.CatalogParser
import app.murmur.core.model.ArtifactId
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.ModelKind
import app.murmur.core.model.UnitPreference
import app.murmur.core.store.CleanupChoice
import app.murmur.core.store.Settings
import app.murmur.testing.FakeBenchmarkRunner
import app.murmur.testing.FakeCatalogRepository
import app.murmur.testing.FakeCustomModelResolver
import app.murmur.testing.FakeDeviceProfileRepository
import app.murmur.testing.FakeInstalledModelsRepository
import app.murmur.testing.FakeUnitHealth
import app.murmur.testing.InMemoryBenchmarkRepository
import app.murmur.testing.InMemorySettingsRepository
import app.murmur.ui.design.MurmurTheme
import kotlinx.coroutines.flow.MutableStateFlow
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.compose.ui.test.captureToImage
import app.murmur.core.store.BenchmarkResultData
import kotlinx.coroutines.flow.first
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/** The real screens over the :testing fakes (Fold7 profile): selection, unit pin, delete, benchmark, add a model. */
@RunWith(AndroidJUnit4::class)
class ModelsInteractionTest {
    @get:Rule
    val rule = createAndroidComposeRule<ComponentActivity>()

    private fun keepScreenOn(): Boolean =
        rule.activity.window.attributes.flags and WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON != 0

    private val res = InstrumentationRegistry.getInstrumentation().targetContext.resources
    private fun s(id: Int, vararg args: Any) = res.getString(id, *args)

    private val catalog = CatalogParser.bundled()
    private val asr = catalog.model(catalog.defaults.asr.getValue("ru+en"))!!
    private val llm = catalog.models.first { it.llm != null && it.artifacts.any { a -> ComputeUnit.NPU in a.units } }
    private val otherLlm = catalog.models.first { it.kind == ModelKind.LLM && it.id != llm.id && !it.noRussian && it.artifacts.size > 1 }

    private inner class World(installed: Set<ArtifactId>, settings: Settings = Settings(), stepMs: Long = 150) {
        val installed = FakeInstalledModelsRepository(catalog.models.flatMap { m -> m.artifacts.map { it.id } }.filterNot { it in installed }.toMutableSet())
        val benchmarks = InMemoryBenchmarkRepository()
        val settings = InMemorySettingsRepository(settings)
        val runner = FakeBenchmarkRunner(stepMs = stepMs, repo = benchmarks)
        val custom = FakeCustomModelResolver()
        val catalogRepo = FakeCatalogRepository()
        val deps = ModelsDeps(
            catalogRepo, this.installed, benchmarks, runner, this.settings, custom,
            MutableStateFlow(FakeDeviceProfileRepository.FOLD7), FakeUnitHealth(),
            ensureDevice = { FakeDeviceProfileRepository.FOLD7 }, deviceName = "Test phone",
        )
    }

    private val allAsr get() = asr.artifacts.map { it.id }.toSet()

    @Test
    fun everyCatalogModelDetailRenders() {
        val w = World(catalog.models.flatMap { m -> m.artifacts.map { it.id } }.toSet())
        var id by mutableStateOf(catalog.models.first().id)
        rule.setContent { MurmurTheme(darkTheme = false) { ModelDetail(w.deps, id, onClose = {}) } }
        assertEquals(27, catalog.models.size)
        for (m in catalog.models) {
            rule.runOnIdle { id = m.id }
            rule.waitForIdle()
            rule.onNode(hasText(m.displayName) and !hasTestTag(HubTags.row(m.id)), useUnmergedTree = true).assertExists()
            rule.onNodeWithTag(HubTags.ACTION_BAR).assertExists()
        }
    }

    /** Integration pass 6 (ADR-087): the shell's one-shot Speech / Cleanup request from a deep link. */
    @Test
    fun aSegmentRequestShowsThatSegmentOnceAndThenKeepsTheUsersChoice() {
        val w = World(allAsr + llm.artifacts.map { it.id })
        var segment by mutableStateOf<ModelKind?>(ModelKind.LLM)
        var shown = 0
        rule.setContent {
            MurmurTheme(darkTheme = false) {
                ModelsScreen(w.deps, onOpenModel = {}, segment = segment, onSegmentShown = { shown++; segment = null })
            }
        }
        rule.waitForIdle()
        assertEquals(1, shown)
        // "None" exists only on the Cleanup segment.
        rule.onNodeWithTag(HubTags.NONE_ROW).assertExists()
        rule.onNodeWithText(s(R.string.models_tab_speech)).performClick()
        rule.waitForIdle()
        assertTrue(rule.onAllNodes(hasTestTag(HubTags.NONE_ROW)).fetchSemanticsNodes().isEmpty())
        assertEquals(1, shown)
        rule.runOnIdle { segment = ModelKind.LLM }
        rule.waitForIdle()
        assertEquals(2, shown)
        rule.onNodeWithTag(HubTags.NONE_ROW).assertExists()
    }

    @Test
    fun noneAndRadioSelectTheCleanupModel() {
        val w = World(allAsr + llm.artifacts.map { it.id } + otherLlm.artifacts.map { it.id })
        rule.setContent { MurmurTheme(darkTheme = false) { ModelsScreen(w.deps, onOpenModel = {}) } }
        rule.onNodeWithText(s(R.string.models_tab_cleanup)).performClick()
        rule.onNodeWithTag(HubTags.NONE_ROW).performClick()
        rule.waitForIdle()
        assertEquals(CleanupChoice.None, w.settings.settings.value.cleanup)
        rule.onNode(hasContentDescriptionExactly(s(R.string.models_cd_select, otherLlm.displayName))).performScrollTo().performClick()
        rule.waitForIdle()
        assertEquals(CleanupChoice.Model(otherLlm.id), w.settings.settings.value.cleanup)
    }

    @Test
    fun theUnitPickerPinsPerModel() {
        val w = World(allAsr)
        rule.setContent { MurmurTheme(darkTheme = false) { ModelDetail(w.deps, asr.id, onClose = {}) } }
        rule.onNodeWithTag(HubTags.unitOption("CPU")).performScrollTo().performClick()
        rule.waitForIdle()
        assertEquals(UnitPreference.Pinned(ComputeUnit.CPU), w.settings.settings.value.unitPreference[asr.id.value])
        rule.onNodeWithTag(HubTags.unitOption("AUTO")).performScrollTo().performClick()
        rule.waitForIdle()
        assertTrue(w.settings.settings.value.unitPreference.isEmpty())
    }

    @Test
    fun deletingTheModelInUseAsksThenSwitches() {
        val w = World(allAsr + llm.artifacts.map { it.id } + otherLlm.artifacts.map { it.id }, Settings(cleanup = CleanupChoice.Model(llm.id)))
        rule.setContent { MurmurTheme(darkTheme = false) { ModelDetail(w.deps, llm.id, onClose = {}) } }
        rule.onNodeWithText(s(R.string.detail_delete)).performClick()
        rule.onNodeWithText(s(R.string.delete_title, llm.displayName)).assertIsDisplayed()
        rule.onNode(hasText(s(R.string.delete_confirm)) and hasAnyAncestor(isDialog())).performClick()
        rule.waitForIdle()
        assertEquals(CleanupChoice.Model(otherLlm.id), w.settings.settings.value.cleanup)
        assertTrue(llm.artifacts.none { w.installed.isInstalled(it.id) })
    }

    @Test
    fun benchmarkSheetRunsAndShowsTheResult() {
        // 2 units x 4 steps x 1 s: long enough to see the running checklist.
        val w = World(allAsr, stepMs = 1_000)
        rule.setContent { MurmurTheme(darkTheme = false) { ModelDetail(w.deps, asr.id, onClose = {}) } }
        rule.onNodeWithText(s(R.string.bench_run)).performScrollTo().performClick()
        rule.waitUntil(5_000) { rule.onAllNodes(hasTestTag(HubTags.BENCH_SHEET)).fetchSemanticsNodes().isNotEmpty() }
        rule.onNodeWithText(s(R.string.bench_overline_run).uppercase()).assertExists()
        rule.onNodeWithText(s(R.string.bench_step_load)).assertExists()
        rule.onNodeWithText(s(R.string.bench_stop)).assertExists()
        // DESIGN 6.4: the phone stays awake while the benchmark runs.
        assertTrue("FLAG_KEEP_SCREEN_ON while running", rule.runOnIdle { keepScreenOn() })
        rule.waitUntil(30_000) { rule.onAllNodes(hasText(s(R.string.bench_overline_result).uppercase())).fetchSemanticsNodes().isNotEmpty() }
        assertFalse("FLAG_KEEP_SCREEN_ON cleared with the result", rule.runOnIdle { keepScreenOn() })
        // Two units measured (NPU and CPU on the Fold7 profile): the segmented control switches between them.
        val inSheet = hasAnyAncestor(hasTestTag(HubTags.BENCH_SHEET))
        assertTrue(rule.onAllNodes(hasText(s(R.string.unit_npu)) and inSheet).fetchSemanticsNodes().isNotEmpty())
        assertTrue(rule.onAllNodes(hasText(s(R.string.unit_cpu)) and inSheet).fetchSemanticsNodes().isNotEmpty())
        // The numbers are the runner's record (FakeBenchmarkRunner: 14.2x, 1 100 MB peak), not frame fixtures.
        val npuId = asr.artifacts.first { ComputeUnit.NPU in it.units }.id
        val rec = kotlinx.coroutines.runBlocking { w.benchmarks.latest(npuId).first() }.single().result as BenchmarkResultData.Asr
        val loc = res.configuration.locales[0]
        rule.onNode(hasText(Format.rtf(rec.rtfx, loc) + s(R.string.bench_unit_x), substring = true) and inSheet).assertExists()
        rule.onNode(hasText(Format.memory(rec.peakMemMb, loc).first, substring = true) and inSheet).assertExists()
        captureSheet("no-frame/bench-result-runner")
        rule.onNodeWithText(s(R.string.bench_done)).performClick()
        rule.waitForIdle()
        rule.onNodeWithText(s(R.string.bench_run_again)).performScrollTo().assertIsDisplayed()
        assertFalse(rule.runOnIdle { keepScreenOn() })
    }

    @Test
    fun stoppingTheBenchmarkClearsKeepScreenOn() {
        val w = World(allAsr, stepMs = 2_000)
        rule.setContent { MurmurTheme(darkTheme = false) { ModelDetail(w.deps, asr.id, onClose = {}) } }
        rule.onNodeWithText(s(R.string.bench_run)).performScrollTo().performClick()
        rule.waitUntil(5_000) { rule.onAllNodes(hasText(s(R.string.bench_stop))).fetchSemanticsNodes().isNotEmpty() }
        assertTrue(rule.runOnIdle { keepScreenOn() })
        rule.onNodeWithText(s(R.string.bench_stop)).performClick()
        rule.waitUntil(5_000) { rule.onAllNodes(hasTestTag(HubTags.BENCH_SHEET)).fetchSemanticsNodes().isEmpty() }
        assertFalse(rule.runOnIdle { keepScreenOn() })
    }

    @Test
    fun deletingACustomModelClosesItsDetail() {
        // Review F9: the entry leaves the catalog; the shell gets onClose instead of an empty page.
        val w = World(allAsr + otherLlm.artifacts.map { it.id })
        val custom = otherLlm.copy(id = app.murmur.core.model.ModelId("custom-x"), displayName = "Custom X", origin = app.murmur.core.catalog.Origin.CUSTOM)
        kotlinx.coroutines.runBlocking { w.catalogRepo.addCustom(custom) }
        var closed = 0
        rule.setContent { MurmurTheme(darkTheme = false) { ModelDetail(w.deps, custom.id, onClose = { closed++ }) } }
        rule.onNodeWithText(s(R.string.detail_delete)).performClick()
        rule.onNode(hasText(s(R.string.delete_confirm)) and hasAnyAncestor(isDialog())).performClick()
        rule.waitUntil(5_000) { closed > 0 }
        assertEquals(1, closed)
        assertTrue(w.catalogRepo.catalog.value.models.none { it.id == custom.id })
    }

    @Test
    fun thePaneHasNoBackArrowAndTheCoverPageDoes() {
        val w = World(allAsr)
        var pane by mutableStateOf(true)
        var closed = 0
        rule.setContent { MurmurTheme(darkTheme = false) { ModelDetail(w.deps, asr.id, onClose = { closed++ }, pane = pane) } }
        val back = androidx.compose.ui.test.hasContentDescription(rule.activity.getString(app.murmur.ui.design.R.string.design_back))
        assertTrue(rule.onAllNodes(back).fetchSemanticsNodes().isEmpty())
        rule.runOnIdle { pane = false }
        rule.onNode(back).performClick()
        rule.runOnIdle { assertEquals(1, closed) }
    }

    /** The sheet as the runner left it, for the design owner (`<externalFiles>/models-shots/<name>--<display>.png`). */
    private fun captureSheet(name: String) {
        val bmp = rule.onNodeWithTag(HubTags.BENCH_SHEET).captureToImage().asAndroidBitmap()
        val wide = res.configuration.screenWidthDp >= 600
        val file = java.io.File(requireNotNull(rule.activity.getExternalFilesDir(null)), "models-shots/$name--${if (wide) "inner" else "cover"}.png")
        file.parentFile!!.mkdirs()
        file.outputStream().use { bmp.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it) }
    }

    @Test
    fun addAModelFromHuggingFace() {
        val w = World(allAsr)
        rule.setContent { MurmurTheme(darkTheme = false) { ModelsScreen(w.deps, onOpenModel = {}) } }
        rule.onNodeWithTag(HubTags.ADD).performScrollTo().performClick()
        rule.onNodeWithTag("add-repo").performTextInput("someone/Speech-Model")
        rule.onNodeWithTag("add-check").performClick()
        rule.waitUntil(5_000) { rule.onAllNodes(hasTestTag("add-save")).fetchSemanticsNodes().isNotEmpty() }
        rule.onNodeWithTag("add-save").performScrollTo().performClick()
        rule.waitForIdle()
        val added = w.catalogRepo.catalog.value.models.single { it.origin == app.murmur.core.catalog.Origin.CUSTOM }
        assertEquals(ModelKind.ASR, added.kind)
        assertEquals("someone/Speech-Model", w.custom.requests.single().repo)
        rule.onNodeWithTag(HubTags.row(added.id)).performScrollTo().assertIsDisplayed()
    }

    private fun hasContentDescriptionExactly(text: String) = androidx.compose.ui.test.hasContentDescription(text, substring = false)
}
