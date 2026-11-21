package app.murmur.core.dictation

import app.murmur.core.engine.PrefixCacheKey
import app.murmur.core.engine.PrefixSpec
import app.murmur.core.model.MurmurError
import app.murmur.core.model.Outcome
import app.murmur.core.model.Stage
import app.murmur.core.model.getOrNull
import app.murmur.core.prompt.ClasspathPromptAssets
import app.murmur.core.prompt.RenderPyPort
import app.murmur.core.store.PromptOverride
import app.murmur.core.store.Settings
import app.murmur.testing.FakeWorld
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FineTunePrefixTest {
    private val world = FakeWorld()
    private val assets = ClasspathPromptAssets()
    private val cleaner = Cleaner(world.engines.llm, RenderPyPort(assets), world.guard, assets, world.clock)
    private val keeper = EngineKeeper(world.engines, world.unitHealth)

    private suspend fun stock() = world.models.resolve(Settings()).getOrNull()!!.llm!!

    private suspend fun load(active: ActiveLlm, settings: Settings = Settings()): Outcome<EngineKeeper.LlmLoaded> {
        val override = cleaner.systemOverride(cleaner.profile(active), settings)
        val prefix = cleaner.prefix(active, settings.preferences, override)
        return keeper.loadLlm(active, active.spec, prefix)
    }

    @Test fun profileAndPreferencesSelectDistinctSnapshotsOfTheSameWarmModel() = runTest {
        val stock = stock()
        val legacy = stock.copy(profile = "finetuned.v1")
        val next = stock.copy(profile = "finetuned.v2")
        val editedStock = Settings(promptOverride = PromptOverride("Replace the stock instructions.", stock.profile))
        val withPreferences = editedStock.copy(preferences = "No emoji.")

        assertTrue(load(stock) is Outcome.Ok)
        assertTrue(load(legacy) is Outcome.Ok)
        assertTrue(load(next) is Outcome.Ok)
        assertTrue(load(next, editedStock) is Outcome.Ok)
        assertTrue(load(next, withPreferences) is Outcome.Ok)
        assertTrue(load(next, withPreferences) is Outcome.Ok)

        val snapshots = world.engines.llm.prefixCalls
        assertEquals(4, snapshots.size)
        assertEquals(4, snapshots.map { it.cacheKey }.distinct().size)
        assertEquals(1, world.engines.llm.loadAttempts.size)
        val prefix = cleaner.prefix(next, null, null)
        val key = PrefixCacheKey.of(next.modelSha256, PrefixCacheKey.sha256Hex(prefix), next.spec.unit, "fake-1", next.nCtx)
        assertEquals(PrefixSpec(prefix, key), snapshots[2])
        assertNotEquals(prefix, snapshots[3].text)
    }

    @Test fun failedProfileSwitchCannotReuseThePreviousWarmPrefix() = runTest {
        val stock = stock()
        assertTrue(load(stock) is Outcome.Ok)
        val error = MurmurError.Timeout(Stage.LLM_PREFIX)
        world.engines.llm.prefixError = error
        assertEquals(Outcome.Err(error), load(stock.copy(profile = "finetuned.v2")))
        world.engines.llm.prefixError = null
        assertTrue(load(stock) is Outcome.Ok)
        val snapshots = world.engines.llm.prefixCalls
        assertEquals(3, snapshots.size)
        assertEquals(snapshots.first(), snapshots.last())
        assertEquals(1, world.engines.llm.loadAttempts.size)
    }
}
