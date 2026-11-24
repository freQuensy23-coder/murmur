package app.murmur.data.db

import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.ModelId
import app.murmur.core.model.UnitPreference
import app.murmur.core.store.BubblePosition
import app.murmur.core.store.BubbleSize
import app.murmur.core.store.BubbleVisibility
import app.murmur.core.store.CleanupChoice
import app.murmur.core.store.Feedback
import app.murmur.core.store.PromptOverride
import app.murmur.core.store.Settings
import app.murmur.core.store.Tone
import androidx.datastore.core.DataStore
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.async
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.emitAll
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.io.IOException
import java.nio.file.Files

/** Typed DataStore on a real file (JVM): round trip, serialised updates, old and corrupt documents. */
class SettingsStoreTest {
    private lateinit var dir: File
    private lateinit var file: File
    private val logged = mutableListOf<String>()
    private val scopes = mutableListOf<CoroutineScope>()

    @Before
    fun setUp() {
        dir = Files.createTempDirectory("settings").toFile()
        file = File(dir, DataStoreSettingsRepository.FILE_NAME)
    }

    @After
    fun tearDown() = runBlocking {
        scopes.forEach { it.coroutineContext[Job]!!.cancelAndJoin() }
        dir.deleteRecursively()
        Unit
    }

    private fun open(): DataStoreSettingsRepository {
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO).also { scopes += it }
        return DataStoreSettingsRepository(file, scope, log = { code, where -> synchronized(logged) { logged += "$code@$where" } })
    }

    /** Closes the store so that the next [open] on the same file is a fresh read (one active DataStore per file). */
    private suspend fun close() {
        scopes.forEach { it.coroutineContext[Job]!!.cancelAndJoin() }
        scopes.clear()
    }

    /** Every field differs from the product default, so a field that is not persisted fails the round trip. */
    private val everything = Settings(
        languages = listOf("en"),
        autoDetectLanguage = false,
        holdToTalk = false,
        feedback = Feedback.BOTH,
        typingWpm = 55,
        bubbleVisibility = BubbleVisibility.ALWAYS,
        tuckIn = false,
        bubbleSize = BubbleSize.LARGE,
        hiddenInPackages = setOf("com.example.bank", "org.example.vault"),
        bubblePositions = mapOf("cover" to BubblePosition(rightEdge = false, yFraction = 0.42f), "inner:2" to BubblePosition(true, 0.8f)),
        hiddenUntilEpochMs = 1_763_500_000_000,
        promptOverride = PromptOverride(systemText = "Исправь ошибки, сохрани смысл.\nKeep English terms.", baseProfile = "default.v4"),
        preferences = "Пиши «ё», не «е». Use Oxford commas.",
        tonePerApp = mapOf("org.telegram.messenger" to Tone.CASUAL, "com.google.android.gm" to Tone.FORMAL, "com.termux" to Tone.VERBATIM),
        defaultTone = Tone.FORMAL,
        useScreen = false,
        retentionDays = 3,
        keepRecordings = false,
        asrModel = ModelId("gigaam-v3-rnnt"),
        hebrewAsrModel = ModelId("zipformer2-66m-streaming-he"),
        cleanup = CleanupChoice.Model(ModelId("lfm3-1.7b")),
        unitPreference = mapOf(
            "lfm3-1.7b" to UnitPreference.Pinned(ComputeUnit.NPU),
            "gigaam-v3-rnnt" to UnitPreference.Pinned(ComputeUnit.CPU),
            "qwen3.5-0.8b" to UnitPreference.Auto,
        ),
        wifiOnlyDownloads = false,
        idleUnloadSec = 30,
        onboardingDone = true,
        debugFakeEngines = true,
        debugTimeoutScale = 4.5f,
        debugIgnoreMemoryGate = true,
    )

    @Test
    fun noFileGivesTheProductDefaults() = runBlocking {
        assertEquals(Settings(), open().current())
        assertEquals(emptyList<String>(), logged)
    }

    @Test
    fun everyFieldRoundTrips() = runBlocking {
        val json = Json { encodeDefaults = true }
        val mine = json.encodeToJsonElement(Settings.serializer(), everything).jsonObject
        val defaults = json.encodeToJsonElement(Settings.serializer(), Settings()).jsonObject
        assertEquals(defaults.keys, mine.keys)
        for (k in mine.keys) assertNotEquals("field $k must differ from its default in this test", defaults[k], mine[k])

        open().update { everything }
        close()
        val reopened = open()
        assertEquals(everything, reopened.current())
        assertEquals(everything, reopened.settings.value)
        // The unit preference is keyed by model id (not artifact id) and keeps each variant.
        assertEquals(UnitPreference.Pinned(ComputeUnit.NPU), reopened.current().unitPreference["lfm3-1.7b"])
        assertEquals(UnitPreference.Auto, reopened.current().unitPreference["qwen3.5-0.8b"])
        assertEquals(emptyList<String>(), logged)
    }

    @Test
    fun cleanupChoiceObjectsRoundTrip() = runBlocking {
        for (choice in listOf(CleanupChoice.None, CleanupChoice.Default, CleanupChoice.Model(ModelId("qwen3.5-0.8b")))) {
            open().update { it.copy(cleanup = choice) }
            close()
            assertEquals(choice, open().current().cleanup)
            close()
        }
    }

    @Test
    fun concurrentUpdatesSerialise() = runBlocking {
        val repo = open()
        val n = 200
        (1..n).map { async(Dispatchers.Default) { repo.update { it.copy(typingWpm = it.typingWpm + 1) } } }.awaitAll()
        assertEquals(40 + n, repo.current().typingWpm)
        withTimeout(5_000) { repo.settings.first { it.typingWpm == 40 + n } }
        close()
        assertEquals(40 + n, open().current().typingWpm)
    }

    @Test
    fun updatesReachTheStateFlow() = runBlocking {
        val repo = open()
        repo.current()
        repo.update { it.copy(retentionDays = 1) }
        assertEquals(1, withTimeout(5_000) { repo.settings.first { it.retentionDays == 1 } }.retentionDays)
    }

    @Test
    fun oldFileWithRemovedLivePreviewKeyStillDecodes() = runBlocking {
        // v1 prototype documents carried "livePreviewModel" (removed by ADR-047); a newer app may add keys we do not know.
        file.writeText("""{"livePreviewModel":"whisper-tiny","retentionDays":3,"typingWpm":50,"futureKey":{"x":[1,2]}}""")
        val s = open().current()
        assertEquals(Settings(retentionDays = 3, typingWpm = 50), s)
        assertEquals(emptyList<String>(), logged)
    }

    @Test
    fun unknownEnumValueFallsBackToThatFieldsDefault() = runBlocking {
        file.writeText("""{"feedback":"LASER","bubbleSize":"HUGE","typingWpm":60}""")
        assertEquals(Settings(typingWpm = 60), open().current())
        assertEquals(emptyList<String>(), logged)
    }

    @Test
    fun corruptFileGivesDefaultsAndLogsTheCodeOnly() = runBlocking {
        file.writeText("{\"typingWpm\": 50, \"languages\": [\"ru\"")
        val repo = open()
        assertEquals(Settings(), repo.current())
        assertEquals(listOf("data.io@settings_corrupt"), logged)
        // The document was replaced by the defaults, and writes work again.
        repo.update { it.copy(typingWpm = 45) }
        close()
        assertEquals(45, open().current().typingWpm)
        assertTrue(file.readText().contains("\"typingWpm\":45"))
    }

    @Test
    fun wrongJsonShapeIsCorruptionToo() = runBlocking {
        file.writeText("""{"languages":"ru","retentionDays":"seven"}""")
        assertEquals(Settings(), open().current())
        assertEquals(listOf("data.io@settings_corrupt"), logged)
    }

    @Test
    fun everyFieldIsWrittenEvenAtItsDefault() = runBlocking {
        val repo = open()
        repo.update { it.copy(typingWpm = 41) }
        repo.update { Settings() } // a real write (the value changed) of a document that is all defaults
        close()
        val written = Json.parseToJsonElement(file.readText()).jsonObject
        // A later release that changes a default must not change what this user has: every key is on disk.
        val descriptor = Settings.serializer().descriptor
        assertEquals((0 until descriptor.elementsCount).map { descriptor.getElementName(it) }.toSet(), written.keys)
        assertEquals(7, written.getValue("retentionDays").jsonPrimitive.int)
        assertEquals(40, written.getValue("typingWpm").jsonPrimitive.int)
    }

    /** A store whose first [failures] reads fail with [error]; a later read sees what [updateData] stored. */
    private class FlakyStore(@Volatile var failures: Int, private val error: () -> Throwable) : DataStore<Settings> {
        private val value = MutableStateFlow(Settings())
        @Volatile var reads = 0
        override val data: Flow<Settings> = flow {
            reads++
            if (failures > 0) {
                failures--
                throw error()
            }
            emitAll(value)
        }
        override suspend fun updateData(transform: suspend (t: Settings) -> Settings): Settings =
            synchronized(value) { value.value }.let { transform(it) }.also { value.value = it }
    }

    private val crashes = mutableListOf<Throwable>()

    private fun openOn(store: DataStore<Settings>, retryDelayMs: Long): DataStoreSettingsRepository {
        val handler = CoroutineExceptionHandler { _, e -> synchronized(crashes) { crashes += e } }
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO + handler).also { scopes += it }
        return DataStoreSettingsRepository(store, scope, { code, where -> synchronized(logged) { logged += "$code@$where" } }) { retryDelayMs }
    }

    @Test
    fun readIoErrorsAreRetriedUntilTheStoreReadsAgain() = runBlocking {
        val store = FlakyStore(failures = 5) { IOException("disk busy") }
        val repo = openOn(store, retryDelayMs = 20)
        // Decisions do not wait for the store to recover: the first failure releases current() with the defaults.
        assertEquals(Settings(), withTimeout(2_000) { repo.current() })
        store.updateData { it.copy(retentionDays = 1) }
        // After five failed reads the collection is still alive and picks up the stored document...
        withTimeout(5_000) { repo.settings.first { it.retentionDays == 1 } }
        // ...and every later update.
        repo.update { it.copy(typingWpm = 70) }
        withTimeout(5_000) { repo.settings.first { it.typingWpm == 70 } }
        assertEquals(6, store.reads)
        assertEquals(List(5) { "data.io@settings_read" }, logged)
        assertEquals(emptyList<Throwable>(), crashes)
    }

    @Test
    fun currentDoesNotWaitForTheRetries() = runBlocking {
        val repo = openOn(FlakyStore(failures = Int.MAX_VALUE) { IOException("no disk") }, retryDelayMs = 60_000)
        assertEquals(Settings(), withTimeout(2_000) { repo.current() })
        assertEquals(listOf("data.io@settings_read"), logged)
    }

    @Test
    fun anyOtherReadErrorIsLoggedAndNeitherCrashesNorHangs() = runBlocking {
        // E.g. DataStore's "multiple DataStores active for the same file" (IllegalStateException).
        val repo = openOn(FlakyStore(failures = 1) { IllegalStateException("second store") }, retryDelayMs = 20)
        assertEquals(Settings(), withTimeout(2_000) { repo.current() })
        assertEquals(listOf("data.io@settings_read"), logged)
        assertEquals(emptyList<Throwable>(), crashes)
        assertEquals(Settings(), repo.settings.value)
    }
}
