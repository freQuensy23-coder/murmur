package app.murmur.platform.a11y

import app.murmur.core.store.InsertMethod
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class InsertionLadderTest {
    /** How a rung behaves on the fake field. */
    enum class Rung { LANDS, LANDS_LATE, LANDS_ON_LATER_READ, REPORTS_OK_BUT_NOTHING, UNAVAILABLE }

    /** A text field with a caret; every action appends at the caret like the real ones do. */
    private class FakeField(
        private val scope: CoroutineScope,
        var text: String = "",
        val commit: Rung = Rung.LANDS,
        val paste: Rung = Rung.LANDS,
        val setText: Rung = Rung.LANDS,
        val readable: Boolean = true,
        val lateMs: Long = 400,
        val nbsp: Boolean = false,
        /** LANDS_ON_LATER_READ: the text shows up only from this read on (a commit that arrives at the window's edge). */
        val landOnRead: Int = Int.MAX_VALUE,
        /** How long one read of the field takes (a binder round trip to the target app under load). */
        val readMs: Long = 0,
    ) : LadderField {
        val calls = mutableListOf<InsertMethod>()
        /** Virtual time of every rung call, from the start of the test. */
        val callAt = mutableListOf<Long>()
        var caret = text.length
        var reads = 0
        private var pending: String? = null

        private fun land(t: String) {
            val s = if (nbsp) t.replace(' ', ' ') else t
            text = text.substring(0, caret) + s + text.substring(caret); caret += s.length
        }

        private fun act(m: InsertMethod, r: Rung, t: String): Boolean {
            if (r == Rung.UNAVAILABLE) return false
            calls += m
            callAt += (scope as TestScope).testScheduler.currentTime
            when (r) {
                Rung.LANDS -> land(t)
                Rung.LANDS_LATE -> scope.launch { delay(lateMs); land(t) }
                Rung.LANDS_ON_LATER_READ -> pending = t
                else -> Unit
            }
            return true
        }

        override suspend fun beforeCursor(maxChars: Int): String? {
            if (readMs > 0) delay(readMs)
            if (++reads >= landOnRead) pending?.let { pending = null; land(it) }
            return if (!readable) null else text.substring(0, caret).takeLast(maxChars)
        }
        override fun commitText(text: String) = act(InsertMethod.COMMIT_TEXT, commit, text)
        override fun paste(text: String) = act(InsertMethod.PASTE, paste, text)
        override fun setText(text: String) = act(InsertMethod.SET_TEXT, setText, text)
    }

    private fun TestScope.ladder() = InsertionLadder(now = { testScheduler.currentTime })

    @Test
    fun commitTextIsTriedFirstAndWinsWhenItLands() = runTest {
        val f = FakeField(this, text = "Hi ")
        val r = ladder().run(f, "there")
        assertEquals(InsertMethod.COMMIT_TEXT, r.method)
        assertEquals(listOf(InsertMethod.COMMIT_TEXT), f.calls)
        assertEquals("Hi there", f.text)
    }

    @Test
    fun ladderOrderIsCommitThenPasteThenSetText() = runTest {
        val f = FakeField(this, commit = Rung.REPORTS_OK_BUT_NOTHING, paste = Rung.REPORTS_OK_BUT_NOTHING)
        val r = ladder().run(f, "hello")
        assertEquals(InsertMethod.SET_TEXT, r.method)
        assertEquals(listOf(InsertMethod.COMMIT_TEXT, InsertMethod.PASTE, InsertMethod.SET_TEXT), f.calls)
        assertEquals(listOf(false, false, true), r.attempts.map { it.verified })
        assertEquals("hello", f.text)
    }

    @Test
    fun unavailableRungsAreSkippedWithoutWaiting() = runTest {
        val f = FakeField(this, commit = Rung.UNAVAILABLE)
        val r = ladder().run(f, "x")
        assertEquals(InsertMethod.PASTE, r.method)
        assertEquals(listOf(false, true), r.attempts.map { it.called })
        assertEquals(0L, r.attempts.first().ms)
    }

    @Test
    fun aLateCommitIsAcceptedAndNeverInsertedTwice() = runTest {
        // The commit becomes visible only after the verify window (1 original read + 13 polls over 600 ms), at the
        // first read of the grace period: the paste must not run, and the attempt is logged as verified (late).
        val f = FakeField(this, commit = Rung.LANDS_ON_LATER_READ, landOnRead = 15)
        val r = ladder().run(f, "once")
        assertEquals(InsertMethod.COMMIT_TEXT, r.method)
        assertEquals(listOf(InsertMethod.COMMIT_TEXT), f.calls)
        assertEquals("once", f.text)
        assertEquals(listOf(true), r.attempts.map { it.verified })
    }

    /**
     * ADR-086 item 5, the emulator finding: the IME accepted the commit, the field showed it only after the 600 ms
     * window, and the old ladder pasted the text a second time. Now the grace period accepts COMMIT_TEXT.
     */
    @Test
    fun aCommitThatShowsUpAfterTheVerifyWindowIsAcceptedInTheGracePeriodNotPastedAgain() = runTest {
        val f = FakeField(this, text = "Note: ", commit = Rung.LANDS_LATE, lateMs = 900)
        val r = ladder().run(f, "buy milk")
        assertEquals(InsertMethod.COMMIT_TEXT, r.method)
        assertEquals(listOf(InsertMethod.COMMIT_TEXT), f.calls)
        assertEquals("Note: buy milk", f.text)
        val a = r.attempts.single()
        assertTrue("verified late", a.verified)
        assertTrue("accepted when it landed (${a.ms} ms)", a.ms in 900..950)
    }

    @Test
    fun anUnreadableFieldAfterATrueCommitIsNeverEscalated() = runTest {
        // The commit was accepted but nothing can read the field: a paste could only double the text, so the ladder
        // stops (the orchestrator then reports Copied, the text is on the clipboard).
        val f = FakeField(this, text = "abc", readable = false)
        val r = ladder().run(f, "x")
        assertNull(r.method)
        assertEquals(listOf(InsertMethod.COMMIT_TEXT), f.calls)
        assertEquals("abcx", f.text)
        assertEquals(listOf(true to false), r.attempts.map { it.called to it.verified })
    }

    @Test
    fun aTrueCommitThatNeverLandsEscalatesToPasteAfterTheGracePeriod() = runTest {
        val f = FakeField(this, text = "a", commit = Rung.REPORTS_OK_BUT_NOTHING)
        val r = ladder().run(f, "b")
        assertEquals(InsertMethod.PASTE, r.method)
        assertEquals(listOf(InsertMethod.COMMIT_TEXT, InsertMethod.PASTE), f.calls)
        assertEquals("ab", f.text)
        val commit = r.attempts.first()
        assertTrue(
            "the commit got the window and the grace period (${commit.ms} ms)",
            commit.ms >= InsertionLadder.VERIFY_TIMEOUT_MS + InsertionLadder.COMMIT_GRACE_MS,
        )
    }

    /**
     * Review F1: the slowest path (every rung reports success, nothing ever shows up) must end inside the orchestrator's
     * delivery timeout with a margin; the old grace (800 ms on top of three 600 ms windows) started SET_TEXT at 2,000 ms,
     * when the delivery timer had already moved the dictation to Copied.
     */
    @Test
    fun theSlowestPathEndsInsideTheDeliveryTimeoutWithAMargin() = runTest {
        val f = FakeField(this, commit = Rung.REPORTS_OK_BUT_NOTHING, paste = Rung.REPORTS_OK_BUT_NOTHING, setText = Rung.REPORTS_OK_BUT_NOTHING)
        val t0 = testScheduler.currentTime
        val r = ladder().run(f, "late")
        val total = testScheduler.currentTime - t0
        assertNull(r.method)
        assertEquals(listOf(InsertMethod.COMMIT_TEXT, InsertMethod.PASTE, InsertMethod.SET_TEXT), r.attempts.map { it.method })
        val commit = r.attempts[0]
        assertEquals("window + grace", InsertionLadder.VERIFY_TIMEOUT_MS + InsertionLadder.COMMIT_GRACE_MS, commit.ms)
        assertTrue("paste and set-text keep their own window (${r.attempts.map { it.ms }})", r.attempts.drop(1).all { it.ms >= InsertionLadder.LATE_RUNG_MS })
        val setTextStart = f.callAt.last() - t0
        assertTrue("set-text starts at $setTextStart ms", setTextStart + r.attempts.last().ms <= InsertionLadder.LADDER_BUDGET_MS)
        assertTrue("whole run $total ms", total <= InsertionLadder.LADDER_BUDGET_MS)
        val delivery = app.murmur.core.dictation.DictationConfig().deliveryTimeoutMs
        assertTrue("budget ${InsertionLadder.LADDER_BUDGET_MS} + 400 ms margin <= delivery $delivery", InsertionLadder.LADDER_BUDGET_MS + 400 <= delivery)
    }

    @Test
    fun withoutAGracePeriodTheLaterRungsKeepTheirFullWindow() = runTest {
        // commitText false (no IME connection): paste and set-text get 600 ms each, as before ADR-086 item 5.
        val f = FakeField(this, commit = Rung.UNAVAILABLE, paste = Rung.REPORTS_OK_BUT_NOTHING, setText = Rung.LANDS_LATE, lateMs = 550)
        val r = ladder().run(f, "z")
        assertEquals(InsertMethod.SET_TEXT, r.method)
        assertEquals(InsertionLadder.VERIFY_TIMEOUT_MS, r.attempts[1].ms)
    }

    @Test
    fun slowReadsNeverPushARungPastTheBudget() = runTest {
        // Each read costs 180 ms: the deadlines are absolute, so the waits shrink instead of adding up, and no rung starts
        // once the budget is spent.
        val f = FakeField(this, commit = Rung.REPORTS_OK_BUT_NOTHING, paste = Rung.REPORTS_OK_BUT_NOTHING, setText = Rung.REPORTS_OK_BUT_NOTHING, readMs = 180)
        val t0 = testScheduler.currentTime
        val r = ladder().run(f, "slow reads")
        val total = testScheduler.currentTime - t0
        assertNull(r.method)
        val starts = f.callAt.map { it - t0 }
        assertTrue("rung starts $starts", starts.isNotEmpty() && starts.all { it < InsertionLadder.LADDER_BUDGET_MS })
        assertTrue("whole run $total ms <= budget + one read + one poll", total <= InsertionLadder.LADDER_BUDGET_MS + 180 + InsertionLadder.POLL_MS)
    }

    @Test
    fun aFalseCommitGoesStraightToPaste() = runTest {
        // commitText false = the IME has no connection: no grace period (nothing can arrive later).
        val f = FakeField(this, commit = Rung.UNAVAILABLE)
        val r = ladder().run(f, "y")
        assertEquals(InsertMethod.PASTE, r.method)
        assertEquals(0L, r.attempts.first().ms)
    }

    /**
     * Review F2: when the text before the caret already repeats the dictated text over the whole read window, a commit
     * that never landed leaves the field unchanged and must not be reported as COMMIT_TEXT (the window-capped reads look
     * the same before and after, so nothing can be verified: the orchestrator reports Copied).
     */
    @Test
    fun aFieldThatAlreadyRepeatsTheTextIsNotReportedAsInsertedWhenNothingLanded() = runTest {
        val f = FakeField(this, text = "ha".repeat(40), commit = Rung.REPORTS_OK_BUT_NOTHING, paste = Rung.UNAVAILABLE, setText = Rung.UNAVAILABLE)
        val r = ladder().run(f, "ha")
        assertNull(r.method)
        assertEquals("ha".repeat(40), f.text)
        assertEquals(listOf(true to false), r.attempts.filter { it.called }.map { it.called to it.verified })
    }

    @Test
    fun aLateCommitInALongFieldIsAcceptedOnTheWindowTails() = runTest {
        // Both reads are capped at the window: a long field compares the window-long tails.
        val f = FakeField(this, text = "x".repeat(500), commit = Rung.LANDS_LATE, lateMs = 800)
        val r = ladder().run(f, "конец")
        assertEquals(InsertMethod.COMMIT_TEXT, r.method)
        assertEquals("x".repeat(500) + "конец", f.text)
    }

    @Test
    fun aSlowCommitWithinTheWindowIsVerifiedByPolling() = runTest {
        val f = FakeField(this, commit = Rung.LANDS_LATE, lateMs = 400)
        val r = ladder().run(f, "slow")
        assertEquals(InsertMethod.COMMIT_TEXT, r.method)
        assertTrue(r.attempts.single().ms in 400..450)
    }

    @Test
    fun allRungsFailingReturnsNoMethod() = runTest {
        val f = FakeField(this, commit = Rung.REPORTS_OK_BUT_NOTHING, paste = Rung.UNAVAILABLE, setText = Rung.REPORTS_OK_BUT_NOTHING)
        val r = ladder().run(f, "lost")
        assertNull(r.method)
        assertEquals(listOf(InsertMethod.COMMIT_TEXT, InsertMethod.SET_TEXT), f.calls)
    }

    @Test
    fun unreadableFieldIsNeverReportedAsInserted() = runTest {
        val f = FakeField(this, readable = false)
        assertNull(ladder().run(f, "x").method)
    }

    @Test
    fun contentEditableNonBreakingSpacesCompareAsSpaces() = runTest {
        val f = FakeField(this, text = "abc", nbsp = true)
        assertEquals(InsertMethod.COMMIT_TEXT, ladder().run(f, " Murmur ok").method)
    }

    @Test
    fun verifiedNeedsAChangeThatEndsWithTheText() {
        assertTrue(InsertionLadder.verified(before = "Hi ", after = "Hi there", text = "there"))
        assertFalse(InsertionLadder.verified(before = "there", after = "there", text = "there")) // already there, nothing changed
        assertFalse(InsertionLadder.verified(before = "", after = "ther", text = "there"))
        assertFalse(InsertionLadder.verified(before = "", after = null, text = "there"))
        assertTrue(InsertionLadder.verified(before = null, after = "there", text = "there"))
        val long = "й".repeat(2000)
        assertTrue(InsertionLadder.verified(before = "", after = long.takeLast(InsertionLadder.readWindow(long)), text = long))
        // Invisible BiDi formatting marks added by RTL/mixed editors do not cause false verification failures.
        assertTrue(InsertionLadder.verified(before = "", after = "\u200Fשלום עולם\u200E", text = "שלום עולם"))
        assertTrue(InsertionLadder.verified(before = "Hi ", after = "Hi \u2067שלום\u2069 \u202Bעולם\u202C", text = "שלום עולם"))
        assertFalse(InsertionLadder.verified(before = "\u200Fשלום", after = "\u200Eשלום", text = "שלום"))
    }

    @Test
    fun rtlCommitWrappedWithBidiMarksDoesNotDoublePaste() = runTest {
        val base = FakeField(this)
        val bidiField = object : LadderField by base {
            override suspend fun beforeCursor(maxChars: Int): String? =
                base.beforeCursor(maxChars)?.let { if (it.isEmpty()) it else "\u200F$it\u2069" }
        }
        val r = ladder().run(bidiField, "שלום עולם")
        assertEquals(InsertMethod.COMMIT_TEXT, r.method)
        assertEquals(listOf(InsertMethod.COMMIT_TEXT), base.calls)
        assertEquals("שלום עולם", base.text)
    }
}
