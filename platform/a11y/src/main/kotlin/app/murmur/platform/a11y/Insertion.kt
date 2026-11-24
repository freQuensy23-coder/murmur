package app.murmur.platform.a11y

import app.murmur.core.store.InsertMethod
import kotlinx.coroutines.delay

/**
 * One focused field as the insertion ladder sees it (ARCHITECTURE 11). The Android implementation wraps the
 * accessibility IME's input connection and the focused node; tests use a fake.
 */
interface LadderField {
    /**
     * Up to [maxChars] characters before the caret, read through the accessibility IME's getSurroundingText (node text
     * is stale on WebView, platform spike e). Only when the IME has no connection to the field does the implementation
     * fall back to the refreshed node text. null = unreadable.
     */
    suspend fun beforeCursor(maxChars: Int): String?
    /** AccessibilityInputConnection.commitText; false when the IME has no connection to this field. */
    fun commitText(text: String): Boolean
    /** Clipboard = [text], then ACTION_PASTE; false when the node refused the action. */
    fun paste(text: String): Boolean
    /** ACTION_SET_TEXT with the text spliced in at the selection, then ACTION_SET_SELECTION after it. */
    fun setText(text: String): Boolean
}

data class LadderAttempt(val method: InsertMethod, val called: Boolean, val verified: Boolean, val ms: Long)

data class LadderResult(val method: InsertMethod?, val attempts: List<LadderAttempt>)

/**
 * commitText -> ACTION_PASTE -> ACTION_SET_TEXT; after each, read back the text before the caret and accept the first
 * method whose result is verified. Before a later step the field is read again: if an earlier step landed late (an
 * async commit), that step wins and nothing is inserted twice.
 *
 * A commitText that returned true is a commit the IME accepted; it reaches the field asynchronously, and under load the
 * text before the caret can show it later than [verifyTimeoutMs] (WP-20's emulator runs: COMMIT_TEXT landed, the read
 * lagged past 600 ms, PASTE inserted the text a second time). So after a true but unverified commit the field is read
 * again for up to [commitGraceMs] (ADR-086 item 5): COMMIT_TEXT is accepted as soon as the text before the caret is
 * [verified]; the ladder escalates to ACTION_PASTE only when the field stays readable and the text never showed up. An
 * unreadable field after a true commit is never escalated: the result is "not inserted" and the orchestrator leaves the
 * text on the clipboard (Copied), because a second method could only be checked by reading the same field.
 *
 * Every wait ends inside one budget, [budgetMs] from the first read: the orchestrator moves the dictation to Copied when
 * its delivery timer (DictationConfig.deliveryTimerMs(): 2 s, never below deliveryMinMs = 1.9 s at any timeout scale)
 * fires, and a rung that lands after that would insert text under a pill that says Copied. A rung waits at most [verifyTimeoutMs] and leaves [lateRungMs] for each rung after it;
 * the grace period also leaves [lateRungMs] for paste and set-text; no rung starts once the budget is spent.
 */
class InsertionLadder(
    private val now: () -> Long,
    private val verifyTimeoutMs: Long = VERIFY_TIMEOUT_MS,
    private val pollMs: Long = POLL_MS,
    private val commitGraceMs: Long = COMMIT_GRACE_MS,
    private val lateRungMs: Long = LATE_RUNG_MS,
    private val budgetMs: Long = LADDER_BUDGET_MS,
) {
    suspend fun run(field: LadderField, text: String): LadderResult {
        val end = now() + budgetMs
        val attempts = mutableListOf<LadderAttempt>()
        val window = readWindow(text)
        val original = field.beforeCursor(window)
        val steps = listOf<Pair<InsertMethod, (String) -> Boolean>>(
            InsertMethod.COMMIT_TEXT to field::commitText,
            InsertMethod.PASTE to field::paste,
            InsertMethod.SET_TEXT to field::setText,
        )
        for ((i, step) in steps.withIndex()) {
            val (method, action) = step
            val lastCalled = attempts.lastOrNull { it.called }
            if (lastCalled != null && verified(original, field.beforeCursor(window), text)) {
                return LadderResult(lastCalled.method, attempts)
            }
            if (now() >= end) break
            val rungsAfter = steps.size - 1 - i
            val t0 = now()
            val called = runCatching { action(text) }.getOrDefault(false)
            val ok = called && awaitVerified(field, original, text, window, minOf(t0 + verifyTimeoutMs, end - rungsAfter * lateRungMs))
            if (ok) {
                attempts += LadderAttempt(method, called, true, now() - t0)
                return LadderResult(method, attempts)
            }
            if (method == InsertMethod.COMMIT_TEXT && called) {
                val late = awaitLateCommit(field, original, text, window, minOf(now() + commitGraceMs, end - rungsAfter * lateRungMs))
                attempts += LadderAttempt(method, called = true, verified = late == LateCommit.LANDED, ms = now() - t0)
                when (late) {
                    LateCommit.LANDED -> return LadderResult(method, attempts)
                    LateCommit.UNREADABLE -> return LadderResult(null, attempts)
                    LateCommit.ABSENT -> continue
                }
            }
            attempts += LadderAttempt(method, called, false, now() - t0)
        }
        return LadderResult(null, attempts)
    }

    private enum class LateCommit { LANDED, UNREADABLE, ABSENT }

    /** ADR-086 item 5: the grace period after a true, unverified commitText, until [deadline]. */
    private suspend fun awaitLateCommit(field: LadderField, original: String?, text: String, window: Int, deadline: Long): LateCommit {
        while (true) {
            val after = field.beforeCursor(window) ?: return LateCommit.UNREADABLE
            if (verified(original, after, text)) return LateCommit.LANDED
            if (now() >= deadline) return LateCommit.ABSENT
            delay(pollMs)
        }
    }

    /** Reads until the text is [verified] or [deadline] passed; always reads at least once. */
    private suspend fun awaitVerified(field: LadderField, original: String?, text: String, window: Int, deadline: Long): Boolean {
        while (true) {
            if (verified(original, field.beforeCursor(window), text)) return true
            if (now() >= deadline) return false
            delay(pollMs)
        }
    }

    companion object {
        /** Long enough for an async commit or a WebView paste to arrive (spike read back after 150 ms). */
        const val VERIFY_TIMEOUT_MS = 600L
        /**
         * The extra wait for a true commitText (ADR-086 item 5): a commit is accepted up to 1 s after it was sent (the
         * slowest verified commit in WP-20's emulator logs took 137 ms).
         */
        const val COMMIT_GRACE_MS = 400L
        /** What a rung leaves for each later rung, and the shortest window paste and set-text get after a grace period. */
        const val LATE_RUNG_MS = 250L
        /**
         * Every rung, read and wait of one run ends inside this: 500 ms of the orchestrator's 2 s delivery timeout
         * (DictationConfig.deliveryTimerMs(), never below 1.9 s; :app TimeoutFloorsTest) stay for re-resolving the
         * target, the clipboard copy and the binder reads.
         */
        const val LADDER_BUDGET_MS = 1_500L
        const val POLL_MS = 50L
        /** Only the tail of a long text is compared: bounded binder payload, same certainty. */
        const val COMPARE_TAIL = 256
        private const val MARGIN = 32

        fun readWindow(text: String) = minOf(text.length, COMPARE_TAIL) + MARGIN

        /**
         * The text before the caret now ends with [text] and differs from what was there before: the insertion landed at
         * the caret. Non-breaking spaces (contenteditable) compare as spaces, and invisible Unicode BiDi formatting
         * marks injected by RTL editors are stripped before comparison.
         */
        fun verified(before: String?, after: String?, text: String): Boolean {
            if (after == null) return false
            val a = norm(after)
            if (!a.endsWith(norm(text).takeLast(COMPARE_TAIL))) return false
            return before == null || norm(before) != a
        }

        private val BIDI_FORMAT = Regex("[\u200E\u200F\u202A-\u202E\u2066-\u2069]")

        private fun norm(s: String) = s.replace('\u00A0', ' ').replace(BIDI_FORMAT, "")
    }
}
