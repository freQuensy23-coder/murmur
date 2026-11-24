package app.murmur.core.benchmark

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class WerTest {
    @Test
    fun werIgnoresCaseAndPunctuation() {
        assertEquals(0.0, WordErrorRate.wer("Привет, мир!", "привет мир"), 1e-9)
        assertEquals(0.5, WordErrorRate.wer("one two", "one three"), 1e-9)
    }

    @Test
    fun normaliserIsTheSpikes() {
        // Expected values printed by asr_lib.norm of the asr-ocr spike for the same inputs.
        assertEquals(listOf("все", "таки", "ее_дом", "2024"), WordErrorRate.normalize("Всё-таки ЕЁ_дом, 2024!"))
        assertEquals(listOf("все"), WordErrorRate.normalize("вс\u0435\u0308")) // decomposed ё: NFC, then folded
        assertEquals(listOf("привет", "мир", "да"), WordErrorRate.normalize("Привет,\u00a0мир — «да»"))
        assertEquals(0.0, WordErrorRate.wer("всё", "все"), 1e-9)
        assertEquals(0.0, WordErrorRate.wer("Её нет", "ее нет"), 1e-9)
    }

    @Test
    fun validityGateRejectsFastButWrongResults() {
        assertNull(BenchValidity.asr(0.074, cpuWer = 0.076))
        assertEquals("wer_high", BenchValidity.asr(0.9, cpuWer = null))
        assertEquals("wer_above_cpu", BenchValidity.asr(0.15, cpuWer = 0.07))
        assertNull(BenchValidity.llm("Привет, как дела?", guardAccepted = true, stopReason = 0, expectCyrillic = true))
        assertEquals("empty", BenchValidity.llm(" ", true, 0, true))
        assertEquals("early_stop", BenchValidity.llm("Привет", true, 1, true))
        assertEquals("wrong_script", BenchValidity.llm("Hello, how are you?", true, 0, true))
        // Soft prompt fallback (e.g. R7_distractor on both CPU and NPU) does not disqualify a unit unless CPU passed.
        assertNull(BenchValidity.llm("Привет, как дела?", false, 0, true, guardReason = "R7_distractor", cpuGuardAccepted = false))
        assertEquals("guard_fallback", BenchValidity.llm("Привет, как дела?", false, 0, true, guardReason = "R7_distractor", cpuGuardAccepted = true))
        // Hard numeric/corruption guard failures always disqualify the unit.
        assertEquals("guard_fallback", BenchValidity.llm("Привет, как дела?", false, 0, true, guardReason = "R4_loop", cpuGuardAccepted = false))
    }
}
