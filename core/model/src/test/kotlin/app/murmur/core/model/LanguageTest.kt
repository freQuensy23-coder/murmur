package app.murmur.core.model

import org.junit.Assert.assertEquals
import org.junit.Test

class LanguageTest {
    @Test
    fun hebrewConstantAndNormalizeHandleLegacyIwCode() {
        assertEquals(Language("he"), Language.HE)
        assertEquals("he", Language.normalize("iw"))
        assertEquals("he", Language.normalize("IW"))
        assertEquals("he", Language.normalize(" he "))
        assertEquals("ru", Language.normalize("RU"))
        assertEquals("en", Language.normalize("en"))
    }
}
