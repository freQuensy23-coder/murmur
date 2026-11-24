package app.murmur.feature.settings

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AppVersionComparatorTest {
    @Test
    fun newerVersionsAreDetected() {
        assertTrue(AppVersionComparator.isNewer("v0.2.1", "0.2.0"))
        assertTrue(AppVersionComparator.isNewer("0.2.1", "0.2.0"))
        assertTrue(AppVersionComparator.isNewer("v0.10.0", "0.2.9"))
        assertTrue(AppVersionComparator.isNewer("v1.0.0", "0.99.99"))
        assertTrue(AppVersionComparator.isNewer("v0.2.1", "0.2.1-rc1"))
    }

    @Test
    fun equalOrOlderVersionsAreNotNewer() {
        assertFalse(AppVersionComparator.isNewer("v0.2.1", "0.2.1"))
        assertFalse(AppVersionComparator.isNewer("v0.2.0", "0.2.1"))
        assertFalse(AppVersionComparator.isNewer("v0.2.1-beta1", "0.2.1"))
        assertFalse(AppVersionComparator.isNewer("test-56e9d4b", "0.2.1"))
        assertFalse(AppVersionComparator.isNewer("", "0.2.1"))
        assertFalse(AppVersionComparator.isNewer("v0.2.2", "invalid"))
    }
}
