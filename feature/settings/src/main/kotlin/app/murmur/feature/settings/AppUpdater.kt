package app.murmur.feature.settings

import kotlinx.coroutines.flow.StateFlow

/** State of a manual "Update" check/download/install triggered from Settings. */
sealed interface AppUpdateState {
    data object Idle : AppUpdateState
    data object Checking : AppUpdateState
    data class Downloading(val version: String, val progressPercent: Int?) : AppUpdateState
    data class Installing(val version: String) : AppUpdateState
    data class UpToDate(val currentVersion: String) : AppUpdateState
    data object PermissionRequired : AppUpdateState
    data class Failed(val reason: Failure) : AppUpdateState {
        enum class Failure { NETWORK, NO_APK, VERIFY, INSTALL }
    }
}

/** Manual GitHub Releases updater wired into Settings; never polls in the background. */
interface AppUpdater {
    val state: StateFlow<AppUpdateState>
    fun checkAndUpdate()
}

/**
 * Compares release tags / version strings like "v0.2.1" and "0.2.0".
 * Unparseable tags return false for [isNewer] rather than throwing or falsely triggering updates.
 */
object AppVersionComparator {
    private data class Parsed(val parts: List<Int>, val preRelease: Boolean)

    fun isNewer(candidate: String, current: String): Boolean {
        val c = parse(candidate) ?: return false
        val cur = parse(current) ?: return false
        val n = maxOf(c.parts.size, cur.parts.size)
        for (i in 0 until n) {
            val a = c.parts.getOrElse(i) { 0 }
            val b = cur.parts.getOrElse(i) { 0 }
            if (a != b) return a > b
        }
        // Same numeric version: a stable release is newer than a pre-release (e.g. 0.2.1 > 0.2.1-rc1).
        return !c.preRelease && cur.preRelease
    }

    private fun parse(raw: String): Parsed? {
        val trimmed = raw.trim().removePrefix("v").removePrefix("V")
        if (trimmed.isEmpty()) return null
        val withoutBuild = trimmed.substringBefore('+')
        val dashIdx = withoutBuild.indexOf('-')
        val core = if (dashIdx >= 0) withoutBuild.substring(0, dashIdx) else withoutBuild
        val preRelease = dashIdx >= 0
        val segments = core.split('.')
        if (segments.isEmpty()) return null
        val nums = segments.map { it.toIntOrNull()?.takeIf { v -> v >= 0 } ?: return null }
        return Parsed(nums, preRelease)
    }
}
