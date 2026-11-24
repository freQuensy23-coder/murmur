package app.murmur.wiring

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class GitHubAppUpdaterTest {
    @Test
    fun parsesReleaseJsonAndSelectsMatchingBuildType() {
        val json = """
            {
              "tag_name": "v0.2.2",
              "assets": [
                {
                  "name": "murmur-v0.2.2-debug.apk",
                  "browser_download_url": "https://github.com/freQuensy23-coder/murmur/releases/download/v0.2.2/murmur-v0.2.2-debug.apk",
                  "size": 89974100,
                  "digest": "sha256:88a3a78bf8fa1b90e329e44751bbc7d2ab55f9be5e530946c05827332242c5b6"
                },
                {
                  "name": "murmur-v0.2.2-release-unsigned.apk",
                  "browser_download_url": "https://github.com/freQuensy23-coder/murmur/releases/download/v0.2.2/murmur-v0.2.2-release-unsigned.apk",
                  "size": 53000000
                },
                {
                  "name": "murmur-v0.2.2-release.apk",
                  "browser_download_url": "https://github.com/freQuensy23-coder/murmur/releases/download/v0.2.2/murmur-v0.2.2-release.apk",
                  "size": 53311651,
                  "digest": "sha256:575d5fa145c32bac456f9b6ab6a8aa5d011e31eac6f706699c8c2cb820e8040e"
                }
              ]
            }
        """.trimIndent()

        val release = GitHubAppUpdater.parseReleaseJson(json)
        assertNotNull(release)
        assertEquals("v0.2.2", release!!.tagName)

        val relAsset = GitHubAppUpdater.selectApkAsset(release.assets, isDebug = false)
        assertEquals("murmur-v0.2.2-release.apk", relAsset?.name)
        assertEquals("575d5fa145c32bac456f9b6ab6a8aa5d011e31eac6f706699c8c2cb820e8040e", relAsset?.sha256)

        val dbgAsset = GitHubAppUpdater.selectApkAsset(release.assets, isDebug = true)
        assertEquals("murmur-v0.2.2-debug.apk", dbgAsset?.name)
        assertEquals("88a3a78bf8fa1b90e329e44751bbc7d2ab55f9be5e530946c05827332242c5b6", dbgAsset?.sha256)
    }

    @Test
    fun rejectsUnsignedReleaseApk() {
        val assets = listOf(
            GitHubAppUpdater.ReleaseAsset(
                name = "murmur-v0.2.2-release-unsigned.apk",
                downloadUrl = "https://example.com/unsigned.apk",
                size = 1000L,
                sha256 = null,
            ),
        )
        assertNull(GitHubAppUpdater.selectApkAsset(assets, isDebug = false))
    }
}
