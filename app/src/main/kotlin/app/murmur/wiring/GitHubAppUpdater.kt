package app.murmur.wiring

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.PackageInstaller
import android.net.Uri
import android.provider.Settings as AndroidSettings
import android.util.Log
import app.murmur.BuildConfig
import app.murmur.feature.settings.AppUpdateState
import app.murmur.feature.settings.AppUpdater
import app.murmur.feature.settings.AppVersionComparator
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * Manual GitHub Releases updater triggered only when the user taps "Update" in Settings (release builds only).
 * Checks the latest release, selects the signed release APK, downloads and verifies SHA-256 when provided, and hands
 * off to Android's [PackageInstaller] confirmation popup.
 */
class GitHubAppUpdater(
    private val context: Context,
    private val scope: CoroutineScope,
    private val currentVersion: String = BuildConfig.VERSION_NAME,
    private val io: CoroutineDispatcher = Dispatchers.IO,
) : AppUpdater {

    private val _state = MutableStateFlow<AppUpdateState>(AppUpdateState.Idle)
    override val state: StateFlow<AppUpdateState> = _state.asStateFlow()

    @Volatile private var activeJob: Job? = null

    init {
        AppUpdateInstallReceiver.onResult = ::onInstallResult
    }

    override fun checkAndUpdate() {
        if (activeJob?.isActive == true) return
        activeJob = scope.launch {
            runCatching { performUpdate() }.onFailure { e ->
                if (e is CancellationException) throw e
                Log.w(TAG, "update_failed ${e.javaClass.simpleName}")
                _state.value = AppUpdateState.Failed(AppUpdateState.Failed.Failure.NETWORK)
            }
        }
    }

    private suspend fun performUpdate() {
        _state.value = AppUpdateState.Checking
        val release = withContext(io) { fetchLatestRelease() } ?: run {
            _state.value = AppUpdateState.Failed(AppUpdateState.Failed.Failure.NETWORK)
            return
        }
        if (!AppVersionComparator.isNewer(release.tagName, currentVersion)) {
            _state.value = AppUpdateState.UpToDate(currentVersion)
            return
        }
        val asset = selectApkAsset(release.assets) ?: run {
            _state.value = AppUpdateState.Failed(AppUpdateState.Failed.Failure.NO_APK)
            return
        }
        if (!asset.downloadUrl.startsWith("https://")) {
            _state.value = AppUpdateState.Failed(AppUpdateState.Failed.Failure.VERIFY)
            return
        }

        val apkFile = withContext(io) { downloadOrReuse(release.tagName, asset) } ?: return

        if (!context.packageManager.canRequestPackageInstalls()) {
            _state.value = AppUpdateState.PermissionRequired
            openUnknownSourcesSettings()
            return
        }

        _state.value = AppUpdateState.Installing(release.tagName)
        val committed = withContext(io) { commitInstallSession(apkFile) }
        if (!committed) {
            _state.value = AppUpdateState.Failed(AppUpdateState.Failed.Failure.INSTALL)
        }
    }

    private fun fetchLatestRelease(): ReleaseInfo? {
        val conn = (URL(LATEST_RELEASE_URL).openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = API_READ_TIMEOUT_MS
            setRequestProperty("Accept", "application/vnd.github+json")
            setRequestProperty("User-Agent", "Murmur-Android/$currentVersion")
        }
        return try {
            conn.connect()
            if (conn.url.protocol != "https" || conn.responseCode != HttpURLConnection.HTTP_OK) return null
            val body = conn.inputStream.bufferedReader().use { it.readText() }
            parseReleaseJson(body)
        } catch (e: Exception) {
            Log.w(TAG, "release_fetch_error ${e.javaClass.simpleName}")
            null
        } finally {
            conn.disconnect()
        }
    }

    private fun downloadOrReuse(version: String, asset: ReleaseAsset): File? {
        val dir = File(context.cacheDir, UPDATES_DIR).apply { mkdirs() }
        val safeName = asset.name.replace(Regex("[^A-Za-z0-9._-]"), "_")
        val target = File(dir, safeName)
        if (target.isFile && asset.sha256 != null && sha256Hex(target) == asset.sha256) {
            return target
        }

        // Remove older cached APKs / partial downloads before fetching the new one.
        dir.listFiles()?.forEach { if (it.name != safeName) it.delete() }
        val part = File(dir, "$safeName.part")
        part.delete()

        _state.value = AppUpdateState.Downloading(version, 0)
        val conn = (URL(asset.downloadUrl).openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = DOWNLOAD_READ_TIMEOUT_MS
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "Murmur-Android/$currentVersion")
        }
        return try {
            conn.connect()
            if (conn.url.protocol != "https" || conn.responseCode != HttpURLConnection.HTTP_OK) {
                _state.value = AppUpdateState.Failed(AppUpdateState.Failed.Failure.NETWORK)
                return null
            }
            val total = conn.contentLengthLong.takeIf { it > 0L } ?: asset.size.takeIf { it > 0L }
            val digest = MessageDigest.getInstance("SHA-256")
            var downloaded = 0L
            var lastPct = -1
            conn.inputStream.use { input ->
                FileOutputStream(part).use { out ->
                    val buf = ByteArray(32 * 1024)
                    while (true) {
                        val read = input.read(buf)
                        if (read < 0) break
                        out.write(buf, 0, read)
                        digest.update(buf, 0, read)
                        downloaded += read
                        val pct = total?.let { ((downloaded * 100L) / it).toInt().coerceIn(0, 100) }
                        if (pct != lastPct) {
                            lastPct = pct ?: -1
                            _state.value = AppUpdateState.Downloading(version, pct)
                        }
                    }
                    out.fd.sync()
                }
            }
            val actualSha = digest.digest().joinToString("") { "%02x".format(it) }
            if (asset.sha256 != null && !actualSha.equals(asset.sha256, ignoreCase = true)) {
                Log.w(TAG, "apk_sha256_mismatch")
                part.delete()
                _state.value = AppUpdateState.Failed(AppUpdateState.Failed.Failure.VERIFY)
                return null
            }
            if (target.exists()) target.delete()
            if (!part.renameTo(target)) {
                part.copyTo(target, overwrite = true)
                part.delete()
            }
            target
        } catch (e: CancellationException) {
            part.delete()
            throw e
        } catch (e: Exception) {
            Log.w(TAG, "apk_download_error ${e.javaClass.simpleName}")
            part.delete()
            _state.value = AppUpdateState.Failed(AppUpdateState.Failed.Failure.NETWORK)
            null
        } finally {
            conn.disconnect()
        }
    }

    private fun commitInstallSession(apkFile: File): Boolean {
        val installer = context.packageManager.packageInstaller
        val params = PackageInstaller.SessionParams(PackageInstaller.SessionParams.MODE_FULL_INSTALL).apply {
            setAppPackageName(context.packageName)
            setSize(apkFile.length())
        }
        var session: PackageInstaller.Session? = null
        var committed = false
        return try {
            val sessionId = installer.createSession(params)
            session = installer.openSession(sessionId)
            session.openWrite("murmur-update.apk", 0, apkFile.length()).use { out ->
                FileInputStream(apkFile).use { input ->
                    input.copyTo(out, bufferSize = 32 * 1024)
                }
                session.fsync(out)
            }
            val intent = Intent(context, AppUpdateInstallReceiver::class.java).apply {
                action = ACTION_INSTALL_RESULT
            }
            val pending = PendingIntent.getBroadcast(
                context,
                sessionId,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_MUTABLE,
            )
            session.commit(pending.intentSender)
            committed = true
            true
        } catch (e: Exception) {
            Log.w(TAG, "install_session_error ${e.javaClass.simpleName}")
            runCatching { session?.abandon() }
            false
        } finally {
            if (!committed) runCatching { session?.abandon() }
            runCatching { session?.close() }
        }
    }

    private fun openUnknownSourcesSettings() {
        val intent = Intent(
            AndroidSettings.ACTION_MANAGE_UNKNOWN_APP_SOURCES,
            Uri.parse("package:${context.packageName}"),
        ).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        runCatching { context.startActivity(intent) }
            .onFailure { Log.w(TAG, "unknown_sources_open_failed ${it.javaClass.simpleName}") }
    }

    private fun onInstallResult(status: Int) {
        _state.value = when (status) {
            PackageInstaller.STATUS_PENDING_USER_ACTION,
            PackageInstaller.STATUS_SUCCESS,
            PackageInstaller.STATUS_FAILURE_ABORTED -> AppUpdateState.Idle
            else -> {
                Log.w(TAG, "install_failed status=$status")
                AppUpdateState.Failed(AppUpdateState.Failed.Failure.INSTALL)
            }
        }
    }

    internal data class ReleaseAsset(val name: String, val downloadUrl: String, val size: Long, val sha256: String?)
    internal data class ReleaseInfo(val tagName: String, val assets: List<ReleaseAsset>)

    internal companion object {
        const val TAG = "MurmurUpdater"
        const val ACTION_INSTALL_RESULT = "app.murmur.action.APP_UPDATE_INSTALL_RESULT"
        private const val LATEST_RELEASE_URL = "https://api.github.com/repos/freQuensy23-coder/murmur/releases/latest"
        private const val UPDATES_DIR = "updates"
        private const val CONNECT_TIMEOUT_MS = 15_000
        private const val API_READ_TIMEOUT_MS = 30_000
        private const val DOWNLOAD_READ_TIMEOUT_MS = 60_000

        fun parseReleaseJson(json: String): ReleaseInfo? {
            val root = runCatching { Json.parseToJsonElement(json).jsonObject }.getOrNull() ?: return null
            val tag = root["tag_name"]?.jsonPrimitive?.contentOrNull?.takeIf { it.isNotBlank() } ?: return null
            val arr = root["assets"]?.jsonArray ?: return ReleaseInfo(tag, emptyList())
            val assets = buildList {
                for (el in arr) {
                    val obj = runCatching { el.jsonObject }.getOrNull() ?: continue
                    val name = obj["name"]?.jsonPrimitive?.contentOrNull?.takeIf { it.isNotBlank() } ?: continue
                    val url = obj["browser_download_url"]?.jsonPrimitive?.contentOrNull?.takeIf { it.isNotBlank() } ?: continue
                    val size = obj["size"]?.jsonPrimitive?.longOrNull ?: -1L
                    val digest = parseSha256Digest(obj["digest"]?.jsonPrimitive?.contentOrNull)
                    add(ReleaseAsset(name, url, size, digest))
                }
            }
            return ReleaseInfo(tag, assets)
        }

        fun selectApkAsset(assets: List<ReleaseAsset>): ReleaseAsset? =
            assets.firstOrNull {
                it.name.endsWith("-release.apk", ignoreCase = true) &&
                    !it.name.endsWith("-unsigned.apk", ignoreCase = true)
            }

        private fun parseSha256Digest(raw: String?): String? {
            if (raw.isNullOrBlank()) return null
            val hex = raw.trim().removePrefix("sha256:")
            return hex.takeIf { it.length == 64 && it.all { c -> c in '0'..'9' || c in 'a'..'f' || c in 'A'..'F' } }?.lowercase()
        }

        private fun sha256Hex(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            FileInputStream(file).use { input ->
                val buf = ByteArray(32 * 1024)
                while (true) {
                    val read = input.read(buf)
                    if (read < 0) break
                    digest.update(buf, 0, read)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }
    }
}

/** Receives [PackageInstaller] session commit results and opens the system install confirmation dialog. */
class AppUpdateInstallReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != GitHubAppUpdater.ACTION_INSTALL_RESULT) return
        val status = intent.getIntExtra(PackageInstaller.EXTRA_STATUS, PackageInstaller.STATUS_FAILURE)
        if (status == PackageInstaller.STATUS_PENDING_USER_ACTION) {
            val confirm = intent.getParcelableExtra(Intent.EXTRA_INTENT, Intent::class.java)
            if (confirm != null) {
                runCatching {
                    context.startActivity(confirm.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                    onResult?.invoke(status)
                }.onFailure {
                    Log.w(GitHubAppUpdater.TAG, "confirm_activity_failed ${it.javaClass.simpleName}")
                    onResult?.invoke(PackageInstaller.STATUS_FAILURE)
                }
            } else {
                onResult?.invoke(PackageInstaller.STATUS_FAILURE)
            }
        } else {
            onResult?.invoke(status)
        }
    }

    internal companion object {
        @Volatile var onResult: ((Int) -> Unit)? = null
    }
}
