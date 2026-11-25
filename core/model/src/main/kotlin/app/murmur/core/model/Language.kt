package app.murmur.core.model

import kotlinx.serialization.Serializable

/** ISO 639-1 code, lowercase ("ru", "en", "he"). */
@Serializable
@JvmInline
value class Language(val code: String) {
    init {
        require(code.length in 2..8 && code == code.lowercase()) { "bad language code: $code" }
    }

    override fun toString() = code

    companion object {
        val RU = Language("ru")
        val EN = Language("en")
        val HE = Language("he")

        /** Normalizes legacy Android/Java ISO 639-1 codes (e.g. "iw" -> "he"). */
        fun normalize(code: String): String = when (val c = code.trim().lowercase()) {
            "iw" -> "he"
            else -> c
        }
    }
}
