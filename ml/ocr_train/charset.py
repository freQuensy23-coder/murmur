"""Character set, dictionary builder, and visual-order BiDi helpers for 3-language (ru + en + he) OCR.

CTC class layout (exact PaddleOCR / ml/ocr/pipeline.py contract):
    0        : CTC blank ("")
    1..850   : Original cyrillic_PP-OCRv5_mobile_rec dictionary (ml/ocr/golden/rec_keys.txt), IDENTICAL indices
    851..881 : Appended Hebrew characters (27 letters א-ת incl. sofits ךםןףץ + ׳ ״ ₪ ־)
    882      : Space (" ")  [transferred from stock index 851 during weight transfer]

Total dictionary entries: 881
Total CTC output classes: 883
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import glob
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

# FriBidiParType exact bitmask constants (fribidi-bidi-types.h):
#   FRIBIDI_MASK_RTL     = 0x00000001
#   FRIBIDI_MASK_STRONG  = 0x00000010
#   FRIBIDI_MASK_NEUTRAL = 0x00000040
#   FRIBIDI_MASK_LETTER  = 0x00000100
_FRIBIDI_PAR_LTR = 0x00000110  # FRIBIDI_MASK_STRONG | FRIBIDI_MASK_LETTER
_FRIBIDI_PAR_RTL = 0x00000111  # FRIBIDI_MASK_STRONG | FRIBIDI_MASK_LETTER | FRIBIDI_MASK_RTL
_FRIBIDI_PAR_ON = 0x00000040   # FRIBIDI_MASK_NEUTRAL (UAX #9 P2/P3 auto-detect)


def _load_fribidi():
    """Load libfribidi (same C library Pillow raqm uses, including UAX #9 Paired Bracket Algorithm)."""
    cands: list[str] = []
    found = ctypes.util.find_library("fribidi")
    if found:
        cands.append(found)
    try:
        import PIL

        pil_dir = os.path.dirname(PIL.__file__)
        for pat in (
            "*fribidi*",
            ".dylibs/*fribidi*",
            "../Pillow.libs/*fribidi*",
            "../pillow.libs/*fribidi*",
        ):
            cands.extend(sorted(glob.glob(os.path.join(pil_dir, pat))))
    except Exception:
        pass
    cands.extend([
        "libfribidi.so.0",
        "libfribidi.so",
        "/usr/lib/x86_64-linux-gnu/libfribidi.so.0",
        "/usr/lib/aarch64-linux-gnu/libfribidi.so.0",
        "/opt/homebrew/lib/libfribidi.0.dylib",
        "/opt/homebrew/lib/libfribidi.dylib",
        "/usr/local/lib/libfribidi.0.dylib",
        "/usr/local/lib/libfribidi.dylib",
    ])
    for cand in cands:
        try:
            lib = ctypes.CDLL(cand)
            fn = lib.fribidi_log2vis
            fn.argtypes = [
                ctypes.POINTER(ctypes.c_uint32),  # const FriBidiChar *str (uint32_t)
                ctypes.c_int,                     # const FriBidiStrIndex len (int)
                ctypes.POINTER(ctypes.c_uint32),  # FriBidiParType *pbase_dir (uint32_t)
                ctypes.POINTER(ctypes.c_uint32),  # FriBidiChar *visual_str (uint32_t)
                ctypes.c_void_p,                  # FriBidiStrIndex *positions_L_to_V
                ctypes.c_void_p,                  # FriBidiStrIndex *positions_V_to_L
                ctypes.c_void_p,                  # FriBidiLevel *embedding_levels
            ]
            # FriBidiLevel is `typedef signed char` (must be c_int8, NOT c_int, on x86_64 Linux)
            fn.restype = ctypes.c_int8
            return fn
        except (OSError, AttributeError):
            continue
    return None


_FRIBIDI_LOG2VIS = _load_fribidi()
USING_FRIBIDI = _FRIBIDI_LOG2VIS is not None

try:
    from bidi.algorithm import get_display as _bidi_get_display  # type: ignore

    USING_PYTHON_BIDI = True
except ImportError:  # pragma: no cover - fallback for minimal local test envs without python-bidi
    _bidi_get_display = None
    USING_PYTHON_BIDI = False

BIDI_BACKEND: str = "fribidi" if USING_FRIBIDI else ("python-bidi" if USING_PYTHON_BIDI else "fallback")

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_REC_KEYS = REPO_ROOT / "ml" / "ocr" / "golden" / "rec_keys.txt"
DEFAULT_DICT_PATH = Path(__file__).resolve().parent / "dict_ru_en_he.txt"

# 27 Hebrew letters (U+05D0..U+05EA, includes all 5 final/sofit forms: ך ם ן ף ץ)
HEBREW_LETTERS: tuple[str, ...] = tuple(chr(cp) for cp in range(0x05D0, 0x05EB))
# Hebrew punctuation & currency: geresh, gershayim, shekel, maqaf
HEBREW_PUNCT: tuple[str, ...] = (
    "\u05F3",  # ׳ GERESH
    "\u05F4",  # ״ GERSHAYIM
    "\u20AA",  # ₪ NEW SHEQEL SIGN
    "\u05BE",  # ־ HEBREW PUNCTUATION MAQAF
)
HEBREW_EXTRA_CHARS: tuple[str, ...] = HEBREW_LETTERS + HEBREW_PUNCT

EXPECTED_BASE_LEN = 850
EXPECTED_TOTAL_LEN = EXPECTED_BASE_LEN + len(HEBREW_EXTRA_CHARS)  # 881

# Hebrew combining marks (cantillation + niqqud), stripped for modern UI OCR
_NIKKUD_RE = re.compile(r"[\u0591-\u05BD\u05BF\u05C1\u05C2\u05C4\u05C5\u05C7]")
# Invisible BiDi control / zero-width chars
_INVISIBLE_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF]")

# Mirrored paired punctuation for RTL visual runs (UAX #9 Bidi_Mirrored subset)
_MIRROR_MAP = str.maketrans(
    "()[]{}<>«»‹›",
    ")([]}{><»«›‹",
)

# YAML scalar chars that need single-quoting to round-trip cleanly through both PyYAML and Kotlin OcrDictionary
_YAML_SPECIAL = set("!\"#%&'*,-:<>?@[]`{}|~\\ ")


def is_hebrew_char(ch: str) -> bool:
    cp = ord(ch)
    return 0x0590 <= cp <= 0x05FF


def has_hebrew(text: str) -> bool:
    return any(0x05D0 <= ord(c) <= 0x05EA for c in text)


def strip_nikkud(text: str) -> str:
    """Remove Hebrew niqqud/cantillation marks and invisible BiDi control characters."""
    text = unicodedata.normalize("NFC", text)
    text = _NIKKUD_RE.sub("", text)
    text = _INVISIBLE_RE.sub("", text)
    return text


def detect_base_dir(text: str) -> Literal["L", "R"]:
    """UAX #9 P2/P3 first-strong-character paragraph base direction."""
    for ch in text:
        bd = unicodedata.bidirectional(ch)
        if bd in ("R", "AL"):
            return "R"
        if bd == "L":
            return "L"
    return "L"


def _fallback_get_display(text: str, base_dir: Literal["L", "R"] | None = None) -> str:
    """Clean UAX #9 subset fallback for Hebrew + Latin + Cyrillic + digits + punctuation.

    Used only when `python-bidi` is not installed. When `python-bidi` IS installed (all training
    environments), `bidi.algorithm.get_display` is called directly.
    """
    if not text:
        return text
    bdir = base_dir if base_dir in ("L", "R") else detect_base_dir(text)
    base_level = 1 if bdir == "R" else 0

    n = len(text)
    types = [unicodedata.bidirectional(c) for c in text]
    # Collapse Arabic/Hebrew R/AL -> R, LRE/etc -> ON
    for i, t in enumerate(types):
        if t == "AL":
            types[i] = "R"
        elif t not in ("L", "R", "EN", "ES", "ET", "AN", "CS", "NSM", "WS", "ON", "B", "S"):
            types[i] = "ON"

    # W4: single ES/CS between EN -> EN
    for i in range(1, n - 1):
        if types[i] in ("ES", "CS") and types[i - 1] == "EN" and types[i + 1] == "EN":
            types[i] = "EN"
    # W5: ET adjacent to EN -> EN
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if types[i] == "ET":
                if (i > 0 and types[i - 1] == "EN") or (i + 1 < n and types[i + 1] == "EN"):
                    types[i] = "EN"
                    changed = True
    # W6: remaining ES/ET/CS -> ON
    for i in range(n):
        if types[i] in ("ES", "ET", "CS", "NSM"):
            types[i] = "ON"
    # W7: EN after strong L (no intervening R) -> L (simplified: keep EN at level 2 under R, 0 under L)

    # N1 / N2: resolve neutrals (WS, ON) between strong types (treat EN as R for N1 per UAX#9)
    def _strong_for_neutral(t: str) -> str:
        if t == "L":
            return "L"
        if t in ("R", "EN", "AN"):
            return "R"
        return ""

    i = 0
    while i < n:
        if _strong_for_neutral(types[i]):
            i += 1
            continue
        j = i
        while j < n and not _strong_for_neutral(types[j]):
            j += 1
        left_s = _strong_for_neutral(types[i - 1]) if i > 0 else bdir
        right_s = _strong_for_neutral(types[j]) if j < n else bdir
        resolved = left_s if left_s == right_s else bdir
        for k in range(i, j):
            types[k] = resolved
        i = j

    # I1 / I2: assign embedding levels
    levels = [base_level] * n
    for i, t in enumerate(types):
        if base_level == 0:
            if t == "R":
                levels[i] = 1
            elif t in ("EN", "AN"):
                levels[i] = 2
        else:
            if t in ("L", "EN", "AN"):
                levels[i] = 2
            else:
                levels[i] = 1

    # L1: trailing whitespace resets to paragraph embedding level
    k = n - 1
    while k >= 0 and text[k].isspace():
        levels[k] = base_level
        k -= 1

    # L4: mirror characters at odd levels
    chars = [c.translate(_MIRROR_MAP) if (levels[i] % 2 == 1) else c for i, c in enumerate(text)]

    # L2: reverse contiguous sequences from max_level down to 1
    max_lvl = max(levels)
    for lvl in range(max_lvl, 0, -1):
        i = 0
        while i < n:
            if levels[i] >= lvl:
                j = i
                while j < n and levels[j] >= lvl:
                    j += 1
                chars[i:j] = reversed(chars[i:j])
                levels[i:j] = reversed(levels[i:j])
                i = j
            else:
                i += 1

    return "".join(chars)


def _fribidi_get_display(text: str, base_dir: Literal["L", "R"] | None = None) -> str:
    """Exact UAX #9 + Paired Bracket Algorithm visual ordering via libfribidi (`fribidi_log2vis`)."""
    assert _FRIBIDI_LOG2VIS is not None
    if not text:
        return text
    if base_dir == "L":
        pbase = ctypes.c_uint32(_FRIBIDI_PAR_LTR)
    elif base_dir == "R":
        pbase = ctypes.c_uint32(_FRIBIDI_PAR_RTL)
    else:
        pbase = ctypes.c_uint32(_FRIBIDI_PAR_ON)
    codes = [ord(c) for c in text]
    n = len(codes)
    in_buf = (ctypes.c_uint32 * n)(*codes)
    out_buf = (ctypes.c_uint32 * n)()
    lvl = _FRIBIDI_LOG2VIS(in_buf, n, ctypes.byref(pbase), out_buf, None, None, None)
    if lvl <= 0:
        raise RuntimeError(f"fribidi_log2vis failed (ret={lvl}) on {text!r}")
    return "".join(chr(out_buf[i]) for i in range(n))


def to_visual_order(text: str, base_dir: Literal["L", "R"] | None = None) -> str:
    """Convert logical Unicode text into left-to-right visual order for CTC training.

    Prefers `libfribidi` (`fribidi_log2vis`, the exact UAX #9 + Paired Bracket Algorithm engine
    used by Pillow `raqm`), guaranteeing 100% parity between rendered pixels and CTC labels even
    on mixed scripts with brackets like `Получил письмо из ביטוח לאומי (23.11.2025)`.
    Falls back to `python-bidi` (`get_display`) or pure-Python UAX #9 only if `libfribidi` is absent.
    """
    text = strip_nikkud(text)
    if not text:
        return text
    if not has_hebrew(text) and base_dir != "R":
        return text
    if _FRIBIDI_LOG2VIS is not None:
        return _fribidi_get_display(text, base_dir=base_dir)
    if _bidi_get_display is not None:
        if base_dir is None:
            return _bidi_get_display(text)
        return _bidi_get_display(text, base_dir=base_dir)
    return _fallback_get_display(text, base_dir=base_dir)


def visual_to_logical(visual_text: str, base_dir: Literal["L", "R"] | None = None) -> str:
    """Recover logical text from visual-order CTC output (involution under UAX #9 for standard UI lines).

    When `base_dir` is omitted, defaults to `"R"` if the line contains Hebrew and starts/ends
    consistently with RTL, or `detect_base_dir` otherwise. Note that in production Android/Python
    postprocess, pure/dominant-Hebrew lines are reordered with `base_dir="R"`.
    """
    if not visual_text or not has_hebrew(visual_text):
        return visual_text
    if base_dir is None:
        # Heuristic matching visual-order output: if Hebrew is present and the rightmost strong
        # char (which was the first logical char in an RTL paragraph) is Hebrew, use base_dir="R".
        right_strong = "L"
        for ch in reversed(visual_text):
            bd = unicodedata.bidirectional(ch)
            if bd in ("R", "AL"):
                right_strong = "R"
                break
            if bd == "L":
                right_strong = "L"
                break
        base_dir = right_strong
    return to_visual_order(visual_text, base_dir=base_dir)


def load_base_keys(base_path: Path = BASE_REC_KEYS) -> list[str]:
    """Load the original 850-char Cyrillic+Latin dictionary from `ml/ocr/golden/rec_keys.txt`."""
    keys = [k for k in base_path.read_text(encoding="utf-8").split("\n") if k != ""]
    if len(keys) != EXPECTED_BASE_LEN:
        raise ValueError(f"Expected {EXPECTED_BASE_LEN} base keys in {base_path}, got {len(keys)}")
    return keys


def build_trilingual_keys(base_path: Path = BASE_REC_KEYS) -> list[str]:
    """Build the 881-char ru+en+he dictionary preserving base indices 0..849 identically."""
    base = load_base_keys(base_path)
    base_set = set(base)
    extras = []
    for ch in HEBREW_EXTRA_CHARS:
        if ch in base_set:
            raise ValueError(f"Hebrew extra char {ch!r} (U+{ord(ch):04X}) already exists in base dict")
        extras.append(ch)
    keys = base + extras
    assert len(keys) == len(set(keys)) == EXPECTED_TOTAL_LEN
    assert keys[:EXPECTED_BASE_LEN] == base
    return keys


def write_dict_txt(keys: Sequence[str], out_path: Path = DEFAULT_DICT_PATH) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(keys) + "\n", encoding="utf-8")
    return out_path


def _yaml_quote_char(ch: str) -> str:
    # Always single-quote so YAML 1.1 loaders (PyYAML) never interpret '=', 'y', 'n', 'Y', 'N',
    # digits, or punctuation as tags/booleans/ints, while matching Kotlin OcrDictionary.kt identically.
    escaped = ch.replace("'", "''")
    return f"'{escaped}'"


def write_inference_yml(keys: Sequence[str], out_path: Path, model_name: str = "ru_en_he_PP-OCRv5_mobile_rec") -> Path:
    """Write PaddleOCR-style `inference.yml` compatible with both `ml/ocr/pipeline.py` and Kotlin `OcrDictionary`."""
    lines = [
        "Global:",
        f"  model_name: {model_name}",
        "PostProcess:",
        "  name: CTCLabelDecode",
        "  character_dict:",
    ]
    for ch in keys:
        lines.append(f"  - {_yaml_quote_char(ch)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


@dataclass(frozen=True)
class Charset:
    keys: tuple[str, ...]
    chars: tuple[str, ...]
    char_to_id: dict[str, int]
    blank_id: int = 0

    @property
    def space_id(self) -> int:
        return len(self.keys) + 1

    @property
    def num_classes(self) -> int:
        return len(self.chars)

    def can_encode(self, text: str) -> bool:
        return all(c in self.char_to_id for c in text)

    def filter_text(self, text: str) -> str:
        """Keep only characters representable in this charset and collapse whitespace."""
        cleaned = strip_nikkud(text)
        kept = [c if c in self.char_to_id else " " for c in cleaned]
        return " ".join("".join(kept).split())

    def encode(self, visual_text: str) -> list[int]:
        ids = []
        for c in visual_text:
            cid = self.char_to_id.get(c)
            if cid is None:
                raise KeyError(f"Character {c!r} (U+{ord(c):04X}) not in charset")
            ids.append(cid)
        return ids

    def decode_ids(self, ids: Sequence[int]) -> str:
        out: list[str] = []
        prev = -1
        for k in ids:
            if k != prev and k != self.blank_id:
                out.append(self.chars[k])
            prev = k
        return "".join(out)

    def decode_probs(self, probs: np.ndarray) -> tuple[str, float]:
        """Greedy CTC decode matching `ml/ocr/pipeline.py:ctc_decode` byte-for-byte."""
        idx = probs.argmax(-1)
        mx = probs.max(-1)
        text: list[str] = []
        confs: list[float] = []
        prev = -1
        for t, k in enumerate(idx.tolist()):
            if k != prev and k != self.blank_id:
                text.append(self.chars[k])
                confs.append(float(mx[t]))
            prev = k
        return "".join(text), (sum(confs) / len(confs) if confs else 0.0)


def load_charset(dict_path: Path = DEFAULT_DICT_PATH) -> Charset:
    if not dict_path.exists():
        keys = build_trilingual_keys()
        write_dict_txt(keys, dict_path)
    else:
        keys = [k for k in dict_path.read_text(encoding="utf-8").split("\n") if k != ""]
    chars = ("", *keys, " ")
    c2i = {k: i + 1 for i, k in enumerate(keys)}
    c2i[" "] = len(keys) + 1
    return Charset(keys=tuple(keys), chars=chars, char_to_id=c2i)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build/verify the 881-char ru+en+he OCR dictionary.")
    ap.add_argument("--base", type=Path, default=BASE_REC_KEYS, help="Path to original rec_keys.txt (850 chars)")
    ap.add_argument("--out", type=Path, default=DEFAULT_DICT_PATH, help="Output dict_ru_en_he.txt path")
    ap.add_argument("--yml", type=Path, default=None, help="Optional PaddleOCR-compatible inference.yml output")
    args = ap.parse_args()

    keys = build_trilingual_keys(args.base)
    write_dict_txt(keys, args.out)
    if args.yml is not None:
        write_inference_yml(keys, args.yml)
    cs = load_charset(args.out)
    print(
        f"Wrote {args.out} ({len(cs.keys)} keys -> {cs.num_classes} CTC classes; "
        f"blank=0, base=1..{EXPECTED_BASE_LEN}, he={EXPECTED_BASE_LEN + 1}..{len(cs.keys)}, "
        f"space={cs.space_id}; bidi_backend={BIDI_BACKEND})"
    )


if __name__ == "__main__":
    main()
