"""Fast multiprocess Android UI crop renderer with strict HarfBuzz/FriBidi RTL verification.

Produces realistic 48xW single-line crops matching `PP-OCRv6_tiny_det` -> `OcrPipeline` crops:
  - Hard-checks Pillow `raqm` (HarfBuzz + FriBidi) AND pixel-level Hebrew RTL glyph ordering at startup.
  - Auto-discovers Android/Hebrew/Cyrillic fonts across `.cache/ml/ocr/fonts/`, `.cache/ml/ocr/`,
    `.cache/ml/ocr_train/fonts/`, `.cache/ml/screens/fonts/`, and system font directories.
  - Uses per-font glyph coverage + Android-style `Roboto + NotoSansHebrew` baseline-aligned run
    compositing so mixed `he+en` / `ru+he` lines NEVER render tofu boxes (`□`).
  - Simulates light/dark/Telegram/WhatsApp/code themes, variable font weights, DB-box crop jitter
    (±2..6 px), realistic native Android font sizes resampled to height=48, and mild UI noise/blur.

Outputs:
  <out_dir>/images/{train,val}/NNN/NNNNNN.png
  <out_dir>/{train,val}_meta.jsonl      (full metadata: image, visual, text, base_dir, cat, width, height)
  <out_dir>/{train,val}_list.txt        (PaddleOCR/PyTorch tab-separated: rel_path\\tvisual_label)
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

from .charset import BIDI_BACKEND, REPO_ROOT, USING_FRIBIDI, detect_base_dir, has_hebrew, is_hebrew_char, to_visual_order

TARGET_HEIGHT = 48
MAX_CROP_WIDTH = 2560

# Realistic Android UI (bg_rgb, fg_rgb, weight_prob) palettes
UI_THEMES: list[tuple[str, tuple[int, int, int], tuple[int, int, int]]] = [
    # Standard light Android / Chrome / Docs / Gmail
    ("light_white", (255, 255, 255), (20, 22, 26)),
    ("light_surface", (247, 248, 250), (32, 33, 36)),
    ("light_muted", (255, 255, 255), (95, 99, 104)),
    ("light_blue_link", (255, 255, 255), (26, 115, 232)),
    ("light_pill_blue", (232, 240, 254), (25, 103, 210)),
    ("light_pill_warn", (254, 247, 224), (60, 64, 67)),
    # Standard dark Android / Material You
    ("dark_surface", (18, 18, 18), (232, 234, 237)),
    ("dark_card", (28, 30, 34), (241, 243, 244)),
    ("dark_elevated", (32, 33, 36), (220, 224, 228)),
    ("dark_muted", (24, 25, 28), (154, 160, 166)),
    ("dark_blue_link", (20, 22, 26), (138, 180, 248)),
    # Telegram light & dark bubbles
    ("tg_light_in", (255, 255, 255), (16, 16, 16)),
    ("tg_light_out", (238, 255, 222), (16, 16, 16)),
    ("tg_light_out2", (227, 250, 206), (20, 28, 20)),
    ("tg_dark_in", (24, 37, 51), (245, 245, 245)),
    ("tg_dark_out", (43, 82, 120), (255, 255, 255)),
    ("tg_dark_bg", (14, 22, 33), (224, 232, 240)),
    # WhatsApp light & dark bubbles
    ("wa_light_out", (231, 255, 219), (17, 27, 33)),
    ("wa_light_in", (255, 255, 255), (17, 27, 33)),
    ("wa_dark_in", (32, 44, 51), (233, 237, 239)),
    ("wa_dark_out", (0, 92, 75), (233, 237, 239)),
    # Code / terminal / GitHub dark & light
    ("term_dark", (13, 17, 23), (201, 209, 217)),
    ("term_accent", (13, 17, 23), (121, 192, 255)),
    ("term_green", (13, 17, 23), (126, 231, 135)),
    ("code_light", (246, 248, 250), (36, 41, 47)),
]

# Skip non-text / symbol / emoji / ancient-script system fonts during auto-discovery
_SKIP_FONT_SUBSTRS = (
    "emoji", "braille", "symbol", "lastresort", "wingding", "webding", "dingbat",
    "zapf", "marlett", "keyboard", "aquakana", "hiragino", "pingfang", "gothic",
    "myanmar", "oriya", "bangla", "gujarati", "telugu", "kannada", "malayalam",
    "sinhala", "tamil", "khmer", "lao", "thai", "tibetan", "devanagari", "gurmukhi",
    "armenian", "georgian", "ethiopic", "cherokee", "cjk", "kr-", "jp-", "sc-", "tc-",
    "mandaic", "lepcha", "avestan", "syriac", "samaritan", "phoenician", "nko",
    "osage", "adlam", "vai", "hanunoo", "buhid", "tagalog", "coptic", "glagolitic",
    "ogham", "runic", "shavian", "deseret", "lydian", "lycian", "carian", "cuneiform",
    "egyptian", "anatolian", "linear", "miao", "bassa", "pahawh", "mende", "saurashtra",
    "cham", "kayah", "multani", "hatran", "pahlavi", "manichaean", "bhaiksuki",
    "albanian", "hungarian", "canadian", "ugaritic", "nabataean", "palmyrene", "imperial",
    "geezapro", "nastaleeq", "damascus", "baghdad", "farah", "mishafi", "nadeem",
    "stix", "kefa", "euphemia", "plantagenet", "silom", "thonburi", "ayuthaya", "sathu",
    "skia", "chalkduster", "papyrus", "herculanum", "luminari", "party", "trattatello",
    "bradley", "brush", "snell", "zapfino", "phosphate", "signpainter", "bodoni ornaments",
)

# Preferred UI font basenames (boosted in sampling so Android/Israeli UI fonts dominate)
_PREFERRED_SUBSTRS = (
    "notosanshebrew", "rubik", "heebo", "assistant", "roboto", "notosans",
    "opensans", "ibmplexsanshebrew", "dejavusans", "freesans", "arial", "tahoma",
    "jetbrainsmono", "robotomono", "varelaround", "frankruhllibre", "sfhebrew",
)


@dataclass(frozen=True)
class FontInfo:
    path: str
    name: str
    has_latin: bool
    has_cyrillic: bool
    has_hebrew: bool
    has_shekel: bool
    has_geresh: bool
    has_maqaf: bool
    is_mono: bool
    is_variable_wght: bool
    wght_min: int
    wght_max: int
    preferred: bool

    def supports_script_of(self, ch: str) -> bool:
        cp = ord(ch)
        if 0x05D0 <= cp <= 0x05EA:
            return self.has_hebrew
        if ch in ("\u05F3", "\u05F4"):
            return self.has_geresh
        if ch == "\u20AA":
            return self.has_shekel
        if ch == "\u05BE":
            return self.has_maqaf
        if 0x0400 <= cp <= 0x04FF:
            return self.has_cyrillic
        if ch.isascii():
            return self.has_latin
        return self.has_latin and self.has_cyrillic


def _render_single_glyph(font: ImageFont.FreeTypeFont, ch: str) -> bytes:
    im = Image.new("L", (36, 36), 255)
    ImageDraw.Draw(im).text((4, 4), ch, fill=0, font=font)
    return im.tobytes()


def _probe_font(path: Path) -> FontInfo | None:
    low = path.name.lower()
    if any(bad in low for bad in _SKIP_FONT_SUBSTRS):
        return None
    try:
        font = ImageFont.truetype(str(path), 24, layout_engine=ImageFont.Layout.RAQM)
    except Exception:
        return None

    notdef = _render_single_glyph(font, "\uFFFE")

    def has_ch(c: str) -> bool:
        g = _render_single_glyph(font, c)
        return g != notdef and min(g) < 210

    has_lat = has_ch("A") and has_ch("g") and has_ch("5")
    has_cyr = has_ch("Ж") and has_ch("д") and has_ch("я")
    has_heb = has_ch("א") and has_ch("ש") and has_ch("ם") and has_ch("ץ")
    if not (has_lat or has_cyr or has_heb):
        return None

    has_shek = has_ch("₪")
    has_ger = has_ch("׳") and has_ch("״")
    has_maq = has_ch("־")

    is_var = False
    wmin, wmax = 400, 400
    try:
        axes = font.get_variation_axes()
        for ax in axes:
            if ax.get("name") in (b"Weight", "Weight"):
                is_var = True
                wmin = int(ax.get("minimum", 400))
                wmax = int(ax.get("maximum", 700))
                break
    except Exception:
        pass

    norm_name = low.replace("-", "").replace("_", "").replace(" ", "")
    pref = any(p in norm_name for p in _PREFERRED_SUBSTRS)
    is_mono = ("mono" in norm_name) or ("courier" in norm_name) or ("console" in norm_name)
    return FontInfo(
        path=str(path),
        name=path.name,
        has_latin=has_lat,
        has_cyrillic=has_cyr,
        has_hebrew=has_heb,
        has_shekel=has_shek,
        has_geresh=has_ger,
        has_maqaf=has_maq,
        is_mono=is_mono,
        is_variable_wght=is_var,
        wght_min=wmin,
        wght_max=wmax,
        preferred=pref,
    )


def default_font_search_dirs(extra_dirs: Sequence[Path] = ()) -> list[Path]:
    dirs: list[Path] = []
    for d in extra_dirs:
        dirs.append(Path(d))
    env_dir = os.environ.get("MURMUR_OCR_TRAIN_FONTS")
    if env_dir:
        dirs.append(Path(env_dir))
    # Priority: coordinator's pre-downloaded openclaw-dp fonts, then ocr_train cache, then screens cache
    dirs.extend([
        REPO_ROOT / ".cache" / "ml" / "ocr" / "fonts",
        REPO_ROOT / ".cache" / "ml" / "ocr",
        REPO_ROOT / ".cache" / "ml" / "ocr_train" / "fonts",
        REPO_ROOT / ".cache" / "ml" / "screens" / "fonts",
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path("/System/Library/Fonts"),
        Path("/System/Library/Fonts/Supplemental"),
        Path("/Library/Fonts"),
        Path.home() / "Library" / "Fonts",
    ])
    seen: set[str] = set()
    out: list[Path] = []
    for d in dirs:
        if d.is_dir():
            rp = str(d.resolve())
            if rp not in seen:
                seen.add(rp)
                out.append(d)
    return out


def discover_fonts(extra_dirs: Sequence[Path] = (), max_system_fonts: int = 24) -> list[FontInfo]:
    """Discover and probe usable UI fonts across project caches and system font dirs."""
    search_dirs = default_font_search_dirs(extra_dirs)
    seen_real: set[str] = set()
    found: list[FontInfo] = []
    project_prefixes = (str((REPO_ROOT / ".cache").resolve()),) + tuple(str(Path(d).resolve()) for d in extra_dirs)

    sys_count = 0
    for d in search_dirs:
        is_proj = any(str(d.resolve()).startswith(pref) for pref in project_prefixes)
        # Project dirs: recursive scan; system dirs: scan preferred UI fonts first
        candidates = sorted(
            p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in (".ttf", ".otf", ".ttc")
        )
        # Prioritize preferred names inside huge system directories
        candidates.sort(
            key=lambda p: (
                0 if any(k in p.name.lower().replace("-", "").replace("_", "").replace(" ", "") for k in _PREFERRED_SUBSTRS) else 1,
                p.name.lower(),
            )
        )
        for p in candidates:
            rp = str(p.resolve())
            if rp in seen_real:
                continue
            seen_real.add(rp)
            if not is_proj and sys_count >= max_system_fonts:
                break
            info = _probe_font(p)
            if info is not None:
                # In system dirs, only keep fonts that look like clean UI fonts (preferred or multi-script)
                if not is_proj and not (info.preferred or (info.has_hebrew and info.has_latin)):
                    continue
                found.append(info)
                if not is_proj:
                    sys_count += 1

    if not found:
        raise RuntimeError(
            "No usable TTF/OTF fonts found! Run `ml/ocr_train/fetch_fonts.sh` or pass `--fonts-dir`."
        )
    if not any(f.has_hebrew for f in found):
        raise RuntimeError(
            "No Hebrew-capable font found! Run `ml/ocr_train/fetch_fonts.sh` (or install fonts-noto-core / "
            "pass `--fonts-dir .cache/ml/ocr/fonts`)."
        )
    if not any(f.has_cyrillic for f in found):
        raise RuntimeError(
            "No Cyrillic-capable font found! Run `ml/ocr_train/fetch_fonts.sh` or `ml/ocr/fetch_assets.sh`."
        )
    return found


def verify_rtl_shaping(hebrew_font_path: str) -> None:
    """Loudly verify that Pillow raqm + HarfBuzz + FriBidi is active and truly renders Hebrew RTL."""
    if not features.check("raqm"):
        raise RuntimeError(
            "Pillow `raqm` (HarfBuzz + FriBidi) text layout is NOT available!\n"
            "Hebrew crops would render BACKWARDS (unshaped LTR), corrupting the entire dataset.\n"
            "Fix on Ubuntu/Debian:\n"
            "  sudo apt-get update && sudo apt-get install -y libfribidi0 libharfbuzz0b libraqm0\n"
            "  pip install --force-reinstall --no-binary :all: pillow  # or modern manylinux wheel with raqm"
        )
    if not USING_FRIBIDI:
        raise RuntimeError(
            f"libfribidi ctypes binding unavailable (active bidi_backend={BIDI_BACKEND!r})!\n"
            "Pillow `raqm` uses libfribidi (with UAX #9 Paired Bracket Algorithm), whereas `python-bidi` "
            "lacks PBA and can scramble brackets on mixed ru+he lines.\n"
            "Ensure `libfribidi0` (Linux) or `fribidi` (macOS) is installed so `charset.py` matches `raqm` 1:1."
        )
    font = ImageFont.truetype(hebrew_font_path, 34, layout_engine=ImageFont.Layout.RAQM)

    # Check 1: 'א' (wide/heavy) vs 'י' (tiny) with spaces between them.
    # Logical: א first, י last -> Visual RTL: י on LEFT, א on RIGHT -> right_ink > left_ink.
    im = Image.new("L", (420, 64), 255)
    ImageDraw.Draw(im).text((20, 10), "א          י", fill=0, font=font, direction="rtl")
    arr = np.asarray(im) < 160
    xs = np.where(arr)[1]
    if len(xs) < 20:
        raise RuntimeError(f"RTL check rendered blank image with font {hebrew_font_path}")
    mid = (int(xs.min()) + int(xs.max())) // 2
    left_ink = int(arr[:, int(xs.min()) : mid].sum())
    right_ink = int(arr[:, mid : int(xs.max()) + 1].sum())
    if right_ink <= left_ink:
        raise RuntimeError(
            f"RTL glyph ordering FAILED for {hebrew_font_path}: 'א          י' had "
            f"left_ink={left_ink} >= right_ink={right_ink} (expected א on right, י on left!)."
        )

    # Check 2: Mixed paragraph direction 'שלום          I' with direction='rtl' puts 'שלום' on right.
    im2 = Image.new("L", (460, 64), 255)
    ImageDraw.Draw(im2).text((20, 10), "שלום          I", fill=0, font=font, direction="rtl")
    arr2 = np.asarray(im2) < 160
    xs2 = np.where(arr2)[1]
    if len(xs2) > 20:
        mid2 = (int(xs2.min()) + int(xs2.max())) // 2
        li2 = int(arr2[:, int(xs2.min()) : mid2].sum())
        ri2 = int(arr2[:, mid2 : int(xs2.max()) + 1].sum())
        if ri2 <= li2:
            raise RuntimeError(
                f"RTL paragraph base-direction FAILED for {hebrew_font_path}: 'שלום          I' had "
                f"left_ink={li2} >= right_ink={ri2}."
            )


# ----------------------------------------------------------------------------- crop renderer


class FontPool:
    """Worker-local cache of loaded `ImageFont` objects + script-aware font pairing."""

    def __init__(self, fonts: Sequence[FontInfo]):
        self.fonts = list(fonts)
        self.he_fonts = [f for f in self.fonts if f.has_hebrew]
        self.ru_fonts = [f for f in self.fonts if f.has_cyrillic]
        self.lat_fonts = [f for f in self.fonts if f.has_latin]
        self.all_three = [f for f in self.fonts if f.has_hebrew and f.has_cyrillic and f.has_latin]
        self.he_lat = [f for f in self.fonts if f.has_hebrew and f.has_latin]
        self.mono_fonts = [f for f in self.fonts if f.is_mono and f.has_latin]
        self.shekel_fonts = [f for f in self.fonts if f.has_shekel]
        self.geresh_fonts = [f for f in self.fonts if f.has_geresh]
        self.maqaf_fonts = [f for f in self.fonts if f.has_maqaf]
        self._cache: dict[tuple[str, int, int | None], ImageFont.FreeTypeFont] = {}

    def _weighted_choice(self, pool: Sequence[FontInfo], rng: random.Random) -> FontInfo:
        weights = [3.0 if f.preferred else 1.0 for f in pool]
        return rng.choices(pool, weights=weights, k=1)[0]

    def get_pil_font(self, info: FontInfo, size: int, weight: int | None = None) -> ImageFont.FreeTypeFont:
        use_w = None
        if info.is_variable_wght and weight is not None:
            use_w = max(info.wght_min, min(info.wght_max, weight))
        key = (info.path, size, use_w)
        f = self._cache.get(key)
        if f is not None:
            return f
        f = ImageFont.truetype(info.path, size, layout_engine=ImageFont.Layout.RAQM)
        if use_w is not None:
            try:
                # Set Weight axis while preserving any other axes (like Width=100) at default
                axes = f.get_variation_axes()
                vals = [
                    use_w if ax.get("name") in (b"Weight", "Weight") else ax.get("default", 100)
                    for ax in axes
                ]
                f.set_variation_by_axes(vals)
            except Exception:
                pass
        self._cache[key] = f
        return f

    def pick_fonts_for_line(self, text: str, cat: str, rng: random.Random) -> tuple[FontInfo, FontInfo]:
        """Pick (primary_font, hebrew_fallback_font) for a line.

        If `primary_font` covers every character in `text`, it renders in a single HarfBuzz call.
        Otherwise (e.g. Android `Roboto` + `NotoSansHebrew` on mixed `he+en` / `ru+he`), composite
        baseline-aligned runs are used with zero tofu boxes.
        """
        if cat == "en" and self.mono_fonts and rng.random() < 0.30:
            prim = self._weighted_choice(self.mono_fonts, rng)
            heb = self._weighted_choice(self.he_fonts, rng)
            return prim, heb

        if cat == "he":
            # 80% dedicated Hebrew fonts (NotoSansHebrew, Rubik, Heebo, Assistant), 20% multi-script
            prim = self._weighted_choice(self.he_fonts, rng)
            lat_fb = self._weighted_choice(self.ru_fonts or self.lat_fonts, rng)
            return prim, lat_fb

        if cat in ("ru", "ru_en"):
            prim = self._weighted_choice(self.ru_fonts, rng)
            heb = self._weighted_choice(self.he_fonts, rng)
            return prim, heb

        if cat == "he_en":
            # Mix: ~55% single-font Hebrew+Latin (Rubik, Heebo, Assistant, Arial), ~45% Android Roboto + NotoSansHebrew
            if self.he_lat and rng.random() < 0.55:
                prim = self._weighted_choice(self.he_lat, rng)
                return prim, prim
            prim = self._weighted_choice(self.ru_fonts or self.lat_fonts, rng)
            heb = self._weighted_choice(self.he_fonts, rng)
            return prim, heb

        # ru_he or fallback
        if self.all_three and rng.random() < 0.50:
            prim = self._weighted_choice(self.all_three, rng)
            return prim, prim
        prim = self._weighted_choice(self.ru_fonts, rng)
        heb = self._weighted_choice(self.he_fonts, rng)
        return prim, heb

    def font_for_char(self, ch: str, primary: FontInfo, secondary: FontInfo) -> FontInfo:
        if ch == " ":
            return primary
        if primary.supports_script_of(ch):
            return primary
        if secondary.supports_script_of(ch):
            return secondary
        if ch == "₪" and self.shekel_fonts:
            return self.shekel_fonts[0]
        if ch in ("׳", "״") and self.geresh_fonts:
            return self.geresh_fonts[0]
        if ch == "־" and self.maqaf_fonts:
            return self.maqaf_fonts[0]
        if is_hebrew_char(ch) and self.he_fonts:
            return self.he_fonts[0]
        if self.all_three:
            return self.all_three[0]
        return self.ru_fonts[0] if self.ru_fonts else primary


def _single_font_covers_all(text: str, info: FontInfo) -> bool:
    return all(ch == " " or info.supports_script_of(ch) for ch in text)


def _split_visual_runs(
    visual_text: str,
    pool: FontPool,
    primary: FontInfo,
    secondary: FontInfo,
) -> list[tuple[str, FontInfo, bool]]:
    """Split visual-order string into contiguous `(visual_substr, font_info, is_hebrew_run)` runs."""
    runs: list[tuple[str, FontInfo, bool]] = []
    cur_chars: list[str] = []
    cur_font: FontInfo | None = None
    cur_heb: bool | None = None

    for ch in visual_text:
        is_heb = 0x05D0 <= ord(ch) <= 0x05EA
        # Keep spaces attached to the current run to avoid awkward inter-font gaps
        if ch == " " and cur_font is not None:
            finfo = cur_font
            heb_flag = False if cur_heb else False
        else:
            finfo = pool.font_for_char(ch, primary, secondary)
            heb_flag = is_heb

        if cur_font is None:
            cur_font = finfo
            cur_heb = heb_flag
            cur_chars = [ch]
        elif finfo.path == cur_font.path and heb_flag == cur_heb:
            cur_chars.append(ch)
        else:
            runs.append(("".join(cur_chars), cur_font, bool(cur_heb)))
            cur_font = finfo
            cur_heb = heb_flag
            cur_chars = [ch]
    if cur_chars and cur_font is not None:
        runs.append(("".join(cur_chars), cur_font, bool(cur_heb)))
    return runs


def _jitter_rgb(rgb: tuple[int, int, int], rng: random.Random, delta: int = 8) -> tuple[int, int, int]:
    return tuple(max(0, min(255, c + rng.randint(-delta, delta))) for c in rgb)  # type: ignore


def render_crop(
    text: str,
    visual: str | None,
    base_dir: Literal["L", "R"] | None,
    cat: str,
    pool: FontPool,
    rng: random.Random,
) -> tuple[Image.Image, dict]:
    """Render one realistic 48xW Android UI crop. Returns `(PIL.Image RGB, info_dict)`."""
    if base_dir not in ("L", "R"):
        base_dir = detect_base_dir(text)
    if visual is None:
        visual = to_visual_order(text, base_dir=base_dir)

    pil_dir = "rtl" if base_dir == "R" else "ltr"
    prim_info, sec_info = pool.pick_fonts_for_line(text, cat, rng)

    # Native Android pixel height before DB crop -> bilinear resize to 48 px
    r_sz = rng.random()
    if r_sz < 0.12:
        font_size = rng.randint(15, 20)   # small timestamp / subtitle / dense UI
    elif r_sz < 0.85:
        font_size = rng.randint(21, 38)   # typical Android body / chat / email
    else:
        font_size = rng.randint(39, 52)   # large header / title

    weight = rng.choices([400, 500, 600, 700], weights=[0.55, 0.22, 0.10, 0.13], k=1)[0]
    theme_name, bg_base, fg_base = rng.choice(UI_THEMES)
    bg = _jitter_rgb(bg_base, rng, delta=6)
    fg = _jitter_rgb(fg_base, rng, delta=8)

    # Render onto oversized scratch canvas, measure exact ink/text bbox, then apply DB-box jitter
    prim_font = pool.get_pil_font(prim_info, font_size, weight)
    single_font = _single_font_covers_all(text, prim_info)

    est_w = max(120, int(len(text) * font_size * 1.15) + 120)
    est_h = int(font_size * 2.6) + 40
    canvas = Image.new("RGB", (est_w, est_h), bg)
    draw = ImageDraw.Draw(canvas)
    baseline_y = int(font_size * 1.55) + 12
    start_x = 32

    if single_font:
        draw.text((start_x, baseline_y), text, fill=fg, font=prim_font, direction=pil_dir, anchor="ls")
        bbox = draw.textbbox((start_x, baseline_y), text, font=prim_font, direction=pil_dir, anchor="ls")
    else:
        # Android-style baseline-aligned font-fallback compositing on visual-order runs
        runs = _split_visual_runs(visual, pool, prim_info, sec_info)
        cur_x = float(start_x)
        x0_min, y0_min, x1_max, y1_max = 1e9, 1e9, -1e9, -1e9
        for run_vis, finfo, is_heb_run in runs:
            rf = pool.get_pil_font(finfo, font_size, weight)
            # HarfBuzz/FriBidi automatically lays out strong-R Hebrew chars RTL, so pass logical order
            # (reverse of visual Hebrew run) and it produces the exact visual sequence left-to-right!
            draw_str = run_vis[::-1] if is_heb_run else run_vis
            rdir = "rtl" if is_heb_run else "ltr"
            draw.text((cur_x, baseline_y), draw_str, fill=fg, font=rf, direction=rdir, anchor="ls")
            rb = draw.textbbox((cur_x, baseline_y), draw_str, font=rf, direction=rdir, anchor="ls")
            if rb[2] > rb[0] and rb[3] > rb[1]:
                x0_min = min(x0_min, rb[0])
                y0_min = min(y0_min, rb[1])
                x1_max = max(x1_max, rb[2])
                y1_max = max(y1_max, rb[3])
            adv = draw.textlength(draw_str, font=rf, direction=rdir)
            cur_x += adv
        if x1_max <= x0_min:
            bbox = (start_x, baseline_y - font_size, int(cur_x) + 4, baseline_y + 4)
        else:
            bbox = (int(math.floor(x0_min)), int(math.floor(y0_min)), int(math.ceil(x1_max)), int(math.ceil(y1_max)))

    # Tight crop + realistic PP-OCRv6_tiny_det unclip margin jitter (±2..6 px, slight asymmetry)
    bx0, by0, bx1, by1 = bbox
    pad_l = rng.randint(1, max(3, font_size // 4))
    pad_r = rng.randint(1, max(3, font_size // 4))
    pad_t = rng.randint(1, max(3, font_size // 5))
    pad_b = rng.randint(1, max(3, font_size // 5))
    # Occasional very tight DB box (0..-1 px edge clip)
    if rng.random() < 0.12:
        pad_t = rng.randint(0, 1)
        pad_b = rng.randint(0, 1)

    cx0 = max(0, bx0 - pad_l)
    cy0 = max(0, by0 - pad_t)
    cx1 = min(canvas.width, bx1 + pad_r)
    cy1 = min(canvas.height, by1 + pad_b)
    if cx1 - cx0 < 4:
        cx1 = min(canvas.width, cx0 + 8)
    if cy1 - cy0 < 4:
        cy1 = min(canvas.height, cy0 + 8)

    crop = canvas.crop((cx0, cy0, cx1, cy1))

    # Occasional subtle 1-px UI divider/bubble border line near top or bottom edge
    if rng.random() < 0.08:
        d2 = ImageDraw.Draw(crop)
        edge_col = tuple(max(0, min(255, int(0.82 * b + 0.18 * f))) for b, f in zip(bg, fg))
        y_edge = 0 if rng.random() < 0.5 else crop.height - 1
        d2.line([(0, y_edge), (crop.width, y_edge)], fill=edge_col, width=1)

    # Mild downscale/upscale anti-aliasing simulation before resizing to TARGET_HEIGHT=48
    if rng.random() < 0.25 and crop.height >= 24:
        scale = rng.uniform(0.68, 0.92)
        sw = max(4, int(round(crop.width * scale)))
        sh = max(8, int(round(crop.height * scale)))
        crop = crop.resize((sw, sh), resample=Image.Resampling.BILINEAR)

    # Final resize to exact height = 48 (preserving aspect ratio, matching ml/ocr/pipeline.py)
    new_w = max(8, min(MAX_CROP_WIDTH, int(math.ceil(TARGET_HEIGHT * crop.width / crop.height))))
    resample = rng.choices(
        [Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS],
        weights=[0.65, 0.20, 0.15],
        k=1,
    )[0]
    crop = crop.resize((new_w, TARGET_HEIGHT), resample=resample)

    # Mild realistic optical / compression augmentations (~30% of crops)
    aug = rng.random()
    if aug < 0.10:
        crop = crop.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 0.70)))
    elif aug < 0.20:
        # Subtle JPEG compression artifacts (screenshot sharing / WhatsApp re-encoding)
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=rng.randint(76, 94))
        crop = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    elif aug < 0.28:
        arr = np.asarray(crop, dtype=np.float32)
        noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0.0, rng.uniform(1.0, 3.5), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        crop = Image.fromarray(arr, mode="RGB")

    meta = {
        "width": crop.width,
        "height": crop.height,
        "font": prim_info.name,
        "theme": theme_name,
        "single_font": single_font,
    }
    return crop, meta


# ----------------------------------------------------------------------------- multiprocess driver


_WORKER_POOL: FontPool | None = None


def _init_worker(fonts: list[FontInfo]) -> None:
    global _WORKER_POOL
    _WORKER_POOL = FontPool(fonts)


def _render_chunk(args: tuple[list[dict], str, str, str, int]) -> list[dict]:
    records, out_dir_str, split, ext, seed_base = args
    assert _WORKER_POOL is not None
    out_dir = Path(out_dir_str)
    results: list[dict] = []
    for rec in records:
        idx = int(rec["id"])
        rng = random.Random(seed_base + idx * 1000003)
        bdir = rec.get("base_dir")
        if bdir not in ("L", "R"):
            bdir = detect_base_dir(rec["text"])
        # Always compute visual label via libfribidi so CTC targets match Pillow raqm pixels 1:1
        # even when re-rendering an existing corpus JSONL.
        vis = to_visual_order(rec["text"], base_dir=bdir)
        img, info = render_crop(
            text=rec["text"],
            visual=vis,
            base_dir=bdir,
            cat=rec.get("cat", "he"),
            pool=_WORKER_POOL,
            rng=rng,
        )
        shard = f"{idx // 1000:04d}"
        rel_path = f"images/{split}/{shard}/{idx:07d}.{ext}"
        abs_path = out_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        if ext == "jpg":
            img.save(abs_path, format="JPEG", quality=94)
        else:
            img.save(abs_path, format="PNG", compress_level=1)
        results.append({
            "id": idx,
            "image": rel_path,
            "visual": vis,
            "text": rec["text"],
            "base_dir": bdir,
            "cat": rec.get("cat", "he"),
            "width": info["width"],
            "height": info["height"],
        })
    return results


def render_split(
    corpus_jsonl: Path,
    out_dir: Path,
    split: str,
    fonts: list[FontInfo],
    workers: int = 8,
    ext: str = "png",
    seed: int = 42,
    limit: int | None = None,
) -> dict:
    records: list[dict] = []
    with corpus_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break

    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = max(16, min(500, len(records) // max(1, workers * 4) + 1))
    chunks = [
        (records[i : i + chunk_size], str(out_dir), split, ext, seed + (0 if split == "train" else 777_777))
        for i in range(0, len(records), chunk_size)
    ]

    all_rows: list[dict] = []
    if workers <= 1 or len(records) <= 64:
        _init_worker(fonts)
        for ch in chunks:
            all_rows.extend(_render_chunk(ch))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(fonts,)) as ex:
            done = 0
            for part in ex.map(_render_chunk, chunks):
                all_rows.extend(part)
                done += len(part)
                if done % 10_000 < len(part) or done == len(records):
                    dt = max(1e-3, time.perf_counter() - t0)
                    print(
                        f"[render_crops:{split}] {done:,}/{len(records):,} "
                        f"({done / dt:,.0f} crops/s, elapsed {dt:.1f}s)",
                        flush=True,
                    )

    all_rows.sort(key=lambda r: r["id"])
    meta_path = out_dir / f"{split}_meta.jsonl"
    list_path = out_dir / f"{split}_list.txt"
    with meta_path.open("w", encoding="utf-8") as fm, list_path.open("w", encoding="utf-8") as fl:
        for row in all_rows:
            fm.write(json.dumps(row, ensure_ascii=False) + "\n")
            fl.write(f"{row['image']}\t{row['visual']}\n")

    elapsed = time.perf_counter() - t0
    return {
        "split": split,
        "count": len(all_rows),
        "elapsed_s": round(elapsed, 2),
        "crops_per_sec": round(len(all_rows) / max(1e-3, elapsed), 1),
        "meta_path": str(meta_path),
        "list_path": str(list_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Render 48xW Android UI OCR crops with verified HarfBuzz RTL.")
    ap.add_argument("--corpus-dir", type=Path, default=Path(".cache/ml/ocr_train/corpus"))
    ap.add_argument("--out-dir", type=Path, default=Path(".cache/ml/ocr_train/crops"))
    ap.add_argument("--fonts-dir", type=Path, action="append", default=[], help="Extra font dir(s)")
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 4))))
    ap.add_argument("--ext", choices=["png", "jpg"], default="png")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-val", type=int, default=None)
    ap.add_argument("--check-only", action="store_true", help="Only verify fonts & RTL shaping then exit")
    args = ap.parse_args()

    fonts = discover_fonts(args.fonts_dir)
    he_font = next(f for f in fonts if f.has_hebrew)
    verify_rtl_shaping(he_font.path)

    print(
        f"[render_crops] RTL shaping VERIFIED (raqm/HarfBuzz/FriBidi OK using {he_font.name}). "
        f"Discovered {len(fonts)} fonts "
        f"(he={sum(f.has_hebrew for f in fonts)}, cyr={sum(f.has_cyrillic for f in fonts)}, "
        f"all3={sum(f.has_hebrew and f.has_cyrillic and f.has_latin for f in fonts)}):",
        flush=True,
    )
    for f in fonts[:16]:
        flags = "".join([
            "H" if f.has_hebrew else "-",
            "C" if f.has_cyrillic else "-",
            "L" if f.has_latin else "-",
            "₪" if f.has_shekel else "-",
            "V" if f.is_variable_wght else "-",
        ])
        print(f"  [{flags}] {f.name} ({f.path})", flush=True)
    if args.check_only:
        return

    stats = {}
    for split, lim in (("train", args.limit_train), ("val", args.limit_val)):
        src = args.corpus_dir / f"{split}_corpus.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"Missing corpus file: {src}. Run `python -m ml.ocr_train.build_corpus` first.")
        stats[split] = render_split(src, args.out_dir, split, fonts, workers=args.workers, ext=args.ext, seed=args.seed, limit=lim)
    (args.out_dir / "render_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
