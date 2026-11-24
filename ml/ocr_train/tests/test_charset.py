"""Unit tests for charset index preservation, dictionary YAML/TXT, and visual-order BiDi helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ml.ocr_train.charset import (
    BASE_REC_KEYS,
    BIDI_BACKEND,
    DEFAULT_DICT_PATH,
    EXPECTED_BASE_LEN,
    EXPECTED_TOTAL_LEN,
    HEBREW_EXTRA_CHARS,
    USING_FRIBIDI,
    build_trilingual_keys,
    detect_base_dir,
    load_base_keys,
    load_charset,
    strip_nikkud,
    to_visual_order,
    visual_to_logical,
    write_inference_yml,
)


def test_charset_preserves_base_indices_identically():
    base = load_base_keys(BASE_REC_KEYS)
    assert len(base) == EXPECTED_BASE_LEN == 850

    keys = build_trilingual_keys(BASE_REC_KEYS)
    assert len(keys) == EXPECTED_TOTAL_LEN == 881
    assert keys[:850] == base, "Original rec_keys.txt indices 0..849 MUST be preserved identically!"
    assert tuple(keys[850:]) == HEBREW_EXTRA_CHARS
    assert len(set(keys)) == len(keys), "Dictionary must not contain duplicates"

    # Checked-in dict_ru_en_he.txt matches build_trilingual_keys() byte-for-byte
    disk_keys = [k for k in DEFAULT_DICT_PATH.read_text(encoding="utf-8").split("\n") if k != ""]
    assert disk_keys == keys


def test_hebrew_alphabet_sofits_and_punctuation_present():
    cs = load_charset(DEFAULT_DICT_PATH)
    assert cs.num_classes == 883
    assert cs.blank_id == 0
    assert cs.space_id == 882
    assert cs.chars[0] == ""
    assert cs.chars[882] == " "
    assert cs.chars[7] == "'"  # rec_keys index 6 -> CTC class 7

    # All 27 Hebrew letters (including final sofit forms ך ם ן ף ץ) + ׳ ״ ₪ ־
    for ch in "אבגדהוזחטיךכלםמןנסעףפץצקרשת׳״₪־":
        assert ch in cs.char_to_id, f"Missing Hebrew char {ch!r}"
        assert 851 <= cs.char_to_id[ch] <= 881


def test_inference_yml_roundtrip(tmp_path: Path):
    cs = load_charset(DEFAULT_DICT_PATH)
    yml_path = tmp_path / "inference.yml"
    write_inference_yml(cs.keys, yml_path)

    # Verify with PyYAML if installed, and also verify minimal Kotlin-compatible block sequence parser
    text = yml_path.read_text(encoding="utf-8")
    assert "PostProcess:" in text
    assert "character_dict:" in text

    # Minimal parser mirroring Kotlin OcrDictionary.fromInferenceYml
    lines = text.splitlines()
    idx = next(i for i, ln in enumerate(lines) if ln.strip() == "character_dict:")
    parsed = []
    for ln in lines[idx + 1 :]:
        if not ln.startswith("  - "):
            break
        raw = ln[4:]
        if raw.startswith("'") and raw.endswith("'"):
            parsed.append(raw[1:-1].replace("''", "'"))
        else:
            parsed.append(raw)
    assert parsed == list(cs.keys)
    assert parsed[6] == "'"

    try:
        import yaml  # type: ignore

        doc = yaml.safe_load(text)
        assert [str(k) for k in doc["PostProcess"]["character_dict"]] == list(cs.keys)
    except ImportError:
        pass


def test_visual_order_roundtrip():
    # Pure Hebrew: visual order reverses word and letter order
    assert to_visual_order("שלום עולם") == "םלוע םולש"
    assert visual_to_logical("םלוע םולש", base_dir="R") == "שלום עולם"

    # Mixed Hebrew + English (RTL paragraph base)
    he_en = "עשיתי push ל-GitHub"
    assert detect_base_dir(he_en) == "R"
    vis = to_visual_order(he_en, base_dir="R")
    assert vis.endswith("יתישע")
    assert "push" in vis and "GitHub" in vis
    assert visual_to_logical(vis, base_dir="R") == he_en

    # Mixed Russian + Hebrew (LTR paragraph base)
    ru_he = "Привет שלום"
    assert detect_base_dir(ru_he) == "L"
    vis_ru_he = to_visual_order(ru_he, base_dir="L")
    assert vis_ru_he == "Привет םולש"
    assert visual_to_logical(vis_ru_he, base_dir="L") == ru_he

    # Prices & mirrored parentheses
    price = "מחיר: 120 ₪"
    assert visual_to_logical(to_visual_order(price, base_dir="R"), base_dir="R") == price
    parens = "שעה 15:30 (תל אביב)"
    assert visual_to_logical(to_visual_order(parens, base_dir="R"), base_dir="R") == parens

    # Paired Bracket Algorithm (UAX #9 PBA) regression: libfribidi matches Pillow raqm 1:1
    assert USING_FRIBIDI and BIDI_BACKEND == "fribidi"
    ru_he_date = "Получил письмо из ביטוח לאומי (23.11.2025)"
    assert to_visual_order(ru_he_date, base_dir="L") == "Получил письмо из (23.11.2025) ימואל חוטיב"
    he_en_pr = "תיקנתי את ה-bug ב-GitHub (#4182)"
    assert to_visual_order(he_en_pr, base_dir="R") == "GitHub (#4182)-ב bug-ה תא יתנקית"

    # Nikkud stripping
    assert strip_nikkud("שָׁלוֹם עוֹלָם") == "שלום עולם"


def test_ctc_encode_and_decode_probs():
    cs = load_charset(DEFAULT_DICT_PATH)
    vis = to_visual_order("שלום Hi 120 ₪")
    ids = cs.encode(vis)
    # Build greedy CTC probability matrix with blanks & repeats
    seq = []
    for k in ids:
        seq.extend([k, k, 0])
    probs = np.full((len(seq), cs.num_classes), 1e-4, dtype=np.float32)
    for t, k in enumerate(seq):
        probs[t, k] = 0.95
    decoded, conf = cs.decode_probs(probs)
    assert decoded == vis
    assert abs(conf - 0.95) < 1e-5
