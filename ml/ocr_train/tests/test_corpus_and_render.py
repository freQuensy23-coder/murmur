"""Unit tests for `--smoke` corpus generation and HarfBuzz RTL Android crop rendering."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from ml.ocr_train.build_corpus import build_corpus
from ml.ocr_train.charset import load_charset, to_visual_order
from ml.ocr_train.render_crops import (
    TARGET_HEIGHT,
    FontPool,
    discover_fonts,
    render_crop,
    render_split,
    verify_rtl_shaping,
)


def test_smoke_corpus_generation(tmp_path: Path):
    stats = build_corpus(
        out_dir=tmp_path,
        num_lines=2000,
        val_ratio=0.10,
        seed=123,
        offline=True,
    )
    assert stats["train_total"] >= 1600
    assert stats["val_total"] >= 150

    # All 6 categories present
    expected_cats = {"he", "he_en", "ru", "ru_en", "en", "ru_he"}
    assert set(stats["train_by_cat"].keys()) == expected_cats
    assert set(stats["val_by_cat"].keys()) == expected_cats

    # Hebrew share ~40% of total
    total = stats["train_total"] + stats["val_total"]
    he_total = stats["train_by_cat"]["he"] + stats["val_by_cat"]["he"]
    assert 0.35 <= he_total / total <= 0.45

    # Zero train/val logical text overlap and 100% encodable visual labels
    cs = load_charset()
    train_rows = [json.loads(ln) for ln in (tmp_path / "train_corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    val_rows = [json.loads(ln) for ln in (tmp_path / "val_corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    train_texts = {r["text"] for r in train_rows}
    val_texts = {r["text"] for r in val_rows}
    assert train_texts.isdisjoint(val_texts), "Train/val text leakage detected!"

    for r in val_rows[:100]:
        assert r["base_dir"] in ("L", "R")
        assert cs.can_encode(r["visual"]), f"Unencodable visual label: {r['visual']!r}"


def test_rtl_shaping_verification_and_crop_dimensions():
    fonts = discover_fonts()
    he_font = next(f for f in fonts if f.has_hebrew)
    # Must pass loudly on HarfBuzz/raqm-enabled Pillow
    verify_rtl_shaping(he_font.path)

    pool = FontPool(fonts)
    rng = random.Random(7)
    samples = [
        ("שלום עולם 120 ₪", "R", "he"),
        ("עשיתי push ל-GitHub", "R", "he_en"),
        ("Привет שלום", "L", "ru_he"),
        ("Ревью PR #4182 в Grafana", "L", "ru_en"),
        ("kubectl get pods -n prod-eu-2", "L", "en"),
    ]
    seen_bg_colors = set()
    for txt, bdir, cat in samples:
        vis = to_visual_order(txt, base_dir=bdir)
        img, meta = render_crop(txt, vis, bdir, cat, pool, rng)
        assert img.mode == "RGB"
        assert img.height == TARGET_HEIGHT == 48
        assert 16 <= img.width <= 2560
        arr = np.asarray(img)
        # Must not be flat/blank (foreground ink must contrast with background)
        assert arr.std() > 8.0, f"Rendered crop looks blank for {txt!r}"
        seen_bg_colors.add(tuple(int(v) for v in arr[0, 0]))
        assert meta["height"] == 48

    assert len(seen_bg_colors) >= 2


def test_rendered_hebrew_crop_rtl_pixel_order():
    """Verify rendered crop itself places 'א' on the right and 'י' on the left (RTL visual order)."""
    fonts = discover_fonts()
    pool = FontPool(fonts)
    rng = random.Random(99)
    txt = "א          י"
    vis = to_visual_order(txt, base_dir="R")
    img, _ = render_crop(txt, vis, "R", "he", pool, rng)

    arr = np.asarray(img, dtype=np.float32)
    bg = np.median(np.concatenate([arr[0, :, :], arr[-1, :, :]], axis=0), axis=0)
    ink = np.linalg.norm(arr - bg[None, None, :], axis=-1) > 40.0
    xs = np.where(ink)[1]
    assert len(xs) > 20
    mid = (int(xs.min()) + int(xs.max())) // 2
    left_ink = int(ink[:, int(xs.min()) : mid].sum())
    right_ink = int(ink[:, mid : int(xs.max()) + 1].sum())
    assert right_ink > left_ink, f"Expected heavy 'א' on RIGHT and tiny 'י' on LEFT, got left={left_ink} right={right_ink}"


def test_render_split_smoke(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    crops_dir = tmp_path / "crops"
    build_corpus(out_dir=corpus_dir, num_lines=200, val_ratio=0.2, seed=5, offline=True)
    # Also inject a stale python-bidi-style scrambled visual label to verify render_split recomputes via libfribidi
    val_jsonl = corpus_dir / "val_corpus.jsonl"
    rows = [json.loads(ln) for ln in val_jsonl.read_text(encoding="utf-8").splitlines()]
    rows[0] = {
        "id": 99999,
        "text": "Получил письмо из ביטוח לאומי (23.11.2025)",
        "visual": "Получил письмо из 23.11.2025) ימואל חוטיב)",
        "base_dir": "L",
        "cat": "ru_he",
    }
    val_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    fonts = discover_fonts()
    st = render_split(val_jsonl, crops_dir, "val", fonts, workers=1, limit=12)
    assert st["count"] == 12
    assert (crops_dir / "val_meta.jsonl").exists()
    assert (crops_dir / "val_list.txt").exists()
    meta_rows = [json.loads(ln) for ln in (crops_dir / "val_meta.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {r["id"]: r for r in meta_rows}
    assert by_id[99999]["visual"] == "Получил письмо из (23.11.2025) ימואל חוטיב"
    lines = (crops_dir / "val_list.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 12
    rel, label = lines[0].split("\t")
    assert (crops_dir / rel).exists()
    assert len(label) >= 2
