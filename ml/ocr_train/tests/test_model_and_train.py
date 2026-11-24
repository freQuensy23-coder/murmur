"""Unit tests for PPOCRv5MobileRec architecture, CTC head weight transfer, and smoke training."""
from __future__ import annotations

from pathlib import Path

import torch

from ml.ocr_train.build_corpus import build_corpus
from ml.ocr_train.export_onnx import load_model_from_checkpoint
from ml.ocr_train.model import PPOCRv5MobileRec, expand_ctc_head_weights
from ml.ocr_train.render_crops import discover_fonts, render_split
from ml.ocr_train.train_rec import build_argparser, train


def test_model_param_count_and_bucket_shapes():
    m = PPOCRv5MobileRec(num_classes=883, export_softmax=True).eval()
    n_params = m.count_parameters()
    assert 1_500_000 <= n_params <= 3_000_000, f"Unexpected mobile param count: {n_params}"

    for bw in (320, 480, 640, 960):
        x = torch.randn(2, 3, 48, bw)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, bw // 8, 883)
        assert torch.allclose(y.sum(dim=-1), torch.ones(2, bw // 8), atol=1e-5)
        assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_ctc_head_weight_transfer_preserves_base_and_space():
    torch.manual_seed(0)
    src_w = torch.randn(852, 120)
    src_b = torch.randn(852)
    dst_w = torch.zeros(883, 120)
    dst_b = torch.zeros(883)

    new_w, new_b = expand_ctc_head_weights(src_w, src_b, dst_w, dst_b, base_dict_len=850)
    assert new_b is not None
    # Blank (0) and all 850 base Cyrillic/Latin keys (1..850) copied identically
    assert torch.equal(new_w[:851], src_w[:851])
    assert torch.equal(new_b[:851], src_b[:851])
    # Space moved cleanly from stock 851 -> expanded 882
    assert torch.equal(new_w[882], src_w[851])
    assert torch.equal(new_b[882], src_b[851])
    # Hebrew rows 851..881 freshly initialized (non-zero finite)
    assert torch.isfinite(new_w[851:882]).all()
    assert float(new_w[851:882].abs().mean()) > 0.0


def test_smoke_training_and_checkpoint_reload(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    crops_dir = tmp_path / "crops"
    run_dir = tmp_path / "run"

    build_corpus(out_dir=corpus_dir, num_lines=120, val_ratio=0.2, seed=11, offline=True)
    fonts = discover_fonts()
    render_split(corpus_dir / "train_corpus.jsonl", crops_dir, "train", fonts, workers=1, limit=24)
    render_split(corpus_dir / "val_corpus.jsonl", crops_dir, "val", fonts, workers=1, limit=8)

    ap = build_argparser()
    args = ap.parse_args([
        "--crops-dir", str(crops_dir),
        "--out-dir", str(run_dir),
        "--epochs", "1",
        "--batch-size", "8",
        "--workers", "0",
        "--device", "cpu",
        "--amp", "none",
        "--ema-decay", "0.0",
        "--log-every", "2",
    ])
    summary = train(args)
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "last.pt").exists()
    assert (run_dir / "history.jsonl").exists()
    assert summary["best_val"]["n"] == 8

    # Reload checkpoint for export
    m, cfg = load_model_from_checkpoint(run_dir / "best.pt", use_ema=False, export_softmax=True)
    assert cfg["num_classes"] == 883
    with torch.no_grad():
        out = m(torch.randn(1, 3, 48, 320))
    assert out.shape == (1, 40, 883)
