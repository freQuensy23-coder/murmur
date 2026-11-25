"""Unit tests for Girgur training config (girgur-q2-lora.yaml) and RTX 5090 pod script."""
from __future__ import annotations

import subprocess
from pathlib import Path

from ml.train import config as cfglib

REPO = Path(__file__).resolve().parents[3]
GIRGUR_CFG = REPO / "ml" / "train" / "configs" / "girgur-q2-lora.yaml"
GIRGUR_POD = REPO / "ml" / "pod" / "run_girgur.sh"


def test_girgur_config_resolves_with_pinned_qwen2b_and_data_dir(monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)
    cfg = cfglib.resolve(GIRGUR_CFG)
    assert cfg["run_name"] == "girgur-q2-lora"
    assert cfg["base_model"] == "Qwen/Qwen3.5-2B"
    assert cfg["method"] == "lora"
    assert cfg["lora"]["r"] == 16 and cfg["lora"]["alpha"] == 32
    assert cfg["data"]["train"][0]["path"] == ".cache/ml/data/girgur/train.jsonl"
    assert cfg["data"]["dev"][0]["path"] == ".cache/ml/data/girgur/dev.jsonl"
    assert cfg["export"]["name"] == "girgur-cleanup-qwen3.5-2b"
    assert "Q4_0" in cfg["export"]["quants"] and cfg["export"]["keep_bf16"] is True
    # Pinned HF revision from base.yaml must resolve cleanly
    rev = cfglib.base_revision(cfg)
    assert rev and len(rev) == 40

    # DATA_DIR env override works for pod volume layout
    monkeypatch.setenv("DATA_DIR", "/workspace/data/girgur")
    cfg2 = cfglib.resolve(GIRGUR_CFG)
    assert cfg2["data"]["train"][0]["path"] == "/workspace/data/girgur/train.jsonl"
    assert cfg2["data"]["dev"][0]["path"] == "/workspace/data/girgur/dev.jsonl"


def test_run_girgur_sh_syntax_and_contract():
    assert GIRGUR_POD.exists()
    r = subprocess.run(["bash", "-n", str(GIRGUR_POD)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    txt = GIRGUR_POD.read_text(encoding="utf-8")
    assert "girgur-cleanup-qwen3.5-2b" in txt
    assert "girgur-q2-lora.yaml" in txt
    assert "ml.train.sft" in txt and "ml.train.merge" in txt and "ml.train.export" in txt
    assert "ml.train.parity" in txt and "ml.eval.run_eval" in txt
