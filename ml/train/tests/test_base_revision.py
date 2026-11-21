"""Base weights are loaded at pinned Hub commits (ml/train/configs/base.yaml base_revisions = TRAINING_RUNBOOK 0.1):
every run config resolves to a full 40-character sha, a local directory needs none, an unpinned Hub repo is refused,
and sft / merge / bootstrap / export_eval pass the revision on."""
from __future__ import annotations

import re
import subprocess
import sys

import pytest

from ml.eval import contract
from ml.train import config as cfglib

CONFIGS = sorted((contract.REPO / "ml/train/configs").glob("*.yaml"))
RUNBOOK_SHORT = {"LiquidAI/LFM3-1.7B": "1da06bec6a81", "Qwen/Qwen3.5-0.8B": "2fc06364715b",
                 "Qwen/Qwen3.5-2B": "15852e8c1636"}


def test_table_matches_runbook_revisions():
    table = cfglib.pinned_revisions()
    assert set(table) == set(RUNBOOK_SHORT)
    for repo, sha in table.items():
        assert re.fullmatch(r"[0-9a-f]{40}", sha) and sha.startswith(RUNBOOK_SHORT[repo]), (repo, sha)
    runbook = (contract.REPO / "docs/ml/TRAINING_RUNBOOK.md").read_text(encoding="utf-8")
    for repo, short in RUNBOOK_SHORT.items():
        assert f"`{repo}` (rev `{short}" in runbook


@pytest.mark.parametrize("path", [p for p in CONFIGS if p.name != "dryrun-cpu.yaml"], ids=lambda p: p.stem)
def test_every_hub_run_config_is_pinned(path):
    cfg = cfglib.resolve(path)
    rev = cfglib.base_revision(cfg)
    assert rev == cfglib.pinned_revisions()[cfg["base_model"]]
    # an override of the base (e.g. full-fallback on Qwen3.5-0.8B) follows the table, not the old base's sha
    assert cfglib.base_revision(cfglib.resolve(path, ["base_model=Qwen/Qwen3.5-0.8B"])).startswith("2fc06364715b")


def test_local_dir_needs_none_and_unpinned_hub_repo_is_refused(tmp_path):
    assert cfglib.base_revision({"base_model": str(tmp_path)}) is None
    with pytest.raises(SystemExit, match="no pinned revision"):
        cfglib.base_revision({"base_model": "someone/unknown-model", "base_revisions": cfglib.pinned_revisions()})
    sha = "a1" * 20
    assert cfglib.base_revision({"base_model": "someone/unknown-model", "base_revision": sha}) == sha
    for ref in ("main", "refs/pr/2", "abc"):
        with pytest.raises(SystemExit, match="invalid pinned revision"):
            cfglib.base_revision({"base_model": "someone/unknown-model", "base_revision": ref})


def test_cli_prints_the_pinned_sha_for_bootstrap():
    r = subprocess.run([sys.executable, "-m", "ml.train.config", "revision", "LiquidAI/LFM3-1.7B"],
                       cwd=contract.REPO, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == cfglib.pinned_revisions()["LiquidAI/LFM3-1.7B"]
    bad = subprocess.run([sys.executable, "-m", "ml.train.config", "revision", "someone/unknown"],
                         cwd=contract.REPO, capture_output=True, text=True)
    assert bad.returncode != 0


def test_loaders_pass_the_revision():
    """Source-level guard: every Hub load of a base model in the training/pod code carries the pinned revision."""
    sft = (contract.REPO / "ml/train/sft.py").read_text()
    assert 'AutoTokenizer.from_pretrained(cfg["base_model"], revision=revision)' in sft
    assert 'from_pretrained(cfg["base_model"], revision=cfglib.base_revision(cfg)' in sft
    assert "AutoConfig.from_pretrained(base, revision=revision)" in sft
    assert "snapshot_download(cfg[\"base_model\"], revision=cfglib.base_revision(cfg)" in sft
    merge = (contract.REPO / "ml/train/merge.py").read_text()
    assert "from_pretrained(base, revision=revision" in merge
    for sh in ("ml/pod/bootstrap.sh", "ml/pod/export_eval.sh"):
        text = (contract.REPO / sh).read_text()
        assert "snapshot_download" in text and "revision='$rev'" in text and "ml.train.config revision" in text, sh
