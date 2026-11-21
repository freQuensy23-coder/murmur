"""End-to-end plumbing of run_eval (+ judge, gates, compare, leaderboard) against the stub teacher
served over HTTP: no model weights needed."""
from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from ml.eval import compare, contract, leaderboard, run_eval, smoke_to_contract

STUB = contract.REPO / "ml/eval/stub_teacher.py"


@pytest.fixture(scope="module")
def stub(tmp_path_factory):
    pf = tmp_path_factory.mktemp("stub") / "port"
    p = subprocess.Popen([sys.executable, str(STUB), "--port", "0", "--port-file", str(pf), "--fail-first", "2"])
    for _ in range(100):
        if pf.exists() and pf.read_text():
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{pf.read_text()}/v1"
    p.terminate()


@pytest.fixture(autouse=True)
def teacher_env(stub, monkeypatch):
    """Responses API, as on the dev server; no real credentials in tests."""
    monkeypatch.setenv("MURMUR_TEACHER_BASE_URL", stub)
    monkeypatch.setenv("MURMUR_TEACHER_API", "responses")
    monkeypatch.setenv("MURMUR_TEACHER_MODEL", "stub-teacher")
    monkeypatch.setenv("MURMUR_TEACHER_API_KEY", "not-a-real-key")


def test_run_eval_with_stub_and_judge(stub, tmp_path):
    rows = [smoke_to_contract.convert(json.loads(l), "test") for l in
            (contract.REPO / "prompts/eval/smoke_cases.v1.jsonl").read_text(encoding="utf-8").splitlines()[:6]]
    data = tmp_path / "d.jsonl"
    contract.write_jsonl(data, rows)
    out = tmp_path / "e"
    args = ["--model", "teacher", "--backend", "teacher", "--data", str(data), "--out", str(out), "--judge"]
    assert run_eval.main(args) == 0
    rep = json.loads((out / "report.json").read_text())
    o = rep["summary"]["overall"]
    assert o["n"] == 6 and o["judge_n"] == 6 and o["judge_mean"] == 4.0 and o["judge_drift"] == 0.0
    assert o["em"] == 0.0  # the stub answers "stub"
    assert rep["gates"]["ship_measurement"] is False and rep["info"]["backend"] == "teacher"
    assert rep["info"]["judge"] == ["stub-teacher"] and rep["info"]["judged_n"] == 6
    preds = (out / "predictions.jsonl").read_text().splitlines()
    assert len(preds) == 6
    # resumable: a second run generates nothing new and gives the same report
    assert run_eval.main(args) == 0
    assert len((out / "predictions.jsonl").read_text().splitlines()) == 6
    # a different system prompt into the same dir is refused
    with pytest.raises(SystemExit):
        run_eval.main(args + ["--system", "default"])
    c = compare.compare(rep, rep)
    assert c["qat_recommended"] is False and c["scopes"]["overall"]["em"]["delta"] == 0
    lb = leaderboard.table(leaderboard.rows([str(tmp_path)]))
    assert lb.count("\n") == 2 and "INCOMPLETE" in lb or "FAIL" in lb


def test_rescore_and_judge_cli_on_synced_dir(tmp_path):
    """Pod writes predictions; dev server judges (ml.eval.judge CLI) and rescores without the model.
    report.json info.data points to a pod path; --data-root remaps it."""
    from ml.eval import judge
    rows = [smoke_to_contract.convert(json.loads(l), "test") for l in
            (contract.REPO / "prompts/eval/smoke_cases.v1.jsonl").read_text(encoding="utf-8").splitlines()[:4]]
    local = tmp_path / "data"
    contract.write_jsonl(local / "test.jsonl", rows)
    ev = tmp_path / "runs/x/eval/Q4_0"
    ev.mkdir(parents=True)
    contract.write_jsonl(ev / "predictions.jsonl", [{"id": r["id"], "out": r["target"]} for r in rows])
    (ev / "report.json").write_text(json.dumps({"info": {
        "model": "/workspace/runs/x/gguf/x-Q4_0.gguf", "data": [str(tmp_path / "unsynced" / "test.jsonl")], "split": None,
        "system": "finetuned", "template": "qwen3.5", "backend": "gguf", "label": "x Q4_0",
        "model_sha256": "ab" * 32, "model_bytes": 123}}))
    assert judge.main([str(ev), "--data-root", str(local)]) == 0
    assert len((ev / "judge.jsonl").read_text().splitlines()) == 4
    assert run_eval.main(["--rescore", str(ev), "--data-root", str(local)]) == 0
    rep = json.loads((ev / "report.json").read_text())
    o = rep["summary"]["overall"]
    assert o["em"] == 1.0 and o["judge_n"] == 4 and rep["info"]["model_sha256"] == "ab" * 32
    assert rep["gates"]["ship_measurement"] is True and rep["info"]["label"] == "x Q4_0"


def test_report_has_pass_by_set_and_stock_comparison(tmp_path):
    """--rescore on two files (dev.v2 + a dataset-style dev file): `pass` (prompts/eval/score.py), a
    per-file `by_set` breakdown, and the comparison with the frozen stock baselines on dev.v2. An oracle
    (output := target) passes every dev.v2 case, raw and guarded, and beats the best stock model."""
    from ml.eval import baseline
    v2 = contract.REPO / "prompts/eval/sets/dev.v2.jsonl"
    own = tmp_path / "data/dev.jsonl"
    rows = [smoke_to_contract.convert(json.loads(l), "dev") for l in
            (contract.REPO / "prompts/eval/smoke_cases.v2.jsonl").read_text(encoding="utf-8").splitlines()[:5]]
    for r in rows:
        r["id"] = "own-" + r["id"]
    contract.write_jsonl(own, rows)
    ev = tmp_path / "ev"
    ev.mkdir()
    data = contract.load([v2, own])
    contract.write_jsonl(ev / "predictions.jsonl", [{"id": r["id"], "out": r["target"]} for r in data])
    (ev / "report.json").write_text(json.dumps({"info": {
        "model": "/workspace/runs/x/gguf/x-Q4_0.gguf", "data": [str(v2), str(own)], "split": None,
        "system": "finetuned.v2", "template": "qwen3.5", "backend": "gguf", "label": "oracle"}}))
    assert run_eval.main(["--rescore", str(ev)]) == 0
    rep = json.loads((ev / "report.json").read_text())
    by_set = rep["summary"]["by_set"]
    assert set(by_set) == {"dev.v2", "dev"} and by_set["dev.v2"]["n"] == 98 and by_set["dev"]["n"] == 5
    assert by_set["dev.v2"]["pass"] == 1.0 and rep["guarded"]["summary"]["by_set"]["dev.v2"]["pass"] == 1.0
    cmp = rep["baseline"]["sets"]["dev.v2"]
    assert cmp["best_stock"] == "Qwen3.5-2B" and cmp["stock"]["Qwen3.5-2B"]["guarded"]["pass"] == 0.5714
    assert cmp["delta_pass_guarded_vs_best"] == round(1.0 - 0.5714, 4)
    assert "Versus the stock models" in (ev / "report.md").read_text()
    md = baseline.table([str(tmp_path)])
    assert "### dev.v2" in md and "stock Qwen3.5-2B" in md and "| oracle |" in md


def test_frozen_baselines_match_prompt_eval_doc():
    """docs/ml/PROMPT_EVAL.md 0.1: pass after the guard, dev.v2 / test.v2 (cases out of 98 / 66)."""
    from ml.eval import baseline
    b = baseline.load()
    want = {"Qwen3.5-2B": (56, 33), "LFM3-1.7B": (36, 21), "Qwen3.5-0.8B": (33, 21)}
    for m, (d, t) in want.items():
        assert b["models"][m]["dev.v2"]["pass_cases"]["guarded"] == d
        assert b["models"][m]["test.v2"]["pass_cases"]["guarded"] == t
