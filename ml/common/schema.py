"""Shared data contract (binding for data, training and eval).

One example = one JSON object per line:
  id        str, stable and unique
  lang      "ru" | "en" | "mixed" | "he" | "he_en" | "he_ru"
  category  str  (taxonomy below; snake_case)
  source    "sim" | "tts" | "manual" | "public:<dataset>"
  split     "train" | "dev" | "test"
  ctx       object accepted by prompts/render.py user_message(ctx, transcript)
  transcript str  raw ASR-like input
  target    str  exact model output ("" is valid)
  meta      object, free-form provenance. Keys with a meaning for loaders:
            preferences (str)    -> render.system_prompt(kind, preferences)
            style_override (str) -> render.user_message(..., style_override)
            licence (str), teacher (str), asr (str), noise_ops (list), filters (list)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .jsonschema_lite import validate as validate_json
from .render_bridge import REPO, render

LANGS = {"ru", "en", "mixed", "he", "he_en", "he_ru"}
SPLITS = {"train", "dev", "test"}
SOURCE_RE = re.compile(r"^(sim|tts|manual|public:[A-Za-z0-9_.\-/]+)$")
CATEGORY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
TOP_KEYS = ["id", "lang", "category", "source", "split", "ctx", "transcript", "target", "meta"]

# Category taxonomy (docs/ml/DATA_CARD.md). Unknown categories are allowed by
# the contract but reported by `validate --strict`.
CATEGORIES = {
    "plain", "self_correction", "context_names", "context_distractor", "no_op", "near_no_op", "numbers",
    "address", "formatting_commands", "list", "dictated_question", "dictated_command", "ai_addressed",
    "injection", "continuation", "asr_artifact", "noise_only", "meaningful_filler", "preferences_conflict",
    "code_switch", "asr_correction", "context_asr_correction",
    # PLAN_V2 synthetic set (ml/data/synth/sampler.py CATEGORY_TARGETS)
    "context_resolution", "context_counterpart", "near_miss", "context_irrelevant",
}

_SCHEMA = json.loads((REPO / "prompts" / "context.schema.v1.json").read_text(encoding="utf-8"))
_V2_OFFICIAL = REPO / "prompts" / "context.schema.v2.json"
_V2_FALLBACK = Path(__file__).with_name("ctx_v2.schema.json")


def _schema_v2() -> dict:
    p = _V2_OFFICIAL if _V2_OFFICIAL.exists() else _V2_FALLBACK
    return json.loads(p.read_text(encoding="utf-8"))


def validate_ctx(ctx) -> list[str]:
    if not isinstance(ctx, dict):
        return ["ctx: not an object"]
    if ctx.get("v") == 2:  # CONTEXT_V2 (OCR screen lines); required keys enforced
        return validate_json(ctx, _schema_v2(), "ctx", enforce_required=True)
    # v1: 'required' is not enforced: the contract allows minimal ctx objects.
    return validate_json(ctx, _SCHEMA, "ctx", enforce_required=False)


def validate_record(rec, check_render: bool = True, strict: bool = False) -> list[str]:
    """Return a list of human-readable errors (empty = valid)."""
    if not isinstance(rec, dict):
        return ["record is not an object"]
    errs = []
    missing = [k for k in TOP_KEYS if k not in rec]
    if missing:
        errs.append(f"missing keys {missing}")
    extra = [k for k in rec if k not in TOP_KEYS]
    if extra:
        errs.append(f"unknown top-level keys {extra} (put them in meta)")
    if errs:
        return errs
    if not isinstance(rec["id"], str) or not rec["id"]:
        errs.append("id: empty")
    if rec["lang"] not in LANGS:
        errs.append(f"lang: {rec['lang']!r}")
    if not isinstance(rec["category"], str) or not CATEGORY_RE.match(rec["category"]):
        errs.append(f"category: {rec['category']!r}")
    elif strict and rec["category"] not in CATEGORIES:
        errs.append(f"category: {rec['category']!r} not in taxonomy")
    if not isinstance(rec["source"], str) or not SOURCE_RE.match(rec["source"]):
        errs.append(f"source: {rec['source']!r}")
    if rec["split"] not in SPLITS:
        errs.append(f"split: {rec['split']!r}")
    for k in ("transcript", "target"):
        if not isinstance(rec[k], str):
            errs.append(f"{k}: not a string")
    if not isinstance(rec["meta"], dict):
        errs.append("meta: not an object")
    else:
        for k in ("preferences", "style_override"):
            if k in rec["meta"] and rec["meta"][k] is not None and not isinstance(rec["meta"][k], str):
                errs.append(f"meta.{k}: not a string")
    errs += validate_ctx(rec["ctx"])
    if check_render and not errs:
        try:
            r = render()
            r.user_message(rec["ctx"], rec["transcript"], style_override=rec["meta"].get("style_override"))
            r.system_prompt("finetuned", rec["meta"].get("preferences"))
        except Exception as e:  # noqa: BLE001
            errs.append(f"render failed: {type(e).__name__}: {e}")
    return errs


# ------------------------------------------------------------------ jsonl io
def read_jsonl(path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dumps(rec: dict) -> str:
    return json.dumps(rec, ensure_ascii=False, separators=(",", ":"))


def write_jsonl(path, recs, key_order: bool = True) -> int:
    """Atomic write (tmp + rename). Records are written with contract key order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r in recs:
            if key_order and all(k in r for k in TOP_KEYS):
                r = {k: r[k] for k in TOP_KEYS} | {k: v for k, v in r.items() if k not in TOP_KEYS}
            f.write(dumps(r) + "\n")
            n += 1
    tmp.replace(path)
    return n


def validate_file(path, strict: bool = False, check_render: bool = True) -> dict:
    """Validate every line + id uniqueness. Returns summary dict."""
    seen, bad, n = set(), [], 0
    for i, rec in enumerate(iter_jsonl(path)):
        n += 1
        e = validate_record(rec, check_render=check_render, strict=strict)
        rid = rec.get("id") if isinstance(rec, dict) else None
        if rid in seen:
            e.append(f"duplicate id {rid}")
        seen.add(rid)
        if e:
            bad.append((i + 1, rid, e))
    return {"path": str(path), "records": n, "invalid": len(bad), "errors": bad[:50]}
