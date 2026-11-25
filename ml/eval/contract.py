"""Shared data contract (one JSON object per line) and the single prompt-rendering entry point.

Every pipeline (training, eval, export checks) renders prompts through `render_prompt()` here,
which delegates to prompts/render.py, so the app, the training data and the eval share one format.

Contract fields (binding; shared with ml/data, ml/roundtrip and prompts/eval):
  id, lang ("ru"|"en"|"mixed"|"he"|"he_en"|"he_ru"), category, source ("sim"|"tts"|"public:<ds>"|"manual"),
  split ("train"|"dev"|"test"), ctx (object for render.user_message), transcript, target, meta (object)
Optional meta keys this code reads:
  meta.preferences      user "My preferences" text the target was written for (always rendered)
  meta.style_override   replaces the Style: line (render.user_message style_override)
  meta.check            {keep, forbid, alts, noanswer, injection} - see ml/eval/metrics.py
  meta.entities / meta.distractors   gold names / near-miss names (alternative to meta.check)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Iterator

REPO = Path(__file__).resolve().parents[2]
PROMPTS = REPO / "prompts"
if str(PROMPTS) not in sys.path:
    sys.path.insert(0, str(PROMPTS))
import render  # noqa: E402  (prompts/render.py, the executable spec of the prompt format)

LANGS = {"ru", "en", "mixed", "he", "he_en", "he_ru"}
SPLITS = {"train", "dev", "test"}
REQUIRED = ("id", "lang", "category", "source", "split", "ctx", "transcript", "target", "meta")

# Model family -> render.chat template name, and the end-of-turn string the model must emit.
TEMPLATES = {"qwen3.5": "<|im_end|>", "lfm3": "<|im_end|>"}
MODEL_TYPE_TO_TEMPLATE = {"qwen3_5": "qwen3.5", "qwen3_5_text": "qwen3.5", "lfm3": "lfm3",
                          "qwen35": "qwen3.5"}  # last one: GGUF general.architecture


class ContractError(ValueError):
    pass


def validate(ex: dict, where: str = "") -> dict:
    """Raise ContractError if `ex` violates the shared contract; return it unchanged otherwise."""
    missing = [k for k in REQUIRED if k not in ex]
    if missing:
        raise ContractError(f"{where} id={ex.get('id')!r}: missing {missing}")
    if not isinstance(ex["id"], str) or not ex["id"]:
        raise ContractError(f"{where}: id must be a non-empty string")
    if ex["lang"] not in LANGS:
        raise ContractError(f"{where} {ex['id']}: lang {ex['lang']!r} not in {sorted(LANGS)}")
    if ex["split"] not in SPLITS:
        raise ContractError(f"{where} {ex['id']}: split {ex['split']!r} not in {sorted(SPLITS)}")
    src = ex["source"]
    if not (src in ("sim", "tts", "manual") or (isinstance(src, str) and src.startswith("public:"))):
        raise ContractError(f"{where} {ex['id']}: bad source {src!r}")
    for k in ("transcript", "target", "category"):
        if not isinstance(ex[k], str):
            raise ContractError(f"{where} {ex['id']}: {k} must be a string")
    for k in ("ctx", "meta"):
        if not isinstance(ex[k], dict):
            raise ContractError(f"{where} {ex['id']}: {k} must be an object")
    return ex


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                yield validate(json.loads(line), f"{path}:{n}")


def load(paths: str | Path | Iterable[str | Path], split: str | None = None) -> list[dict]:
    """Load one or more contract JSONL files; optionally keep one split. Duplicate ids are an error."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out, seen = [], set()
    for p in paths:
        for ex in iter_jsonl(p):
            if split and ex["split"] != split:
                continue
            if ex["id"] in seen:
                raise ContractError(f"duplicate id {ex['id']!r} in {p}")
            seen.add(ex["id"])
            out.append(ex)
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def render_prompt(ex: dict, system: str = "finetuned", template: str = "qwen3.5",
                  preferences: str | None = None) -> str:
    """Full raw prompt string up to and including the assistant header (thinking disabled).

    `system` is a prompt profile (prompts/profiles/<name>.json: system prompt + few-shot turns +
    answer prefix, e.g. "default.v2") or a bare system prompt name ("finetuned", "default",
    "default.v1"), exactly as render.load_profile resolves it. Without a style override the result
    equals render.render_prompt(profile, ctx, transcript, template, prefs) byte for byte.
    meta.preferences of the example always wins over the `preferences` argument, because the
    target was written for those preferences."""
    meta = ex.get("meta") or {}
    prefs = meta.get("preferences") or preferences
    prof = render.load_profile(system)
    sysmsg = render.system_prompt(prof["system"], prefs, prof.get("preferences_block", "preferences_block"))
    user = render.user_message(ex.get("ctx") or {}, ex["transcript"], meta.get("style_override"), ctx2=prof.get("ctx2"))
    return render.chat_turns(sysmsg, render.profile_turns(prof), user, template) + prof.get("answer_prefix", "")


def render_messages(ex: dict, system: str = "finetuned", preferences: str | None = None
                    ) -> tuple[str, list[tuple[str, str]], str]:
    """(system text, few-shot (user, assistant) turns, user text) for chat-style APIs (teacher B3).
    Same pieces as render_prompt, before the chat template is applied."""
    meta = ex.get("meta") or {}
    prefs = meta.get("preferences") or preferences
    prof = render.load_profile(system)
    sysmsg = render.system_prompt(prof["system"], prefs, prof.get("preferences_block", "preferences_block"))
    user = render.user_message(ex.get("ctx") or {}, ex["transcript"], meta.get("style_override"), ctx2=prof.get("ctx2"))
    return sysmsg, render.profile_turns(prof), user


def completion_text(ex: dict, template: str) -> str:
    """What the model must generate after the prompt: target + end-of-turn marker."""
    return ex["target"] + TEMPLATES[template]
