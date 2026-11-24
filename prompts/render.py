#!/usr/bin/env python3
"""Reference renderer for Murmur post-processing prompts (v1).

This file is the executable specification of the prompt layout. The Android
(Kotlin) implementation must produce byte-identical strings for the golden
cases in prompts/examples/ (see prompts/README.md). The synthetic-data and
eval pipelines import this module so that training, evaluation and the app
all see exactly the same text.

No third-party dependencies. Python >= 3.9.

Usage:
  python3 render.py CONTEXT.json TRANSCRIPT.txt [--template qwen3.5|lfm3|lfm2.5|qwen3|messages]
                    [--system default|finetuned|default.v2] [--profile NAME] [--preferences PREFS.txt]
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPT_VERSION = "v1"  # default system prompt version for bare names ("default" -> default.v1)

# ---------------------------------------------------------------- budgets
# Character budgets (not tokens) so that every tokenizer sees the same text.
# Russian costs ~2.5-2.9 chars/token on Qwen3.5 / LFM3 tokenizers, English ~4.2
# (measured in docs/ml/CONTEXT_AND_PROMPT.md), so 1400 chars <= ~560 tokens.
MAX_CONTEXT_CHARS = 1400
MAX_BEFORE_CURSOR = 300
MAX_AFTER_CURSOR = 100
MAX_MESSAGES = 3
MAX_MESSAGE_CHARS = 160
MAX_TERMS = 40
MAX_TERMS_CHARS = 400
MAX_DICTIONARY = 20
MAX_SNIPPETS = 3
MAX_SNIPPET_CHARS = 120
MAX_TITLE_CHARS = 60

LANG_NAMES = {"ru": "Russian", "en": "English", "he": "Hebrew", "uk": "Ukrainian", "de": "German",
              "fr": "French", "es": "Spanish"}

FIELD_PHRASES = {
    "message": "message field",
    "email_body": "email body",
    "email_subject": "email subject",
    "search": "search field",
    "address": "address field",
    "note": "note",
    "code": "code or terminal input",
    "generic": "text field",
    "none": "no text field focused",
}


# ---------------------------------------------------------------- sanitising
_CTRL = re.compile(r"[\u0000-\u0008\u000b-\u001f\u007f-\u009f​-‏‪-‮⁦-⁩﻿]")
_WS = re.compile(r"[ \t ]+")


def clean(s: str | None, keep_newlines: bool = False) -> str:
    """Normalise untrusted text: NFC, strip control/bidi chars, neutralise
    angle brackets so screen text can never open or close our tags."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = _CTRL.sub("", s)
    s = s.replace("<", "‹").replace(">", "›")  # ‹ ›
    if keep_newlines:
        s = "\n".join(_WS.sub(" ", line).strip() for line in s.splitlines())
        s = re.sub(r"\n{3,}", "\n\n", s).strip()
    else:
        s = _WS.sub(" ", s.replace("\r", " ").replace("\n", " ")).strip()
    return s


def clip_end(s: str, n: int) -> str:
    """Keep the first n chars, cut at a word boundary, mark with an ellipsis."""
    if len(s) <= n:
        return s
    cut = s[:n]
    sp = cut.rfind(" ")
    if sp > n * 0.6:
        cut = cut[:sp]
    return cut.rstrip() + "…"


def clip_start(s: str, n: int) -> str:
    """Keep the last n chars (text right before the cursor matters most)."""
    if len(s) <= n:
        return s
    cut = s[-n:]
    sp = cut.find(" ")
    if 0 <= sp < n * 0.4:
        cut = cut[sp + 1:]
    return "…" + cut.lstrip()


# ---------------------------------------------------------------- context
def _context_lines(ctx: dict) -> list[tuple[int, str]]:
    """Return (drop_priority, line). Higher priority number = dropped first
    when the block is over budget. Priority 0 lines are never dropped."""
    out: list[tuple[int, str]] = []
    privacy = (ctx.get("privacy") or {}).get("level", "full")

    app = ctx.get("app") or {}
    field = ctx.get("field") or {}
    label = clean(app.get("label")) or "unknown app"
    kind = field.get("kind") or "generic"
    if privacy == "none":
        label = "unknown app"
    out.append((0, f"App: {label} — {FIELD_PHRASES.get(kind, 'text field')}"))

    lang = (ctx.get("language") or {}).get("asr")
    if lang:
        out.append((0, f"Dictation language: {LANG_NAMES.get(lang, lang)}"))

    if privacy in ("app_only", "none"):
        return out

    conv = ctx.get("conversation") or {}
    title = clip_end(clean(conv.get("title")), MAX_TITLE_CHARS)
    if title and privacy == "full":
        label = {"chat": "Conversation with", "email_thread": "Email thread", "document": "Document",
                 "page": "Page"}.get(conv.get("kind") or "chat", "Conversation with")
        out.append((1, f"{label}: {title}"))

    if privacy == "full":
        msgs = [m for m in (conv.get("messages") or []) if clean(m.get("text"))][-MAX_MESSAGES:]
        if msgs:
            out.append((3, "Recent messages (oldest first):"))
            for i, m in enumerate(msgs):
                who = "Me" if m.get("is_me") else (clip_end(clean(m.get("author")), 40) or "Them")
                # older messages get a higher drop priority
                out.append((3 + (len(msgs) - 1 - i), f"- {who}: {clip_end(clean(m.get('text')), MAX_MESSAGE_CHARS)}"))

        before = clip_start(clean(field.get("before_cursor"), keep_newlines=False), MAX_BEFORE_CURSOR)
        after = clip_end(clean(field.get("after_cursor"), keep_newlines=False), MAX_AFTER_CURSOR)
        if before:
            out.append((2, f"Text already in the field before the cursor: «{before}»"))
        if after:
            out.append((5, f"Text in the field after the cursor: «{after}»"))

    user = ctx.get("user") or {}
    dic = [clean(t) for t in (user.get("dictionary") or []) if clean(t)][:MAX_DICTIONARY]
    if dic:
        out.append((1, "User's dictionary: " + "; ".join(dic)))

    terms, seen, used = [], {t.casefold() for t in dic}, 0
    for t in sorted(ctx.get("terms") or [], key=lambda x: -float(x.get("score", 0))):
        txt = clip_end(clean(t.get("text")), 60)
        if not txt or txt.casefold() in seen:
            continue
        if len(terms) >= MAX_TERMS or used + len(txt) + 2 > MAX_TERMS_CHARS:
            break
        seen.add(txt.casefold())
        terms.append(txt)
        used += len(txt) + 2
    if terms:
        out.append((2, "Names and terms on screen: " + "; ".join(terms)))

    if privacy == "full":
        snips = [clip_end(clean(s), MAX_SNIPPET_CHARS) for s in (ctx.get("screen_snippets") or [])]
        snips = [s for s in snips if s][:MAX_SNIPPETS]
        if snips:
            out.append((6, "Other text on screen: " + " | ".join(snips)))
    return out


def render_context(ctx: dict | None, ctx2: dict | None = None) -> str:
    """<context> block. ctx v1 (no "v" or v=1): unchanged v1 layout. ctx v2 (v=2, screen = OCR lines, see
    docs/ml/CONTEXT_V2.md): render_context_v2 with options ctx2 (default CTX2_DEFAULT)."""
    if (ctx or {}).get("v") == 2:
        return render_context_v2(ctx, ctx2)
    lines = _context_lines(ctx or {})
    # Drop lowest-value lines until the block fits the budget.
    while sum(len(l) + 1 for _, l in lines) > MAX_CONTEXT_CHARS:
        worst = max(range(len(lines)), key=lambda i: (lines[i][0], i))
        if lines[worst][0] == 0:
            break
        lines.pop(worst)
    # A "Recent messages" header without messages is useless.
    body = [l for _, l in lines]
    if "Recent messages (oldest first):" in body:
        i = body.index("Recent messages (oldest first):")
        if i + 1 >= len(body) or not body[i + 1].startswith("- "):
            body.pop(i)
    return "<context>\n" + "\n".join(body) + "\n</context>"


# ---------------------------------------------------------------- context v2 (screen = raw OCR lines)
# Every step is integer/string arithmetic on the ctx v2 object so that the Kotlin port is byte-identical.
# Options (a profile may override them under "ctx2"; the shipped values are CTX2_DEFAULT):
#   budget     max characters of screen text (sum of len(line) + 1 over the shown lines)
#   order      "near":    lines ranked by vertical distance to the input field, shown nearest first
#              "near_reading": the same selection, shown in reading order
#              "reading": lines taken top to bottom (pipeline order), shown in reading order
#   min_conf   drop OCR lines with conf below this
#   min_chars  drop lines with fewer letters+digits than this
#   drop_clock drop clock/status-like blocks (after merge): <= 12 chars, digits/punctuation plus at most one of
#              am/pm/ч/мин/LTE/5G/4G/G ("12:52", "5G 87%", "9:41 PM")
#   dedup      drop a line whose casefolded text was already kept (repeated UI labels)
#   merge      join wrapped lines into one block: same left edge within 2 % of width, line heights within 10 %,
#              vertical gap below 0.5 line height (measured on real screenshots: wrapped lines 5-13 px apart at
#              42-48 px line height, chat bubbles ~95 px)
#   side       prefix "» " to blocks that are right-aligned (x0 >= 15 % of width, right margin < 10 %)
#   terms      add "Terms on screen: a; b; …" derived from the shown lines (see ocr_terms)
#   line_max   clip every line/block to this many characters
#   place      "before" (shipped) | "after": <context> after the <transcript> (ablation; see user_message)
CTX2_DEFAULT = {"budget": 1400, "order": "near", "min_conf": 0.5, "min_chars": 2, "drop_clock": True,
                "dedup": True, "merge": True, "side": False, "terms": False, "line_max": 240, "place": "before"}
# ctx v2 without a screen: privacy.level the phone sends for each privacy.reason (binding for app, data and eval;
# context.schema.v2.json). app_only hides the field text before the cursor (kind and hint stay); full shows it.
#   secure_window (FLAG_SECURE), denylisted_app                -> app_only: the app/user marked the content sensitive
#   password_field                                             -> none: the app never runs the LLM there (ADR-070,
#                                                                 skipLlm); no eval or training example uses it
#   screenshot_error, ocr_empty                                -> full: nothing sensitive, the screenshot just failed
#   user_setting (screen context switched off by the user)     -> app_only (the only "off" the settings offer; not in
#                                                                 the table, core/context CtxV2JsonTest pins its keys)
NO_SCREEN_LEVEL = {"secure_window": "app_only", "denylisted_app": "app_only", "password_field": "none",
                   "screenshot_error": "full", "ocr_empty": "full"}
_CLOCK = re.compile(r"^[\d\s:.,/%+\-–—()|·•]*(?:[ap]\.?m\.?|ч|мин|lte|5g|4g|g)?[\d\s:.,/%+\-–—()|·•]*$", re.I)
_ALNUM = re.compile(r"[^\W_]", re.U)


def _ocr_blocks(ctx: dict, o: dict) -> list[dict]:
    """Filtered OCR lines -> blocks {"text", "box", "i"} in reading order (pipeline order)."""
    scr = ctx.get("screen") or {}
    w, h = (scr.get("size") or [1080, 2520])[:2]
    fbox = (ctx.get("field") or {}).get("box")
    hint = clean((ctx.get("field") or {}).get("hint")).casefold()
    kept, seen = [], set()
    for ln in scr.get("lines") or []:
        t = clean(ln.get("text"))
        box = ln.get("box") or [0, 0, 0, 0]
        if not t or float(ln.get("conf", 1.0)) < o["min_conf"] or len(_ALNUM.findall(t)) < o["min_chars"]:
            continue
        if fbox and fbox[0] <= (box[0] + box[2]) // 2 <= fbox[2] and fbox[1] <= (box[1] + box[3]) // 2 <= fbox[3]:
            continue  # inside the focused field: its text is given exactly as before_cursor / hint
        if hint and t.casefold() == hint:
            continue
        if o["dedup"]:
            if t.casefold() in seen:
                continue
            seen.add(t.casefold())
        kept.append({"text": t, "box": list(box)})
    blocks: list[dict] = []
    for ln in kept:
        b = blocks[-1] if blocks else None
        lh, bh = ln["box"][3] - ln["box"][1], b["last"][3] - b["last"][1] if b else 0
        aligned = (b is not None and (abs(ln["box"][0] - b["last"][0]) * 50 <= w
                                      or abs(ln["box"][2] - b["last"][2]) * 50 <= w))
        if (o["merge"] and aligned
                and abs(lh - bh) * 10 <= bh and 0 <= ln["box"][1] - b["last"][3]
                and (ln["box"][1] - b["last"][3]) * 2 < max(1, bh)):
            b["text"] += " " + ln["text"]
            b["box"] = [min(b["box"][0], ln["box"][0]), b["box"][1], max(b["box"][2], ln["box"][2]), ln["box"][3]]
            b["last"] = ln["box"]
        else:
            blocks.append({"text": ln["text"], "box": list(ln["box"]), "last": ln["box"]})
    if o["drop_clock"]:  # after merging: a wrapped "15:30" that ends a sentence stays, bubble timestamps go
        blocks = [b for b in blocks if not (len(b["text"]) <= 12 and _CLOCK.match(b["text"]))]
    for i, b in enumerate(blocks):
        b["i"] = i
        b["text"] = clip_end(b["text"], o["line_max"])
        b["right"] = b["box"][0] * 100 >= 15 * w and (w - b["box"][2]) * 10 < w
    return blocks


def _field_anchor_y(ctx: dict, blocks: list[dict]) -> int:
    """Vertical position of the input field: field.box centre; else the screen bottom."""
    fbox = (ctx.get("field") or {}).get("box")
    if fbox:
        return (fbox[1] + fbox[3]) // 2
    return int(((ctx.get("screen") or {}).get("size") or [1080, 2520])[1])


_TERM_STRIP = ".,;:!?()«»\"“”‹›'’-–—…"


def ocr_terms(texts: list[str], limit: int = 20) -> list[str]:
    """Deterministic name/term candidates from OCR text, first occurrence order, casefold-unique, >= 3 chars:
    runs of capitalised words (joined: "Тимур Галиев"; a run at a sentence start needs >= 2 words), tokens with
    an inner digit/#/@/_/./- ("prod-eu-2", "MUR-231"), CamelCase ("ClickHouse"), ALL-CAPS ("NDA"), and Latin
    words of >= 4 letters in a line that is mostly Cyrillic ("nginx")."""
    out, seen = [], set()

    def add(x: str):
        if len(x) >= 3 and x.casefold() not in seen:
            seen.add(x.casefold())
            out.append(x)
    for t in texts:
        lat_n = sum(1 for ch in t if "a" <= ch.lower() <= "z")
        cyr_line = sum(1 for ch in t if "Ѐ" <= ch <= "ӿ") > lat_n
        heb_line = sum(1 for ch in t if "\u0590" <= ch <= "\u05ff" and ch.isalpha()) > lat_n
        toks = t.split()
        run: list[str] = []
        run_start = False

        def flush():
            if run and (len(run) >= 2 or not run_start):
                add(" ".join(run))
            run.clear()
        for j, tok in enumerate(toks):
            core = tok.strip(_TERM_STRIP)
            sent_start = j == 0 or toks[j - 1][-1:] in ".!?…"
            letters = [ch for ch in core if ch.isalpha()]
            special = bool(re.search(r"\w[#@_./\-]\w|[^\W\d_]\d|\d[^\W\d_]", core))
            camel = bool(re.search(r"[a-zа-яё][A-ZА-ЯЁ]", core))
            caps = len(letters) >= 2 and all(ch.isupper() for ch in letters)
            latin = (cyr_line or heb_line) and len(letters) >= 4 and all("a" <= ch.lower() <= "z" for ch in letters)
            if not core:
                flush()
            elif special or camel or caps:
                flush()
                add(core)
            elif core[:1].isupper() and core[1:2].islower():
                if not run:
                    run_start = sent_start
                run.append(core)
                if tok[-1:] in ",.!?…;:":
                    flush()
            else:
                flush()
                if latin:
                    add(core)
        flush()
    return out[:limit]


def render_context_v2(ctx: dict, ctx2: dict | None = None) -> str:
    o = {**CTX2_DEFAULT, **(ctx2 or {})}
    privacy = (ctx.get("privacy") or {}).get("level", "full")
    app, field = ctx.get("app") or {}, ctx.get("field") or {}
    label = clean(app.get("label")) or "unknown app"
    if privacy == "none":
        label = "unknown app"
    kind = field.get("kind") or ("none" if not field else "generic")
    out = [f"App: {label} — {FIELD_PHRASES.get(kind, 'text field')}"]
    hint = clip_end(clean(field.get("hint")), 60)
    if hint and privacy != "none":
        out.append(f"Field hint: «{hint}»")
    lang = (ctx.get("language") or {}).get("asr")
    if lang:
        out.append(f"Dictation language: {LANG_NAMES.get(lang, lang)}")
    if privacy in ("app_only", "none"):
        return "<context>\n" + "\n".join(out) + "\n</context>"
    before = clip_start(clean(field.get("before_cursor")), MAX_BEFORE_CURSOR)
    if before:
        out.append(f"Text already in the field before the cursor: «{before}»")
    blocks = _ocr_blocks(ctx, o)
    if o["order"] == "reading":
        ranked = blocks
    else:
        ay = _field_anchor_y(ctx, blocks)
        ranked = sorted(blocks, key=lambda b: (abs((b["box"][1] + b["box"][3]) // 2 - ay), b["i"]))
    shown, used = [], 0
    for b in ranked:
        s = ("» " if o["side"] and b["right"] else "") + b["text"]
        if used + len(s) + 1 > o["budget"]:
            break
        shown.append((b["i"], s))
        used += len(s) + 1
    if o["order"] == "near_reading":
        shown.sort()
    if shown:
        head = {"near": "Screen text (OCR, nearest to the field first):",
                "near_reading": "Screen text (OCR, top to bottom):",
                "reading": "Screen text (OCR, top to bottom):"}[o["order"]]
        out.append(head)
        out += [s for _, s in shown]
        if o["terms"]:
            terms = ocr_terms([s for _, s in sorted(shown)])
            if terms:
                out.append("Terms on screen: " + "; ".join(terms))
    return "<context>\n" + "\n".join(out) + "\n</context>"


# ---------------------------------------------------------------- prompt
def load_tones() -> dict:
    return json.loads((HERE / "tones" / "tones.v1.json").read_text(encoding="utf-8"))


def style_line(ctx: dict | None, tones: dict | None = None, override: str | None = None) -> str:
    if override:
        return "Style: " + clean(override)
    tones = tones or load_tones()
    cat = ((ctx or {}).get("app") or {}).get("category") or "default"
    variation = ((ctx or {}).get("field") or {}).get("input_type_variation")
    cat = tones.get("field_overrides", {}).get(variation, cat) if variation else cat
    entry = tones["categories"].get(cat) or tones["categories"]["default"]
    return "Style: " + entry["text"]


def _versioned(name: str) -> str:
    """'default' -> 'default.v1' (backward compatible); 'default.v2' stays as is."""
    return name if re.search(r"\.v\d+$", name) else f"{name}.{PROMPT_VERSION}"


def system_prompt(kind: str = "default", preferences: str | None = None,
                  preferences_block: str = "preferences_block") -> str:
    """kind: 'default' | 'finetuned' (v1) or an explicit file stem such as 'default.v2'."""
    base = (HERE / "system" / f"{_versioned(kind)}.md").read_text(encoding="utf-8").rstrip("\n")
    prefs = clean(preferences, keep_newlines=True) if preferences else ""
    if prefs:
        block = (HERE / "system" / f"{_versioned(preferences_block)}.md").read_text(encoding="utf-8").rstrip("\n")
        base += block.replace("{{PREFERENCES}}", prefs[:1000])
    return base


def user_message(ctx: dict | None, transcript: str, style_override: str | None = None,
                 ctx2: dict | None = None) -> str:
    """ctx2: v2 context options of the profile (ignored for ctx v1). ctx2["place"] == "after" (ablation only)
    puts the <context> block after the transcript."""
    if (ctx or {}).get("v") == 2 and (ctx2 or {}).get("place") == "after":
        return (style_line(ctx, override=style_override) + "\n<transcript>\n" + clean(transcript)
                + "\n</transcript>\n" + render_context(ctx, ctx2))
    return (style_line(ctx, override=style_override) + "\n"
            + render_context(ctx, ctx2) + "\n"
            + "<transcript>\n" + clean(transcript) + "\n</transcript>")


def chat(system: str, user: str, template: str = "qwen3.5") -> str:
    """Raw prompt strings, thinking disabled. Verified against the official
    chat_template.jinja of Qwen/Qwen3.5-0.8B and LiquidAI/LFM3-1.7B
    (enable_thinking=False, add_generation_prompt=True) on 2026-11-18."""
    if template == "qwen3.5":
        return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
    if template == "lfm3":
        return (f"<|startoftext|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n<think></think>")
    if template in ("lfm2.5", "qwen3"):
        return chat_turns(system, [], user, template)
    if template == "messages":
        return json.dumps([{"role": "system", "content": system}, {"role": "user", "content": user}],
                          ensure_ascii=False, indent=2)
    raise ValueError(template)


# Chat-template pieces, thinking disabled. Checked against the official chat_template.jinja of
# Qwen/Qwen3.5-0.8B, LiquidAI/LFM3-1.7B, LiquidAI/LFM2.5-1.2B-Instruct, LiquidAI/LFM2.5-350M and
# Qwen/Qwen3-0.6B (enable_thinking=False, add_generation_prompt=True, earlier assistant turns as the
# template renders them) — see prompts/tests/test_render.py.
TEMPLATES = {
    #           bos                prior assistant turn opener          final assistant opener
    "qwen3.5": ("", "<|im_start|>assistant\n", "<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    "qwen3": ("", "<|im_start|>assistant\n", "<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    "lfm3": ("<|startoftext|>", "<|im_start|>assistant\n<think></think>", "<|im_start|>assistant\n<think></think>"),
    "lfm2.5": ("<|startoftext|>", "<|im_start|>assistant\n", "<|im_start|>assistant\n"),
}


def chat_turns(system: str, turns: list[tuple[str, str]], user: str, template: str = "qwen3.5") -> str:
    """System + few-shot (user, assistant) turns + the real user turn, ending with the assistant opener.
    With turns=[] this equals chat(system, user, template)."""
    bos, prior, final = TEMPLATES[template]
    out = [f"{bos}<|im_start|>system\n{system}<|im_end|>\n"]
    for u, a in turns:
        out.append(f"<|im_start|>user\n{u}<|im_end|>\n{prior}{a}<|im_end|>\n")
    out.append(f"<|im_start|>user\n{user}<|im_end|>\n{final}")
    return "".join(out)


# ---------------------------------------------------------------- profiles (v2+)
# A profile = which system prompt, which few-shot turns and which answer prefix. It is the unit the app
# ships and the eval measures: prompts/profiles/<name>.json.
def load_profile(name: str) -> dict:
    p = HERE / "profiles" / f"{name}.json"
    if not p.exists():  # bare system prompt name, e.g. "default" or "default.v1"
        return {"name": name, "system": name, "shots": None}
    prof = json.loads(p.read_text(encoding="utf-8"))
    prof.setdefault("name", name)
    return prof


def profile_turns(profile: dict) -> list[tuple[str, str]]:
    shots = profile.get("shots")
    if not shots:
        return []
    rows = [json.loads(l) for l in (HERE / shots).read_text(encoding="utf-8").splitlines() if l.strip()]
    return [(user_message(r["ctx"], r["transcript"], ctx2=profile.get("ctx2")), r["target"]) for r in rows]


def render_prompt(profile: dict | str, ctx: dict | None, transcript: str, template: str = "qwen3.5",
                  preferences: str | None = None) -> str:
    """Full raw prompt for one request under a profile (system + shots + user + assistant opener)."""
    prof = load_profile(profile) if isinstance(profile, str) else profile
    system = system_prompt(prof["system"], preferences, prof.get("preferences_block", "preferences_block"))
    return (chat_turns(system, profile_turns(prof), user_message(ctx, transcript, ctx2=prof.get("ctx2")), template)
            + prof.get("answer_prefix", ""))


def cache_prefix(profile: dict | str, template: str = "qwen3.5", preferences: str | None = None) -> str:
    """The request-independent prefix (everything before the real user message's content). Its KV state can
    be computed once and restored for every request."""
    prof = load_profile(profile) if isinstance(profile, str) else profile
    system = system_prompt(prof["system"], preferences, prof.get("preferences_block", "preferences_block"))
    full = chat_turns(system, profile_turns(prof), "", template)
    marker = "<|im_start|>user\n"
    return full[: full.rindex(marker) + len(marker)]


def regen_goldens() -> None:
    """Rewrite prompts/examples/golden_*.txt from golden_*.input.json."""
    for inp in sorted((HERE / "examples").glob("golden_*.input.json")):
        g = json.loads(inp.read_text(encoding="utf-8"))
        stem = inp.name[: -len(".input.json")]
        for kind in ("default", "finetuned"):
            for tpl in ("qwen3.5", "lfm3"):
                text = chat(system_prompt(kind, g.get("preferences")), user_message(g["context"], g["transcript"]), tpl)
                (HERE / "examples" / f"{stem}.{tpl}.{kind}.txt").write_text(text, encoding="utf-8")


def regen_goldens_v2() -> None:
    """prompts/examples/ctxv2_*.input.json -> ctxv2_*.context.txt (render_context, shipped CTX2_DEFAULT) and
    ctxv2_*.<template>.<profile>.txt (full prompt) for the shipped v2 profiles."""
    for inp in sorted((HERE / "examples").glob("ctxv2_*.input.json")):
        g = json.loads(inp.read_text(encoding="utf-8"))
        stem = inp.name[: -len(".input.json")]
        (HERE / "examples" / f"{stem}.context.txt").write_text(render_context(g["context"]), encoding="utf-8")
        for prof in GOLDEN_V2_PROFILES:
            for tpl in ("qwen3.5", "lfm3"):
                (HERE / "examples" / f"{stem}.{tpl}.{prof}.txt").write_text(
                    render_prompt(prof, g["context"], g["transcript"], tpl), encoding="utf-8")


GOLDEN_V2_PROFILES = ("default.v1", "default.v4")  # shipped stock profile first (PROMPT_EVAL.md §0), v4 kept selectable


def main(argv: list[str]) -> int:
    import argparse
    if argv[:1] == ["--regen-goldens"]:
        regen_goldens()
        regen_goldens_v2()
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("context")
    ap.add_argument("transcript")
    ap.add_argument("--template", default="qwen3.5")
    ap.add_argument("--system", default="default")
    ap.add_argument("--preferences")
    ap.add_argument("--profile")
    a = ap.parse_args(argv)
    ctx = json.loads(Path(a.context).read_text(encoding="utf-8"))
    tr = Path(a.transcript).read_text(encoding="utf-8")
    prefs = Path(a.preferences).read_text(encoding="utf-8") if a.preferences else None
    if a.profile:
        sys.stdout.write(render_prompt(a.profile, ctx, tr, a.template, prefs))
        return 0
    sys.stdout.write(chat(system_prompt(a.system, prefs), user_message(ctx, tr), a.template))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
