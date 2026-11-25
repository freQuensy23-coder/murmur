#!/usr/bin/env python3
"""Deterministic output guard for Murmur post-processing — executable reference (v1).

The Android (Kotlin) implementation is a port of this file and must pass prompts/tests/test_guard.py
(the cases there are the contract). No third-party dependencies.

Frozen as GUARD_VERSION (below). Tuning protocol: a rule change is accepted or rejected on dev runs only
(prompts/eval/sets/dev*.jsonl); held-out test runs are scored with the frozen guard and never consulted for a
rule change. Any behaviour change bumps GUARD_VERSION (run metadata records it).

Two entry points:

  precheck(transcript) -> Pre(skip: bool, transcript: str, reason: str)
      Runs BEFORE the LLM. skip=True: insert nothing and do not call the LLM (silence/filler-only input or a
      known ASR hallucination). Otherwise returns the transcript with known hallucination tails removed.

  guard(transcript, output, ctx) -> Verdict(text, action, reason)
      Runs AFTER the LLM. action is "accept" (model text, possibly repaired), "fallback" (insert the
      transcript, see fallback_text) or "empty" (insert nothing). `reason` is a short machine code for counters.

Order of the post-LLM rules (first match wins):
  repair  strip <think>…</think>, chat tokens, a leading label ("Output:", "Текст:" …), wrapping quotes/backticks
          that the transcript did not have, and a repeated copy of the field text before the cursor
  R1 empty            output empty but the transcript has a non-filler word                    -> fallback
  R2 leaked_tags      prompt markup in the output (<context>, <transcript>, Style:, <|im_…)    -> fallback
  R3 preamble         assistant preamble ("Sure", "Here is", "Конечно,", "Вот исправленный" …) -> fallback
  R4 loop / length    a word 3-gram repeated >= 4 times that is not repeated in the transcript, or a 1-6 character
                      unit repeated >= 8 times in a row that the transcript does not contain;
                      or non-filler words: out > 1.5*in + 4, or out < 0.3*in when in >= 8;
                      or characters: out > 2.5*in + 40                                         -> fallback
  R5 script           translation: Cyrillic<->Latin share flipped, or a script present in the
                      transcript (>= 8 letters and >= 20 %) vanished from the output (not for
                      code/address fields)                                                      -> fallback
  R5b translated_span  >= 1 word (>= 3 letters) of one script vanished from the output with no phonetic
                      counterpart in it (key similarity < 0.7 to every output word and adjacent word pair),
                      AND >= 1 new word of the other script appeared that sounds like
                      nothing in the transcript (an English half rendered in Russian or vice versa; Cyrillic
                      tech terms rewritten in Latin are phonetic counterparts and do not count)  -> fallback
  R6 novel            >= 3 and > 50 % of output content words (>= 4 letters) absent from the
                      transcript (5-letter stem, or phonetic key similarity >= 0.75) and the context -> fallback
  R7 distractor       a screen name/term (ctx v2: render.ocr_terms of all OCR lines) in the output that was not spoken: its phonetic key matches
                      no 1-4 word span of the transcript (sim < 0.6), or another screen term not
                      in the output matches the transcript better by >= 0.15 (near-miss swap)  -> fallback
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

FILLERS = frozenset("""э ээ эээ э-э э-э-э м мм ммм м-м хм гм а-а ну эм ам
um uh uhm uh-huh erm er hmm hm mm mmm ah eh
אה אהה אההה אה-אה אמ אממ אמממ כאילו המ הממ""".split())

# Whole-transcript ASR hallucinations on silence (Whisper family) and markers. Matched case-insensitively
# against the transcript with punctuation stripped at the ends.
_HALLU_FULL = re.compile(
    r"^(?:\[?\s*(?:blank_audio|silence|music|музыка|тишина|аплодисменты|смех|מוזיקה|שקט|מחיאות כפיים|צחוק)\s*\]?|"
    r"\(?\s*(?:music|музыка|מוזיקה)\s*\)?|"
    r"thanks? (?:you )?for watching!?|thank you for watching!?|please subscribe.*|"
    r"продолжение следует\.*|спасибо за просмотр!?|подписывайтесь на канал.*|"
    r"субтитры (?:сделал|делал|создавал|подготовил)\w*\s.*|редактор субтитров\s.*|"
    r"корректор\s+\S+\.\s*\S+|dimatorzok|"
    r"תודה (?:רבה )?שצפיתם!?|תודה על הצפייה!?|הירשמו לערוץ.*|תרשמו לערוץ.*|"
    r"המשך יבוא\.*|כתוביות.*|עריכת כתוביות\s.*)$", re.I)
# The same phrases when they trail real speech ("Буду через 10 минут. Субтитры сделал DimaTorzok").
_HALLU_TAIL = re.compile(
    r"(?:(?<=[.!?…])\s*|\s+)(?:продолжение следует\.*|субтитры (?:сделал|делал|создавал|подготовил)\w*\s.*|"
    r"редактор субтитров\s.*|thanks? for watching!?|thank you for watching!?|\[blank_audio\]|"
    r"תודה (?:רבה )?שצפיתם!?|תודה על הצפייה!?|המשך יבוא\.*|"
    r"כתוביות\s*(?::|על ידי|ע״י|ע\"י|מאת|נוצרו).*|עריכת כתוביות\s.*)\s*$", re.I)

_THINK = re.compile(r"<think>.*?</think>\s*|^.*?</think>\s*", re.S)
_CHAT_TOKENS = re.compile(r"<\|[a-z_]+\|>|<\|im_end\|>|<\|endoftext\|>")
_LABEL = re.compile(r"^\s*(?:output|edited text|edited|result|corrected text|text|вывод|результат|текст|"
                    r"исправленный текст|отредактированный текст|"
                    r"טקסט מתוקן|טקסט ערוך|טקסט|תוצאה|פלט)\s*:\s*", re.I)
_LEAK = re.compile(r"</?(?:context|transcript|preferences)>|^\s*Style:|<\|im_|<think|Names and terms on screen|"
                   r"Dictation language:|Text already in the field", re.I | re.M)
_PREAMBLE = re.compile(
    r"^\s*(?:sure\b|certainly\b|of course\b|here is\b|here's\b|here are\b|okay, here\b|i'm sorry\b|i cannot\b|"
    r"as an ai\b|конечно[,!]|вот (?:исправленный|ваш|отредактированный|готовый)|исправленный текст|"
    r"отредактированный текст|извините, (?:я|но)|я не могу\b|"
    r"בטח\b|בוודאי\b|כמובן\b|הנה (?:הטקסט|הגרסה)|טקסט מתוקן|אני מצטער[,!]|אני לא יכול\b)", re.I)
_HEB_MARKS = re.compile(r"[\u0591-\u05bd\u05bf\u05c1\u05c2\u05c4\u05c5\u05c7]")
_WORD = re.compile(r"[\w]+(?:[-'’׳״][\w]+)*׳?")
_QUOTE_PAIRS = {'"': '"', "«": "»", "“": "”", "'": "'", "`": "`"}

GUARD_VERSION = "guard.v1.2"  # v1.1 + Hebrew script/phonetics/tokenization/fillers/hallucinations/labels/preambles

GROW_K, GROW_ADD = 1.5, 4
CHAR_GROW_K, CHAR_GROW_ADD = 2.5, 40           # characters: catches blow-ups inside one hyphenated "word"
_CHAR_LOOP = re.compile(r"(.{1,6}?)\1{7,}", re.S)  # a 1-6 char unit repeated >= 8 times in a row ("-5-5-5-5…")
SHRINK_K, SHRINK_MIN_IN = 0.3, 8
NOVEL_MIN, NOVEL_SHARE, NOVEL_SIM = 3, 0.5, 0.75
SCRIPT_MIN_LETTERS = 8
DISTRACTOR_SIM = 0.6
RIVAL_MARGIN = 0.15
NO_SCRIPT_CHECK = {"code", "address"}
SPAN_LOST_MIN, SPAN_SIM, SPAN_KEEP_SIM = 1, 0.5, 0.7


@dataclass(frozen=True)
class Pre:
    skip: bool
    transcript: str
    reason: str


@dataclass(frozen=True)
class Verdict:
    text: str
    action: str  # accept | fallback | empty
    reason: str


# ---------------------------------------------------------------- helpers
def _word_tokens(s: str) -> list[str]:
    return _WORD.findall(_HEB_MARKS.sub("", s or ""))


def _words(s: str) -> list[str]:
    return [w.casefold().replace("ё", "е") for w in _word_tokens(s)]


def _content(ws: list[str]) -> list[str]:
    return [w for w in ws if w not in FILLERS]


def _letters(s: str) -> tuple[int, int, int]:
    lat = sum(1 for ch in s if "a" <= ch.lower() <= "z")
    cyr = sum(1 for ch in s if "Ѐ" <= ch <= "ӿ")
    heb = sum(1 for ch in s if "\u0590" <= ch <= "\u05ff" and ch.isalpha())
    return lat, cyr, heb


_TR = {
    **dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
               ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "i", "k", "l", "m", "n", "o", "p", "r", "s", "t",
                "u", "f", "h", "ts", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"])),
    **dict(zip("אבגדהוזחטיךכלםמןנסעףפץצקרשת",
               ["a", "b", "g", "d", "", "u", "z", "k", "t", "i",
                "k", "k", "l", "m", "m", "n", "n", "s", "a", "f",
                "p", "ts", "ts", "k", "r", "sh", "t"])),
}
_HEB_DIGRAPHS = [("וו", "v"), ("ג׳", "j"), ("ג'", "j"), ("צ׳", "ch"), ("צ'", "ch")]
_PHON = [("sch", "sh"), ("ph", "f"), ("ck", "k"), ("qu", "kv"), ("w", "v"), ("th", "t"), ("x", "ks"), ("c", "k"),
         ("y", "i"), ("j", "dj"), ("q", "k"), ("ou", "au"), ("ee", "i"), ("oo", "u"), ("h", "")]


def phonetic_key(s: str) -> str:
    """Script-independent rough key: Cyrillic and Hebrew transliterated, simple English phonetics, vowels folded, no
    separators, doubled letters collapsed. 'клик хаус' -> 'klakas', 'ClickHouse' -> 'klakas', 'גיטהאב' -> 'gatab'."""
    s = unicodedata.normalize("NFKD", s.casefold())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    for a, b in _HEB_DIGRAPHS:
        s = s.replace(a, b)
    s = "".join(_TR.get(ch, ch) for ch in s)
    s = re.sub(r"[^a-z0-9]", "", s)
    for a, b in _PHON:
        s = s.replace(a, b)
    s = re.sub(r"[aeiou]+", "a", s)
    return re.sub(r"(.)\1+", r"\1", s)


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return 1.0 - _lev(a, b) / max(len(a), len(b))


def best_sim(term: str, transcript: str) -> float:
    """Best similarity between the term's phonetic key and the key of any 1-4 word transcript span."""
    tk = phonetic_key(term)
    if not tk:
        return 1.0
    keys = [phonetic_key(w) for w in _word_tokens(transcript)]
    best = 0.0
    for i in range(len(keys)):
        for n in range(1, 5):
            if i + n <= len(keys):
                best = max(best, _sim(tk, "".join(keys[i:i + n])))
    return best


def spoken(term: str, transcript: str) -> bool:
    """True if `term` plausibly was said."""
    return best_sim(term, transcript) >= DISTRACTOR_SIM


def _render():
    """prompts/render.py (same directory), imported by path so the guard works from any working directory."""
    import importlib.util
    import sys
    from pathlib import Path
    mod = sys.modules.get("render")
    if mod is not None and hasattr(mod, "ocr_terms"):
        return mod
    spec = importlib.util.spec_from_file_location("murmur_prompt_render", Path(__file__).with_name("render.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["render"] = mod
    return mod


def _ocr_lines(ctx: dict) -> list[str]:
    """ctx v2: the texts of all OCR lines (what the model may have seen, before budget trimming)."""
    return [ln.get("text") or "" for ln in ((ctx.get("screen") or {}).get("lines") or [])]


def _screen_terms(ctx: dict) -> list[str]:
    if ctx.get("v") == 2:  # names/terms derived from raw OCR text, the same function as the prompt's Terms line
        return [t for t in _render().ocr_terms(_ocr_lines(ctx), limit=60) if len(t.strip()) >= 3]
    out = [t.get("text", "") for t in (ctx.get("terms") or [])]
    conv = ctx.get("conversation") or {}
    out += [conv.get("title") or ""] + [m.get("author") or "" for m in (conv.get("messages") or [])]
    return [t for t in out if len(t.strip()) >= 3]


def _screen_words(ctx: dict) -> set[str]:
    if ctx.get("v") == 2:
        f = ctx.get("field") or {}
        parts = _ocr_lines(ctx) + [f.get("before_cursor") or "", f.get("hint") or ""]
        return {w[:5] for p in parts for w in _words(p)}
    conv = ctx.get("conversation") or {}
    parts = _screen_terms(ctx) + [m.get("text") or "" for m in (conv.get("messages") or [])]
    parts += list(ctx.get("screen_snippets") or []) + [(ctx.get("field") or {}).get("before_cursor") or ""]
    parts += list((ctx.get("user") or {}).get("dictionary") or [])
    return {w[:5] for p in parts for w in _words(p)}


def _script(w: str) -> str:
    lat, cyr, heb = _letters(w)
    if lat > cyr and lat > heb:
        return "lat"
    if cyr > lat and cyr > heb:
        return "cyr"
    if heb > lat and heb > cyr:
        return "heb"
    return ""


def _translated_span(transcript: str, output: str) -> bool:
    """R5b. Words = casefolded tokens of >= 3 letters, hyphens/underscores split. A lost word has no output word or
    adjacent output pair with key similarity >= SPAN_KEEP_SIM; a new word has no 1-4 word transcript span with key
    similarity >= SPAN_SIM."""
    flat_t, flat_o = re.sub(r"[-_'’]", " ", transcript), re.sub(r"[-_'’]", " ", output)
    tw = [w for w in _content(_words(flat_t)) if len(w) >= 3]
    ow = [w for w in _content(_words(flat_o)) if len(w) >= 3]
    tset, oset = set(tw), set(ow)
    ok1 = [phonetic_key(w) for w in ow]
    okeys = ok1 + [a + b for a, b in zip(ok1, ok1[1:])]  # single output words and adjacent pairs
    for src, dst in (("lat", "cyr"), ("cyr", "lat"), ("lat", "heb"), ("heb", "lat"), ("cyr", "heb"), ("heb", "cyr")):
        lost = [w for w in tw if _script(w) == src and w not in oset
                and max((_sim(phonetic_key(w), k) for k in okeys), default=0.0) < SPAN_KEEP_SIM]
        new = [w for w in ow if _script(w) == dst and w not in tset and best_sim(w, flat_t) < SPAN_SIM]
        if len(set(lost)) >= SPAN_LOST_MIN and new:
            return True
    return False


# ---------------------------------------------------------------- pre-LLM
def precheck(transcript: str) -> Pre:
    t = unicodedata.normalize("NFC", transcript or "").strip()
    bare = t.strip(" .,!?…-—\"'«»()[]")
    if not _content(_words(bare)):
        return Pre(True, "", "filler_only" if bare else "empty")
    if _HALLU_FULL.match(bare):
        return Pre(True, "", "asr_hallucination")
    cut = _HALLU_TAIL.sub("", t).rstrip()
    if cut != t and cut:
        return Pre(False, cut, "tail_removed")
    return Pre(False, t, "ok")


# ---------------------------------------------------------------- post-LLM
def fallback_text(transcript: str, ctx: dict | None = None) -> str:
    """Raw transcript, first letter capitalised unless the field is search/address/code or the cursor is
    mid-sentence."""
    t = (transcript or "").strip()
    ctx = ctx or {}
    cat = (ctx.get("app") or {}).get("category")
    before = ((ctx.get("field") or {}).get("before_cursor") or "").rstrip()
    mid = bool(before) and not re.search(r"[.!?…:\n]$", before)
    if t and cat not in ("search", "address", "code") and not mid:
        t = t[0].upper() + t[1:]
    return t


def repair(output: str, transcript: str, ctx: dict | None = None) -> str:
    o = _THINK.sub("", output or "")
    o = _CHAT_TOKENS.sub("", o).strip()
    o = _LABEL.sub("", o, count=1).strip()
    t = (transcript or "").strip()
    if o.startswith("```") and o.endswith("```") and len(o) > 6 and not t.startswith("```"):
        o = re.sub(r"^```\w*\n?|\n?```$", "", o).strip()
    while len(o) >= 2 and o[0] in _QUOTE_PAIRS and o.endswith(_QUOTE_PAIRS[o[0]]) and not t.startswith(o[0]) \
            and o.count(o[0]) == (2 if o[0] == _QUOTE_PAIRS[o[0]] else 1):
        o = o[1:-1].strip()
    before = (((ctx or {}).get("field") or {}).get("before_cursor") or "").strip()
    if len(before) >= 15 and o.startswith(before):
        o = o[len(before):].lstrip()
    return o


def guard(transcript: str, output: str, ctx: dict | None = None) -> Verdict:
    ctx = ctx or {}
    pre = precheck(transcript)
    if pre.skip:
        return Verdict("", "empty", pre.reason)
    tr = pre.transcript
    fb = fallback_text(tr, ctx)
    o = repair(output, tr, ctx)
    cat = (ctx.get("app") or {}).get("category")
    field_kind = (ctx.get("field") or {}).get("kind")
    tw, ow = _content(_words(tr)), _content(_words(o))

    if not o:
        return Verdict(fb, "fallback", "R1_empty")
    if _LEAK.search(o):
        return Verdict(fb, "fallback", "R2_leaked_tags")
    if _PREAMBLE.search(o) and not _PREAMBLE.search(tr):
        return Verdict(fb, "fallback", "R3_preamble")
    grams = [" ".join(ow[i:i + 3]) for i in range(len(ow) - 2)]
    tgrams = [" ".join(tw[i:i + 3]) for i in range(len(tw) - 2)]
    for g in set(grams):
        if grams.count(g) >= 4 and tgrams.count(g) < 2:
            return Verdict(fb, "fallback", "R4_loop")
    m = _CHAR_LOOP.search(o)
    if m and m.group(0) not in tr:
        return Verdict(fb, "fallback", "R4_loop")
    if len(o) > CHAR_GROW_K * len(tr) + CHAR_GROW_ADD:
        return Verdict(fb, "fallback", "R4_length")
    if len(ow) > GROW_K * len(tw) + GROW_ADD or (len(tw) >= SHRINK_MIN_IN and len(ow) < SHRINK_K * len(tw)):
        return Verdict(fb, "fallback", "R4_length")
    if cat not in NO_SCRIPT_CHECK and field_kind not in NO_SCRIPT_CHECK:
        (tl, tc, th), (ol, oc, oh) = _letters(tr), _letters(o)
        tn, on = tl + tc + th, ol + oc + oh
        if tn >= SCRIPT_MIN_LETTERS and on:
            for t_s, o_s in ((tl, ol), (tc, oc), (th, oh)):
                if t_s >= SCRIPT_MIN_LETTERS and t_s / tn >= 0.2 and o_s / on < 0.05:
                    return Verdict(fb, "fallback", "R5_script")
            for t_s, o_s in ((tc, oc), (th, oh)):
                if t_s / tn >= 0.7 and o_s / on < 0.3:
                    return Verdict(fb, "fallback", "R5_script")
        if _translated_span(tr, o):
            return Verdict(fb, "fallback", "R5b_translated_span")
    allowed = {w[:5] for w in _words(tr)} | _screen_words(ctx)
    content = [w for w in ow if len(w) >= 4 and not w.isdigit()]
    novel = [w for w in content if w[:5] not in allowed and best_sim(w, tr) < NOVEL_SIM]
    if len(novel) >= NOVEL_MIN and len(novel) / len(content) > NOVEL_SHARE:
        return Verdict(fb, "fallback", "R6_novel")
    o_fold, t_fold = o.casefold(), tr.casefold()
    terms = _screen_terms(ctx)
    for term in terms:
        if term.casefold() not in o_fold or term.casefold() in t_fold:
            continue
        s_term = best_sim(term, tr)
        rivals = [best_sim(x, tr) for x in terms if x != term and x.casefold() not in o_fold]
        if s_term < DISTRACTOR_SIM or any(r >= s_term + RIVAL_MARGIN for r in rivals):
            return Verdict(fb, "fallback", "R7_distractor")
    return Verdict(o, "accept", "ok" if o == (output or "").strip() else "repaired")


if __name__ == "__main__":
    import json
    import sys
    a = json.loads(sys.stdin.read())
    print(json.dumps(guard(a["transcript"], a["output"], a.get("ctx")).__dict__, ensure_ascii=False))
