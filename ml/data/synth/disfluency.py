"""Disfluency injection (code, not teacher): fillers, hesitations, stutters and restarts added to the clean
verbatim `spoken` text before the ASR channel. Self-corrections are semantic and come from the teacher.

All insertions are recorded as ops so filters can tell injected fillers from the speaker's own words.

What the speaker says is not what the recogniser writes: Whisper-family models mostly drop hesitation sounds
("э-э", "um") and some repeated words, transducers keep more of them. `render` (per ASR model, measured by the
disfluency probe in ml/data/probe/, see disfluency_render.v1.json) gives the probability that a hesitation sound
and a repeated word survive in the hypothesis; suppressed ops stay in meta with "suppressed": true.

Injected lexical fillers are only words the deployed prompt deletes (prompts/system/default.v5.md rule 1:
э, ну, короче, типа, как бы, значит, вот, um, uh, like, you know, I mean, so). "Слушай", "так", "okay" are
not injected: they often carry meaning in a message (the eval sets keep "Слушай, ...").
"""
from __future__ import annotations

import random
import re

from ml.common import text as T

HES = {"ru": ["э", "э-э", "эм", "м-м", "ээ"], "en": ["um", "uh", "uhm", "er", "hmm"],
       "he": ["אממ", "אמ", "אהה", "אה", "הממ", "ממ"]}
FILLERS = {"ru": ["ну", "вот", "короче", "типа", "как бы", "в общем", "значит", "это самое"],
           "en": ["so", "like", "you know", "I mean", "basically"],
           "he": ["כאילו", "בעצם", "יעני", "נו", "אז", "בקיצור", "כזה"]}
START_FILLERS = {"ru": ["ну", "короче", "в общем", "значит", "ну вот"], "en": ["so", "um so", "so like"],
                 "he": ["אז", "נו", "בקיצור", "כאילו", "טוב אז", "אממ"]}
FULL_RENDER = {"hesitation_keep": 1.0, "repeat_keep": 1.0}
_TOK = re.compile(r"\S+")


def _lang_of(tok: str, default: str) -> str:
    return "he" if T.is_heb(tok) else ("ru" if T.is_cyr(tok) else ("en" if T.is_lat(tok) else default))


def inject(spoken: str, lang: str, messiness: float, r: random.Random, sounds_only: bool = False,
           keep_caps: set[str] = frozenset(), render: dict | None = None) -> tuple[str, list]:
    """Return (what the recogniser hears and writes, ops). messiness 0 = unchanged, ~0.35 light, ~0.8 heavy.
    keep_caps: words that keep their capital after an inserted utterance-initial filler (names, terms).
    render: {"hesitation_keep", "repeat_keep"} of the ASR model (default: everything is written)."""
    rd = render or FULL_RENDER
    rr = random.Random(r.random())  # survival draws on their own stream: the speaker's choices do not change
    if messiness <= 0 or not spoken.strip():
        return spoken, []
    toks = _TOK.findall(spoken)
    base = "he" if lang in ("he", "he_en", "he_ru") else ("en" if lang == "en" else "ru")
    ops: list = []
    n = len(toks)
    budget = max(1, round(messiness * (1 + n / 7)))
    out: list[str] = []
    # utterance-initial filler
    if not sounds_only and r.random() < 0.45 * messiness + 0.15:
        f = r.choice(START_FILLERS[_lang_of(toks[0], base)])
        out.append(f[:1].upper() + f[1:] + ("," if r.random() < 0.3 else ""))
        ops.append({"op": "start_filler", "text": f})
        budget -= 1
        w0 = re.sub(r"\W+$", "", toks[0])
        if w0[:1].isupper() and w0[1:].islower() and w0 not in keep_caps and w0 != "I":
            toks[0] = toks[0][:1].lower() + toks[0][1:]
    for i, tok in enumerate(toks):
        lg = _lang_of(tok, base)
        word = re.sub(r"^\W+|\W+$", "", tok)
        clause_start = i > 0 and re.search(r"[,.!?;:]$", toks[i - 1])
        if budget > 0 and i > 0 and clause_start and r.random() < 0.5 * messiness:
            sound = sounds_only or r.random() < 0.4
            f = r.choice(HES[lg] if sound else FILLERS[lg])
            op = {"op": "filler", "text": f, "at": i}
            if sound and rr.random() >= rd["hesitation_keep"]:
                op["suppressed"] = True
            else:
                out.append(f)
            ops.append(op)
            budget -= 1
        elif budget > 0 and i > 0 and r.random() < 0.08 * messiness:
            f = r.choice(HES[lg])
            op = {"op": "hesitation", "text": f, "at": i}
            if rr.random() >= rd["hesitation_keep"]:
                op["suppressed"] = True
            else:
                out.append(f)
            ops.append(op)
            budget -= 1
        out.append(tok)
        # stutter / repetition of a short word (не "не не": negation doubling can change meaning)
        if (budget > 0 and not sounds_only and word and len(word) <= 6 and word.lower() not in T.NEGATIONS_RU
                and word.lower() not in T.NEGATIONS_EN and word not in T.NEGATIONS_HE
                and not re.search(r"\W$", tok) and r.random() < 0.07 * messiness):
            op = {"op": "repeat", "text": word, "at": i}
            if rr.random() >= rd["repeat_keep"]:
                op["suppressed"] = True
            else:
                out.append(word.lower() if i else word)
            ops.append(op)
            budget -= 1
    # restart: the first two words said twice ("мы сделали мы сделали ...")
    if not sounds_only and n >= 6 and r.random() < 0.12 * messiness:
        k = 2 if not re.search(r"\W$", toks[1]) else 1
        lead = [re.sub(r"\W+$", "", t) for t in toks[:k]]
        start = 1 if ops and ops[0]["op"] == "start_filler" else 0
        op = {"op": "restart", "text": " ".join(lead)}
        if rr.random() >= rd["repeat_keep"]:
            op["suppressed"] = True
        else:
            out[start:start] = lead
        ops.append(op)
    return " ".join(out), ops


def injected_words(ops: list) -> set[str]:
    return {T.norm_word(w) for o in ops if not o.get("suppressed") for w in T.words(o.get("text", ""))}


def hesitation_tokens(text: str) -> int:
    """Pure hesitation sounds in a text (э, э-э, эм, um, uh ...): the per-model filler-token rate in stats."""
    return sum(1 for w in T.words(text) if T.norm_word(w) in T.HESITATION)
