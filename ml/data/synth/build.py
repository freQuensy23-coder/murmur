"""Scenario + teacher example -> contract record(s) (PLAN_V2 stage C, steps 3-5).

  ctx         CONTEXT_V2: teacher screen content -> render spec (injection placed in it) -> ml.screens.build_ctx
              (render -> real OCR); category rules checked on the OCR lines and on the rendered <context>
              block; one re-render with another theme/font scale, then OcrReject. No-screen / no-field cases
              come from the scenario (accessibility side).
  transcript  spoken -> [garble the hard term / misspell the surname] -> disfluencies -> ASR channel
              (mined noise.py or builtin) -> [hallucinated tail]; retried with new seeds until the
              intended text is still recoverable from transcript + context
  target      the teacher's target; for a no-context counterpart the hard term is replaced by the
              transcript's garbled form (the model must not invent a term the screen does not support)

Raises Reject(reason) when an example cannot be built faithfully.
"""
from __future__ import annotations

import copy
import random
import re
from collections import Counter

from ml.common import text as T
from ml.common.render_bridge import render
from ml.common.util import rng as mkrng, stable_id
from ml.data.synth import asr_noise as AN
from ml.data.synth import disfluency as D
from ml.data.synth import screen as SCR
from ml.data.synth import terms as TT
from ml.data.synth.channel import Channel, _lang_runs, _term_garble_implausible
from ml.data.synth.lexicon import KNOWN_TECH
from ml.data.synth.teacher_prompts import GEN_VERSION

NOISE_HES = {"ru": ["э", "э-э", "ээ", "эм", "м-м", "ммм", "хм", "а-а", "э-э-э"],
             "en": ["uh", "um", "uhm", "hmm", "er", "erm", "mm", "ah"],
             "he": ["אממ", "אמ", "אהה", "אה", "הממ", "ממ", "אמממ"]}  # no 'mm-hmm' / 'uh-huh': they can mean yes
NOISE_JOIN = [" ", ", ", "… ", "... "]
NOISE_END = ["", "", "…", ".", "...", "?"]


def noise_only_text(lg: str, r: random.Random, split: str = "train") -> str:
    """What a recogniser writes for a breath, a hesitation or silence: 1-3 hesitation sounds in a recogniser's
    spelling, case and punctuation (Whisper capitalises and adds dots, CTC models write bare lowercase), or a
    subtitle hallucination from the split's own pool (held-out splits never see a train string). Varied, so that the
    exact-duplicate filter does not collapse the category."""
    if r.random() < 0.3:
        tail = r.choice(tails_for(lg, split) + ["…"])
        if tail != "…" and r.random() < 0.4:  # a hesitation the recogniser wrote, then the hallucinated credit
            tail = r.choice(NOISE_HES[lg]).capitalize() + r.choice(["... ", "… ", ". ", ", "]) + tail
        return tail
    toks = [r.choice(NOISE_HES[lg]) for _ in range(r.choice([1, 1, 2, 2, 3]))]
    if lg == "ru" and len(toks) > 1 and r.random() < 0.25:
        toks.insert(1, "ну")
    text = toks[0]
    for t in toks[1:]:
        text += r.choice(NOISE_JOIN) + t
    text += r.choice(NOISE_END)
    if r.random() < 0.4:
        text = text[:1].upper() + text[1:]
    return text


# Whisper-family hallucinations on silence (public reports; the TTS round trip has no silent stretches and shows none:
# 0 of 10,605 hypotheses). Unmistakable credits and channel promos only ('Thank you.' or 'Спасибо за внимание!' can
# be dictated). Train and held-out pools are disjoint, so dev/test measure removal of an unseen string; every string
# the project's eval sets use (prompts/eval: 'Субтитры сделал DimaTorzok', 'Продолжение следует...', 'Thanks for
# watching!', the А.Синецкая credit) and its close variants are held-out only.
TAIL_POOLS = {
    "ru": {"train": ["Подписывайтесь на канал!", "ПОДПИСЫВАЙТЕСЬ НА КАНАЛ", "Спасибо за просмотр!",
                     "Ставьте лайки и подписывайтесь на канал!", "До встречи в следующем видео!"],
           "heldout": ["Продолжение следует...", "Субтитры сделал DimaTorzok", "Субтитры создавал DimaTorzok",
                       "Субтитры делал DimaTorzok", "Редактор субтитров А.Синецкая Корректор А.Егорова",
                       "Редактор субтитров А.Семкин Корректор А.Егорова"]},
    "en": {"train": ["Thank you for watching.", "Thank you for watching!", "Please subscribe to my channel.",
                     "Please like and subscribe.", "See you in the next video!"],
           "heldout": ["Thanks for watching!", "Thanks for watching, and I'll see you next time.",
                       "Subtitles by the Amara.org community", "Transcription by CastingWords"]},
    "he": {"train": ["הירשמו לערוץ!", "תודה רבה שצפיתם!", "נתראה בסרטון הבא!",
                     "עשו לייק והירשמו לערוץ!", "כתוביות אוטומטיות בעברית"],
           "heldout": ["תודה שצפיתם!", "תודה שצפיתם בסרטון.", "כתוביות בעברית",
                       "המשך יבוא...", "עריכת כתוביות בעברית"]},
}
# a decoder stutter loop seen in the real round trip (Nemotron 3.5 streaming: 'через с с с с с сентенс Трансформерс')
LOOP_MODEL_HINT = "nemotron-3.5"
LOOP_SHARE = 0.25


def tails_for(lg: str, split: str) -> list[str]:
    key = "he" if lg in ("he", "he_en", "he_ru") else ("en" if lg == "en" else "ru")
    return TAIL_POOLS[key]["train" if split == "train" else "heldout"]


def stutter_loop(hyp: str, r: random.Random) -> tuple[str, str | None]:
    """Insert a decoder loop of the next word's first letter ('с с с с сентенс') before a word of 4+ letters."""
    ws = hyp.split(" ")
    cand = [i for i in range(1, len(ws)) if len(re.sub(r"\W", "", ws[i])) >= 4 and ws[i][:1].isalpha()]
    if not cand:
        return hyp, None
    i = r.choice(cand)
    loop = " ".join([ws[i][:1].lower()] * r.randint(3, 5))
    ws.insert(i, loop)
    return " ".join(ws), loop


INJECTION_FORBID = {
    "OK": ["OK", "ОК"], "APPROVED": ["APPROVED"], "принято": ["Принято"], "Sent from my iPhone": ["Sent from my iPhone"],
    "Отправлено с Android": ["Отправлено с Android"], "I agree to the terms": ["I agree to the terms"],
    "Согласен с условиями": ["Согласен с условиями"], "bit.ly": ["bit.ly"], "t.me/bonus_promo": ["t.me/bonus_promo"],
    "YES": ["YES"], "заказ отменён": ["заказ отменён"],
    "אישור": ["אישור"], "מאושר": ["מאושר"], "נשלח מהאייפון שלי": ["נשלח מהאייפון שלי"],
    "אני מסכים לתנאים": ["אני מסכים לתנאים"],
}
NOANSWER = {"dictated_question", "dictated_command", "ai_addressed"}


class Reject(Exception):
    pass


def fold(s: str) -> str:
    return re.sub(r"[\s\-_.]+", " ", T.nfc(s).lower().replace("ё", "е")).strip()


def _contains(hay: str, needle: str) -> bool:
    n = fold(needle)
    return bool(n) and re.search(r"(?<!\w)" + re.escape(n), fold(hay)) is not None


# ------------------------------------------------------------------ context (v2: render -> OCR)
class OcrReject(Reject):
    """A category rule failed on the OCR output even after the one allowed re-render."""

    def __init__(self, rule: str, term: str | None):
        super().__init__("ocr:" + rule)
        self.rule, self.term = rule, term


def rendered_block(ctx: dict) -> str | None:
    """The <context> block of the CURRENT prompts/render.py, or None while it does not render ctx v2 screens."""
    return render().render_context(ctx) if SCR.renderer_shows_screen() else None


def ctx_text(ctx: dict) -> str:
    """Everything the model can see about the screen (for the recoverability check)."""
    blk = rendered_block(ctx)
    return blk if blk is not None else "\n".join(SCR.lines_of(ctx)) + "\n" + ((ctx.get("field") or {}).get("before_cursor") or "")


def rule_failure(sc: dict, ctx: dict, target: str) -> tuple[str, str | None] | None:
    """CONTEXT_V2 "Category rules checked on OCR output": on ctx.screen.lines AND on the rendered <context> block
    (the renderer trims: a needed line that was trimmed away teaches the model to invent). -> (rule, term) | None."""
    cat = sc["category"]
    blk = rendered_block(ctx)

    def shown(s: str) -> bool:
        return SCR.on_screen(ctx, s) and (blk is None or SCR.exact_in_text(blk, s))

    def hidden(s: str) -> bool:  # neither s nor a close variant anywhere
        return not SCR.close_on_screen(ctx, s) and (blk is None or not SCR.close_in_text(blk, s))

    if cat in SCR_NEEDED and not ctx.get("screen"):
        return "no_screen_lines", None
    if cat == "context_resolution":
        t = sc["spoken_terms"][0]["text"]
        if not shown(t):
            return "term_not_readable", t
    elif cat == "context_names":
        for p in sc["people"]:
            if not shown(p):
                return "name_not_readable", p
    elif cat == "near_miss":
        said = [t["text"] for t in sc["spoken_terms"]] or [p.split()[-1] for p in sc["people"]]
        for d in sc["distractors"]:
            if not shown(d):
                return "distractor_not_readable", d
        for x in said:
            hay = " ".join(SCR.lines_of(ctx)) + " " + (blk or "")
            if re.search(r"(?<!\w)" + re.escape(SCR._cf(x)) + r"(?!\w)", SCR._cf(hay)):
                return "said_item_on_screen", x
    elif cat == "injection":
        inj = sc["injection"]["text"]
        if not SCR.fuzzy_on_screen(ctx, inj) or (blk is not None and not SCR.fuzzy_on_screen(
                {"screen": {"lines": [{"text": blk}]}}, inj)):
            return "injection_not_visible", inj[:40]
    return None


SCR_NEEDED = {"context_resolution", "context_names", "near_miss", "injection"}


def make_ctx(sc: dict, content: dict, backend: SCR.Backend, target: str, check: bool = True):
    """-> (ctx, screen_meta). Real render + OCR; on a rule failure ONE re-render (other theme/font), then OcrReject."""
    bc = (content.get("before_cursor") or "").strip()
    if sc.get("screen_absent"):
        return SCR.absent_ctx(sc, bc), {"absent": sc["screen_absent"]}
    fail = None
    for variant in (0, 1):
        spec = SCR.make_spec(sc, content, variant)
        ctx = SCR.finalize(backend.build_ctx(spec), sc)
        if sc["category"] == "continuation" and bc and "field" in ctx:
            ctx["field"]["before_cursor"] = bc  # accessibility gives the field text exactly
        fail = rule_failure(sc, ctx, target) if check else None
        if not fail:
            return ctx, {"spec": spec, "variant": variant, "backend": backend.version, "render": backend.last_meta,
                         "render_check":
                         "render.py" if SCR.renderer_shows_screen() else "skipped:render.py_has_no_v2_screen"}
    raise OcrReject(*fail)


def counterpart_ctx(sc: dict, spec: dict, term: str, backend: SCR.Backend):
    """Re-render the paired screen without the supporting item(s) and run OCR again (lines are never edited)."""
    try:
        base = SCR.strip_spec(spec, term)
    except ValueError as e:
        raise Reject(str(e))
    for variant in (0, 1):
        sp = base if variant == 0 else base | {"theme": "dark" if base["theme"] == "light" else "light",
                                               "seed": (base["seed"] * 7 + 1) % (1 << 30)}
        ctx = SCR.finalize(backend.build_ctx(sp), sc)
        blk = rendered_block(ctx)
        if not SCR.close_on_screen(ctx, term) and (blk is None or not SCR.close_in_text(blk, term)):
            return ctx, {"spec": sp, "variant": variant, "backend": backend.version, "render": backend.last_meta,
                         "stripped": term}
    raise OcrReject("counterpart_term_still_visible", term)


# ------------------------------------------------------------------ garbling
def _term_variants_mined(channel: Channel, model: str, term: str, lang: str) -> list[tuple[str, int]]:
    if channel.nm is None:
        return []
    by_lang = channel.nm._terms.get(model, {}).get(term) or {}
    # which mined sentences may lend a form: English speech only from English sentences (a mixed sentence's
    # Cyrillic 'декод' is not how a recogniser writes 'decode' inside English speech); Russian speech from Russian
    # and mixed ones; mixed speech from all
    ok = {"en": {"en"}, "ru": {"ru", "mixed"}}.get(lang, {"en", "ru", "mixed"})
    out = []
    for lg, cnt in by_lang.items():
        if lg in ok:
            out += [(h, c) for h, c in cnt.items() if h and h.lower() != term.lower()]
    return out


def _term_variants_other(channel: Channel, model: str, term: str, lang: str) -> list[tuple[str, int]]:
    """The term's mined manglings by the OTHER recognisers (same speech-language rule as above): real renderings
    of this very term, used when the chosen recogniser was never heard saying it."""
    if channel.nm is None:
        return []
    ok = {"en": {"en"}, "ru": {"ru", "mixed"}}.get(lang, {"en", "ru", "mixed"})
    out: Counter = Counter()
    for m, terms in channel.nm._terms.items():
        if m == model:
            continue
        for lg, cnt in (terms.get(term) or {}).items():
            if lg in ok:
                for h, c in cnt.items():
                    if h and h.lower() != term.lower():
                        out[h] += c
    return sorted(out.items())


def _has_lat(s: str) -> bool:
    return bool(re.search(r"[A-Za-z]", s))


def latin_garble(term: str, hb: dict, r: random.Random) -> str:
    """A Latin misspelling made the way the mined sampler garbles an unseen term (vowel swap, dropped double or
    inner letter: ml/roundtrip/mine/noise.perturb_word), kept only if the aligner counts it as a recognition error and
    the recogniser's mined habits allow that kind of letter drop."""
    from ml.roundtrip.mine.align import is_error_variant
    from ml.roundtrip.mine.noise import perturb_word
    for _ in range(12):
        ws = term.split()
        i = r.randrange(len(ws))
        ws[i] = perturb_word(ws[i], r)
        v = " ".join(ws)
        if v != term and is_error_variant(term, v) and not _term_garble_implausible(term, v, hb):
            return v
    return ""


LATIN_BOOST = 2.0


def garble_form(term: dict, lang: str, channel: Channel, model: str, prev_word: str, r: random.Random,
                need_cyrillic: bool = False) -> tuple[str, str]:
    """How the recogniser spells a hard term it does not know. -> (form, source).

    Russian / mixed speech: the script follows the recogniser's mined habit (share of Latin misspellings among its
    term errors: 0.02 for T-one, 0.3-0.45 for Whisper / Parakeet / GigaAM), Cyrillic only for a scenario that needs
    a no-context counterpart. Forms: this recogniser's mined manglings of the term, else another recogniser's mined
    manglings of it, else a Latin misspelling made like the mined sampler's (Latin) or the probe lexicon's reading /
    a transliteration (Cyrillic). English speech: mined manglings, the lexicon's spoken forms, CamelCase split, else
    the sampler-style misspelling (no generic respelling rules: no recogniser in the round trip shows them)."""
    text = term["text"]
    hb = channel.habits(model, lang)
    own = _term_variants_mined(channel, model, text, lang)
    other = _term_variants_other(channel, model, text, lang)
    if lang in ("he", "he_en", "he_ru"):
        form = AN.heb_form(text, r)
        return form, "heb_translit"
    if lang in ("ru", "mixed"):
        # the paired scenarios (210 of 330 train records) must stay Cyrillic, so the others draw Latin at twice the
        # recogniser's measured share (cap 0.8): the set keeps a real number of 'Latin misspelling resolved from
        # the screen' examples
        p_lat = 0.0 if need_cyrillic else min(0.8, LATIN_BOOST * ((1.0 - hb["cyr_share"]) if hb.get("err") else 0.3))
        want_lat = r.random() < p_lat

        def ok(h: str) -> bool:
            if want_lat:
                return _has_lat(h) and not T.is_cyr(h) and not _term_garble_implausible(text, h, hb)
            return T.is_cyr(h) and not _has_lat(h)
        for src, forms in (("mined", own), ("mined_other", other)):
            fs = [(h, c) for h, c in forms if ok(h)]
            if fs:
                hs, w = zip(*fs)
                return r.choices(hs, weights=w)[0], src
        if want_lat:
            form = latin_garble(text, hb, r)
            if form:
                return form, "garbled_latin"
        if term.get("ru_spoken"):
            form = r.choice(term["ru_spoken"])
            src = "lexicon"
        else:
            form = " ".join(T.lat2cyr_rough(w) for w in T.words(text))
            src = "translit"
        return AN._case_ending(prev_word, form, r), src
    for src, forms in (("mined", own), ("mined_other", other)):
        fs = [(h, c) for h, c in forms if _has_lat(h) and not _term_garble_implausible(text, h, hb)]
        if fs and (src == "mined" or r.random() < 0.7):
            hs, w = zip(*fs)
            return r.choices(hs, weights=w)[0], src
    if term.get("en_spoken") and r.random() < 0.6:
        form = r.choice(term["en_spoken"])
        if form.lower() != text.lower():
            return form, "lexicon"
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", text)
    if len(parts) > 1 and r.random() < 0.5:
        return " ".join(p.lower() for p in parts), "split"
    form = latin_garble(text, hb, r)
    if not form and len(parts) > 1:
        form = " ".join(p.lower() for p in parts)
    return form, "garbled_latin"


_SPOKEN_SYM = {"dot": ".", "точка": ".", "נקודה": ".", "slash": "/", "слэш": "/", "слеш": "/", "סלאש": "/",
              "dash": "-", "дефис": "-", "מקף": "-", "underscore": "_", "плюс": "+", "plus": "+", "פלוס": "+",
              "שטרודל": "@"}


def term_key(s: str) -> str:
    """'Android Manifest dot XML' and 'AndroidManifest.xml' get the same key."""
    ws = [_SPOKEN_SYM.get(w.lower(), w.lower()) for w in re.findall(r"[\w+#]+|[./]", T.nfc(s))]
    return "".join(ws).replace("ё", "е").replace("-", "").replace("_", "")


def find_term(spoken: str, term: str) -> list[tuple[int, int]]:
    """Character spans of `term` in `spoken`: exact (case-insensitive) or a spoken spelling of it."""
    pat = re.compile(r"(?:(?<![\w-])|(?<=[\u05d0-\u05ea]-))" + re.escape(term) + r"(?![\w])", re.I)
    spans = [(m.start(), m.end()) for m in pat.finditer(spoken)]
    if spans:
        return spans
    key = term_key(term)
    toks = [(m.start(), m.end()) for m in re.finditer(r"\S+", spoken)]
    for i in range(len(toks)):
        for j in range(i + 1, min(len(toks), i + 6) + 1):
            a, b = toks[i][0], toks[j - 1][1]
            chunk = spoken[a:b].rstrip(",.!?;:»)\"'")
            if term_key(chunk) == key:
                return [(a, a + len(chunk))]
    return []


def plausible_garble(form: str, term: str) -> bool:
    """A garble must still sound like the whole term: 'клик хаус' for ClickHouse yes, 'пуш' for
    'push notification' or 'кубер' for Kubernetes no (restoring those would teach the model to add words)."""
    if re.fullmatch(r"[A-Z0-9/]{2,6}", term):  # acronyms are spelled letter by letter: 'си ви и' for CVE
        return len(form.split()) >= len(re.sub(r"\W", "", term)) - 1 or len(form) >= 3
    kf, kt = T.phonetic_key(form), T.phonetic_key(term)
    min_len = 0.45 if T.is_heb(form) else 0.6
    return len(kf) >= min_len * len(kt) and T.similarity(kf, kt) >= 0.35


def _replace_term(spoken: str, term: str, form_fn) -> tuple[str, int]:
    spans = find_term(spoken, term)
    out, last = [], 0
    for a, b in spans:
        prev = spoken[:a].split()[-1:] or [""]
        out += [spoken[last:a], form_fn(re.sub(r"\W", "", prev[0]))]
        last = b
    out.append(spoken[last:])
    return "".join(out), len(spans)


def misspell_surname(spoken: str, person: str, r: random.Random) -> tuple[str, str | None, str | None]:
    sur = person.split()[-1]
    stem = T.norm_word(sur)[: max(3 if T.is_heb(sur) else 4, len(sur) - 2)]
    toks = spoken.split()
    for i, tok in enumerate(toks):
        core = re.sub(r"^\W+|\W+$", "", tok)
        if core and T.norm_word(core).startswith(stem):
            for _ in range(6):
                if T.is_heb(core):
                    new = AN.respell_he(core, r)
                elif T.is_cyr(core):
                    new = AN.respell_ru(core, r)
                else:
                    new = AN.respell_en(core, r)
                if new != core and fold(new) != fold(core):
                    toks[i] = tok.replace(core, new)
                    return " ".join(toks), core, new
            return spoken, None, None
    return spoken, None, None


def is_well_known(term: str) -> bool:
    """A term a writer spells right with no screen support (terms.WELL_KNOWN, lexicon.KNOWN_TECH, plain English
    words): the documented exception to CONTEXT_V2's binding rule. A Cyrillic inflection ('NLP-сервиса') counts by
    its Latin part. Such terms never get a no-context counterpart (terms.py), so no pair contradicts the rule."""
    core = re.sub(r"[-‐]?[А-Яа-яЁё\u05d0-\u05ea]+", "", term).strip(" -")
    if not core:
        return False
    low = {t.lower() for t in TT.WELL_KNOWN | KNOWN_TECH}
    return core.lower() in low or TT._plain_english(core)


def term_restore_ok(term: str, form: str) -> bool:
    """May the target write `term` where the transcript has the recogniser's `form`, with no screen support?
    The mechanical form of CONTEXT_V2's binding rule (a garbled word is never fixed towards a spelling the context
    does not contain): the target keeps the transcript's rendering unless the term is on screen (checked by the
    caller) or WELL-KNOWN (is_well_known: exactly the terms that never get a no-context counterpart, so no pair
    contradicts it). Always yes for a spelling variant of the same word ('docusign', 'Elastic Search', 'Zero Shot').
    For a well-known term also a near-identical Latin misspelling ('Confloence'; spelling similarity >= 0.8, or one
    letter off for 4+ letters: 'Zom', 'Jera'), a Cyrillic rendering that sounds like it ('докир' for Docker,
    'Джира') or a letter-by-letter acronym ('ди эн эс' for DNS). Never another word ('Traffic' for Traefik,
    'Equalization' for quantization, 'EWS' for AWS), and nothing else for any other term ('трефик' for Traefik,
    'апентабл' for OpenTable, 'крем' for cream, 'Parqet' for Parquet): the counterparts keep 'трэфик'."""
    if term_key(form) == term_key(term) or re.sub(r"\W", "", form).lower() == re.sub(r"\W", "", term).lower():
        return True
    if T.is_cyr(form) or T.is_heb(form):
        if not is_well_known(term):
            return False
        if re.fullmatch(r"[A-Z0-9/]{2,6}", term) and len(form.split()) >= len(re.sub(r"\W", "", term)) - 1:
            return True
        return T.phon_sim(term, form) >= (0.55 if T.is_heb(form) else 0.7)
    if not is_well_known(term):
        return False  # 'Parqet', 'tstflight': only the screen may say how a product name is spelled
    a, b = term.lower(), form.lower()
    return T.similarity(a, b) >= 0.8 or (len(a) >= 4 and _edit1(a, b))


# ------------------------------------------------------------------ real-word substitutions (guessing)
# Russian noun / adjective endings: a swap between two of them on the same stem is an agreement repair grammar
# resolves ('до понедельник' -> 'до понедельника'); verb person, mood, tense or gender is not ('Проверю' for
# 'Проверь', 'можешь' for 'можете', 'успел' for 'успела')
NOMINAL_ENDINGS = {"", "а", "я", "ы", "и", "у", "ю", "е", "о", "ь", "ой", "ей", "ом", "ем", "ам", "ям", "ах", "ях",
                   "ами", "ями", "ов", "ев", "ий", "ый", "ая", "яя", "ое", "ее", "ую", "юю", "ых", "их", "ым", "им",
                   "ого", "его", "ому", "ему", "ые", "ие", "ою", "ею", "ью"}
VERBAL_PAIRS = {frozenset(p) for p in (("ь", "ю"), ("ь", "у"), ("й", "ю"), ("й", "у"))}
EN_HOMOPHONES = [{"their", "there", "they're"}, {"your", "you're"}, {"its", "it's"}, {"then", "than"},
                 {"here", "hear"}, {"whether", "weather"}, {"know", "no"}, {"right", "write"}, {"buy", "by"},
                 {"new", "knew"}, {"wait", "weight"}, {"where", "wear", "were"}, {"whose", "who's"}]
VOCAB: set[str] = set()  # words of every teacher 'spoken' text (generate.set_vocab), plus the channel's real words


def set_vocab(words) -> None:
    VOCAB.clear()
    VOCAB.update(words)


def same_lemma(a: str, b: str) -> bool:
    """Two forms of one word whose difference grammar restores: a Russian noun/adjective ending swap on a stem of
    3+ letters, an English plural / possessive / -ed / -ing, or an English homophone pair (their/there)."""
    a, b = T.norm_word(a), T.norm_word(b)
    if T.is_cyr(a) and T.is_cyr(b):
        for k in range(3, min(len(a), len(b)) + 1):
            if a[:k] != b[:k]:
                break
            ea, eb = a[k:], b[k:]
            if ea in NOMINAL_ENDINGS and eb in NOMINAL_ENDINGS and frozenset((ea, eb)) not in VERBAL_PAIRS \
                    and not (a[:k].endswith("л") and {ea, eb} <= {"", "а", "о", "и"}) \
                    and not (a[:k].endswith("т") and {ea, eb} == {"ь", ""}):  # успел / успела, проверить / проверит
                return True
        return False
    s, l = sorted((a, b), key=len)
    if any(l == s + suf for suf in ("s", "es", "'s", "ed", "ing")) or (s.endswith("e") and l in (s + "d", s[:-1] + "ing")):
        return True
    return any(a in g and b in g for g in EN_HOMOPHONES)


def real_word_guess(spoken_w: str, hyp_w: str, target: str, ctx_text: str, vocab_extra=frozenset()) -> bool:
    """The channel replaced a spoken word (4+ letters) by ANOTHER real word, and the target restores the spoken one
    with no support on screen: 'Would' for 'Could', 'появилось' for 'отвалилось', 'Проверю' for 'Проверь'. A cleanup
    model cannot know which word was said, so the example would teach it to guess."""
    a, b = T.norm_word(spoken_w), T.norm_word(hyp_w)
    if len(a) < 4 or a == b or len(b) < 3:
        return False
    tw = set(T.norm_words(target))
    if a not in tw or b in tw:
        return False
    if a in set(T.norm_words(ctx_text)):
        return False
    if b not in VOCAB and b not in vocab_extra:
        return False
    return not same_lemma(a, b)


def guessed_subs(events: list[str], target: str, ctx_text: str, vocab_extra=frozenset()) -> list[str]:
    out = []
    for e in events:
        if e.startswith("sub:") and "->" in e:
            a, b = e[4:].split("->", 1)
            if len(a.split()) == 1 and len(b.split()) == 1 and real_word_guess(a, b, target, ctx_text, vocab_extra):
                out.append(e)
    return out


def _edit1(a: str, b: str) -> bool:
    """One insertion, deletion or substitution apart ('zom' / 'zoom', 'lest' / 'list')."""
    if abs(len(a) - len(b)) > 1 or a == b:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))


def unsupported_term_garbles(events: list[str], target: str, ctx: dict, skip: tuple = ()) -> list[str]:
    """Channel garbles of terms (the planner's spoken terms and any term the mined tables recognise in the speech)
    that the target writes correctly although the term is not on screen and the garble is not a safe spelling
    variant (term_restore_ok). `skip`: the context_resolution term (its garble is the point of the example)."""
    bad = []
    for e in events:
        m = re.match(r"term:(.+?)->(.*?):[a-z_]+$", e)
        if not m or m.group(1) in skip or not m.group(2) or m.group(2) == "∅":
            continue
        if _contains(target, m.group(1)) and not SCR.on_screen(ctx, m.group(1)) and not term_restore_ok(m.group(1), m.group(2)):
            bad.append(e)
    return bad


# ------------------------------------------------------------------ recoverability
_LIST_MARK = re.compile(r"(?m)^\s*\d+[.)]\s+")


def _vouches(w: str, h: str) -> bool:
    """Can the transcript word h be a garbled hearing of the target word w? Never a single letter ('п' is no
    evidence for 'Pro'); same script: spelling similarity >= 0.6; other script ('докер'/'דוקר' for 'docker'):
    phonetic similarity >= 0.75 (0.55 when Hebrew omits short vowels)."""
    if len(h) < 2 or (len(w) >= 4 and len(h) < 3):
        return False
    if (T.is_cyr(w), T.is_heb(w)) == (T.is_cyr(h), T.is_heb(h)):
        if T.similarity(w, h) >= 0.6:
            return True
        # Hebrew homophone slip ('טודה'/'תודה') or joined/split prefix ('בבית' vs 'בית')
        if T.is_heb(w) and T.phon_sim(w, h) >= 0.65:
            return True
        return False
    return T.phon_sim(w, h) >= (0.55 if (T.is_heb(w) or T.is_heb(h)) else 0.75)


def unrecoverable(target: str, hyp: str, ctx_text: str, allow: set[str] = frozenset()) -> list[str]:
    """Target content words and numbers that neither the transcript nor the context supports."""
    hw = T.norm_words(hyp)
    hs = set(hw)
    joined = {a + b for a, b in zip(hw, hw[1:])} | {a + "-" + b for a, b in zip(hw, hw[1:])}
    cs = set(T.norm_words(ctx_text))
    hnums = set(T.numbers_in(hyp))
    tw = sorted(set(T.norm_words(target)))
    free = hs - set(tw)
    bad = []
    for w in T.content_words(target):
        nums = T.numbers_in(w)
        if nums and set(nums) <= hnums:  # '15:30' <- 'пятнадцать тридцать'; 'первом' needs a 1 in the hypothesis
            continue
        if w in hs or w in cs or w in joined or w in allow:
            continue
        # Hebrew single-letter prefix attached to a term/word: 'ל-github' or 'בסלאק'
        he_stem = w[2:] if len(w) > 3 and w[0] in "בלהומשכ" and w[1] == "-" else None
        if he_stem and (he_stem in hs or he_stem in cs or he_stem in joined or he_stem in allow):
            continue
        # a hypothesis word that is itself an exact target word cannot vouch for another one
        # ('Saturday' must not make a garbled 'Sunday' look recoverable)
        # and a garbled word vouches only for the target word it is closest to ('Seturday' is Saturday's)
        if any(_vouches(w, h) and _closest(h, tw) == w for h in free):
            continue
        if he_stem and any(_vouches(he_stem, h[1:] if len(h) > 3 and h[0] == w[0] else h) for h in free):
            continue
        bad.append(w)
    # every number in the target must be heard (digits or number words) or be on screen: 'в съем вечера' is no
    # evidence for '6 вечера'
    missing = set(T.numbers_in(_LIST_MARK.sub("", target))) - hnums - set(T.numbers_in(ctx_text))
    bad += [str(n) for n in sorted(missing)]
    return bad


def _closest(h: str, words: list[str]) -> str:
    return max(words, key=lambda t: (max(T.similarity(t, h), T.phon_sim(t, h)), t)) if words else ""


# words whose loss a reader restores from grammar alone (a preposition, conjunction or article); losing anything
# else (negation, a question word or auxiliary, a pronoun, a content word, a spoken command) makes the target say
# what the transcript does not ('So ask Dmitri to drive' -> 'Can you ask Dmitri to drive?')
DELETABLE = {"в", "во", "у", "и", "с", "со", "к", "ко", "из", "на", "при", "по", "о", "об", "от", "до", "за", "а",
             "the", "a", "an", "of", "to", "in", "on", "at", "by", "for", "and", "it",
             "את", "של", "על", "אל", "עם", "או", "גם", "כי", "אז"}


def unrecoverable_deletions(events: list[str], target: str) -> list[str]:
    """Deleted words the target still contains and grammar cannot restore (fillers the target drops are fine)."""
    tw = set(T.norm_words(target))
    bad = []
    for e in events:
        if e.startswith("del:"):
            for w in T.norm_words(e[4:]):
                if w in tw and w not in DELETABLE:
                    bad.append(e)
                    break
    return bad


def implausible_subs(events: list[str]) -> list[str]:
    """Substitutions and insertions no recogniser makes from the audio ('Sunday' -> 'Utxpyri', 'по' -> 'свенство',
    'на' -> 'аж', inserted 'папы'). Mined confusion tables contain such alignment artefacts; an example built on them
    teaches the model to guess, so the attempt is retried with another seed. (Channel.implausible adds the
    model-specific term-garble checks.)"""
    from ml.data.synth.channel import plausible_ins, plausible_sub
    out = []
    for e in events:
        if e.startswith("sub:") and "->" in e:
            a, b = e[4:].split("->", 1)
            wa, wb = a.split(), b.split()
            pairs = list(zip(wa, wb)) if len(wa) == len(wb) else [(a.replace(" ", ""), b.replace(" ", ""))]
            if not all(plausible_sub(x, y) for x, y in pairs):
                out.append(e)
        elif e.startswith("ins:") and not all(plausible_ins(w) for w in e[4:].split()):
            out.append(e)
    return out


# ------------------------------------------------------------------ target consistency
# teacher examples dropped by name after the 2026-11-20 data review: semantically odd ('include SRE in the grocery
# order') or a guessed transliteration of a Russian word in a URL path ('расписание' -> 'raspisanie')
BAD_TEACHER_EXAMPLES = {"sc-6a9f1251f168": "SRE in a grocery order", "sc-af6c2f4d893a": "ritmfit.ru/raspisanie"}
_EN_NUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}
_RU_NUM = {}
for _v, _forms in {1: "один одна одну одной одного одним одном", 2: "два две двух двум двумя",
                   3: "три трёх трех трём трем тремя", 4: "четыре четырёх четырех четырём четырем четырьмя",
                   5: "пять пяти пятью", 6: "шесть шести шестью", 7: "семь семи семью", 8: "восемь восьми восемью",
                   9: "девять девяти девятью", 10: "десять десяти десятью"}.items():
    for _f in _forms.split():
        _RU_NUM[_f] = _v
_EN_UNITS = r"(?:minutes?|hours?|seconds?)"
_RU_UNITS = r"(?:минут[аыуе]?|минутами|минутам|час(?:а|ов|у|ом|ам|ами|ах)?|секунд[аыуе]?|секундами|секундам)"
_EN_DUR = re.compile(r"\b(" + "|".join(_EN_NUM) + r")(?:( or | to )(" + "|".join(_EN_NUM) + r"))?([ -])(" + _EN_UNITS + r")\b",
                     re.I)
_RU_DUR = re.compile(r"(?<![\w-])(" + "|".join(sorted(_RU_NUM, key=len, reverse=True)) + r")(?:( или | - |-)(" +
                     "|".join(sorted(_RU_NUM, key=len, reverse=True)) + r"))? (" + _RU_UNITS + r")(?!\w)", re.I)


def digit_durations(target: str) -> str:
    """Prompt rule 4 (digits for measurements): 'ten minutes' -> '10 minutes', 'two or three hours' -> '2 or 3 hours',
    'через пять минут' -> 'через 5 минут'. The teacher wrote small durations both ways (en minutes: 34 digits,
    7 words); one style for all."""
    def en(m):
        a = str(_EN_NUM[m.group(1).lower()])
        b = (m.group(2) + str(_EN_NUM[m.group(3).lower()])) if m.group(3) else ""
        return a + b + m.group(4) + m.group(5)

    def ru(m):
        if m.group(1).lower() in ("одна", "одну", "одной", "одного", "одним", "одном", "один"):
            return m.group(0)  # 'одну минуту' / 'один час': the speaker's article-like 'one' stays a word
        a = str(_RU_NUM[m.group(1).lower()])
        b = (m.group(2) + str(_RU_NUM[m.group(3).lower()])) if m.group(3) else ""
        return a + b + " " + m.group(4)
    return _RU_DUR.sub(ru, _EN_DUR.sub(en, target))


def digits_spelled_out(hyp: str, target: str, lang: str) -> list[int]:
    """Numbers the transcript already gives in digits that the target writes in words ('10 minutes' ->
    'ten minutes'): a cleanup never turns a recogniser's digit back into a word."""
    lg = None if lang == "mixed" else lang
    hd = set(int(d) for d in re.findall(r"(?<![\d.,:])\d+(?![\d.,:]\d)", hyp))
    if not hd:
        return []
    toks = [t if not re.search(r"\d", t) else "\0" for t in T.norm_words(target)]
    td = set(int(d) for d in re.findall(r"\d+", target))
    return [v for a, b, v in T.number_spans(toks, lg) if toks[a] != "\0" and v in hd and v not in td]


def address_slug_guess(spoken: str, target: str) -> str | None:
    """A URL path segment written in Latin for a Russian word the user said ('слэш расписание' -> '/raspisanie',
    'слэш каталог' -> '/catalog'): neither the transliteration nor the translation is knowable from the audio."""
    m = re.match(r"^(?:https?://)?[^/\s]+/(.+)$", target.strip())
    if not m:
        return None
    lat_words = [T.norm_word(w) for w in T.words(spoken) if T.is_lat(w)]
    lat_spoken, joined = set(lat_words), "".join(lat_words)
    for seg in re.split(r"[/?=&#.\-_]+", m.group(1)):
        if len(seg) >= 4 and re.fullmatch(r"[A-Za-z]+", seg) and seg.lower() not in lat_spoken \
                and seg.lower() not in joined:
            return seg
    return None


# ------------------------------------------------------------------ records
def _near_no_op(target: str, r: random.Random) -> tuple[str, str]:
    opts = []
    if "," in target:
        opts.append("drop_comma")
    if target[:1].isupper():
        opts.append("lower_first")
    if re.search(r"[.!?]$", target):
        opts.append("drop_final")
    if not opts:  # a lowercase query without punctuation: the recogniser capitalised it or added a period
        opts = ["upper_first", "add_final"] if target[:1].islower() else ["add_final"]
    op = r.choice(opts)
    if op == "upper_first":
        return target[:1].upper() + target[1:], op
    if op == "add_final":
        return target + ".", op
    if op == "drop_comma":
        idx = [m.start() for m in re.finditer(",", target)]
        i = r.choice(idx)
        return target[:i] + target[i + 1:], op
    if op == "drop_final":
        return target[:-1], op
    return target[:1].lower() + target[1:], op


def _record(sc: dict, rid: str, category: str, ctx: dict, transcript: str, target: str, meta: dict) -> dict:
    return {"id": rid, "lang": sc["lang"], "category": category, "source": "sim", "split": sc["split"],
            "ctx": ctx, "transcript": transcript, "target": target, "meta": meta}


def _base_meta(sc: dict, teacher: str, gen: str | None = None) -> dict:
    return {"scenario": sc["sid"], "group": sc["group"], "teacher": teacher, "gen": gen or GEN_VERSION,
            "app_category": sc["app"]["category"], "persona": sc["persona"], "topic": sc.get("topic"),
            "privacy": sc["privacy"], "mix": sc["mix"]}


def build_noise_only(sc: dict, channel: Channel, backend: SCR.Backend) -> list[dict]:
    r = mkrng("noise_only", sc["sid"])
    lg = "he" if sc["lang"] in ("he", "he_en", "he_ru") else ("en" if sc["lang"] == "en" else "ru")
    hyp = noise_only_text(lg, r, sc["split"])
    ctx, smeta = make_ctx(sc, {"title": "", "messages": [], "before_cursor": "", "snippets": []}, backend, "")
    meta = _base_meta(sc, "code") | {"noise": {"source": "code:noise_only"}, "check": {"empty": True}, "screen": smeta}
    return [_record(sc, stable_id("syn", sc["sid"]), "noise_only", ctx, hyp, "", meta)]


def build(sc: dict, ex: dict, channel: Channel, teacher: str, backend: SCR.Backend, attempts: int = 5,
          gen: str | None = None) -> list[dict]:
    """One teacher example -> 1 record (2 with the no-context counterpart)."""
    cat = sc["category"]
    spoken, target, screen = ex["spoken"].strip(), ex["target"].strip(), ex["screen"]
    if not spoken or not target:
        raise Reject("empty_spoken_or_target")
    if sc["sid"] in BAD_TEACHER_EXAMPLES:
        raise Reject("teacher_semantics")
    # straight apostrophes (no recogniser writes ’, and a phone keyboard types ' ): one typography for both sides
    spoken, target = spoken.replace("’", "'"), digit_durations(target.replace("’", "'"))
    if cat == "address" and address_slug_guess(spoken, target):
        raise Reject("address_slug_guess")
    r = mkrng("build", sc["sid"])
    term = sc["spoken_terms"][0] if sc["spoken_terms"] else None
    # cheap teacher-side checks before paying for render + OCR
    if cat == "context_resolution":
        if not find_term(spoken, term["text"]):
            raise Reject("term_not_in_spoken")
        if not _contains(target, term["text"]):
            raise Reject("term_not_in_target")
    if cat == "near_miss" and term and not find_term(spoken, term["text"]):
        raise Reject("term_not_in_spoken")
    ctx, smeta = make_ctx(sc, screen, backend, target)
    rend = ctx_text(ctx)
    meta = _base_meta(sc, teacher, gen) | {"spoken": spoken, "screen": smeta}
    check: dict = {}
    lang = sc["lang"]

    if cat == "no_op":
        meta["noise"] = {"source": "code:no_op"}
        return [_record(sc, stable_id("syn", sc["sid"]), cat, ctx, target, target, meta)]
    if cat == "near_no_op":
        hyp, op = _near_no_op(target, r)
        meta["noise"] = {"source": "code:near_no_op", "events": [op]}
        return [_record(sc, stable_id("syn", sc["sid"]), cat, ctx, hyp, target, meta)]

    last = "no_attempt"
    for k in range(attempts):
        rk = mkrng("noise", sc["sid"], k)
        artifact = None
        if cat == "asr_artifact":
            artifact = "loop" if lang != "en" and rk.random() < LOOP_SHARE and channel.has_model(LOOP_MODEL_HINT, lang) \
                else "tail"
            model = channel.pick_family(lang, rk, (LOOP_MODEL_HINT,) if artifact == "loop" else ("whisper", "distil"))
        else:
            # one recogniser for every attempt but the last: re-drawing it on each retry would let the noisiest
            # recognisers (Qwen3-ASR on Russian, Nemotron streaming on English) drop out of the set
            mr = mkrng("model", sc["sid"]) if k < attempts - 1 else rk
            model = channel.pick_model(lang, mr, None, switch=sc.get("mix") == "switch")
        render = channel.disfluency_render(model)
        sev = sc["severity"] if k < attempts - 2 else min(sc["severity"], 0.5)
        sp, events, garble = spoken, [], None
        chan_terms = [t for t in sc["spoken_terms"] if _contains(spoken, t["text"])]  # exact Latin spelling only
        if cat == "context_resolution":
            form, src = None, None
            # a term inside an English clause of code-switched speech is misheard the English way
            tlang = "en" if lang == "mixed" and channel.can_switch(model) and any(
                k == "en" and find_term(seg, term["text"]) for seg, k in _lang_runs(spoken)) else lang

            def ff(prev, _r=rk):
                nonlocal form, src
                if form is None:
                    form, src = garble_form(term, tlang, channel, model, prev, _r,
                                            need_cyrillic=bool(sc.get("counterpart")))
                return form
            sp, n = _replace_term(sp, term["text"], ff)
            if not form or form.lower() == term["text"].lower():
                last = "no_garble"
                continue
            if not plausible_garble(form, term["text"]):
                last = "garble_is_abbreviation"
                continue
            if sc.get("counterpart") and re.search(r"[A-Za-z]", form) and k < attempts - 2:
                # the pair needs a Cyrillic rendering (a Latin misspelling has no no-context answer): try another
                # recogniser / seed first; the last attempts accept it and keep the parent without its counterpart
                last = "counterpart_latin_garble"
                continue
            garble = {"term": term["text"], "form": form, "source": src, "count": n}
            chan_terms = [t for t in chan_terms if t["text"] != term["text"]]
        if cat == "context_names" and rk.random() < 0.85:
            for p in sc["people"]:
                sp, old, new = misspell_surname(sp, p, rk)
                if new:
                    garble = (garble or {}) | {"person": p, "from": old, "to": new}
        caps = {w for t in sc["spoken_terms"] for w in T.words(t["text"])} | {w for p in sc["people"] for w in p.split()}
        sp2, dops = D.inject(sp, lang, sc["messiness"], rk, sounds_only=cat == "meaningful_filler", keep_caps=caps,
                             render=render)
        hyp, events = channel.transcribe(sp2, target, lang, model, chan_terms, sc["noise_seed"] + k, sev)
        if artifact == "tail":
            tail = rk.choice(tails_for(lang, sc["split"]))
            hyp = (hyp.rstrip() + " " + tail).strip()
            events = events + [f"tail:{tail}"]
        elif artifact == "loop":
            hyp, loop = stutter_loop(hyp, rk)
            if not loop:
                last = "no_loop_site"
                continue
            events = events + [f"loop:{loop}"]
        if not hyp.strip():
            last = "empty_hypothesis"
            continue
        bad_noise = channel.implausible(events, model, lang)
        if bad_noise:
            hyp, events, bad_noise = channel.repair_terms(hyp, events, model, lang, bad_noise, sc["spoken_terms"])
        if bad_noise:
            last = "implausible_noise"
            continue
        if unrecoverable_deletions(events, target):
            last = "unrecoverable_deletion"
            continue
        if unsupported_term_garbles(events, target, ctx, skip=tuple(t["text"] for t in sc["spoken_terms"]) if cat == "context_resolution" else ()):
            last = "unsupported_term_garble"
            continue
        if guessed_subs(events, target, rend, channel.real_words()):
            last = "real_word_guess"
            continue
        if digits_spelled_out(hyp, target, lang):
            last = "digit_spelled_out"
            continue
        if garble and garble.get("form") and not _contains(hyp, garble["form"]):
            last = "garble_lost"
            continue
        if garble and garble.get("to") and not _contains(hyp, garble["to"]):
            last = "misspelling_lost"
            continue
        if cat == "context_resolution" and _contains(hyp, term["text"]):
            last = "term_in_transcript"
            continue
        if cat == "near_miss" and term and not _contains(hyp, term["text"]):
            last = "said_term_garbled"
            continue
        allow = {T.norm_word(w) for t in sc["spoken_terms"] for w in T.words(t["text"])} if cat == "context_resolution" else set()
        if cat == "context_names":
            allow |= {T.norm_word(w) for p in sc["people"] for w in T.words(p)}
            allow |= {T.norm_word(w) for w in T.words(target) if any(T.norm_word(w).startswith(T.norm_word(p.split()[-1])[:4])
                                                                    for p in sc["people"])}
        bad = unrecoverable(target, hyp, rend, allow)
        if bad:
            last = "unrecoverable:" + ",".join(bad[:3])
            continue
        break
    else:
        raise Reject(last.split(":")[0] if not last.startswith("unrecoverable") else "asr_unrecoverable")

    meta["asr_model"] = model
    meta["noise"] = {"source": channel.source, "severity": sev, "attempt": k, "events": events[:20]}
    meta["disfluency"] = dops
    meta["disfluency_render"] = render
    if garble:
        meta["garble"] = garble
    if cat == "context_resolution":
        check["keep"] = [term["text"]]
        meta["entities"] = [term["text"]]
        meta["distractors"] = list(sc["distractors"])
    elif cat == "context_names":
        keep = []
        for p in sc["people"]:
            sur = p.split()[-1]
            keep.append(sur[: max(4, len(sur) - 1)] if T.is_cyr(sur) else sur)
        check["keep"] = keep
        meta["entities"] = list(sc["people"])
        meta["distractors"] = list(sc["distractors"])
    elif cat == "near_miss":
        meta["entities"] = [term["text"]] if term else list(sc["people"])
        meta["distractors"] = list(sc["distractors"])
        check["keep"] = [term["text"]] if term else [sc["people"][0].split()[-1][:5]]
    elif cat == "context_irrelevant":
        meta["distractors"] = [x for x in list(sc["screen_people"]) + list(sc["screen_terms"])
                               if SCR.on_screen(ctx, x) and not _contains(target, x)]
    if cat in NOANSWER:
        check["noanswer"] = True
    if cat == "injection":
        check["injection"] = True
        inj = sc["injection"]["text"]
        check["forbid"] = [f for k2, fs in INJECTION_FORBID.items() if k2 in inj for f in fs]
        meta["injection"] = SCR.effective_injection(sc)
    if check:
        meta["check"] = check
    rec = _record(sc, stable_id("syn", sc["sid"]), cat, ctx, hyp, target, meta)
    out = [rec]
    if cat == "context_resolution" and sc.get("counterpart") and garble:
        if re.search(r"[A-Za-z]", garble["form"]):  # a Latin misspelling has no sensible no-context answer
            out.append({"_reject": "counterpart_latin_garble", "category": "context_counterpart", "term": term["text"]})
            return out
        try:
            cp_ctx, cp_smeta = counterpart_ctx(sc, smeta["spec"], term["text"], backend)
        except Reject as rj:  # the parent stays; the counterpart is reported as dropped
            out.append({"_reject": str(rj), "category": "context_counterpart", "term": getattr(rj, "term", None)})
            return out
        if unsupported_term_garbles(events, target, cp_ctx, skip=(term["text"],)):
            out.append({"_reject": "unsupported_term_garble", "category": "context_counterpart", "term": term["text"]})
            return out
        m = re.search(r"(?<!\w)" + re.escape(garble["form"]).replace(r"\ ", r"[\s-]+"), hyp, re.I)
        hyp_form = m.group(0) if m else garble["form"]
        pat = re.compile(r"(?:(?<![\w-])|(?<=[\u05d0-\u05ea]-))" + re.escape(term["text"]) + r"(?![\w])")
        cp_target = pat.sub(lambda mm: hyp_form[:1].upper() + hyp_form[1:] if _sentence_start(target, mm.start())
                            else hyp_form, target)
        if _contains(cp_target, term["text"]):
            raise Reject("counterpart_target_has_term")
        # the parent's checks used the parent's screen; the re-rendered one may no longer show a word they relied on
        cp_text = ctx_text(cp_ctx)
        if guessed_subs(events, cp_target, cp_text, channel.real_words()):
            out.append({"_reject": "real_word_guess", "category": "context_counterpart", "term": term["text"]})
            return out
        if unrecoverable(cp_target, hyp, cp_text):
            out.append({"_reject": "asr_unrecoverable", "category": "context_counterpart", "term": term["text"]})
            return out
        cp_meta = copy.deepcopy(meta) | {"counterpart_of": rec["id"], "screen": cp_smeta,
                                         "check": {"keep": [hyp_form], "forbid": [term["text"]]},
                                         "entities": [hyp_form], "distractors": [term["text"]] + list(sc["distractors"])}
        out.append(_record(sc, stable_id("syn", sc["sid"], "cp"), "context_counterpart", cp_ctx, hyp, cp_target, cp_meta))
    return out


def _sentence_start(text: str, pos: int) -> bool:
    before = text[:pos].rstrip()
    return not before or before[-1] in ".!?\n"
