"""ASR channel adapter: clean spoken text -> simulated raw hypothesis of one ASR model.

Two back ends behind one interface, recorded in meta.noise.source:
  mined:<dir>  ml/roundtrip/mine/noise.py (NoiseModel) with error tables mined from the real TTS -> ASR probe
               (owned by ml/roundtrip; read-only here). Used when --mined DIR (or $MURMUR_MINED_DIR) has
               profiles.json + confusions.jsonl + term_manglings.jsonl.
  builtin      ml/data/synth/asr_noise.py, hand-set priors per engine (fallback while mined tables are absent).

The teacher writes `spoken` with numbers in words. Engines whose mined profile prints digits get the
numbers digitised first (at the profile's digit rate); noise.py then applies the model's own style,
term manglings, confusions, deletions and hallucinations.

Realism guards on top of noise.py (review 2026-11-19; noise.py itself is read-only here):
  * code-switched speech: English clauses (>= 3 Latin words in a row) of a model that writes Latin are simulated
    as English (word-level confusions), not word by word as unknown "terms" with a term error rate
    ('the' -> 'tha', 'you' -> 'йау'); the Russian part as Russian. One casing/punctuation style for the whole text.
  * `implausible(events)`: substitutions that do not sound like the word, inserted content words (only function
    words and fillers are inserted by real decoders), generic term garbles that the model's mined manglings never
    show (single letters; interior letter drops of a non-acronym from a model with no such observed drop).
  * `repair_terms`: a model that writes many Latin terms in Cyrillic gets a phonetic transliteration instead of an
    implausible Latin garble. What cannot be repaired makes build.py retry with another seed.
  * `disfluency_render(model)`: how often the model writes hesitation sounds and repeated words
    (ml/data/probe/disfluency_render.v1.json, measured with the disfluency probe).
"""
from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from ml.common import text as T
from ml.data.synth import asr_noise as AN

PROBE_DIR = Path(__file__).resolve().parents[1] / "probe"
RENDER_FILE = PROBE_DIR / "disfluency_render.v1.json"
STYLE_FILE = PROBE_DIR / "asr_style.v1.json"           # ml.data.realism measure (real round trip, clean channel)
SCALE_FILE = PROBE_DIR / "word_noise_scale.v1.json"    # ml.data.realism calibrate
# Insertions a decoder plausibly makes: function words and fillers (noise.py's own fallback vocabulary)
INS_OK_EXTRA = {"и", "а", "в", "ну", "вот", "это", "the", "a", "and", "uh", "to", "it", "i", "у", "с", "к", "о", "я",
                "да", "но", "же", "ли", "an", "in", "on", "of", "is", "that", "so", "um",
                "את", "של", "על", "עם", "אז", "נו", "אממ", "אה", "זה"}
# function-word swaps a recogniser makes although they do not share sounds (noise.py _SHORT_SWAPS, ml/roundtrip)
SHORT_SWAPS = {
    "в": ["во", "и", "на"], "во": ["в"], "и": ["а", "в"], "а": ["и", "о"], "на": ["но", "а"], "не": ["ни", "и"],
    "с": ["со", "из"], "со": ["с"], "к": ["ко", "и"], "по": ["па", "под"], "о": ["об", "а"], "у": ["в"],
    "что": ["чтобы", "то"], "это": ["то", "эта"], "the": ["a", "that"], "a": ["the", "uh"], "to": ["the", "do"],
    "in": ["and", "on"], "on": ["in", "an"], "and": ["in", "an"], "of": ["off", "a"], "it": ["that", "is"],
    "is": ["was", "his"], "for": ["four", "from"], "at": ["that", "it"],
    "עם": ["אם"], "אם": ["עם"], "על": ["אל"], "אל": ["על"], "כאן": ["כן"], "כן": ["כאן"],
}
_LAT_RUN_MIN = 3


def _fw(w: str) -> str:
    return T.norm_word(w).replace("’", "'")


def plausible_sub(a: str, b: str) -> bool:
    """A substitution a recogniser makes from the audio: sounds like the word (phonetic similarity >= 0.55 for
    words of 8+ letters, >= 0.45 for 4-7, >= 0.34 for shorter ones), a known function-word swap, or one letter
    for one letter. ('вопросительный' -> 'пользователей' is 0.50: rejected.)"""
    fa, fb = _fw(a), _fw(b)
    if not fb or fa == fb:
        return True
    if len(fa) <= 1 and len(fb) <= 1:
        return True
    if fb in SHORT_SWAPS.get(fa, ()) or fa in SHORT_SWAPS.get(fb, ()):
        return True
    return T.phon_sim(a, b) >= (0.55 if len(fa) >= 8 else 0.45 if len(fa) >= 4 else 0.34)


def plausible_ins(w: str) -> bool:
    f = _fw(w)
    return f in INS_OK_EXTRA or f in T.FUNCTION_WORDS or f in T.HESITATION


def _is_acronym(t: str) -> bool:
    letters = [c for c in t if c.isalpha()]
    return 0 < len(letters) <= 6 and all(c.isupper() for c in letters if c.isalpha()) and len(t.split()) == 1


def _letter_drop(term: str, v: str) -> str | None:
    """v is term with 1-2 letters deleted (case and separators ignored)? -> None, 'phonetic' (a doubled letter, a
    word edge, a vowel next to a vowel: 'Ollama' -> 'Olama', 'htop' -> 'top', 'Traefik' -> 'trafik') or 'interior'
    (a letter inside the word that is pronounced: 'Kubernetes' -> 'Kuberetes', 'GitLab' -> 'Gtlab')."""
    a, b = re.sub(r"[\s\-_.]", "", term.lower()), re.sub(r"[\s\-_.]", "", v.lower())
    if not b or len(b) >= len(a) or len(a) - len(b) > 2 or not re.fullmatch(r"[a-z0-9]+", b):
        return None
    kept, j = [], 0
    for i, c in enumerate(a):
        if j < len(b) and c == b[j]:
            kept.append(i)
            j += 1
    if j < len(b):
        return None
    vow = set("aeiouy")
    for i in sorted(set(range(len(a))) - set(kept)):
        doubled = (i > 0 and a[i - 1] == a[i]) or (i + 1 < len(a) and a[i + 1] == a[i])
        edge = i == 0 or i == len(a) - 1
        vcluster = a[i] in vow and ((i > 0 and a[i - 1] in vow) or (i + 1 < len(a) and a[i + 1] in vow))
        if not (doubled or edge or vcluster):
            return "interior"
    return "phonetic"


def _lang_runs(text: str) -> list[tuple[str, str]]:
    """[(segment, 'en'|'ru')]: maximal runs of >= 3 consecutive Latin-script words are English clauses."""
    toks = text.split(" ")
    lat = [bool(re.search(r"[A-Za-z]", t)) and not re.search(r"[А-Яа-яЁё]", t) for t in toks]
    segs, i = [], 0
    while i < len(toks):
        j = i
        while j < len(toks) and lat[j] == lat[i]:
            j += 1
        kind = "en" if lat[i] and j - i >= _LAT_RUN_MIN else "ru"
        piece = " ".join(toks[i:j])
        if segs and segs[-1][1] == kind:
            segs[-1] = (segs[-1][0] + " " + piece, kind)
        else:
            segs.append((piece, kind))
        i = j
    return segs

BUILTIN_LANGS = {"whisper": ("ru", "en", "mixed"), "parakeet_v3": ("ru", "en", "mixed"),
                 "nemotron_3_5": ("ru", "en", "mixed"), "gigaam_v3_e2e": ("ru", "mixed"), "t_one": ("ru",),
                 "qwen3_asr": ("ru", "en", "mixed"),
                 "zipformer2_he": ("he", "he_en", "he_ru"), "whisper_he": ("he", "he_en", "he_ru")}
HE_LANGS = {"he", "he_en", "he_ru"}
HE_MODELS = ("zipformer2_he", "whisper_he")
RU_ONLY_HINTS = ("gigaam", "t-one", "t_one")


class Channel:
    def __init__(self, mined_dir: str | os.PathLike | None = None):
        mined_dir = mined_dir or os.environ.get("MURMUR_MINED_DIR") or None
        self.nm = None
        if mined_dir and (Path(mined_dir) / "profiles.json").exists():
            from ml.roundtrip.mine.noise import NoiseModel  # read-only dependency on the round-trip team
            self.nm = NoiseModel.load(mined_dir)
            self.source = "mined:" + Path(mined_dir).name
        elif mined_dir:
            raise FileNotFoundError(f"no profiles.json in {mined_dir}")
        else:
            self.source = "builtin"
        self.mined_dir = Path(mined_dir) if self.nm is not None else None
        self._habits = _term_habits(str(self.mined_dir / "term_manglings.jsonl")) if self.nm is not None else {}
        self._render = _render_table()
        self._style = _json_table(str(STYLE_FILE)).get("models", {})
        if self.nm is not None:
            _install_word_scale(self.nm, _json_table(str(SCALE_FILE)).get("factors", {}))
        self._real_words = None

    def real_words(self) -> set[str]:
        """Real words of the mined tables (reference words, and the words language-model decoders put in)."""
        if self._real_words is None:
            self._real_words = {T.norm_word(w) for ws in (self.nm._words.values() if self.nm is not None else [])
                                for w in ws}
        return self._real_words

    def style(self, model: str, lang: str) -> dict:
        return (self._style.get(model) or {}).get(lang) or {}

    # ------------------------------------------------------------------ per-model habits
    def habits(self, model: str, lang: str) -> dict:
        """Mined term-error habits of `model` in this speech group (ru for ru/mixed): Cyrillic share of Latin-term
        errors, observed letter drops of acronyms and of other terms."""
        g = "en" if lang == "en" else "ru"
        return self._habits.get((model, g)) or {"err": 0, "cyr_share": 0.0, "drop_acronym": 0, "drop_other": 0,
                                                "drop_interior": 0}

    def writes_latin(self, model: str, lang: str) -> bool:
        if self.nm is None:
            return bool(AN.PROFILES.get(model, {}).get("latin", True))
        lb = self.nm.profiles[model].get("latin", {})
        keep = (lb.get("by_lang", {}).get("mixed") or lb.get("by_lang", {}).get("ru") or {}).get("kept_latin_rate")
        keep = lb.get("kept_latin_rate") if keep is None else keep
        return keep is not None and keep >= 0.8

    def can_switch(self, model: str) -> bool:
        """Model writes an English clause inside Russian speech as English (code-switching scenarios)."""
        if self.nm is None:
            return "en" in BUILTIN_LANGS.get(model, ())
        return self.nm.supports(model, "en") and self.writes_latin(model, "mixed")

    def disfluency_render(self, model: str) -> dict:
        """{"hesitation_keep", "repeat_keep"} for this ASR model (probe-measured; family prior otherwise)."""
        return _render_for(self._render, model)

    # ------------------------------------------------------------------ models
    def models(self, lang: str) -> list[str]:
        if lang in HE_LANGS:
            return list(HE_MODELS)
        if self.nm is None:
            return [m for m, ls in BUILTIN_LANGS.items() if lang in ls]
        out = []
        for m in self.nm.models():
            langs = set((self.nm.profiles[m].get("langs") or {}).keys())
            if langs:
                ok = lang in langs or (lang == "mixed" and "ru" in langs) or (lang == "ru" and "mixed" in langs)
            else:
                ok = not (lang == "en" and any(h in m for h in RU_ONLY_HINTS))
            if lang == "en" and any(h in m for h in RU_ONLY_HINTS):
                ok = False
            if ok:
                out.append(m)
        return out

    def has_model(self, hint: str, lang: str) -> bool:
        return any(hint in m for m in self.models(lang))

    def pick_family(self, lang: str, r: random.Random, hints: tuple[str, ...]) -> str:
        """A random recogniser of the named families that serves this language (asr_artifact: every Whisper-family
        model, not always the first one)."""
        ms = [m for m in self.models(lang) if any(h in m for h in hints)]
        return r.choice(ms) if ms else self.pick_model(lang, r)

    def pick_model(self, lang: str, r: random.Random, prefer: str | None = None, switch: bool = False) -> str:
        if lang in HE_LANGS:
            if prefer:
                hit = [m for m in HE_MODELS if prefer in m]
                if hit:
                    return hit[0]
            # zipformer2-66m-streaming-he is the shipping Hebrew ASR (80%), whisper_he covers punctuated/mixed fallback
            return "zipformer2_he" if r.random() < 0.80 else "whisper_he"
        ms = self.models(lang)
        if switch:  # true code-switching: only recognisers that write the English clause in English
            ms = [m for m in ms if self.can_switch(m)] or ms
        if prefer:
            hit = [m for m in ms if prefer in m]
            if hit:
                return hit[0]
        return r.choice(ms)

    def _digit_rate(self, model: str) -> float:
        if self.nm is None:
            return AN.PROFILES[model]["digits"]
        d = self.nm.profiles[model].get("digits") or {}
        if d.get("label") == "digits":
            return float(d.get("hyp_digit_rate") or 0.9) if d.get("hyp_digit_rate") else 0.9
        return 0.0

    def is_lowercase_engine(self, model: str) -> bool:
        if self.nm is None:
            return not AN.PROFILES[model]["case"]
        return ((self.nm.profiles[model].get("casing") or {}).get("label") == "lowercase")

    # ------------------------------------------------------------------ run
    def transcribe(self, spoken: str, target: str, lang: str, model: str, terms: list[dict], seed: int,
                   severity: float) -> tuple[str, list[str]]:
        """terms: [{"text", "category"}] Latin terms present in `spoken` (the channel may mangle them)."""
        r = random.Random(seed)
        if self.nm is None or lang in HE_LANGS or model in HE_MODELS:
            ents = [{"text": t["text"], "kind": "tech"} for t in terms]
            # punctuation and casing are projected from the punctuated spoken text (what the audio supports),
            # not from the edited target
            out = AN.noise(_strip_punct(spoken), spoken, lang, model, min(1.0, severity / 2), r, entities=ents)
            return out.hyp, [_fmt_op(o) for o in out.ops]
        text = spoken
        ev = []
        if r.random() < self._digit_rate(model):
            toks = text.split()
            dig = AN._digitise(toks, "en" if lang == "en" else "ru", r)
            if dig != toks:
                text = " ".join(dig)
                ev.append("digits")
        nterms = [{"term": t["text"], "category": t.get("category", "")} for t in terms]
        text, dropped = self._edge_drop(text, model, lang, r)
        if dropped:
            ev.append(f"del:{dropped}")
        segs = _lang_runs(text) if lang == "mixed" and self.can_switch(model) else [(text, lang)]
        if len(segs) == 1:
            sim = self.nm.simulate(text, model, nterms, "mixed" if lang == "mixed" else lang, seed=seed,
                                   severity=severity)
            hyp, evs = sim.hypothesis, ev + list(sim.events)
        else:
            hyp, evs = self._simulate_segments(segs, model, nterms, seed, severity, ev)
        return self._final_style(hyp, spoken, model, lang, evs, r), evs

    def _edge_drop(self, text: str, model: str, lang: str, r: random.Random) -> tuple[str, str | None]:
        """Streaming recognisers lose the first word far more often than any other (real round trip: Nemotron
        streaming English 41-44 % vs 5 % for the rest). The excess rate of the measured table; whether the example
        stays recoverable is the build's recoverability check."""
        p = float(self.style(model, lang).get("first_word_drop") or 0.0)
        ws = text.split(" ")
        if p <= 0 or len(ws) < 4 or r.random() >= p:
            return text, None
        first = re.sub(r"^\W+|\W+$", "", ws[0])
        if not first or re.search(r"\d", first):
            return text, None
        return " ".join(ws[1:]), first

    def _final_style(self, hyp: str, spoken: str, model: str, lang: str, events: list[str], r: random.Random) -> str:
        """The recogniser's own sentence-final mark at its measured rate (given that it punctuates this utterance at
        all), also after a spoken command ('...вопросительный знак.'), and straight apostrophes only (0 curly ones
        in 10,605 real hypotheses)."""
        hyp = hyp.replace("’", "'")
        st = self.style(model, lang)
        if not hyp or not st or any(e.startswith("tail:") or e in ("empty", "foreign_script") for e in events):
            return hyp
        if "style:no_punct" in events:  # this utterance came out without any punctuation (the model's own rate)
            return hyp
        p_end = float(st.get("end_given_punct", 1.0))
        has_end = hyp.rstrip()[-1:] in ".!?…"
        if r.random() < p_end:
            if not has_end:
                hyp = hyp.rstrip(" ,;:") + ("?" if spoken.rstrip()[-1:] == "?" else ".")
                events.append("style:final_mark")
        elif has_end:
            hyp = re.sub(r"[.!?…]+$", "", hyp.rstrip())
            events.append("style:no_final_mark")
        return hyp

    def _simulate_segments(self, segs, model, nterms, seed, severity, ev):
        """Code-switched text: each language run simulated in its own language; one style for the whole text."""
        outs, evs = [], list(ev) + ["segments:" + "+".join(k for _, k in segs)]
        for i, (seg, kind) in enumerate(segs):
            lg = "en" if kind == "en" else "mixed"
            ts = [t for t in nterms if t["term"] in seg]
            for k in range(4):  # no hallucinated tail / empty output / foreign script in the middle of a sentence
                sim = self.nm.simulate(seg, model, ts, lg, seed=seed * 31 + i * 7 + k * 1009, severity=severity)
                bad = any(e == "empty" or e == "foreign_script" for e in sim.events) or \
                    (i < len(segs) - 1 and any(e.startswith("tail:") for e in sim.events))
                if not bad:
                    break
            else:
                sim = type(sim)(seg, [])
            outs.append([sim.hypothesis, seg])
            evs += list(sim.events)
        style_lower = any(e == "style:lowercase" for e in evs)
        style_nopunct = any(e == "style:no_punct" for e in evs)
        joined = []
        for i, (h, seg) in enumerate(outs):
            if style_nopunct:
                h = " ".join(w for w in (re.sub(r"^[^\w]+|[^\w%'’-]+$", "", x) for x in h.split()) if w)
            if style_lower:
                h = h.lower()
            elif i > 0 and h[:1].isupper() and seg[:1].islower():
                h = h[:1].lower() + h[1:]  # a later run continues the sentence: no new capital
            if i < len(outs) - 1 and not style_nopunct and h[-1:] in ".!?" and not seg.rstrip()[-1:] in ".!?":
                h = h[:-1]  # the run ends mid-sentence: no sentence-final mark
            joined.append(h)
        return " ".join(x for x in joined if x).strip(), evs

    # ------------------------------------------------------------------ realism guards
    def implausible(self, events: list[str], model: str, lang: str) -> list[str]:
        """Noise events no real recogniser produces from the audio (see module docstring)."""
        out = []
        hb = self.habits(model, lang)
        for e in events:
            if e.startswith("sub:") and "->" in e:
                a, b = e[4:].split("->", 1)
                if not all(plausible_sub(x, y) for x, y in _pair_words(a, b)):
                    out.append(e)
            elif e.startswith("ins:"):
                if not all(plausible_ins(w) for w in e[4:].split()):
                    out.append(e)
            elif e.startswith("term:") and e.endswith(":garbled"):
                term, v = e[5:-len(":garbled")].split("->", 1)
                if _term_garble_implausible(term, v, hb):
                    out.append(e)
        return out

    def repair_terms(self, hyp: str, events: list[str], model: str, lang: str, bad: list[str],
                     terms: list[dict]) -> tuple[str, list[str], list[str]]:
        """Replace implausible Latin garbles of a Cyrillic-writing model by a phonetic transliteration.
        -> (hyp, events, still_bad)."""
        hb = self.habits(model, lang)
        if lang == "en" or hb["cyr_share"] < 0.35:
            return hyp, events, bad
        rest, ev = [], list(events)
        for e in bad:
            if not (e.startswith("term:") and e.endswith(":garbled")):
                rest.append(e)
                continue
            term, v = e[5:-len(":garbled")].split("->", 1)
            if not v or v == "∅" or not re.search(r"[A-Za-z]", v):
                rest.append(e)
                continue
            hits = [m for m in re.finditer(r"(?<![\w-])" + re.escape(v) + r"(?![\w])", hyp, re.I)]
            if len(hits) != 1:
                rest.append(e)
                continue
            reading = _reading(term, terms)
            if not reading:
                rest.append(e)
                continue
            if hyp[hits[0].start():][:1].isupper() and not _lowercase_text(hyp):
                reading = reading[:1].upper() + reading[1:]
            hyp = hyp[:hits[0].start()] + reading + hyp[hits[0].end():]
            ev[ev.index(e)] = f"term:{term}->{reading}:translit_repair"
        return hyp, ev, rest


@lru_cache(maxsize=8)
def _json_table(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _install_word_scale(nm, factors: dict) -> None:
    """Scale the mined non-term word substitution / deletion rates per (model, language) (ml.data.realism calibrate:
    the synthetic transcripts missed fewer words than the real ones, 0.4-0.9 of the real rate). noise.py is read-only
    here, so the instance's rate lookup is wrapped: base rates via _word_rates, word-specific counts via _conf."""
    if not factors or getattr(nm, "_word_scale", None) is not None:
        return
    nm._word_scale = factors
    base_rates, base_sim = nm._word_rates, nm.simulate
    scaled_conf: dict = {}
    state = {"f": 1.0}

    def word_rates(model, lang):
        ps, pd, pi, near = base_rates(model, lang)
        f = state["f"]
        return ps * f, pd * f, pi, near

    def simulate(text, asr_model, terms=None, lang=None, seed=0, severity=1.0):
        f = float((factors.get(asr_model) or {}).get(lang or "", 1.0))
        if f == 1.0:
            return base_sim(text, asr_model, terms, lang, seed=seed, severity=severity)
        key = (asr_model, f)
        if key not in scaled_conf:
            scaled_conf[key] = {w: [(hk, c * f) for hk, c in lst] for w, lst in nm._conf.get(asr_model, {}).items()}
        orig = nm._conf.get(asr_model)
        nm._conf[asr_model] = scaled_conf[key]
        state["f"] = f
        try:
            return base_sim(text, asr_model, terms, lang, seed=seed, severity=severity)
        finally:
            state["f"] = 1.0
            if orig is not None:
                nm._conf[asr_model] = orig

    nm._word_rates = word_rates
    nm.simulate = simulate


def _pair_words(a: str, b: str) -> list[tuple[str, str]]:
    wa, wb = a.split(), b.split()
    if len(wa) == len(wb):
        return list(zip(wa, wb))
    return [(a.replace(" ", ""), b.replace(" ", ""))]


def _term_garble_implausible(term: str, v: str, hb: dict) -> bool:
    if v in ("", "∅"):
        return False  # a dropped term is a deletion; recoverability decides
    letters = re.sub(r"[^\w]", "", v)
    if len(letters) <= 1:
        return True  # 'Pro' -> 'п', 'PR' -> 'P'
    kind = _letter_drop(term, v) if re.search(r"[A-Za-z]", term) and re.search(r"[A-Za-z]", v) else None
    if kind is None:
        return False
    if _is_acronym(term):
        return hb["drop_acronym"] < 1
    if kind == "interior":
        return hb["drop_interior"] < 2
    return hb["drop_acronym"] + hb["drop_other"] < 1


def _reading(term: str, terms: list[dict]) -> str:
    for t in terms:
        if t.get("text") == term and t.get("ru_spoken"):
            return t["ru_spoken"][0]
    try:
        from ml.roundtrip.mine.translit import lat2cyr, lexicon_readings
    except Exception:  # noqa: BLE001
        return T.lat2cyr_rough(term)
    rd = [x for x in lexicon_readings(term) if not re.search(r"[A-Za-z]", x)]
    return rd[0] if rd else lat2cyr(term)


def _lowercase_text(s: str) -> bool:
    return not any(c.isupper() for c in s)


@lru_cache(maxsize=4)
def _term_habits(path: str) -> dict:
    """(model, speech group) -> Latin-term error habits from the mined term_manglings.jsonl."""
    out: dict = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        term = d["term"]
        if not re.search(r"[A-Za-z]", term):
            continue
        key = (d["asr_model"], "en" if d.get("lang_context") == "en" else "ru")
        h = out.setdefault(key, Counter())
        for v in list(d.get("variants") or []) + list(d.get("merged") or []):
            hyp, c = v.get("hyp") or "", int(v.get("count") or 1)
            if hyp.lower().replace(" ", "") == term.lower().replace(" ", ""):
                continue
            h["err"] += c
            if hyp and re.search(r"[А-Яа-яЁё]", hyp) and not re.search(r"[A-Za-z]", hyp):
                h["cyr"] += c
            elif (kind := _letter_drop(term, hyp)):
                h["drop_acronym" if _is_acronym(term) else "drop_other"] += c
                if kind == "interior" and not _is_acronym(term):
                    h["drop_interior"] += c
    return {k: {"err": v["err"], "cyr_share": round(v["cyr"] / v["err"], 3) if v["err"] else 0.0,
                "drop_acronym": v["drop_acronym"], "drop_other": v["drop_other"], "drop_interior": v["drop_interior"]}
            for k, v in out.items()}


# recogniser families: prior survival rates when the probe has not measured a model
RENDER_PRIORS = [("whisper", 0.1, 0.5), ("distil", 0.1, 0.5), ("qwen3", 0.1, 0.5), ("parakeet", 0.3, 0.9),
                 ("nemotron", 0.3, 0.9), ("gigaam", 0.5, 0.9), ("t-one", 0.5, 0.9), ("t_one", 0.5, 0.9),
                 ("zipformer2", 0.7, 0.9)]


@lru_cache(maxsize=1)
def _render_table() -> dict:
    try:
        return json.loads(RENDER_FILE.read_text(encoding="utf-8")).get("models", {})
    except (OSError, ValueError):
        return {}


def _render_for(table: dict, model: str) -> dict:
    m = table.get(model)
    if m and m.get("n_hesitations", 0) >= 10:
        return {"hesitation_keep": float(m["hesitation_keep"]), "repeat_keep": float(m.get("repeat_keep", 0.9)),
                "source": "probe"}
    for fam, h, rp in RENDER_PRIORS:
        if fam in model:
            return {"hesitation_keep": h, "repeat_keep": rp, "source": "prior:" + fam}
    return {"hesitation_keep": 0.5, "repeat_keep": 0.9, "source": "prior:default"}


def _strip_punct(s: str) -> str:
    return " ".join(re.sub(r"^[^\w]+|[^\w%]+$", "", w) for w in s.split() if re.sub(r"\W", "", w))


def _fmt_op(o: dict) -> str:
    return ":".join(str(o.get(k)) for k in ("op", "from", "to") if o.get(k) is not None)
