"""Simulated ASR channel (stage D2) for Russian and English.

spoken text (what the user said) -> hypothesis (what a given ASR engine would
plausibly print). Every operation is recorded so build.py can check that the
target is still recoverable (faithfulness) and repair it when it is not.

Engine profiles follow docs/research/asr-models.md: the app ships Nemotron 3.5
streaming, Parakeet TDT v3, GigaAM v3 e2e (all punctuate and case), Whisper
(punctuates, hallucinates tails on silence), T-one (lowercase Cyrillic only,
no digits, no Latin, no punctuation) and Qwen3-ASR. Rates are hand-set priors:
this is only the FALLBACK channel. Once ml/roundtrip has mined real error tables,
ml/data/synth/channel.py uses ml/roundtrip/mine/noise.py instead (--mined DIR).
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from ml.common import text as T
from ml.common.render_bridge import REPO
from ml.data.synth import lexicon as L

DERIVED = REPO / ".cache" / "ml" / "data" / "derived"

PROFILES = {
    #               punct case digits keep_hes translit name_err conf  bound  ending fdel  tail   loop   comma_drop
    "whisper":       dict(punct=True, case=True, digits=0.85, keep_hes=0.25, translit=0.35, name_err=0.5, conf=0.04,
                          bound=0.010, ending=0.010, fdel=0.010, tail=0.03, loop=0.004, comma_drop=0.35, latin=True),
    "parakeet_v3":   dict(punct=True, case=True, digits=0.90, keep_hes=0.55, translit=0.60, name_err=0.5, conf=0.03,
                          bound=0.015, ending=0.015, fdel=0.010, tail=0.0, loop=0.0, comma_drop=0.40, latin=True),
    "nemotron_3_5":  dict(punct=True, case=True, digits=0.90, keep_hes=0.50, translit=0.55, name_err=0.5, conf=0.04,
                          bound=0.020, ending=0.020, fdel=0.015, tail=0.0, loop=0.0, comma_drop=0.45, latin=True),
    "gigaam_v3_e2e": dict(punct=True, case=True, digits=0.90, keep_hes=0.65, translit=0.90, name_err=0.4, conf=0.03,
                          bound=0.015, ending=0.015, fdel=0.010, tail=0.0, loop=0.0, comma_drop=0.35, latin=True),
    "t_one":         dict(punct=False, case=False, digits=0.0, keep_hes=0.75, translit=1.0, name_err=0.6, conf=0.05,
                          bound=0.020, ending=0.025, fdel=0.015, tail=0.0, loop=0.0, comma_drop=1.0, latin=False),
    "qwen3_asr":     dict(punct=True, case=True, digits=0.90, keep_hes=0.20, translit=0.20, name_err=0.4, conf=0.03,
                          bound=0.010, ending=0.010, fdel=0.010, tail=0.0, loop=0.002, comma_drop=0.30, latin=True),
    # ivrit-ai/zipformer2-66m-streaming-he: Hebrew-only transducer, no punctuation, transliterates Latin tech terms
    # to Hebrew, frequent homophone slips (א/ע/ה, ט/ת, כ/ק, כ/ח, ס/ש, ב/ו) and split/joined prefixes (ב/ל/ה/ו/מ/ש).
    "zipformer2_he": dict(punct=False, case=False, digits=0.25, keep_hes=0.70, translit=0.92, name_err=0.5, conf=0.055,
                          bound=0.045, ending=0.0, fdel=0.015, tail=0.0, loop=0.004, comma_drop=1.0, latin=False),
    # Whisper / multilingual fallback on Hebrew: punctuates lightly, keeps some Latin, occasional silence tails.
    "whisper_he":    dict(punct=True, case=True, digits=0.75, keep_hes=0.30, translit=0.45, name_err=0.45, conf=0.035,
                          bound=0.025, ending=0.0, fdel=0.010, tail=0.03, loop=0.003, comma_drop=0.45, latin=True),
}

WHISPER_TAILS = {
    "ru": ["Продолжение следует...", "Субтитры сделал DimaTorzok", "Субтитры создавал DimaTorzok",
           "Редактор субтитров А.Синецкая Корректор А.Егорова", "Спасибо за просмотр!", "Подписывайтесь на канал!"],
    "en": ["Thanks for watching!", "Thank you for watching.", "Please subscribe to my channel.", "Thank you.",
           "Subtitles by the Amara.org community"],
    "he": ["תודה שצפיתם!", "הירשמו לערוץ!", "כתוביות בעברית", "תודה רבה שצפיתם בסרטון!", "נתראה בסרטון הבא!"],
}
HES_RENDER = {"ru": ["э", "э-э", "ээ", "эм", "м-м", "а-а"], "en": ["uh", "um", "uhm", "er", "hmm"],
              "he": ["אממ", "אמ", "אהה", "אה", "הממ", "ממ"]}

# Hebrew homophone letter confusions (medial/safe positions) characteristic of zipformer2-66m-streaming-he
HE_HOMOPHONE_SWAPS = [
    ("א", "ע"), ("ע", "א"), ("א", "ה"), ("ה", "א"), ("ע", "ה"), ("ה", "ע"),
    ("ט", "ת"), ("ת", "ט"), ("כ", "ק"), ("ק", "כ"), ("כ", "ח"), ("ח", "כ"),
    ("ס", "ש"), ("ש", "ס"), ("ב", "ו"), ("ו", "ב"),
]
HE_PREFIXES = ("ב", "ל", "ה", "ו", "מ", "ש", "כ")
_HE_FINAL_FIX = {"כ": "ך", "מ": "ם", "נ": "ן", "פ": "ף", "צ": "ץ"}
_HE_MEDIAL_FIX = {v: k for k, v in _HE_FINAL_FIX.items()}

BUILTIN_CONF = {
    "ru": [("тся", "ться"), ("ться", "тся")],
    "ru_words": [("не", "ни"), ("ни", "не"), ("чтобы", "что бы"), ("также", "так же"), ("тоже", "то же"),
                 ("в течение", "в течении"), ("потому что", "потомучто"), ("вообще", "в общем"), ("сейчас", "щас"),
                 ("что", "че"), ("ничего", "ниче"), ("сегодня", "седня"), ("когда", "када"), ("здравствуйте", "здрасте"),
                 ("пожалуйста", "пожалста"), ("тебе", "те"), ("его", "ево"), ("конечно", "конешно")],
    "en_words": [("their", "there"), ("there", "their"), ("they're", "there"), ("to", "too"), ("too", "to"),
                 ("your", "you're"), ("you're", "your"), ("its", "it's"), ("it's", "its"), ("then", "than"),
                 ("than", "then"), ("affect", "effect"), ("accept", "except"), ("whether", "weather"),
                 ("here", "hear"), ("know", "no"), ("right", "write"), ("buy", "by"), ("for", "four"),
                 ("new", "knew"), ("wait", "weight"), ("we'll", "will"), ("could have", "could of"),
                 ("going to", "gonna"), ("want to", "wanna"), ("all right", "alright"), ("a lot", "alot")],
    "he_words": [("עם", "אם"), ("אם", "עם"), ("על", "אל"), ("אל", "על"), ("כל כך", "כלכך"), ("בסדר", "ב סדר"),
                 ("אחר כך", "אחרכך"), ("בגלל", "ב גלל"), ("אצל", "עצל"), ("קרה", "קרא"), ("קרא", "קרה"),
                 ("עכשיו", "אכשיו"), ("כאן", "כן"), ("מאוד", "מעוד"), ("טוב", "תוב"), ("פשוט", "פשות"),
                 ("תודה", "טודה"), ("כמובן", "חמובן"), ("שלום", "סלום"), ("מספר", "משפר")],
}
RU_ENDINGS = [("ами", "ам"), ("ого", "ой"), ("ыми", "ым"), ("ими", "им"), ("ется", "ет"), ("ются", "ют"),
              ("ала", "ал"), ("или", "ил"), ("ую", "ая"), ("ых", "ый"), ("ому", "ом"), ("ться", "ть"), ("ешь", "ет"),
              ("ов", "а"), ("ах", "ам")]
PREP_CASE = {**{p: "е" for p in ("в", "во", "на", "о", "об", "при")}, **{p: "у" for p in ("к", "ко", "по")},
             **{p: "ом" for p in ("с", "со", "над", "под", "за", "перед", "между")},
             **{p: "а" for p in ("из", "от", "до", "у", "без", "для", "около", "после", "из-за")}}


@dataclass
class Noised:
    hyp: str
    ops: list = field(default_factory=list)
    corruptions: list = field(default_factory=list)   # {"entity","kind","to"}
    profile: str = ""


_CONF_CACHE: dict = {}


def mined_confusions(lang: str) -> dict[str, list[tuple[str, int]]]:
    """ref word -> [(hyp word, count)] mined by public/convert.py (real GigaAM / HyPoradise errors)."""
    if lang in _CONF_CACHE:
        return _CONF_CACHE[lang]
    p = DERIVED / f"confusions_{lang}.json"
    table: dict[str, list] = {}
    if p.exists():
        for it in json.loads(p.read_text(encoding="utf-8")):
            if it["n"] >= 3 and len(it["ref"]) > 2:
                table.setdefault(it["ref"], []).append((it["hyp"], it["n"]))
    _CONF_CACHE[lang] = table
    return table


def load_profiles(path: str | Path) -> None:
    """Override profile rates with measured D1 statistics {engine: {rate: value}}."""
    for eng, vals in json.loads(Path(path).read_text(encoding="utf-8")).items():
        PROFILES.setdefault(eng, dict(PROFILES["parakeet_v3"])).update(vals)


# ------------------------------------------------------------------ entity rendering
def respell_en(name: str, r: random.Random) -> str:
    """How an ASR spells an unfamiliar Latin name it heard: diacritics gone,
    phonetic simplifications ('Raghunathan' -> 'ragunathan', 'Jürgen' -> 'yurgen')."""
    s = name
    if s[:1] in "Jj" and len(s) > 1 and s[1] in "üu":
        s = "Y" + s[1:]
    s = T.strip_diacritics(s.replace("ß", "ss").replace("ø", "o").replace("æ", "ae").replace("ł", "l"))
    rules = [("gh", "g"), ("ph", "f"), ("th", "t"), ("ck", "k"), ("qu", "kw"), ("ei", "ai"), ("ie", "ee"),
             ("oo", "u"), ("y", "i"), ("c", "k"), ("w", "v"), ("sch", "sh"), ("tz", "ts")]
    r.shuffle(rules)
    for a, b in rules[:3]:
        if a in s.lower():
            s = re.sub(a, b, s, count=1, flags=re.I)
            break
    s = re.sub(r"([a-z])\1", r"\1", s)
    return s


def respell_ru(word: str, r: random.Random) -> str:
    """Surname mis-hearing: one vowel/consonant swap ('Галиев' -> 'Галеев')."""
    swaps = [("и", "е"), ("е", "и"), ("о", "а"), ("а", "о"), ("д", "т"), ("т", "д"), ("б", "п"), ("з", "с"),
             ("в", "ф"), ("г", "к"), ("ё", "е"), ("ц", "тс"), ("щ", "ш"), ("ы", "и"), ("я", "е")]
    cands = [(i, a, b) for i in range(1, len(word) - 2) for a, b in swaps if word[i] == a]
    if not cands:
        return word
    i, a, b = r.choice(cands)
    return word[:i] + b + word[i + len(a):]


def respell_he(word: str, r: random.Random) -> str:
    """Hebrew ASR homophone slip in a word or surname (א/ע/ה, ט/ת, כ/ק, כ/ח, ס/ש, ב/ו),
    preserving Hebrew final-letter forms (ךםןףץ)."""
    if len(word) < 3:
        return word
    cands = [(i, a, b) for i in range(len(word)) for a, b in HE_HOMOPHONE_SWAPS if word[i] == a]
    if not cands:
        return word
    i, a, b = r.choice(cands)
    if i == len(word) - 1 and b in _HE_FINAL_FIX:
        b = _HE_FINAL_FIX[b]
    elif i < len(word) - 1 and b in _HE_MEDIAL_FIX:
        b = _HE_MEDIAL_FIX[b]
    return word[:i] + b + word[i + 1:]


def heb_form(e_text: str, r: random.Random) -> str:
    """How a Hebrew ASR engine (e.g. zipformer2-66m-streaming-he) transliterates a Latin tech term."""
    forms = L.TECH_HE.get(e_text)
    if forms:
        return r.choice(forms)
    return " ".join(T.lat2heb_rough(w) for w in T.words(e_text))


def cyr_form(e_text: str, r: random.Random) -> str:
    e = L.BY_TEXT.get(e_text)
    if e and e.spoken_ru:
        return r.choice(list(e.spoken_ru))
    return " ".join(T.lat2cyr_rough(w) for w in T.words(e_text))


# ------------------------------------------------------------------ main entry
def _case_ending(prev: str, form: str, r: random.Random) -> str:
    end = PREP_CASE.get(prev.lower())
    if end and form and form[-1] in "бвгджзклмнпрстфхцчшщ" and r.random() < 0.75:
        return form + end
    return form


def _scale(p: float, difficulty: float) -> float:
    return p * (0.3 + 1.4 * difficulty)


def _he_translit_token(tok: str, r: random.Random) -> str:
    """Transliterate a Latin token (optionally with Hebrew prefix like 'ל-GitHub') into Hebrew."""
    m = re.match(r"^([\u05d0-\u05ea])-([A-Za-z][A-Za-z0-9._+-]*)([^\w]*)$", tok)
    if m:
        pref, lat, trail = m.groups()
        hf = heb_form(lat, r)
        # zipformer2 either glues prefix ('לגיטהאב') or splits it ('ל גיטהאב')
        sep = " " if r.random() < 0.3 else ""
        return pref + sep + hf + trail
    if T.is_lat(tok):
        core = re.sub(r"[^\w.+_-]+$", "", tok)
        return heb_form(core, r) + tok[len(core):]
    return tok


def noise(spoken: str, target: str, lang: str, profile: str, difficulty: float, r: random.Random,
          entities: list[dict] | None = None, lower: bool = False, force_tail: bool = False) -> Noised:
    P = PROFILES[profile]
    out = Noised(hyp="", profile=profile)
    toks = spoken.split()
    tlang = "he" if lang in ("he", "he_en", "he_ru") else ("ru" if lang in ("ru", "mixed") else "en")

    # 1) entities: Latin terms in Russian/Hebrew speech -> Cyrillic/Hebrew; names mis-heard
    for ent in entities or []:
        etext, kind = ent["text"], ent["kind"]
        ew = etext.split()
        low = [t.lower().strip(",.!?") for t in toks]
        # also match Hebrew-prefixed single-word Latin terms like "ל-GitHub"
        for i in range(len(toks) - len(ew) + 1):
            he_pref = ""
            if len(ew) == 1 and tlang == "he":
                mp = re.match(r"^([\u05d0-\u05ea])-(.+)$", low[i])
                if mp and mp.group(2) == ew[0].lower():
                    he_pref = mp.group(1)
                elif low[i] != ew[0].lower():
                    continue
            elif low[i:i + len(ew)] != [w.lower() for w in ew]:
                continue
            if T.is_lat(etext) and tlang == "he" and (r.random() < P["translit"] or not P["latin"]) and kind != "person_en":
                form = heb_form(etext, r)
                if he_pref:
                    form = he_pref + (" " if r.random() < 0.3 else "") + form
                toks[i:i + len(ew)] = form.split()
                out.corruptions.append({"entity": etext, "kind": kind, "to": form, "op": "translit"})
                out.ops.append({"op": "translit", "from": etext, "to": form})
            elif T.is_lat(etext) and tlang == "ru" and (r.random() < P["translit"] or not P["latin"]) and kind != "person_en":
                form = cyr_form(etext, r)
                form = _case_ending(toks[i - 1] if i else "", form, r)
                toks[i:i + len(ew)] = form.split()
                out.corruptions.append({"entity": etext, "kind": kind, "to": form, "op": "translit"})
                out.ops.append({"op": "translit", "from": etext, "to": form})
            elif kind in ("person_ru", "person_en", "person_he") and r.random() < P["name_err"]:
                sur_i = i + len(ew) - 1
                old = toks[sur_i]
                core = re.sub(r"[,.!?]+$", "", old)
                if T.is_heb(core):
                    new = respell_he(core, r)
                elif T.is_cyr(core):
                    new = respell_ru(core, r)
                else:
                    new = respell_en(core, r)
                if not P["latin"] and T.is_lat(new):
                    new = T.lat2heb_rough(new) if tlang == "he" else T.lat2cyr_rough(new)
                if new != core:
                    toks[sur_i] = new + old[len(core):]
                    out.corruptions.append({"entity": etext, "kind": kind, "to": new, "from_token": core, "op": "name"})
                    out.ops.append({"op": "name", "from": core, "to": new})
            elif T.is_lat(etext) and tlang == "en" and kind == "tech" and r.random() < 0.4:
                parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", etext)
                form = " ".join(p.lower() for p in parts) if len(parts) > 1 else etext.lower()
                toks[i:i + len(ew)] = form.split()
                out.corruptions.append({"entity": etext, "kind": kind, "to": form, "op": "split"})
                out.ops.append({"op": "split", "from": etext, "to": form})
            break

    # 2) word-level confusions, boundary errors, dropped endings, function-word deletions
    mined = mined_confusions(tlang)
    words_conf = BUILTIN_CONF["he_words" if tlang == "he" else "ru_words" if tlang == "ru" else "en_words"]
    res: list[str] = []
    i = 0
    protected = {w.lower() for e in (entities or []) for w in e["text"].split()}
    while i < len(toks):
        w = toks[i]
        wl = w.lower()
        if wl in protected or wl in T.NEGATIONS_RU or wl in T.NEGATIONS_EN or wl in T.NEGATIONS_HE:
            res.append(w)
            i += 1
            continue
        two = (wl + " " + toks[i + 1].lower()) if i + 1 < len(toks) else ""
        hit = next((b for a, b in words_conf if a == two), None) if two else None
        if hit and r.random() < _scale(P["conf"] * 3, difficulty):
            res.extend(hit.split())
            out.ops.append({"op": "conf2", "from": two, "to": hit})
            i += 2
            continue
        if r.random() < _scale(P["conf"], difficulty):
            alts = mined.get(T.norm_word(wl))
            b = None
            if alts:
                b = r.choices([a for a, _ in alts], weights=[n for _, n in alts])[0]
            else:
                b = next((y for x, y in words_conf if x == wl), None)
                if b is None and tlang == "ru":
                    for a2, b2 in BUILTIN_CONF["ru"]:
                        if wl.endswith(a2):
                            b = wl[: -len(a2)] + b2
                            break
                elif b is None and tlang == "he" and T.is_heb(wl) and len(wl) >= 4:
                    core = re.sub(r"[,.!?]+$", "", w)
                    swapped = respell_he(core, r)
                    if swapped != core:
                        b = swapped + w[len(core):]
            if b and b != wl:
                res.append(b)
                out.ops.append({"op": "conf", "from": wl, "to": b})
                i += 1
                continue
        # Hebrew prefix split ('בבית' -> 'ב בית', 'לגיטהאב' -> 'ל גיטהאב') or join
        if tlang == "he" and T.is_heb(wl) and len(wl) >= 4 and wl[0] in HE_PREFIXES and "-" not in wl and r.random() < _scale(P["bound"], difficulty):
            res.extend([w[0], w[1:]])
            out.ops.append({"op": "split_prefix", "from": w, "to": w[0] + " " + w[1:]})
            i += 1
            continue
        if len(wl) >= 8 and r.random() < _scale(P["bound"], difficulty):
            k = r.randint(3, len(wl) - 3)
            res.extend([w[:k], w[k:]])
            out.ops.append({"op": "split_word", "from": w, "to": w[:k] + " " + w[k:]})
            i += 1
            continue
        if tlang == "ru" and T.is_cyr(wl) and len(wl) >= 6 and r.random() < _scale(P["ending"], difficulty):
            for a, b in RU_ENDINGS:
                if wl.endswith(a):
                    res.append(w[: -len(a)] + b)
                    out.ops.append({"op": "ending", "from": w, "to": w[: -len(a)] + b})
                    break
            else:
                res.append(w)
            i += 1
            continue
        if tlang == "en" and len(wl) >= 5 and wl.endswith(("ed", "s")) and r.random() < _scale(P["ending"], difficulty):
            nw = w[:-2] if wl.endswith("ed") else w[:-1]
            res.append(nw)
            out.ops.append({"op": "ending", "from": w, "to": nw})
            i += 1
            continue
        he_fdel_ok = {"את", "של", "על", "אל", "עם", "או", "גם", "כי", "אז"}
        if ((tlang != "he" and wl in T.FUNCTION_WORDS and len(wl) <= 3) or (tlang == "he" and wl in he_fdel_ok)) and r.random() < _scale(P["fdel"], difficulty):
            out.ops.append({"op": "fdel", "from": w})
            i += 1
            continue
        res.append(w)
        i += 1
    toks = res

    # 3) hesitations: dropped or rendered per engine
    res = []
    for w in toks:
        if w.lower() in T.HESITATION:
            if r.random() < P["keep_hes"]:
                res.append(r.choice(HES_RENDER[tlang]))
            else:
                out.ops.append({"op": "hes_drop", "from": w})
            continue
        res.append(w)
    toks = res

    # 4) T-one / zipformer2_he (single-script engines): transliterate Latin tokens
    if not P["latin"]:
        if tlang == "he":
            new_toks = []
            for w in toks:
                if T.is_lat(w) or re.match(r"^[\u05d0-\u05ea]-[A-Za-z]", w):
                    tw = _he_translit_token(w, r)
                    new_toks.extend(tw.split())
                    if tw != w:
                        out.ops.append({"op": "translit", "from": w, "to": tw})
                else:
                    new_toks.append(w)
            toks = new_toks
        else:
            toks = [T.lat2cyr_rough(w) if T.is_lat(w) else w for w in toks]

    # 5) numbers: words -> digits per engine
    hyp_words = toks
    if r.random() < P["digits"]:
        hyp_words = _digitise(toks, tlang, r)
        if hyp_words != toks:
            out.ops.append({"op": "digits"})

    # 6) punctuation / casing projected from the target, then degraded
    if P["punct"] and not lower:
        hyp = _project_punct(hyp_words, target, P, r, tlang)
    else:
        hyp = " ".join(re.sub(r"[^\w\s'’\-׳״]", "", w).lower() for w in hyp_words)
        if not P["latin"]:
            hyp = hyp.replace("-", " ")
        out.ops.append({"op": "lower_nopunct"})

    # 7) Whisper-style artifacts
    if force_tail or (P["tail"] and r.random() < P["tail"]):
        tail = r.choice(WHISPER_TAILS[tlang])
        hyp = (hyp + " " + tail).strip()
        out.ops.append({"op": "tail", "to": tail})
    if P["loop"] and r.random() < P["loop"]:
        ws = hyp.split()
        if len(ws) >= 4:
            k = r.randint(2, min(5, len(ws)))
            hyp = hyp + (" " + " ".join(ws[-k:])) * r.randint(2, 4)
            out.ops.append({"op": "loop"})
    out.hyp = re.sub(r"\s+", " ", hyp).strip()
    return out


def _digitise(toks: list[str], lang: str, r: random.Random) -> list[str]:
    norm = [T.norm_word(re.sub(r"[^\w'’-]", "", t)) for t in toks]
    spans = T.number_spans(norm, lang)
    out, i = [], 0
    starts = {a: (b, v) for a, b, v in spans if not norm[a].isdigit()}
    while i < len(toks):
        if i in starts and _keep_as_words(toks, norm, i, starts[i][0], starts[i][1], starts):
            out.extend(toks[i:starts[i][0]])
            i = starts[i][0]
            continue
        if i in starts:
            b, v = starts[i]
            s = str(v)
            if v >= 10000 and r.random() < 0.5:
                s = f"{v:,}".replace(",", " " if lang == "ru" else ",")
            # "пятнадцать тридцать" -> 15:30 (time), "семь сорок пять" -> 7:45
            if b in starts and v <= 23 and (10 <= starts[b][1] < 60 or norm[b] == "ноль") and (r.random() < 0.7 or norm[b] == "ноль"):
                b2, v2 = starts[b]
                s = f"{v}:{v2:02d}"
                b = b2
                if v2 == 0 and b < len(norm) and norm[b] == "ноль":  # "двенадцать ноль ноль"
                    b += 1
            trail = re.sub(r"^.*?([,.!?]*)$", r"\1", toks[b - 1])
            nxt = norm[b] if b < len(norm) else ""
            if nxt.startswith(("процент", "percent")) and r.random() < 0.6:
                s += "%"
                b += 1
            out.append(s + trail)
            i = b
        else:
            out.append(toks[i])
            i += 1
    return out


UNIT_PREFIXES = ("процент", "рубл", "долл", "евро", "час", "минут", "секунд", "км", "метр", "килограмм", "грамм",
                 "гигабайт", "мегабайт", "тысяч", "миллион", "percent", "dollar", "euro", "hour", "minute", "second",
                 "pm", "am", "gb", "mb", "kg", "km", "thousand", "million", "%", "$", "₽", "€",
                 "שקל", "שקלים", "אחוז", "אחוזים", "דקה", "דקות", "שעה", "שעות", "שניה", "שניות", "קילו", "מטר", "₪")


def _keep_as_words(toks: list[str], norm: list[str], a: int, b: int, v: int, starts: dict) -> bool:
    """ASR engines that print digits still write small bare counts and ordinal adverbs in words:
    'о трёх вещах', 'во-вторых', 'one of them'."""
    if any("-" in t for t in toks[a:b]):
        return True
    if b - a == 1 and v < 10:
        if b in starts and v <= 23 and (10 <= starts[b][1] < 60 or norm[b] == "ноль"):  # "семь сорок пять" -> 7:45
            return False
        nxt = norm[b] if b < len(norm) else ""
        return not nxt.startswith(UNIT_PREFIXES)
    return False


def _tok_punct(target: str):
    """target -> [(key, trailing_punct, capitalised)] for alignment."""
    out = []
    initial = True
    for m in re.finditer(r"([«\"(]*)([\w'’%:./@#-]*\w[\w'’%]*|\d[\d:.,%]*)([^\w\s]*)", target.replace("\n", " ")):
        word, trail = m.group(2), m.group(3)
        trail = re.sub(r"[»\")]", "", trail)
        # only mid-sentence capitals (proper nouns) are projected; sentence-initial ones are not word properties
        out.append((T.norm_word(word), trail, word[:1].isupper() and (not initial or word.isupper() and len(word) > 1)))
        initial = bool(re.search(r"[.!?…:]$", trail)) or (initial and not word)
    return out


def _project_punct(words: list[str], target: str, P: dict, r: random.Random, lang: str) -> str:
    tgt = _tok_punct(target)
    hyp_keys = [T.norm_word(re.sub(r"[^\w'’%:.-]", "", w).strip(".")) for w in words]
    ops = T.align([k for k, _, _ in tgt], hyp_keys)
    punct = [""] * len(words)
    caps = [False] * len(words)
    for op, ti, hj in ops:
        if hj is None or ti is None:
            continue
        if op in ("eq", "sub"):
            punct[hj] = tgt[ti][1]
            caps[hj] = tgt[ti][2] and (op == "eq" or r.random() < 0.5)
    # fillers kept by ASR get commas around them sometimes ("Ну, короче, давай")
    for j, w in enumerate(words):
        if (w.lower() in T.HESITATION or w.lower() in ("ну", "короче", "типа", "so", "like", "well")) and r.random() < 0.5:
            punct[j] = punct[j] or ","
    out = []
    sentence_start = True
    for j, w in enumerate(words):
        p = punct[j]
        if p.startswith(",") and r.random() < P["comma_drop"]:
            p = p[1:]
        if p.startswith("?") and r.random() < 0.15:
            p = "."
        if ":" in p and r.random() < 0.5:
            p = p.replace(":", ",")
        base = re.sub(r"[,.!?;:…]+$", "", w) if not re.match(r"^\d", w) else w.rstrip(",.!?")
        if sentence_start or caps[j]:
            base = base[:1].upper() + base[1:]
        elif not caps[j] and not re.search(r"[A-Z].*[A-Z]", base):  # keep acronyms / CamelCase
            base = base[:1].lower() + base[1:] if base[:1].isupper() and j > 0 and not caps[j] else base
        out.append(base + p)
        sentence_start = bool(re.search(r"[.!?…]$", p))
    hyp = " ".join(out)
    if r.random() < 0.1:
        hyp = hyp.rstrip(".")
    return hyp
