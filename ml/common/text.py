"""Shared text utilities for Murmur data and eval (ru + en). Stdlib only.

Everything that compares two texts (filters, N-WER, consistency checks,
context ranking) goes through these functions, so data and eval agree on what
"the same word" means.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# ------------------------------------------------------------------ basics
_NIQQUD = re.compile(r"[\u0591-\u05c7]")
_WORD = re.compile(
    r"[0-9A-Za-zÀ-ɏͰ-ϿЀ-ӿ\u05d0-\u05ea]+(?:['’\-׳״][0-9A-Za-zÀ-ɏͰ-ϿЀ-ӿ\u05d0-\u05ea]+)*׳?"
)
_CYR = re.compile(r"[Ѐ-ӿ]")
_LAT = re.compile(r"[A-Za-zÀ-ɏ]")
_HEB = re.compile(r"[\u05d0-\u05ea]")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def strip_niqqud(s: str) -> str:
    """Remove Hebrew cantillation/vowel combining marks (U+0591..U+05C7)."""
    return _NIQQUD.sub("", s or "")


def words(s: str) -> list[str]:
    """Surface word tokens (letters/digits, inner hyphen/apostrophe/geresh/gershayim kept)."""
    return _WORD.findall(strip_niqqud(nfc(s)))


def norm_word(w: str) -> str:
    """Case/ё/apostrophe/niqqud-insensitive key of one word."""
    w = strip_niqqud(nfc(w)).lower().replace("ё", "е").replace("’", "'")
    return w


def norm_words(s: str) -> list[str]:
    return [norm_word(w) for w in words(s)]


def strip_diacritics(s: str) -> str:
    """Drop combining marks but keep Cyrillic й/ё (they are letters, not accents)."""
    keep = {"й": "\0y", "Й": "\0Y", "ё": "\0e", "Ё": "\0E"}
    for k, v in keep.items():
        s = s.replace(k, v)
    s = "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))
    s = unicodedata.normalize("NFC", s)
    for k, v in keep.items():
        s = s.replace(v, k)
    return s


def script_counts(s: str) -> tuple[int, int]:
    """(cyrillic letters, latin letters)."""
    return len(_CYR.findall(s or "")), len(_LAT.findall(s or ""))


def script_counts_3(s: str) -> tuple[int, int, int]:
    """(cyrillic letters, latin letters, hebrew letters)."""
    s = s or ""
    return len(_CYR.findall(s)), len(_LAT.findall(s)), len(_HEB.findall(s))


def cyr_share(s: str) -> float:
    c, l = script_counts(s)
    return c / (c + l) if c + l else 0.0


def heb_share(s: str) -> float:
    c, l, h = script_counts_3(s)
    tot = c + l + h
    return h / tot if tot else 0.0


def guess_lang(s: str) -> str:
    """'ru' | 'en' | 'mixed' | 'he' | 'he_en' | 'he_ru' | '' by script share.
    Latin product names inside Russian/Hebrew text keep it 'ru'/'he' as long as
    the primary script dominates."""
    ws = [w for w in words(s) if not w.isdigit()]
    if not ws:
        return ""
    n_heb = sum(1 for w in ws if is_heb(w))
    if n_heb:
        n_cyr = sum(1 for w in ws if is_cyr(w))
        n_lat = sum(1 for w in ws if _LAT.search(w))
        n_heb_only = sum(1 for w in ws if is_heb(w) and not _LAT.search(w) and not _CYR.search(w))
        if n_cyr >= 1 and n_heb >= 1 and n_heb_only / len(ws) < 0.85:
            return "he_ru"
        if n_heb_only / len(ws) >= 0.75 and n_cyr == 0:
            return "he"
        if n_lat >= 1:
            return "he_en"
        return "he_ru" if n_cyr else "he"
    share = sum(1 for w in ws if is_cyr(w)) / len(ws)
    if share >= 0.75:
        return "ru"
    if share <= 0.1:
        return "en"
    return "mixed"


def is_cyr(w: str) -> bool:
    return bool(_CYR.search(w))


def is_heb(w: str) -> bool:
    return bool(_HEB.search(w))


def is_lat(w: str) -> bool:
    return bool(_LAT.search(w)) and not _CYR.search(w) and not _HEB.search(w)


# ------------------------------------------------------------------ lexicons
FILLERS_EN = {"um", "uh", "uhm", "umm", "er", "erm", "ah", "eh", "hmm", "mm", "mhm", "huh"}
FILLER_PHRASES_EN = ["you know", "i mean", "kind of", "sort of", "like", "basically", "so", "well",
                     "okay so", "let's see", "actually", "literally"]
FILLERS_RU = {"э", "ээ", "эээ", "э-э", "э-э-э", "м", "мм", "ммм", "м-м", "а-а", "ам", "эм", "хм", "ну-у"}
FILLER_PHRASES_RU = ["ну", "вот", "короче", "типа", "как бы", "в общем", "значит", "слушай", "это самое",
                     "так сказать", "в принципе", "ну вот", "ну короче", "как его"]
FILLERS_HE = {"אמ", "אממ", "אמממ", "אה", "אהה", "אההה", "המ", "הממ", "ממ", "מממ", "א-א"}
FILLER_PHRASES_HE = ["כאילו", "בעצם", "יעני", "נו", "אז", "בקיצור", "סתם", "כזה", "אתה יודע", "תכלס", "טוב אז"]
SC_MARKERS_EN = ["no wait", "sorry", "i mean", "scratch that", "actually", "make that", "or rather",
                 "no no", "correction", "let me rephrase", "wait no"]
SC_MARKERS_RU = ["ой нет", "нет", "точнее", "вернее", "в смысле", "то есть", "не-не", "стоп", "ой",
                 "нет-нет", "я хотел сказать", "я хотела сказать", "поправка", "не так"]
SC_MARKERS_HE = ["לא רגע", "סליחה", "כלומר", "בעצם", "זאת אומרת", "תמחק את זה", "לא לא", "אופס", "רגע לא", "תיקון"]
HESITATION = FILLERS_EN | FILLERS_RU | FILLERS_HE  # pure sounds, never meaningful

STOP_EN = set("""a an the and or but if then so of to in on at by for with from up down out over under
about into onto than as is are was were be been being am do does did done have has had having i you he
she it we they me him her us them my your his its our their this that these those there here what which
who whom whose when where why how not no yes all any some each every both few more most other such only
own same too very can could will would shall should may might must just also again further once
s t don't i'm it's that's i've i'll i'd we're they're you're isn't aren't wasn't weren't can't won't
let's""".split())
STOP_RU = set("""и а но или да же ли бы б то это этот эта эти тот та те так как что чтобы когда где куда
откуда почему зачем кто чем чего кого кому ком в во на по к ко с со у о об обо от до из за под над при
про для без через между перед около после я ты он она оно мы вы они меня тебя его её ее нас вас их мне
тебе ему ей нам вам им мной тобой ним ней нами вами ними мой моя моё мое мои твой твоя твоё твое твои
свой своя своё свое свои наш наша наше наши ваш ваша ваше ваши не ни нет есть был была было были быть
будет будут уже еще ещё только вот ну тоже также все всё весь вся всего всех очень там тут здесь сейчас
потом тогда если же ведь вон""".split())
STOP_HE = set("""של את על אל עם או גם כי אם אז אבל רק כל זה זו זאת אלה אלו אני אתה את הוא היא אנחנו אתם
הן הם לי לך לו לה לנו לכם להם שלי שלך שלו שלה שלנו יש אין לא כן מה מי איפה מתי למה איך כמה כדי
כמו בין לפני אחרי תחת מעל מול אצל מתוך דרך עד מאז עוד כבר עכשיו אחר כך שם פה כאן מאוד יותר פחות
פה היה הייתה היו יהיה תהיה להיות בשביל בגלל למרות""".split())
FUNCTION_WORDS = STOP_EN | STOP_RU | STOP_HE

NEGATIONS_EN = {"not", "no", "never", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
                "can't", "cannot", "won't", "wouldn't", "shouldn't", "nothing", "nobody", "none", "neither", "nor",
                "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "cant", "wont", "wouldnt", "shouldnt",
                "couldn't", "couldnt", "haven't", "havent", "hasn't", "hasnt"}
NEGATIONS_RU = {"не", "нет", "ни", "никогда", "ничего", "никто", "нельзя", "нисколько", "никак", "нигде"}
NEGATIONS_HE = {"לא", "אין", "אל", "כלום", "בלי", "שום", "ללא", "אף", "איננו", "איננה", "אינם"}
HEDGES_EN = ["i think", "maybe", "probably", "perhaps", "i guess", "i believe", "might"]
HEDGES_RU = ["мне кажется", "наверное", "может быть", "возможно", "вроде", "по-моему", "кажется", "думаю"]
HEDGES_HE = ["נראה לי", "אולי", "כנראה", "אני חושב", "אני חושבת", "יכול להיות"]


def content_words(s: str) -> list[str]:
    """Normalised words that carry content: not function words, not fillers,
    not pure numbers (numbers are compared separately)."""
    out = []
    for w in norm_words(s):
        if w in FUNCTION_WORDS or w in HESITATION or w.isdigit():
            continue
        if len(w) == 1 and not w.isdigit():
            continue
        out.append(w)
    return out


# ------------------------------------------------------------------ numbers
_EN_UNITS = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
             "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
            "ninety": 90}
_EN_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
           "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
           "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
           "twentieth": 20, "thirtieth": 30}
_EN_SCALES = {"hundred": 100, "thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9}

_ADJ = ["ый", "ой", "ий", "ого", "его", "ому", "ему", "ым", "им", "ом", "ем", "ая", "яя", "ую", "юю", "ое", "ее",
        "ые", "ие", "ых", "их", "ыми", "ими", "ей"]


def _ru_forms() -> dict[str, tuple[int, str]]:
    """word form -> (value, kind) where kind in unit/teen/ten/hundred/scale/ord."""
    f: dict[str, tuple[int, str]] = {}

    def add(forms, v, kind):
        for x in forms:
            f[x] = (v, kind)

    def ordinal(stem, v):
        add([stem + e for e in _ADJ], v, "ord")

    add(["ноль", "нуль", "ноля", "нуля", "нолю", "нулю", "нолем", "нулем", "нулём", "нолём"], 0, "unit")
    add(["один", "одна", "одно", "одни", "одного", "одной", "одному", "одним", "одном", "одну", "одною", "одних",
         "одними"], 1, "unit")
    add(["два", "две", "двух", "двум", "двумя"], 2, "unit")
    add(["три", "трех", "трём", "трем", "тремя", "трёх"], 3, "unit")
    add(["четыре", "четырех", "четырёх", "четырем", "четырём", "четырьмя"], 4, "unit")
    for stem, v in [("пят", 5), ("шест", 6), ("сем", 7), ("девят", 9), ("десят", 10)]:
        add([stem + "ь", stem + "и", stem + "ью"], v, "unit" if v < 10 else "teen")
    add(["восемь", "восьми", "восемью", "восьмью"], 8, "unit")
    ordinal("перв", 1); ordinal("втор", 2); ordinal("четверт", 4); ordinal("четвёрт", 4)
    ordinal("пят", 5); ordinal("шест", 6); ordinal("седьм", 7); ordinal("восьм", 8); ordinal("девят", 9)
    ordinal("десят", 10)
    add(["третий", "третьего", "третьему", "третьим", "третьем", "третья", "третью", "третьей", "третье",
         "третьи", "третьих", "третьими"], 3, "ord")
    teens = [("одиннадцат", 11), ("двенадцат", 12), ("тринадцат", 13), ("четырнадцат", 14), ("пятнадцат", 15),
             ("шестнадцат", 16), ("семнадцат", 17), ("восемнадцат", 18), ("девятнадцат", 19)]
    for stem, v in teens:
        add([stem + "ь", stem + "и", stem + "ью"], v, "teen")
        ordinal(stem, v)
    for stem, v in [("двадцат", 20), ("тридцат", 30)]:
        add([stem + "ь", stem + "и", stem + "ью"], v, "ten")
        ordinal(stem, v)
    add(["сорок", "сорока"], 40, "ten"); ordinal("сороков", 40)
    for nom, obl, ordst, v in [("пятьдесят", "пятидесяти", "пятидесят", 50), ("шестьдесят", "шестидесяти", "шестидесят", 60),
                               ("семьдесят", "семидесяти", "семидесят", 70), ("восемьдесят", "восьмидесяти", "восьмидесят", 80)]:
        add([nom, obl, nom.replace("десят", "юдесятью")], v, "ten")
        ordinal(ordst, v)
    add(["девяносто", "девяноста"], 90, "ten"); ordinal("девяност", 90)
    add(["сто", "ста"], 100, "hundred"); ordinal("сот", 100)
    hundreds = [(["двести", "двухсот", "двумстам", "двумястами", "двухстах"], 200, "двухсот"),
                (["триста", "трехсот", "трёхсот", "тремстам", "тремястами", "трехстах", "трёхстах"], 300, "трехсот"),
                (["четыреста", "четырехсот", "четырёхсот", "четырьмстам", "четырьмястами", "четырехстах"], 400, "четырехсот"),
                (["пятьсот", "пятисот", "пятистам", "пятьюстами", "пятистах"], 500, "пятисот"),
                (["шестьсот", "шестисот", "шестистам", "шестьюстами", "шестистах"], 600, "шестисот"),
                (["семьсот", "семисот", "семистам", "семьюстами", "семистах"], 700, "семисот"),
                (["восемьсот", "восьмисот", "восьмистам", "восемьюстами", "восьмистах"], 800, "восьмисот"),
                (["девятьсот", "девятисот", "девятистам", "девятьюстами", "девятистах"], 900, "девятисот")]
    for forms, v, ordst in hundreds:
        add(forms, v, "hundred")
        ordinal(ordst, v)
    add(["тысяча", "тысячи", "тысяче", "тысячу", "тысячей", "тысячею", "тысяч", "тысячам", "тысячами", "тысячах"],
        1000, "scale")
    ordinal("тысячн", 1000)
    for stem, v in [("миллион", 10 ** 6), ("миллиард", 10 ** 9)]:
        add([stem, stem + "а", stem + "у", stem + "ом", stem + "е", stem + "ы", stem + "ов", stem + "ам",
             stem + "ами", stem + "ах"], v, "scale")
    return f


RU_NUM = _ru_forms()


def _en_value(w: str):
    if w in _EN_UNITS:
        return _EN_UNITS[w], "unit" if _EN_UNITS[w] < 10 else "teen"
    if w in _EN_TENS:
        return _EN_TENS[w], "ten"
    if w in _EN_ORD:
        return _EN_ORD[w], "ord"
    if w == "hundred":
        return 100, "hundred_mul"
    if w in _EN_SCALES:
        return _EN_SCALES[w], "scale"
    if "-" in w:  # twenty-five, twenty-fifth
        a, _, b = w.partition("-")
        if a in _EN_TENS and (b in _EN_UNITS or b in _EN_ORD):
            vb = _EN_UNITS.get(b, _EN_ORD.get(b))
            if 0 < vb < 10:
                return _EN_TENS[a] + vb, "ord" if b in _EN_ORD else "ten"
    return None


def _he_forms() -> dict[str, tuple[int, str]]:
    """Hebrew cardinal/ordinal forms -> (value, kind), including ו-conjunction prefixes
    used in compound numbers ('עשרים וחמש' = 25, 'מאה ושלושים' = 130)."""
    f: dict[str, tuple[int, str]] = {}

    def add(forms, v, kind, with_vav=True):
        for x in forms:
            f[x] = (v, kind)
            if with_vav and not x.startswith("ו"):
                f["ו" + x] = (v, kind)

    add(["אפס"], 0, "unit")
    add(["אחד", "אחת"], 1, "unit")
    add(["שתיים", "שניים", "שתים", "שנים", "שתי", "שני"], 2, "unit")
    add(["שלוש", "שלושה", "שלושת"], 3, "unit")
    add(["ארבע", "ארבעה", "ארבעת"], 4, "unit")
    add(["חמש", "חמישה", "חמשת"], 5, "unit")
    add(["שש", "שישה", "ששת"], 6, "unit")
    add(["שבע", "שבעה", "שבעת"], 7, "unit")
    add(["שמונה", "שמונת"], 8, "unit")
    add(["תשע", "תשעה", "תשעת"], 9, "unit")
    add(["עשר", "עשרה", "עשרת"], 10, "teen")
    add(["עשרים"], 20, "ten")
    add(["שלושים"], 30, "ten")
    add(["ארבעים"], 40, "ten")
    add(["חמישים"], 50, "ten")
    add(["שישים"], 60, "ten")
    add(["שבעים"], 70, "ten")
    add(["שמונים"], 80, "ten")
    add(["תשעים"], 90, "ten")
    add(["מאה"], 100, "hundred")
    add(["מאתיים"], 200, "hundred")
    add(["מאות"], 100, "hundred_mul", with_vav=False)
    add(["אלף"], 1000, "scale")
    add(["אלפיים"], 2000, "scale")
    add(["אלפים"], 1000, "scale")
    add(["מיליון", "מיליונים"], 10 ** 6, "scale")
    add(["מיליארד", "מיליארדים"], 10 ** 9, "scale")
    for w, v in [("ראשון", 1), ("ראשונה", 1), ("שנייה", 2), ("שלישי", 3), ("שלישית", 3),
                 ("רביעי", 4), ("רביעית", 4), ("חמישי", 5), ("חמישית", 5), ("שישי", 6), ("שישית", 6),
                 ("שביעי", 7), ("שביעית", 7), ("שמיני", 8), ("שמינית", 8), ("תשיעי", 9), ("תשיעית", 9),
                 ("עשירי", 10), ("עשירית", 10)]:
        add([w], v, "ord", with_vav=False)
    return f


HE_NUM = _he_forms()


def _he_value(tok: str):
    if tok in HE_NUM:
        return HE_NUM[tok]
    # Hebrew preposition prefixes on numbers: בשלוש (at 3), לשבע (to 7), מעשר (from 10), כעשרים (~20)
    if len(tok) >= 4 and tok[0] in "בלהמכ" and tok[1:] in HE_NUM:
        v, k = HE_NUM[tok[1:]]
        if k != "ord":
            return v, k
    return None


def _num_info(tok: str, lang: str | None):
    if lang == "he":
        return _he_value(tok)
    if lang in ("he_en", "he_ru"):
        inf = _he_value(tok)
        if inf is not None:
            return inf
        if lang == "he_en":
            return _en_value(tok)
        return RU_NUM.get(tok)
    inf = RU_NUM.get(tok) if lang != "en" else None
    if inf is None and lang != "ru":
        inf = _en_value(tok)
    if inf is None and lang is None:
        inf = _he_value(tok)
    return inf


def _slot(v: int) -> str:
    return "unit" if v < 10 else "teen" if v < 20 else "ten" if v < 100 else "hundred"


def number_spans(tokens: list[str], lang: str | None = None) -> list[tuple[int, int, int]]:
    """Find maximal runs of number words in normalised tokens.
    Returns (start, end_exclusive, value). ru: all cases + ordinals; en:
    "two hundred and five", "twenty-five". A run breaks when a slot repeats
    ("один два три" = three numbers, as in dictated phone digits). Digit
    tokens are their own spans. "oh" counts only after another number."""
    spans: list[tuple[int, int, int]] = []
    i, n = 0, len(tokens)
    while i < n:
        t = tokens[i]
        if t.isdigit():
            spans.append((i, i + 1, int(t)))
            i += 1
            continue
        inf = _num_info(t, lang)
        if inf is None or (t == "oh" and not (spans and spans[-1][1] == i)):
            i += 1
            continue
        total, cur, last, j = 0, 0, None, i
        while j < n:
            tok = tokens[j]
            inf = _num_info(tok, lang)
            if inf is None:
                if tok == "and" and last in ("hundred",) and j + 1 < n and _num_info(tokens[j + 1], lang):
                    j += 1
                    continue
                break
            v, kind = inf
            if kind == "scale":
                if last == "scale_done":
                    break
                total += (cur or 1) * v
                cur, last = 0, "scale"
                j += 1
                continue
            if kind == "hundred_mul":
                if last not in ("unit", "teen") or cur >= 100:
                    break
                cur, last = cur * 100, "hundred"
                j += 1
                continue
            if tok in ("עשר", "עשרה") and last == "unit" and 1 <= cur % 10 <= 9:
                cur += 10
                last = "teen"
                j += 1
                continue
            slot = _slot(v)
            ok = {"unit": last in (None, "ten", "hundred", "scale") and cur % 10 == 0,
                  "teen": last in (None, "hundred", "scale") and cur % 100 == 0,
                  "ten": last in (None, "hundred", "scale") and cur % 100 == 0,
                  "hundred": last in (None, "scale") and cur == 0}[slot]
            if not ok:
                break
            cur += v
            last = slot
            j += 1
            if kind == "ord":
                break
        if j == i:
            i += 1
            continue
        spans.append((i, j, total + cur))
        i = j
    return spans


def numbers_in(s: str, lang: str | None = None) -> list[int]:
    """Multiset of numbers mentioned in s: digit groups ("15 200", "15,200",
    "15:30" -> 15, 30) and number words."""
    s2 = nfc(s)
    # join thousands groups written with space / nbsp / comma: 15 200, 15,200
    s2 = re.sub(r"(?<=\d)[   ,](?=\d{3}\b)", "", s2)
    toks = norm_words(s2)
    out = []
    for tok in toks:
        # split alnum tokens like 3pm, 25-го, v17
        for d in re.findall(r"\d+", tok):
            out.append(int(d))
    alpha = [t if not re.search(r"\d", t) else "\0" for t in toks]
    for a, b, v in number_spans(alpha, lang):
        if alpha[a] != "\0":
            out.append(v)
    return sorted(out)


def spelled_numbers_over(s: str, limit: int = 10, lang: str | None = None) -> list[int]:
    """Numbers > limit that are written in words (style violation for targets)."""
    toks = norm_words(s)
    toks = [t if not re.search(r"\d", t) else "\0" for t in toks]
    return [v for a, b, v in number_spans(toks, lang) if toks[a] != "\0" and v > limit]


# ------------------------------------------------------------------ verbalisation (for code-made spoken forms)
_EN_U = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_EN_T = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def en_cardinal(n: int) -> str:
    if n < 20:
        return _EN_U[n]
    if n < 100:
        return _EN_T[n // 10] + ("" if n % 10 == 0 else "-" + _EN_U[n % 10])
    if n < 1000:
        return _EN_U[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + en_cardinal(n % 100))
    for scale, name in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if n >= scale:
            return en_cardinal(n // scale) + " " + name + ("" if n % scale == 0 else " " + en_cardinal(n % scale))
    raise ValueError(n)


_RU_U = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять",
         "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать",
         "восемнадцать", "девятнадцать"]
_RU_T = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
_RU_H = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]


def ru_plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(n)
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def ru_cardinal(n: int, feminine: bool = False) -> str:
    """Nominative cardinal. feminine=True gives одна/две (тысяча, минута)."""
    if n == 0:
        return "ноль"
    parts = []
    for scale, forms, fem in ((10 ** 9, ("миллиард", "миллиарда", "миллиардов"), False),
                              (10 ** 6, ("миллион", "миллиона", "миллионов"), False),
                              (1000, ("тысяча", "тысячи", "тысяч"), True)):
        if n >= scale:
            q = n // scale
            parts.append(ru_cardinal(q, feminine=fem) + " " + ru_plural(q, *forms))
            n %= scale
    if n >= 100:
        parts.append(_RU_H[n // 100])
        n %= 100
    if n >= 20:
        parts.append(_RU_T[n // 10])
        n %= 10
    if n:
        w = _RU_U[n]
        if feminine and n == 1:
            w = "одна"
        elif feminine and n == 2:
            w = "две"
        parts.append(w)
    return " ".join(parts)


# ------------------------------------------------------------------ transliteration / phonetic key
_CYR2LAT = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
            "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
            "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "",
            "э": "e", "ю": "yu", "я": "ya"}


def cyr2lat(s: str) -> str:
    return "".join(_CYR2LAT.get(c, c) for c in s.lower())


# English letters/digraphs -> how a Russian speaker would write what they hear (rough; used for noise)
_LAT2CYR_MULTI = [("tion", "шн"), ("sion", "жн"), ("ough", "аф"), ("ight", "айт"), ("ee", "и"), ("oo", "у"),
                  ("ea", "и"), ("ou", "ау"), ("ow", "оу"), ("ai", "ей"), ("ay", "ей"), ("ey", "и"), ("oa", "оу"),
                  ("sh", "ш"), ("ch", "ч"), ("th", "т"), ("ph", "ф"), ("ck", "к"), ("qu", "кв"), ("kn", "н"),
                  ("wh", "у"), ("gh", "г"), ("ng", "нг"), ("ts", "ц"), ("ya", "я"), ("yu", "ю"), ("yo", "ё"),
                  ("zh", "ж"), ("x", "кс"), ("j", "дж"), ("w", "у"), ("c", "к"), ("q", "к")]
_LAT2CYR_1 = {"a": "а", "b": "б", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х", "i": "и", "k": "к", "l": "л",
              "m": "м", "n": "н", "o": "о", "p": "п", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "y": "и",
              "z": "з"}


def lat2cyr_rough(s: str) -> str:
    """Rough phonetic rendering of an English word in Cyrillic (noise model
    fallback when the lexicon has no spoken form)."""
    w = s.lower()
    out, i = [], 0
    while i < len(w):
        for a, b in _LAT2CYR_MULTI:
            if w.startswith(a, i):
                out.append(b)
                i += len(a)
                break
        else:
            out.append(_LAT2CYR_1.get(w[i], w[i]))
            i += 1
    r = "".join(out)
    if r.endswith("е") and len(r) > 3 and w.endswith("e"):  # silent final e: "make" -> "мейк" approx
        r = r[:-1]
    return r


# Hebrew -> Latin rough consonantal/vowel transliteration (for phonetic_key)
_HEB2LAT_MULTI = [("ג׳", "dzh"), ("ג'", "dzh"), ("צ׳", "ch"), ("צ'", "ch"), ("ז׳", "j"), ("ז'", "j"),
                  ("וו", "v"), ("יי", "i")]
_HEB2LAT = {"א": "a", "ב": "b", "ג": "g", "ד": "d", "ה": "h", "ו": "o", "ז": "z", "ח": "h", "ט": "t",
            "י": "i", "כ": "k", "ך": "k", "ל": "l", "מ": "m", "ם": "m", "נ": "n", "ן": "n", "ס": "s",
            "ע": "a", "פ": "p", "ף": "p", "צ": "ts", "ץ": "ts", "ק": "k", "ר": "r", "ש": "s", "ת": "t",
            "׳": "", "״": ""}


def heb2lat(s: str) -> str:
    w = strip_niqqud(nfc(s))
    out, i = [], 0
    while i < len(w):
        for a, b in _HEB2LAT_MULTI:
            if w.startswith(a, i):
                out.append(b)
                i += len(a)
                break
        else:
            out.append(_HEB2LAT.get(w[i], w[i]))
            i += 1
    return "".join(out)


# English letters/digraphs -> how a Hebrew ASR/speaker writes what they hear (fallback when no lexicon entry)
_LAT2HEB_MULTI = [("tion", "שן"), ("sion", "ז'ן"), ("ough", "אף"), ("ight", "ייט"), ("ee", "י"), ("oo", "ו"),
                  ("ea", "י"), ("ou", "או"), ("ow", "או"), ("ai", "יי"), ("ay", "יי"), ("ey", "י"), ("oa", "ו"),
                  ("sh", "ש"), ("ch", "צ'"), ("th", "ת"), ("ph", "פ"), ("ck", "ק"), ("qu", "קו"), ("kn", "נ"),
                  ("wh", "וו"), ("gh", "ג"), ("ng", "נג"), ("ts", "צ"), ("ya", "יא"), ("yu", "יו"), ("yo", "יו"),
                  ("zh", "ז'"), ("x", "קס"), ("j", "ג'"), ("w", "וו"), ("c", "ק"), ("q", "ק")]
_LAT2HEB_1 = {"a": "א", "b": "ב", "d": "ד", "e": "א", "f": "פ", "g": "ג", "h": "ה", "i": "י", "k": "ק", "l": "ל",
              "m": "מ", "n": "נ", "o": "ו", "p": "פ", "r": "ר", "s": "ס", "t": "ט", "u": "ו", "v": "ו", "y": "י",
              "z": "ז"}
_HEB_SOFIT = {"כ": "ך", "מ": "ם", "נ": "ן", "פ": "ף", "צ": "ץ"}


def lat2heb_rough(s: str) -> str:
    """Rough phonetic rendering of an English word in Hebrew (noise model fallback)."""
    w = s.lower()
    out, i = [], 0
    while i < len(w):
        for a, b in _LAT2HEB_MULTI:
            if w.startswith(a, i):
                out.append(b)
                i += len(a)
                break
        else:
            ch = w[i]
            # drop medial short 'e' except at start; keep 'a' at start or after consonant sparingly
            if ch == "e" and i > 0 and i == len(w) - 1 and len(w) > 3:
                i += 1
                continue
            if ch == "e" and 0 < i < len(w) - 1:
                i += 1
                continue
            out.append(_LAT2HEB_1.get(ch, ch))
            i += 1
    r = "".join(out)
    if r and r[-1] in _HEB_SOFIT:
        r = r[:-1] + _HEB_SOFIT[r[-1]]
    return r


def phonetic_key(s: str) -> str:
    """Script-independent key: 'клик хаус', 'קליקהאוס' and 'ClickHouse' end up close.
    lower -> cyr->lat / heb->lat -> simplifications -> strip separators/vowel runs."""
    k = strip_diacritics(strip_niqqud(nfc(s))).lower()
    k = cyr2lat(k)
    k = heb2lat(k)
    for a, b in (("ph", "f"), ("ck", "k"), ("w", "v"), ("th", "t"), ("x", "ks"), ("c", "k"), ("q", "k"),
                 ("ou", "au"), ("ee", "i"), ("oo", "u"), ("y", "i"), ("j", "dzh"), ("ts", "ts"), ("zh", "j")):
        k = k.replace(a, b)
    k = re.sub(r"[^a-z0-9]", "", k)
    k = re.sub(r"(.)\1+", r"\1", k)
    return k


# ------------------------------------------------------------------ distances
def levenshtein(a, b) -> int:
    """Edit distance over sequences (str or list)."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


@lru_cache(maxsize=65536)
def phon_sim(a: str, b: str) -> float:
    return similarity(phonetic_key(a), phonetic_key(b))


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Word alignment by Levenshtein with backtrace.
    Returns ops: ('eq'|'sub'|'del'|'ins', ref_idx, hyp_idx). del = ref word
    missing in hyp; ins = extra hyp word."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prow = d[i], d[i - 1]
        for j in range(1, m + 1):
            c = 0 if ri == hyp[j - 1] else 1
            row[j] = min(prow[j] + 1, row[j - 1] + 1, prow[j - 1] + c)
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            ops.append(("eq" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    return ops[::-1]


def wer(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return sum(op != "eq" for op, _, _ in align(ref, hyp)) / len(ref)


# ------------------------------------------------------------------ N-WER normalisation (shared with eval)
def nwer_tokens(s: str, lang: str | None = None) -> list[str]:
    """Normalised tokens for N-WER: lowercase, ё->е, no punctuation, number
    words -> digits, hyphenated words split."""
    s2 = re.sub(r"(?<=\d)[   ,](?=\d{3}\b)", "", nfc(s))
    toks = []
    for w in norm_words(s2):
        toks.extend([p for p in re.split(r"[-']", w) if p] if not w.isdigit() else [w])
    out, i = [], 0
    spans = {a: (b, v) for a, b, v in number_spans(toks, lang)}
    while i < len(toks):
        if i in spans and not toks[i].isdigit():
            b, v = spans[i]
            out.append(str(v))
            i = b
        else:
            out.append(toks[i])
            i += 1
    return out


def nwer(ref: str, hyp: str, lang: str | None = None) -> float:
    return wer(nwer_tokens(ref, lang), nwer_tokens(hyp, lang))


# ------------------------------------------------------------------ misc
PREAMBLE_RE = re.compile(
    r"^\s*(?:(?:sure|certainly|of course|here is|here's|okay,? here|the (?:edited|corrected|cleaned) text|"
    r"конечно|вот (?:исправленный|ваш|отредактированный|текст)|исправленный текст|"
    r"отредактированный текст|готово)\b|"
    r"בטח[,!:]|הנה הטקסט|טקסט מתוקן[:\s]|תוצאה:|פלט:)",
    re.I,
)


def has_preamble(s: str) -> bool:
    return bool(PREAMBLE_RE.match(s or ""))


def sentence_count(s: str) -> int:
    return max(1, len(re.findall(r"[.!?…]+(?=\s|$)", s.strip())))


def contains_phrase(text_norm_words: list[str], phrase: str) -> int:
    """Count occurrences of a normalised multi-word phrase."""
    p = phrase.split()
    k = len(p)
    return sum(1 for i in range(len(text_norm_words) - k + 1) if text_norm_words[i:i + k] == p)
