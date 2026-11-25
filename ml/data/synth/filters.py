"""Deterministic quality filters, near-duplicate removal and split-leakage checks for synthetic records.

Strictness rule: an example that teaches the model to answer, paraphrase, translate, drop or invent content
is worse than no example, so any doubt rejects. `check(rec)` returns a list of reasons (empty = keep).
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

from ml.common import text as T
from ml.common.render_bridge import rendered_context
from ml.common.schema import validate_record

CMD_PHRASES = ["новый абзац", "новая строка", "с новой строки", "вопросительный знак", "восклицательный знак",
               "открыть кавычки", "закрыть кавычки", "открой кавычки", "закрой кавычки", "двоеточие",
               "new paragraph", "new line", "question mark", "exclamation mark", "exclamation point", "open quote",
               "close quote", "open quotes", "close quotes",
               "פסקה חדשה", "שורה חדשה", "סימן שאלה", "סימן קריאה", "פתח מרכאות", "סגור מרכאות"]
CMD_WORDS = {"точка", "запятая", "запятую", "двоеточие", "тире", "абзац", "строка", "строки", "кавычки", "знак",
             "вопросительный", "восклицательный", "period", "comma", "colon", "paragraph", "line", "quote", "quotes",
             "mark", "point", "dash", "semicolon", "открыть", "закрыть", "открой", "закрой", "новый", "новая", "new",
             "open", "close", "full", "stop", "скобка", "скобку", "bracket", "question", "exclamation",
             "נקודה", "פסיק", "נקודתיים", "מקף", "פסקה", "שורה", "חדשה", "סימן", "שאלה", "קריאה", "מרכאות",
             "פתח", "סגור"}
ORDINALS = {"во-первых", "во-вторых", "в-третьих", "в-четвертых", "в-пятых", "первое", "второе", "третье", "четвертое",
            "пятое", "first", "second", "third", "fourth", "fifth", "firstly", "secondly", "thirdly", "пункт", "item",
            "number", "раз", "два", "три", "четыре", "пять", "one", "two", "three", "four", "five", "во", "первых",
            "вторых", "третьих", "next", "finally", "последнее", "lastly", "и", "and", "also", "then", "потом",
            "дальше", "ещё", "еще", "plus",
            "ראשון", "ראשונה", "שני", "שנייה", "שלישי", "שלישית", "רביעי", "חמישי", "קודם", "דבר", "אחת", "שתיים",
            "שלוש", "ארבע", "חמש", "סעיף", "ובסוף", "בסוף"}
SC_MARKERS = ["нет, ", "ой, ", "точнее", "вернее", "в смысле", "то есть", "не-не", "no wait", "scratch that",
              "i mean", "sorry, ", "make that", "or rather",
              "לא רגע", "תמחק את זה", "זאת אומרת", "לא לא,"]
SYMBOL_WORDS = {"плюс", "plus", "решётка", "решетка", "hash", "собака", "at", "dot", "точка", "слэш", "слеш", "slash",
                "дефис", "hyphen", "dash", "underscore", "подчёркивание", "подчеркивание", "амперсанд", "ampersand",
                "тильда", "tilde", "sharp", "шарп", "шарпе", "шарпа", "шарпом", "си",
                "נקודה", "שטרודל", "סלאש", "מקף", "פלוס", "סולמית"}
# number-scale words: 'eight hundred' -> '800'
SCALE_WORDS = {"hundred", "hundreds", "thousand", "thousands", "million", "millions", "billion", "сто", "тысяча",
               "тысячи", "тысяч", "тысячу", "миллион", "миллиона", "миллионов", "номер", "номера", "номеру",
               "מאה", "מאתיים", "מאות", "אלף", "אלפיים", "אלפים", "מיליון", "מיליארד", "מספר"}
_SYMBOL_IN_TOKEN = re.compile(r"\w[#./~@+]|[#./~@+]\w")
SYMBOL_CATS = {"address", "formatting_commands", "numbers"}  # where a spoken symbol may become a symbol
UNIT_PREFIXES = ("процент", "percent", "доллар", "dollar", "бакс", "buck", "рубл", "евро", "euro", "cent", "цент",
                 "pound", "фунт", "тенге", "o'clock",
                 "אחוז", "שקל", "דולר", "אירו", "יורו", "אגורה", "אגורות")
MONTHS_DAYS = {w.lower() for w in "January February March April May June July August September October November "
               "December Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()}


def _parts(words) -> set[str]:
    out = set()
    for w in words:
        out.add(w)
        out.update(p for p in w.split("-") if p)
    return out


def _no_numbers(s: str) -> str:
    return " ".join(w for w in s.split() if not T.numbers_in(w.strip(".,!?;:()«»\"'")))


def _numberish(w: str) -> bool:
    return bool(T.numbers_in(w)) or w.startswith(UNIT_PREFIXES) or all(T.numbers_in(p) for p in w.split("-") if p)


ADDRESS_RE = re.compile(r"^(?:[\w.+-]+@[\w-]+(?:\.[\w-]+)+|(?:https?://)?(?:www\.)?[\w-]+(?:\.[\w-]+)+(?:/[\w./?=&%#-]*)?)$")
TAIL_MARKERS = ["субтитры", "продолжение следует", "thanks for watching", "amara.org", "подписывайтесь", "dimatorzok",
                "редактор субтитров", "спасибо за просмотр", "thank you for watching", "castingwords",
                "like and subscribe", "subscribe to my channel", "next video", "следующем видео",
                "ставьте лайки",
                "תודה שצפיתם", "הירשמו לערוץ", "כתוביות בעברית", "כתוביות אוטומטיות", "נתראה בסרטון הבא"]
_LIST_MARK = re.compile(r"(?m)^\s*\d+[.)]\s+")


def _num_multiset(s: str, lang: str) -> Counter:
    s = _LIST_MARK.sub("", s)
    return Counter(T.numbers_in(s, None if lang in ("mixed", "he_en", "he_ru") else lang))


def _matches(w: str, pool: set[str]) -> bool:
    if w in pool:
        return True
    return any(T.similarity(w, p) >= 0.75 or (len(w) >= 5 and len(p) >= 5 and w[:5] == p[:5]) for p in pool)


def check(rec: dict) -> list[str]:
    errs = validate_record(rec, check_render=True)
    if errs:
        return ["contract:" + errs[0][:60]]
    cat, lang, meta = rec["category"], rec["lang"], rec["meta"]
    tgt, tr = rec["target"], rec["transcript"]
    spoken = meta.get("spoken", "")
    out: list[str] = []
    if cat == "noise_only":
        return [] if tgt == "" else ["noise_only_not_empty"]
    if not tgt.strip():
        return ["empty_target"]
    if cat == "no_op" and tr != tgt:
        out.append("no_op_not_identical")
    if len(tgt) > 1200:
        out.append("too_long")
    if T.has_preamble(tgt) and not T.has_preamble(spoken):  # 'Here's my plan:' the user said is not a preamble
        out.append("preamble")
    low = tgt.lower()
    if any(m in low for m in TAIL_MARKERS):
        out.append("artifact_in_target")
    # language: never translate
    # script shares without numbers: 'в семь' -> 'в 7' is not a script shift
    if lang in ("he", "he_en", "he_ru"):
        hs_t, hs_s = T.heb_share(_no_numbers(tgt)), T.heb_share(_no_numbers(spoken or tr))
        c3_t = T.script_counts_3(_no_numbers(tgt))
        if lang == "he" and cat != "address" and hs_t < 0.5:
            out.append("he_not_hebrew")
        if cat not in ("address", "context_counterpart") and abs(hs_t - hs_s) > 0.30:
            out.append("script_shift")
        if lang == "he_en" and meta.get("mix") == "he_en_switch" and not (c3_t[2] >= 4 and c3_t[1] >= 4):
            out.append("switch_without_both_languages")
        if lang == "he_ru" and cat != "address" and not (c3_t[2] >= 3 and c3_t[0] >= 3):
            out.append("switch_without_both_languages")
    else:
        cs_t, cs_s = T.cyr_share(_no_numbers(tgt)), T.cyr_share(_no_numbers(spoken or tr))
        if lang == "en" and cs_t > 0:
            out.append("cyrillic_in_en")
        if lang == "ru" and cat != "address" and cs_t < 0.5:
            out.append("ru_not_cyrillic")
        if cat not in ("address", "context_counterpart") and abs(cs_t - cs_s) > 0.25:
            out.append("script_shift")
        if lang == "mixed" and meta.get("mix") == "switch" and not (T.script_counts(tgt)[0] and T.script_counts(tgt)[1] >= 6):
            out.append("switch_without_both_languages")
    # content: target words come from what was spoken (+ the on-screen spelling of the restored term)
    ents = list(meta.get("entities", [])) + (chk_forbid := list((meta.get("check") or {}).get("forbid", [])
                                                              if cat == "context_counterpart" else []))
    if cat != "address":
        allow = {T.norm_word(w) for e in ents for w in T.words(e)}
        sw = T.norm_words(spoken)
        sp_words = _parts(sw) | allow | {a + "-" + b for a, b in zip(sw, sw[1:])} | {a + b for a, b in zip(sw, sw[1:])}
        novel = [w for w in T.content_words(tgt)
                 if not _matches(w, sp_words) and not all(_matches(p, sp_words) for p in w.split("-") if p)
                 and not re.fullmatch(r"[\u05d0-\u05ea]-\d+", w)]
        if cat == "context_names":
            novel = [w for w in novel if not any(w[:4] == a[:4] for a in allow)]
        if novel:
            out.append("novel_content")
    if cat not in ("self_correction", "address"):
        tw = T.norm_words(tgt)
        tg = _parts(tw) | {a + b for a, b in zip(tw, tw[1:])} | _parts(T.norm_word(w) for e in ents for w in T.words(e))
        sym_ok = cat in SYMBOL_CATS or bool(_SYMBOL_IN_TOKEN.search(tgt))  # 'config dot yaml' -> 'config.yaml'
        dropped = [w for w in _parts(T.content_words(spoken)) if "-" not in w and not _matches(w, tg)
                   and w not in CMD_WORDS and not (w in SYMBOL_WORDS and sym_ok) and not _numberish(w)
                   and w not in SCALE_WORDS
                   and (cat in ("list", "formatting_commands") or w not in ORDINALS)]
        if cat in ("list", "formatting_commands"):  # 'в' is the first part of 'в-третьих'
            dropped = [w for w in dropped if w not in ORDINALS and w != "в"]
        if dropped:
            out.append("dropped_content")
    # numbers
    if cat != "address":
        nt, ns = _num_multiset(tgt, lang), _num_multiset(spoken, lang)
        if cat == "self_correction":
            if nt - ns:
                out.append("number_mismatch")
        elif nt != ns and (nt - ns):
            out.append("number_mismatch")
        if T.spelled_numbers_over(tgt, 10, None if lang in ("mixed", "he_en", "he_ru") else lang):
            out.append("number_words_in_target")
        from ml.data.synth.build import digits_spelled_out
        if cat not in ("no_op", "near_no_op") and digits_spelled_out(tr, tgt, lang):
            out.append("digit_spelled_out")  # the transcript has the digit, the target writes the word
    # length ratio (words)
    nt_w, ns_w = len(T.words(tgt)), max(1, len(T.words(spoken)))
    lo, hi = {"self_correction": (0.25, 1.2), "list": (0.4, 1.3), "formatting_commands": (0.4, 1.3),
              "address": (0.0, 9.0), "numbers": (0.3, 1.2)}.get(cat, (0.6, 1.25))
    if not lo <= nt_w / ns_w <= hi:
        out.append("length_ratio")
    # context discipline
    tgt_f = low.replace("ё", "е")
    for d in meta.get("distractors") or []:
        df = d.lower().replace("ё", "е")
        if re.search(r"(?<!\w)" + re.escape(df) + r"(?!\w)", tgt_f) and df not in (spoken or "").lower().replace("ё", "е"):
            out.append("distractor_in_target")
            break
    chk = meta.get("check") or {}
    if any(f.lower() in low and f.lower() not in (spoken or "").lower() for f in chk.get("forbid", [])):
        out.append("forbidden_in_target")
    if meta.get("injection") and meta["injection"]["text"][:25].lower() in low:
        out.append("injection_in_target")
    # category-specific
    if cat == "formatting_commands" and any(p in low for p in CMD_PHRASES):
        out.append("command_left")
    if cat == "list" and not re.search(r"(?m)^\s*1[.)]\s", tgt) or cat == "list" and not re.search(r"(?m)^\s*2[.)]\s", tgt):
        out.append("list_not_formatted")
    if cat == "self_correction" and any(m in low for m in SC_MARKERS):
        out.append("correction_marker_left")
    if cat == "continuation":
        bc = (rec["ctx"].get("field") or {}).get("before_cursor", "")
        if not bc or "before the cursor" not in rendered_context(rec["ctx"]):
            out.append("before_cursor_not_rendered")
        if bc and bc.strip()[:18].lower() in low:
            out.append("repeats_before_cursor")
        if bc and not re.search(r"[.!?:…]\s*$", bc) and tgt[:1].isupper() and not _starts_with_name(tgt, rec):
            out.append("continuation_capitalised")
    if cat == "dictated_question" and "?" not in tgt and rec["ctx"].get("field", {}).get("kind") != "search":
        out.append("question_without_mark")
    if cat == "address" and not ADDRESS_RE.match(tgt):
        out.append("not_an_address")
    if cat == "context_resolution":
        term = (meta.get("entities") or [""])[0]
        if term and term.lower() in tr.lower():
            out.append("term_in_transcript")
    if cat == "context_counterpart":
        if (chk.get("forbid") or [""])[0].lower() in low:
            out.append("counterpart_has_term")
    if _copies_prompt_example(spoken + " " + tgt):
        out.append("copied_prompt_example")  # the teacher reused an illustration from its instructions
    if rec["ctx"].get("v") == 2:  # CONTEXT_V2 rules on OCR lines + the current rendered block (same as recheck)
        from ml.data.recheck import rule_errors
        out += ["ctx_rule:" + e.split(":")[1] for e in rule_errors(rec)]
    return out


@lru_cache(maxsize=1)
def _prompt_examples() -> tuple[str, ...]:
    from ml.data.contam import prompt_phrases
    # rule examples about numbers and times ('к девяти утра', 'at six thirty pm') are everyday phrases, not copies
    return tuple(p.lower().replace("ё", "е") for p in prompt_phrases()
                 if len(p) >= 12 and len(p.split()) >= 3 and not T.numbers_in(p))


def _copies_prompt_example(text: str) -> bool:
    t = text.lower().replace("ё", "е")
    return any(p in t for p in _prompt_examples())


def review_flags(rec: dict) -> list[str]:
    """Target content words that occur (exactly, after normalisation) in none of: the spoken text, the transcript,
    the screen lines, the text before the cursor, the record's entities. Such a word passed the fuzzy checks above
    but may be invented ('п' -> 'Pro', an inserted "I'll"); the record is kept and listed in review.jsonl for a human."""
    if rec["category"] in ("noise_only", "address"):
        return []
    ctx = rec["ctx"]
    pool_text = " ".join([rec["meta"].get("spoken", ""), rec["transcript"],
                          " ".join(ln.get("text", "") for ln in (ctx.get("screen") or {}).get("lines", [])),
                          (ctx.get("field") or {}).get("before_cursor") or "",
                          " ".join(rec["meta"].get("entities") or [])])
    pw = T.norm_words(pool_text)
    pool = _parts(pw) | {a + b for a, b in zip(pw, pw[1:])}
    return [w for w in T.content_words(rec["target"]) if w not in pool and not all(p in pool for p in w.split("-") if p)
            and not T.numbers_in(w)]


def _starts_with_name(tgt: str, rec: dict) -> bool:
    first = (T.words(tgt) or [""])[0]
    ctx = rec["ctx"]
    names = {w for t in ctx.get("terms", []) for w in T.words(t["text"])}  # v1
    names |= {w for ln in (ctx.get("screen") or {}).get("lines", []) for w in T.words(ln.get("text", ""))
              if w[:1].isupper()}  # v2: capitalised words on screen
    names |= {w for e in rec["meta"].get("entities") or [] for w in T.words(e)}
    return (first in names or first.isupper() or first.lower() in MONTHS_DAYS or first == "I"
            or (first[:1].isupper() and T.is_lat(first) and rec["lang"] != "en"))


# ------------------------------------------------------------------ near-duplicates
def shingles(s: str, n: int = 5) -> frozenset:
    s = re.sub(r"\s+", " ", T.nfc(s).lower().replace("ё", "е")).strip()
    return frozenset(s[i:i + n] for i in range(max(1, len(s) - n + 1)))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedup(recs: list[dict], thr: float = 0.8) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """Drop exact (normalised target + transcript) and near-duplicate targets (char 5-gram Jaccard >= thr) within a
    language. Records of the same scenario group (a context case and its counterpart) never knock each other out.
    Held-out splits win: a train record that near-duplicates a dev/test record is the one dropped."""
    order = sorted(range(len(recs)), key=lambda i: (recs[i]["split"] == "train", i))
    kept_idx: list[int] = []
    dropped: list[tuple[str, str, str]] = []
    seen_exact: dict = {}
    n_same: dict = {}
    noise_split: dict = {}
    buckets: dict[tuple, list[tuple[int, frozenset]]] = {}
    for i in order:
        r = recs[i]
        tgt = r["target"]
        key = (re.sub(r"\W+", " ", tgt.lower()).strip(), re.sub(r"\W+", " ", r["transcript"].lower()).strip())
        grp = r["meta"].get("group")
        if r["category"] == "noise_only":  # a few spellings of silence repeat by nature: at most 3 of each,
            n_same[key] = n_same.get(key, 0) + 1  # all in one split (held-out splits come first and win)
            first_split = noise_split.setdefault(key, r["split"])
            if n_same[key] > 3 or first_split != r["split"]:
                dropped.append((r["id"], "exact_duplicate", seen_exact.get(key, ("",))[0]))
                continue
            seen_exact.setdefault(key, (r["id"], grp))
            kept_idx.append(i)
            continue
        if key in seen_exact and seen_exact[key][1] != grp:
            dropped.append((r["id"], "exact_duplicate", seen_exact[key][0]))
            continue
        seen_exact.setdefault(key, (r["id"], grp))
        if len(tgt) < 12:  # short texts legitimately repeat ("Ок, спасибо!"); exact check above is enough
            kept_idx.append(i)
            continue
        sh = shingles(tgt)
        bkey = (r["lang"],)
        hit = None
        for j, sj in buckets.get(bkey, []):
            if recs[j]["meta"].get("group") == grp:
                continue
            if jaccard(sh, sj) >= thr:
                hit = recs[j]["id"]
                break
        if hit:
            dropped.append((r["id"], "near_duplicate", hit))
            continue
        buckets.setdefault(bkey, []).append((i, sh))
        kept_idx.append(i)
    keep = set(kept_idx)
    return [r for i, r in enumerate(recs) if i in keep], dropped


# ------------------------------------------------------------------ split hygiene
def leakage(recs: list[dict]) -> dict:
    """Entities and scenario groups must not cross splits (dev/test vs train)."""
    by_split: dict[str, set] = {"train": set(), "dev": set(), "test": set()}
    groups: dict[str, set] = {"train": set(), "dev": set(), "test": set()}
    for r in recs:
        if r["category"] in ("context_resolution", "context_names", "near_miss", "context_counterpart"):
            for e in r["meta"].get("entities") or []:
                if r["category"] != "context_counterpart":
                    by_split[r["split"]].add(e)
        groups[r["split"]].add(r["meta"].get("group"))
    return {"entity_overlap_train_dev": sorted(by_split["train"] & by_split["dev"]),
            "entity_overlap_train_test": sorted(by_split["train"] & by_split["test"]),
            "group_overlap": sorted((groups["train"] & (groups["dev"] | groups["test"])) - {None})}
