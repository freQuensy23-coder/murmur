"""Post-processing of recognised lines: pure Python on plain lists, so it ports line by line to Kotlin.

A line is a dict {"text": str, "conf": float, "box": [x0, y0, x1, y1]} (integer screenshot pixels).
Order of operations (pipeline.py): exclusion -> recognition -> filter -> homoglyph repair -> geometric
de-duplication -> reading order -> textual de-duplication.
"""
from __future__ import annotations

import re
import unicodedata

from .config import CYR2LAT_STRONG, CYR2LAT_WEAK, LAT2CYR_STRONG, LAT2CYR_WEAK, NON_RU_TO_LAT, WEAK_WEIGHT, OcrConfig

# ----------------------------------------------------------------------------- geometry


def area(b) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def inter(a, b) -> int:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def excluded(box, rects, min_overlap: float) -> bool:
    a = area(box)
    if a == 0:
        return True
    return any(inter(box, r) >= min_overlap * a for r in rects)


# ----------------------------------------------------------------------------- filters


def norm_text(s: str) -> str:
    """NFC, collapse whitespace runs to one space, strip."""
    return " ".join(unicodedata.normalize("NFC", s).split())


def keep(line: dict, cfg: OcrConfig) -> bool:
    if line["conf"] < cfg.min_conf:
        return False
    return sum(1 for c in line["text"] if not c.isspace()) >= cfg.min_chars


# ----------------------------------------------------------------------------- homoglyph repair


def _is_cyr(c: str) -> bool:
    return "Ѐ" <= c <= "ӿ"


def _is_lat(c: str) -> bool:
    return ("a" <= c <= "z") or ("A" <= c <= "Z") or ("À" <= c <= "ɏ" and c.isalpha())


def _is_heb(c: str) -> bool:
    return "\u0590" <= c <= "\u05ff"


def _is_heb_letter(c: str) -> bool:
    return "\u0590" <= c <= "\u05ff" and c.isalpha()


def script_score(tok: str) -> tuple[float, float]:
    """(cyrillic, latin) evidence: 1 per letter that has no twin, WEAK_WEIGHT per weak twin, 0 per strong twin."""
    cy = la = 0.0
    for c in tok:
        if c in NON_RU_TO_LAT:
            la += 1.0
        elif _is_cyr(c):
            cy += 0.0 if c in CYR2LAT_STRONG else WEAK_WEIGHT if c in CYR2LAT_WEAK else 1.0
        elif _is_lat(c):
            la += 0.0 if c in LAT2CYR_STRONG else WEAK_WEIGHT if c in LAT2CYR_WEAK else 1.0
    return cy, la


def _to_cyr(tok: str) -> str:
    return "".join(LAT2CYR_STRONG.get(c) or LAT2CYR_WEAK.get(c) or c for c in tok)


def _to_lat(tok: str) -> str:
    return "".join(CYR2LAT_STRONG.get(c) or CYR2LAT_WEAK.get(c) or NON_RU_TO_LAT.get(c) or c for c in tok)


def _has_unambiguous(tok: str) -> bool:
    """True when the token has a letter with no Latin/Cyrillic twin (strong or weak)."""
    return any(c in NON_RU_TO_LAT or (_is_cyr(c) and c not in CYR2LAT_STRONG and c not in CYR2LAT_WEAK) or
               (_is_lat(c) and c not in LAT2CYR_STRONG and c not in LAT2CYR_WEAK) for c in tok)


def _has_both(tok: str) -> bool:
    return any(_is_cyr(c) for c in tok) and any(_is_lat(c) for c in tok)


def screen_script(texts: list[str], ratio: float) -> str | None:
    """'cyr' / 'lat' when one script's evidence over all lines is >= ratio times the other's, else None."""
    cy = la = 0.0
    for t in texts:
        a, b = script_score(t)
        cy, la = cy + a, la + b
    if cy >= ratio * max(la, 1e-9) and cy > 0:
        return "cyr"
    if la >= ratio * max(cy, 1e-9) and la > 0:
        return "lat"
    return None


_CODE = re.compile(r"[0-9#+@$=<>\\]")
_CAMEL = re.compile(r"[a-zа-яё][A-ZА-ЯЁ]|[A-ZА-ЯЁ]{2}[a-zа-яё]")


def _twin_only(t: str, i: int, toks: list[str], scores: list, default: str | None, seps: list[str] = ()) -> str:
    """Decide a token made only of twin letters (e.g. "Bce", "по", "НАРгоху", "А100", "promрt")."""
    letters = [c for c in t if _is_cyr(c) or _is_lat(c)]
    if not letters:
        return t
    if _CODE.search(t):                              # A100, x86, p99, C#, BM25: identifiers are Latin
        return _to_lat(t)
    ncy = sum(1 for c in letters if _is_cyr(c))
    if 0 < ncy < len(letters):                       # both scripts: the recogniser's majority reading
        if 2 * ncy > len(letters):
            return _to_cyr(t)
        if 2 * ncy < len(letters):
            return _to_lat(t)
    if _CAMEL.search("".join(letters)):              # НАРгоху, НеМо: Russian words have no inner capitals
        return _to_lat(t)
    # glued to a decisive token by "-", "/", "_" or "." (RNN-T, llama.cpp, SIM-карта): the same word, its script
    glued = set()
    for j, sep_i in ((i - 1, i - 1), (i + 1, i)):
        if 0 <= j < len(toks) and 0 <= sep_i < len(seps) and seps[sep_i] != " " and scores[j][0] != scores[j][1]:
            glued.add("cyr" if scores[j][0] > scores[j][1] else "lat")
    if len(glued) == 1:
        return _to_cyr(t) if glued == {"cyr"} else _to_lat(t)
    votes = set()
    for j in range(max(0, i - 2), min(len(toks), i + 3)):
        if j != i and scores[j][0] != scores[j][1]:
            votes.add("cyr" if scores[j][0] > scores[j][1] else "lat")
    if not votes and default and any(c.islower() for c in letters):
        votes = {default}                            # no decisive neighbour: the screen's dominant script
    if votes == {"cyr"}:
        return _to_cyr(t)
    return t   # never flip to Latin on context alone: "по", "с", "и" between English terms are Russian


def repair_homoglyphs(text: str, default: str | None = None) -> str:
    r"""Make every token single-script where the evidence says so. Tokens are separated by space, "-", "/",
    "_", "." (kept). Rules, in order (README.md section 6.3):
    1. a token with an unambiguous letter (no twin; non-Russian Cyrillic counts as Latin) and both scripts takes
       the script with more evidence (script_score; ties unchanged); single-script tokens are kept;
    2. a twin-only token (_twin_only): identifiers with digits or #+@$=<>\ -> Latin; both scripts -> majority of
       its letters; camel case -> Latin; glued by -/_. to decisive tokens of one script -> that script; else the
       unanimous script of the 2 tokens on each side, or the screen's
       dominant script for tokens with a lowercase letter, applied only towards Cyrillic.
    """
    parts = re.split(r"([ \-/_.])", text)          # tokens and the separators between them
    toks = parts[0::2]
    seps = parts[1::2]
    scores = [script_score(t) for t in toks]
    out = []
    for i, t in enumerate(toks):
        cy, la = scores[i]
        if not _has_unambiguous(t):
            out.append(_twin_only(t, i, toks, scores, default, seps))
        elif (not _has_both(t) and not any(c in NON_RU_TO_LAT for c in t)) or cy == la:
            out.append(t)
        elif cy > la:
            out.append(_to_cyr(t))
        else:
            out.append(_to_lat(t))
    return "".join(t + (seps[i] if i < len(seps) else "") for i, t in enumerate(out))


# ----------------------------------------------------------------------------- BiDi visual-to-logical

_MIRROR = {
    "(": ")", ")": "(",
    "[": "]", "]": "[",
    "{": "}", "}": "{",
    "<": ">", ">": "<",
    "«": "»", "»": "«",
}
_NUM_ET = frozenset("%₪$€£+")


def _reverse_rtl_run(seg: str) -> str:
    """Reverse an RTL run by base+combining clusters and mirror paired brackets."""
    clusters: list[str] = []
    for ch in seg:
        if clusters and unicodedata.category(ch) == "Mn":
            clusters[-1] += ch
        else:
            clusters.append(_MIRROR.get(ch, ch))
    return "".join(reversed(clusters))


def visual_to_logical(text: str) -> str:
    """Convert visual left-to-right CTC output to logical Unicode order when a line contains Hebrew (U+0590..U+05FF).
    100% no-op on lines without Hebrew characters. Preserves LTR Latin/Cyrillic/digit runs while reversing RTL Hebrew
    runs and RTL run ordering when the line is RTL-based (rightmost strong char is RTL)."""
    if not any(_is_heb(c) for c in text):
        return text
    n = len(text)
    cls = [
        "R" if _is_heb(c)
        else "L" if (_is_lat(c) or _is_cyr(c) or c.isalpha())
        else "D" if "0" <= c <= "9"
        else "N"
        for c in text
    ]
    base = "R"
    for cl in cls:
        if cl in ("R", "L"):
            base = cl
    dirs = ["L" if cl in ("L", "D") else cl for cl in cls]
    for i in range(1, n):
        if text[i] in _NUM_ET and (cls[i - 1] == "D" or (text[i - 1] in _NUM_ET and dirs[i - 1] == "L")):
            dirs[i] = "L"
    for i in range(n - 2, -1, -1):
        if text[i] in _NUM_ET and (cls[i + 1] == "D" or (text[i + 1] in _NUM_ET and dirs[i + 1] == "L")):
            dirs[i] = "L"
    i = 0
    while i < n:
        if dirs[i] != "N":
            i += 1
            continue
        j = i
        while j < n and dirs[j] == "N":
            j += 1
        left = dirs[i - 1] if i > 0 else base
        right = dirs[j] if j < n else base
        fill = left if left == right else base
        for k in range(i, j):
            dirs[k] = fill
        i = j
    runs: list[str] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and dirs[j] == dirs[i]:
            j += 1
        seg = text[i:j]
        runs.append(_reverse_rtl_run(seg) if dirs[i] == "R" else seg)
        i = j
    if base == "R":
        runs.reverse()
    return "".join(runs)


# ----------------------------------------------------------------------------- de-duplication


def iou(a, b) -> float:
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u else 0.0


def dedup_geometric(lines: list[dict], thr: float) -> list[dict]:
    """Drop a line that overlaps an already kept line with IoU >= thr, or lies >= 80 % inside it.
    Lines are visited by descending conf, then by (y0, x0), so the result does not depend on input order."""
    order = sorted(range(len(lines)), key=lambda i: (-lines[i]["conf"], lines[i]["box"][1], lines[i]["box"][0]))
    kept: list[int] = []
    for i in order:
        b = lines[i]["box"]
        if any(iou(b, lines[k]["box"]) >= thr or inter(b, lines[k]["box"]) >= 0.8 * max(1, area(b))
               for k in kept):
            continue
        kept.append(i)
    return [lines[i] for i in sorted(kept)]


def fold(s: str) -> str:
    return " ".join(s.casefold().split())


def dedup_text(lines: list[dict]) -> list[dict]:
    """Keep the first occurrence (in reading order) of each case/space-folded text."""
    seen, out = set(), []
    for ln in lines:
        k = fold(ln["text"])
        if k in seen:
            continue
        seen.add(k)
        out.append(ln)
    return out


# ----------------------------------------------------------------------------- reading order


def _is_rtl_row(row: list[dict]) -> bool:
    heb = ltr = 0
    rtl_starts = ltr_starts = 0
    for ln in row:
        first_strong = ""
        for c in ln["text"]:
            if _is_heb_letter(c):
                heb += 1
                if not first_strong:
                    first_strong = "R"
            elif _is_lat(c) or _is_cyr(c):
                ltr += 1
                if not first_strong:
                    first_strong = "L"
        if first_strong == "R":
            rtl_starts += 1
        elif first_strong == "L":
            ltr_starts += 1
    return heb > 0 and (heb > ltr or rtl_starts > ltr_starts)


def _rows(lines: list[dict], overlap: float) -> list[dict]:
    """Group into rows (vertical overlap >= overlap * min height with the row's first box), rows top to
    bottom by their first box's y0, boxes inside a row left to right (right to left for RTL-dominant rows)."""
    order = sorted(lines, key=lambda ln: (ln["box"][1], ln["box"][0]))
    rows: list[list[dict]] = []
    for ln in order:
        b = ln["box"]
        for row in rows:
            r = row[0]["box"]
            ov = min(b[3], r[3]) - max(b[1], r[1])
            if ov >= overlap * min(b[3] - b[1], r[3] - r[1]) and not any(inter(b, o["box"]) for o in row):
                row.append(ln)
                break
        else:
            rows.append([ln])
    out = []
    for row in rows:
        if _is_rtl_row(row):
            out.extend(sorted(row, key=lambda ln: (-ln["box"][2], ln["box"][1])))
        else:
            out.extend(sorted(row, key=lambda ln: (ln["box"][0], ln["box"][1])))
    return out


def find_gutter(lines: list[dict], width: int, height: int, cfg: OcrConfig):
    """Two-pane layout: a vertical line x that no box crosses in the band where both sides have content.
    Returns (x, band_y0, band_y1) or None. The widest free gap inside pane_gutter_range is used."""
    if not lines or width / height < cfg.pane_min_aspect:
        return None
    lo, hi = int(cfg.pane_gutter_range[0] * width), int(cfg.pane_gutter_range[1] * width)
    best = None
    # candidate gutters: gaps between box edges, scanned on a 4-px grid (deterministic, cheap)
    for x in range(lo, hi + 1, 4):
        left = [ln for ln in lines if ln["box"][2] <= x]
        right = [ln for ln in lines if ln["box"][0] >= x]
        if len(left) < cfg.pane_min_lines or len(right) < cfg.pane_min_lines:
            continue
        y0 = max(min(ln["box"][1] for ln in left), min(ln["box"][1] for ln in right))
        y1 = min(max(ln["box"][3] for ln in left), max(ln["box"][3] for ln in right))
        if y1 - y0 < cfg.pane_min_span * height:
            continue
        crossing = [ln for ln in lines if ln["box"][0] < x < ln["box"][2] and ln["box"][3] > y0 and ln["box"][1] < y1]
        if crossing:
            continue
        lspan = [ln for ln in left if ln["box"][1] >= y0 and ln["box"][3] <= y1]
        rspan = [ln for ln in right if ln["box"][1] >= y0 and ln["box"][3] <= y1]
        if len(lspan) < cfg.pane_min_lines or len(rspan) < cfg.pane_min_lines:
            continue
        # gap width around x
        gl = max((ln["box"][2] for ln in left), default=0)
        gr = min((ln["box"][0] for ln in right), default=width)
        gap = gr - gl
        if best is None or gap > best[0]:
            best = (gap, (gl + gr) // 2, y0, y1)
    return None if best is None else best[1:]


def reading_order(lines: list[dict], width: int, height: int, cfg: OcrConfig) -> list[dict]:
    g = find_gutter(lines, width, height, cfg)
    if g is None:
        return _rows(lines, cfg.row_overlap)
    x, y0, y1 = g
    above = [ln for ln in lines if ln["box"][3] <= y0 or (ln["box"][1] < y0 and ln["box"][0] < x < ln["box"][2])]
    below = [ln for ln in lines if ln not in above and ln["box"][1] >= y1]
    rest = [ln for ln in lines if ln not in above and ln not in below]
    left = [ln for ln in rest if (ln["box"][0] + ln["box"][2]) / 2 < x]
    right = [ln for ln in rest if (ln["box"][0] + ln["box"][2]) / 2 >= x]
    return (_rows(above, cfg.row_overlap) + _rows(left, cfg.row_overlap) + _rows(right, cfg.row_overlap)
            + _rows(below, cfg.row_overlap))
