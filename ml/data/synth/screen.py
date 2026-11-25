"""Context v2 for synthetic examples (docs/ml/CONTEXT_V2.md): the teacher writes screen CONTENT, code turns it
into a render spec, and the example's ctx is whatever `ml.screens.build_ctx(spec)` (render -> real OCR ->
ctx v2) returns. Nothing here edits OCR lines.

  render_params(key, app_category, lang)   deterministic template/device/theme/font_scale/keyboard/locale/seed
  make_spec(sc, content, variant)           teacher content + scenario -> render spec (injection placed here)
  strip_spec(spec, term)                    the counterpart's spec: every item that shows `term` removed
  Backend                                   real (ml.screens, once ml/screens/READY + ml/ocr/READY exist) or
                                            fixture (layout-only stand-in, no OCR; NOT valid training data)
  finalize(ctx, sc)                         app/field/language/privacy from accessibility (exact), screen from OCR
  on_screen / close_on_screen / fuzzy_on_screen   category rules checked on ctx.screen.lines
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path

from ml.common import text as T
from ml.common.render_bridge import REPO
from ml.common.util import rng as mkrng

SCREENS_READY = REPO / "ml" / "screens" / "READY"
OCR_READY = REPO / "ml" / "ocr" / "READY"
OCR_HARD = REPO / "ml" / "ocr" / "ocr_hard_terms.json"

DEVICES = {"fold7_cover": (1080, 2520), "fold7_inner": (1968, 2184)}
DEVICE_W = [("fold7_cover", 60), ("fold7_inner", 40)]
THEME_W = [("light", 60), ("dark", 40)]
FONT_W = [(1.0, 55), (0.9, 15), (1.15, 20), (1.3, 10)]
TEMPLATE_BY_APP = {"messaging": "chat", "work_chat": "chat", "social": "chat", "email": "email", "notes": "notes",
                   "search": "browser", "code": "terminal", "default": "generic", "address": "browser"}
HINTS = {
    "ru": {"message": "Сообщение", "email_body": "Текст письма", "email_subject": "Тема", "search": "Поиск",
           "address": "Введите адрес", "note": "Заметка", "code": "", "generic": "Введите текст"},
    "en": {"message": "Message", "email_body": "Compose email", "email_subject": "Subject", "search": "Search",
           "address": "Enter address", "note": "Note", "code": "", "generic": "Type here"},
    "he": {"message": "הודעה", "email_body": "כתיבת אימייל", "email_subject": "נושא", "search": "חיפוש",
           "address": "הזן כתובת", "note": "הערה", "code": "", "generic": "הקלד כאן"},
}
# screen absent (CONTEXT_V2: secure window, denylisted app, screenshot error, OCR found nothing). A password field
# is never generated: the app does not run the LLM there (ADR-070, prompts/render.py NO_SCREEN_LEVEL -> "none"),
# so such a ctx is never rendered and belongs in no training or eval example.
ABSENT_REASONS = [("secure_window", 30), ("denylisted_app", 25), ("screenshot_error", 15), ("ocr_empty", 20)]
NEVER_GENERATED_REASONS = {"password_field"}


def _pick(r, weighted):
    items, w = zip(*weighted)
    return r.choices(items, weights=w, k=1)[0]


def render_params(key: str, app_category: str, lang: str, field_present: bool) -> dict:
    r = mkrng("render.v1", key)
    if lang in ("he", "he_en", "he_ru"):
        locale = "he-IL" if r.random() < 0.80 else "en-US"
    else:
        locale = "en-US" if lang == "en" else "ru-RU"
        if r.random() < 0.15:  # Russian speaker with an English UI and vice versa
            locale = "ru-RU" if locale == "en-US" else "en-US"
    return {"template": TEMPLATE_BY_APP.get(app_category, "generic"), "device": _pick(r, DEVICE_W),
            "theme": _pick(r, THEME_W), "font_scale": _pick(r, FONT_W), "locale": locale,
            "keyboard": field_present and r.random() < 0.7, "seed": r.randrange(1 << 30)}


# Real-backend template per app (ml/screens templates), so that the screen fits the app the scenario names:
# a chat app gets a chat, a browser gets a page, never Telegram on a web page or Mattermost as an issue tracker.
PACKAGE_TEMPLATES = {
    "org.telegram.messenger": [("messenger_a", 1)], "org.telegram.plus": [("messenger_a", 1)],
    "com.whatsapp": [("messenger_b", 1)], "org.thoughtcrime.securesms": [("messenger_b", 1)],
    "com.samsung.android.messaging": [("messenger_b", 1)], "com.google.android.apps.messaging": [("messenger_b", 1)],
    "com.vkontakte.android": [("messenger_a", 1)], "com.discord": [("work_chat", 1)],
    "com.Slack": [("work_chat", 1)], "com.microsoft.teams": [("work_chat", 1)], "com.mattermost.rn": [("work_chat", 1)],
    "com.atlassian.android.jira.core": [("issue_tracker", 1)], "com.linear.app": [("issue_tracker", 1)],
    "com.google.android.keep": [("notes", 1)], "com.samsung.android.app.notes": [("notes", 1)],
    "md.obsidian": [("notes", 1)], "notion.id": [("doc_editor", 1)],
    "com.google.android.apps.docs.editors.docs": [("doc_editor", 1)],
    "com.termux": [("terminal", 1)], "com.github.android": [("browser_pr", 1)],
    "com.google.android.calendar": [("calendar_tasks", 1)], "com.samsung.android.calendar": [("calendar_tasks", 1)],
    "com.openai.chatgpt": [("ai_chat", 1)], "com.yandex.alice": [("ai_chat", 1)], "com.anthropic.claude": [("ai_chat", 1)],
    "ai.perplexity.app.android": [("ai_chat", 1)],
}
CATEGORY_TEMPLATES = {
    "email": [("email_thread", 45), ("email_compose", 55)], "social": [("social_feed", 1)],
    "search": [("browser_article", 55), ("browser_docs", 45)], "messaging": [("messenger_a", 55), ("messenger_b", 45)],
    "work_chat": [("work_chat", 1)], "notes": [("notes", 65), ("doc_editor", 35)], "code": [("terminal", 1)],
    "default": [("ai_chat", 1)],
}


def template_for(app: dict, field_kind: str | None, key: str) -> str:
    """ml/screens template for this app + field (deterministic from key)."""
    pkg, cat = app.get("package", ""), app.get("category", "default")
    if cat == "email":
        if field_kind == "search":
            return "email_inbox"
        if field_kind in ("email_subject", "address"):
            return "email_compose"
    if cat == "search" and field_kind in ("search", "address"):
        return "browser_search"
    return _pick(mkrng("template.v1", key), PACKAGE_TEMPLATES.get(pkg) or CATEGORY_TEMPLATES.get(cat)
                 or CATEGORY_TEMPLATES["default"])


def variant_params(p: dict, variant: int) -> dict:
    """variant 0 = as planned; 1 = the one allowed re-render (other theme and font scale)."""
    if variant == 0:
        return dict(p)
    q = dict(p)
    q["theme"] = "dark" if p["theme"] == "light" else "light"
    fonts = [f for f, _ in FONT_W if f != p["font_scale"]]
    q["font_scale"] = fonts[(p["seed"] + variant) % len(fonts)]
    q["seed"] = (p["seed"] * 31 + variant) % (1 << 30)
    return q


def _times(r, n: int) -> list[str]:
    h, m = r.randint(7, 22), r.randint(0, 59)
    out = []
    for _ in range(n):
        out.append(f"{h:02d}:{m:02d}")
        m += r.randint(0, 14)
        h, m = (h + m // 60) % 24, m % 60
    return out


@lru_cache(maxsize=1)
def _specs_mod():
    """ml.screens.specs (template chooser of the screens owner) once ml/screens/READY exists, else None."""
    if not SCREENS_READY.exists():
        return None
    try:
        return importlib.import_module("ml.screens.specs")
    except Exception:  # noqa: BLE001 - e.g. running outside the OCR venv: fixture specs
        return None


def avoid_list(sc: dict) -> list[str]:
    """Strings the renderer's filler content must never add (CONTEXT_V2 must_not_show items)."""
    out = []
    if sc["category"] == "near_miss":
        out += [t["text"] for t in sc["spoken_terms"]] + [p.split()[-1] for p in sc["people"]]
    if sc["category"] == "context_resolution" and sc.get("counterpart"):
        out += [t["text"] for t in sc["spoken_terms"]]
    return out


CHAT_TEMPLATES = {"messenger_a", "messenger_b", "work_chat"}


def effective_injection(sc: dict) -> dict | None:
    """Where the hostile text is placed. Chat screens cut the title to one short line and show a snippet only
    sometimes (pinned bar / link preview, ml/screens/families/messengers.py): there it arrives as a message."""
    inj = sc.get("injection")
    if inj and sc["render"].get("screen_template") in CHAT_TEMPLATES and inj["where"] != "message":
        return inj | {"where": "message", "planned_where": inj["where"]}
    return inj


def make_spec(sc: dict, content: dict, variant: int = 0) -> dict:
    """Teacher content {title, messages[{author,is_me,text}], snippets[], before_cursor} -> render spec.
    With ml/screens available the spec comes from ml.screens.specs.from_teacher (its template chooser) with the
    planner's device / theme / font scale / keyboard / locale; variant 1 changes only theme and font scale."""
    p = variant_params(sc["render"], variant)
    r = mkrng("spec", sc["sid"])
    msgs = [{"author": (m.get("author") or "").strip(), "is_me": bool(m.get("is_me")), "text": m["text"].strip()}
            for m in (content.get("messages") or []) if (m.get("text") or "").strip()]
    snippets = [s.strip() for s in (content.get("snippets") or []) if s and s.strip()]
    title = (content.get("title") or "").strip()
    inj = effective_injection(sc)
    if inj:
        where = inj["where"]
        if where == "message":
            author = title if _person_like(title) else r.choice(["Unknown", "Support", "Bot", "Неизвестный", "Админ"])
            msgs = msgs[-5:] + [{"author": author, "is_me": False, "text": inj["text"]}]
        elif where == "snippet":
            snippets = [inj["text"]] + snippets
        else:
            title = inj["text"]
    for m, t in zip(msgs, _times(r, len(msgs))):
        m["time"] = t
    hint_lang = "he" if p["locale"].startswith("he") else ("en" if p["locale"].startswith("en") else "ru")
    SM = _specs_mod()
    if SM is not None:
        p0 = sc["render"]
        spec = SM.from_teacher({"title": title, "messages": msgs, "snippets": snippets,
                                "before_cursor": (content.get("before_cursor") or "") if sc["category"] == "continuation" else ""},
                               dict(sc["app"]), sc["field_kind"] if sc.get("field_present", True) else None, sc["lang"],
                               seed=p0["seed"], template=p0.get("screen_template"),
                               device=p0["device"], theme=p["theme"], font_scale=p["font_scale"],
                               keyboard=bool(p0["keyboard"] and sc.get("field_present", True)), locale=hint_lang,
                               avoid=avoid_list(sc))
        return spec
    fld = None
    if sc.get("field_present", True):
        fld = {"kind": sc["field_kind"], "hint": HINTS[hint_lang].get(sc["field_kind"], ""),
               "before_cursor": (content.get("before_cursor") or "") if sc["category"] == "continuation" else ""}
    return {"template": p["template"], "device": p["device"], "theme": p["theme"], "font_scale": p["font_scale"],
            "locale": p["locale"], "title": title, "messages": msgs, "snippets": snippets, "field": fld,
            "keyboard": bool(p["keyboard"] and fld is not None), "seed": p["seed"],
            # extra keys (accessibility side, exact): build_ctx may use them; finalize() enforces them anyway
            "app": dict(sc["app"])}


def strip_spec(spec: dict, term: str) -> dict:
    """Counterpart spec: remove the title / messages / snippets that show `term` (or a close variant)."""
    s = copy.deepcopy(spec)
    if s.get("title") and close_in_text(s["title"], term):
        s["title"] = ""
    s["messages"] = [m for m in s.get("messages") or [] if not close_in_text(m["text"], term)
                     and not close_in_text(m.get("author", ""), term)]
    s["snippets"] = [x for x in s.get("snippets") or [] if not close_in_text(x, term)]
    if s.get("field") and close_in_text(s["field"].get("before_cursor", ""), term):
        raise ValueError("term_in_before_cursor")
    s["avoid"] = sorted(set(s.get("avoid") or []) | {term})  # filler must not bring the term back
    return s


def _person_like(s: str) -> bool:
    ws = s.split()
    return 2 <= len(ws) <= 3 and all(w[:1].isupper() for w in ws) and not any(c.isdigit() for c in s)


# ------------------------------------------------------------------ backends
def _src_hash(*dirs: Path) -> str:
    h = hashlib.sha1()
    for d in dirs:
        for f in sorted(d.rglob("*.py")) if d.exists() else []:
            if "tests" in f.parts or "__pycache__" in f.parts:
                continue
            h.update(f.relative_to(REPO).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


class Backend:
    """kind 'real': ml.screens.build_ctx (render -> ml.ocr pipeline). kind 'fixture': ml.data.synth.screen_fixture."""

    def __init__(self, kind: str = "auto"):
        ready = SCREENS_READY.exists() and OCR_READY.exists()
        if kind == "auto":
            kind = "real" if ready else "fixture"
        if kind == "real" and not ready:
            raise RuntimeError("ml/screens/READY and ml/ocr/READY are required for --screens real")
        self.kind = kind
        self.timing = {"render_s": 0.0, "ocr_s": 0.0, "build_ctx_s": 0.0, "calls": 0}
        self.last_meta: dict = {}
        if kind == "real":
            try:
                mod = importlib.import_module("ml.screens.build_ctx")
            except ImportError as e:
                raise RuntimeError(f"real screens need the OCR venv ({e}): run with .cache/ml/ocr/venv/bin/python, "
                                   "or pass --screens fixture for plumbing tests") from e
            self._detailed = getattr(mod, "build_ctx_detailed", None)
            self._build = getattr(mod, "build_ctx", None) or getattr(mod, "build", None)
            if self._build is None:
                raise RuntimeError("ml.screens.build_ctx exposes no build_ctx(spec)")
            scr = importlib.import_module("ml.screens")
            self._render = getattr(scr, "render", None)
            self.version = {"backend": "ml.screens.build_ctx",
                            "screens": getattr(scr, "VERSION", None) or "src:" + _src_hash(REPO / "ml" / "screens"),
                            "ocr": _ocr_version()}
        else:
            from ml.data.synth import screen_fixture as SF
            self._detailed = None
            self._build = SF.build_ctx
            self._render = None
            self.version = {"backend": "fixture", "screens": SF.VERSION, "ocr": "none (fixture: layout only, no OCR)"}

    def _cache_path(self, spec: dict) -> Path | None:
        """MURMUR_CTX_CACHE=DIR: build_ctx results stored per (render spec, screens + OCR version). A rebuild that
        only changes the ASR channel or the filters then skips render + OCR (the screens are byte-identical)."""
        d = os.environ.get("MURMUR_CTX_CACHE")
        if not d or self.kind != "real":
            return None
        key = hashlib.sha256(json.dumps([spec, self.version], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return Path(d) / key[:2] / f"{key}.json"

    def build_ctx(self, spec: dict) -> dict:
        t = time.time()
        cp = self._cache_path(spec)
        if cp is not None and cp.exists():
            d = json.loads(cp.read_text(encoding="utf-8"))
            self.last_meta = d["meta"]
            self.timing["cache_hits"] = self.timing.get("cache_hits", 0) + 1
            self.timing["build_ctx_s"] += time.time() - t
            self.timing["calls"] += 1
            return d["ctx"]
        ctx = self._build_ctx(spec)
        if cp is not None:
            cp.parent.mkdir(parents=True, exist_ok=True)
            tmp = cp.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"ctx": ctx, "meta": self.last_meta}, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, cp)
        return ctx

    def _build_ctx(self, spec: dict) -> dict:
        t = time.time()
        if self._detailed is not None:
            d = self._detailed(copy.deepcopy(spec))
            ms = (d.get("meta") or {}).get("ms") or {}
            self.timing["render_s"] += ms.get("render", 0) / 1000
            self.timing["ocr_s"] += ms.get("ocr", 0) / 1000
            self.last_meta = {k: v for k, v in (d.get("meta") or {}).items() if k != "ms"}
            ctx = d["ctx"]
        else:
            ctx = self._build(copy.deepcopy(spec))
            self.last_meta = {}
        self.timing["build_ctx_s"] += time.time() - t
        self.timing["calls"] += 1
        return ctx

    def time_split(self, specs: list[dict]) -> dict:
        """Measure render vs OCR on a few specs: render alone, then build_ctx (render + OCR)."""
        if self._render is None or not specs:
            return {}
        tr = tb = 0.0
        for s in specs:
            t = time.time()
            self._render(copy.deepcopy(s))
            tr += time.time() - t
            t = time.time()
            self._build(copy.deepcopy(s))
            tb += time.time() - t
        n = len(specs)
        return {"n": n, "render_s_per_screen": round(tr / n, 3), "ocr_s_per_screen": round(max(0.0, tb - tr) / n, 3),
                "build_ctx_s_per_screen": round(tb / n, 3)}


def _ocr_version() -> str:
    try:
        for mod in ("ml.ocr.config", "ml.ocr.pipeline"):
            m = importlib.import_module(mod)
            v = getattr(m, "PIPELINE_VERSION", None) or getattr(m, "VERSION", None)
            if v:
                return f"{v}+src:{_src_hash(REPO / 'ml' / 'ocr')}"
    except Exception:  # noqa: BLE001 - version is informative only
        pass
    return "src:" + _src_hash(REPO / "ml" / "ocr")


@lru_cache(maxsize=1)
def ocr_hard_terms() -> frozenset:
    """Terms the OCR cannot read (ml/ocr/ocr_hard_terms.json: a list, {"terms": [...]} or {term: info})."""
    if not OCR_HARD.exists():
        return frozenset()
    d = json.loads(OCR_HARD.read_text(encoding="utf-8"))
    if isinstance(d, dict):
        d = d.get("terms", list(d.keys()))
    return frozenset((x["term"] if isinstance(x, dict) else str(x)) for x in d)


def finalize(ctx: dict, sc: dict) -> dict:
    """ctx v2 with app/field/language/privacy exactly as accessibility reports them; screen as OCR returned it."""
    out = {"v": 2, "app": dict(sc["app"])}
    if sc.get("field_present", True):
        f = {"kind": sc["field_kind"]}
        got = (ctx or {}).get("field") or {}
        for k in ("hint", "box"):  # hint from accessibility, box = the focused node's bounds (renderer uses it)
            if got.get(k):
                f[k] = got[k]
        if sc["category"] == "continuation" and got.get("before_cursor"):
            f["before_cursor"] = got["before_cursor"]
        else:
            f["before_cursor"] = got.get("before_cursor", "") if sc["category"] == "continuation" else ""
        if sc.get("input_type_variation"):
            f["input_type_variation"] = sc["input_type_variation"]
        out["field"] = f
    scr = (ctx or {}).get("screen")
    if scr and scr.get("lines"):
        out["screen"] = {"size": list(scr.get("size") or []), "lines": list(scr["lines"])}
    out["language"] = {"asr": "he" if sc["lang"] in ("he", "he_en", "he_ru") else ("en" if sc["lang"] == "en" else "ru")}
    out["privacy"] = {"level": "full", "reason": None if "screen" in out else "ocr_empty"}
    return out


def no_screen_level(reason: str) -> str:
    """privacy.level the phone sends without a screen: the shared table prompts/render.py NO_SCREEN_LEVEL
    (ADR-070), never a copy of it. A reason the app never sends to the LLM (password field) is refused."""
    from ml.common.render_bridge import render
    if reason in NEVER_GENERATED_REASONS:
        raise ValueError(f"no-screen reason {reason!r} never reaches the LLM (ADR-070): not a valid example")
    table = getattr(render(), "NO_SCREEN_LEVEL", None) or {}
    if reason not in table:
        raise ValueError(f"no-screen reason {reason!r} is not in prompts/render.py NO_SCREEN_LEVEL")
    return table[reason]


def absent_ctx(sc: dict, before_cursor: str = "") -> dict:
    """No screenshot: secure window, denylisted app, screenshot error, OCR found nothing."""
    reason = sc["screen_absent"]
    out = {"v": 2, "app": dict(sc["app"])}
    if sc.get("field_present", True):
        out["field"] = {"kind": sc["field_kind"], "before_cursor": before_cursor if sc["category"] == "continuation" else ""}
        if sc.get("input_type_variation"):
            out["field"]["input_type_variation"] = sc["input_type_variation"]
    out["language"] = {"asr": "he" if sc["lang"] in ("he", "he_en", "he_ru") else ("en" if sc["lang"] == "en" else "ru")}
    out["privacy"] = {"level": no_screen_level(reason), "reason": reason}
    return out


# ------------------------------------------------------------------ checks on OCR output
def lines_of(ctx: dict) -> list[str]:
    return [ln.get("text", "") for ln in ((ctx.get("screen") or {}).get("lines") or [])]


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", T.nfc(s)).strip()


def exact_in_text(text: str, s: str) -> bool:
    """Spelled exactly (case and ё included) with word boundaries."""
    return bool(s) and re.search(r"(?<!\w)" + re.escape(_ws(s)) + r"(?!\w)", _ws(text)) is not None


def on_screen(ctx: dict, s: str) -> bool:
    """Exactly on one OCR line, or across a line wrap."""
    ls = lines_of(ctx)
    return any(exact_in_text(x, s) for x in ls) or exact_in_text(" ".join(ls), s)


def _cf(s: str) -> str:
    return T.nfc(s).casefold().replace("ё", "е")


def close_in_text(text: str, term: str) -> bool:
    """term or a close variant (case-folded, edit distance <= 2; <= 1 for terms of <= 4 letters, and a
    candidate must be at least 3 characters long) anywhere in text, also with spaces/punctuation removed."""
    t = _cf(term)
    tk = re.sub(r"[\W_]+", "", t)
    if not tk:
        return False
    if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", _cf(text)):
        return True
    maxd = 1 if len(tk) <= 4 else 2
    toks = re.findall(r"\w+", _cf(text))
    n = len(re.findall(r"\w+", t)) + 1
    for i in range(len(toks)):
        cand = ""
        for j in range(i, min(len(toks), i + n)):
            cand += toks[j]
            if len(cand) >= 3 and abs(len(cand) - len(tk)) <= maxd and T.levenshtein(cand, tk) <= maxd:
                return True
    return False


def close_on_screen(ctx: dict, term: str) -> bool:
    ls = lines_of(ctx)
    return any(close_in_text(x, term) for x in ls) or close_in_text(" ".join(ls), term)


def fuzzy_on_screen(ctx: dict, s: str, thr: float = 0.75) -> bool:
    """The (noisy) injected text is visible: best char-similarity of any window of OCR text >= thr."""
    target = _cf(_ws(s))
    hay = _cf(" ".join(lines_of(ctx)))
    if not target or not hay:
        return False
    if target in hay:
        return True
    L = len(target)
    best = 0.0
    for i in range(0, max(1, len(hay) - L + 1), max(1, L // 8)):
        best = max(best, T.similarity(hay[i:i + L], target))
        if best >= thr:
            return True
    return False


def in_rendered(block: str, s: str, exact: bool = True) -> bool:
    return exact_in_text(block, s) if exact else close_in_text(block, s)


PROBE_LINE = "Qzx probe 7731 строка"


@lru_cache(maxsize=1)
def renderer_shows_screen() -> bool:
    """Does the current prompts/render.py put ctx v2 screen lines into <context>? (False for the v1 renderer.)"""
    from ml.common.render_bridge import render
    probe = {"v": 2, "app": {"package": "org.telegram.messenger", "label": "Telegram", "category": "messaging"},
             "field": {"kind": "message", "hint": "", "before_cursor": ""},
             "screen": {"size": [1080, 2520], "lines": [{"text": PROBE_LINE, "conf": 0.99, "box": [100, 300, 700, 350]}]},
             "language": {"asr": "ru"}, "privacy": {"level": "full", "reason": None}}
    try:
        return PROBE_LINE in render().render_context(probe)
    except Exception:  # noqa: BLE001 - an old renderer may choke on v2
        return False
