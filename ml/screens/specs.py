"""Teacher screen -> render spec. The teacher writes {title, messages, before_cursor, snippets}
(ml/data/synth/teacher_prompts.py); this module picks a template that fits the app, and the device,
theme, font scale, locale and keyboard, deterministically from the seed. Nothing of the content is dropped:
every message and snippet goes into the spec (whether it fits on the screen is the render's business,
and `visible` reports what did)."""
from __future__ import annotations

import random

# package -> template (exact app look-alike), then category -> weighted templates
BY_PACKAGE = {
    "org.telegram.messenger": "messenger_a", "org.telegram.plus": "messenger_a", "com.whatsapp": "messenger_b",
    "com.facebook.orca": "messenger_a", "com.viber.voip": "messenger_b",
    "com.slack": "work_chat", "com.Slack": "work_chat", "com.microsoft.teams": "work_chat", "com.discord": "work_chat",
    "com.atlassian.android.jira.core": "issue_tracker", "com.linear.app": "issue_tracker",
    "com.monday.monday": "issue_tracker", "com.trello": "issue_tracker", "com.asana.app": "issue_tracker",
    "com.termux": "terminal", "com.github.android": "browser_pr", "com.google.android.keep": "notes",
    "com.samsung.android.app.notes": "notes", "com.google.android.apps.docs.editors.docs": "doc_editor",
    "com.microsoft.office.word": "doc_editor", "notion.id": "doc_editor", "md.obsidian": "notes",
    "com.google.android.calendar": "calendar_tasks", "com.samsung.android.calendar": "calendar_tasks",
    "com.todoist": "calendar_tasks", "com.google.android.apps.tasks": "calendar_tasks",
    "com.waze": "browser_search", "com.google.android.apps.maps": "browser_search", "com.tranzmate": "browser_search",
    "com.yad2.yad2": "browser_search", "com.booking": "browser_search",
    "com.vkontakte.android": "social_feed", "com.instagram.android": "social_feed",
    "com.facebook.katana": "social_feed", "com.instagram.barcelona": "social_feed", "com.google.android.youtube": "social_feed",
    "com.openai.chatgpt": "ai_chat", "com.anthropic.claude": "ai_chat",
    "com.yandex.alice": "ai_chat", "ai.perplexity.app.android": "ai_chat", "com.google.android.apps.bard": "ai_chat",
}
BY_CATEGORY = {
    "messaging": [("messenger_a", 55), ("messenger_b", 45)],
    "work_chat": [("work_chat", 75), ("issue_tracker", 25)],
    "email": [("email_thread", 40), ("email_compose", 45), ("email_inbox", 15)],
    "notes": [("notes", 65), ("doc_editor", 35)],
    "search": [("browser_search", 45), ("browser_article", 35), ("browser_docs", 20)],
    "social": [("social_feed", 70), ("messenger_a", 30)],
    "code": [("code_editor", 35), ("terminal", 30), ("browser_pr", 35)],
    "default": [("ai_chat", 60), ("calendar_tasks", 20), ("browser_article", 20)],
}
FONT_SCALES = [(0.85, 12), (0.9, 12), (1.0, 40), (1.1, 16), (1.15, 10), (1.3, 10)]
HINTS = {
    "ru": {"message": "Сообщение", "email_body": "Текст письма", "email_subject": "Тема", "note": "Заметка",
           "search": "Поиск", "address": "Адрес", "code": "", "generic": ""},
    "en": {"message": "Message", "email_body": "Compose email", "email_subject": "Subject", "note": "Note",
           "search": "Search", "address": "Address", "code": "", "generic": ""},
    "he": {"message": "הודעה", "email_body": "כתיבת אימייל", "email_subject": "נושא", "note": "הערה",
           "search": "חיפוש", "address": "כתובת", "code": "", "generic": ""},
}


def _pick(r: random.Random, items):
    tot = sum(w for _, w in items)
    x = r.random() * tot
    for v, w in items:
        x -= w
        if x < 0:
            return v
    return items[-1][0]


def choose_template(app: dict, field_kind: str | None, r: random.Random) -> str:
    pkg, cat = app.get("package", ""), app.get("category", "default")
    if field_kind == "address":
        return "email_compose" if cat == "email" else r.choice(["browser_search", "browser_article"])
    if field_kind == "search" and cat in ("search", "default"):
        return "browser_search"
    if field_kind == "search" and cat == "email":
        return "email_inbox"
    if pkg in BY_PACKAGE and r.random() < 0.85:
        return BY_PACKAGE[pkg]
    return _pick(r, BY_CATEGORY.get(cat, BY_CATEGORY["default"]))


def from_teacher(screen: dict, app: dict, field_kind: str | None = "message", lang: str = "ru", seed: int = 0,
                 template: str | None = None, device: str | None = None, theme: str | None = None,
                 font_scale: float | None = None, keyboard: bool | None = None, locale: str | None = None,
                 privacy: str = "full", avoid=()) -> dict:
    """Render spec for a teacher screen. field_kind None = the app exposes no focused field."""
    r = random.Random(f"spec|{seed}")
    tpl = template or choose_template(app, field_kind, r)
    if locale:
        loc = locale
    elif lang in ("he", "he_en", "he_ru"):
        loc = "he" if r.random() < 0.8 else "en"
    else:
        loc = "ru" if lang in ("ru", "mixed") else ("ru" if r.random() < 0.35 else "en")
    field = None
    if field_kind is not None:
        field = {"kind": field_kind, "hint": HINTS.get(loc, HINTS["en"]).get(field_kind, ""), "before_cursor": screen.get("before_cursor") or ""}
    return {
        "template": tpl,
        "device": device or ("fold7_cover" if r.random() < 0.6 else "fold7_inner"),
        "theme": theme or ("light" if r.random() < 0.55 else "dark"),
        "font_scale": font_scale if font_scale is not None else _pick(r, FONT_SCALES),
        "locale": loc,
        "title": screen.get("title") or "",
        "messages": [dict(m) for m in (screen.get("messages") or [])],
        "snippets": list(screen.get("snippets") or []),
        "field": field,
        "keyboard": keyboard if keyboard is not None else (field is not None and r.random() < 0.85),
        "seed": int(seed),
        "app": dict(app),
        "lang": lang,
        "privacy": privacy,
        "avoid": list(avoid),
    }
