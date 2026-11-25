"""Template registry: name -> builder, the app each template stands for, and spec defaults."""
from __future__ import annotations

from .families import messengers as M

TEMPLATES = {
    "messenger_a": M.messenger_a,
    "messenger_b": M.messenger_b,
    "work_chat": M.work_chat,
}

# (package, label, category, field kind, family) per template: the app a template looks like by default.
TEMPLATE_APPS = {
    "messenger_a": ("org.telegram.messenger", "Telegram", "messaging", "message", "messenger"),
    "messenger_b": ("com.whatsapp", "WhatsApp", "messaging", "message", "messenger"),
    "work_chat": ("com.slack", "Slack", "work_chat", "message", "work_chat"),
}


def _register_more():
    """Families in other modules register here (keeps import order explicit)."""
    from .families import mail, docs, web, dev, misc
    for mod in (mail, docs, web, dev, misc):
        TEMPLATES.update(mod.TEMPLATES)
        TEMPLATE_APPS.update(mod.TEMPLATE_APPS)


_register_more()

DEFAULTS = {"device": "fold7_cover", "theme": "light", "font_scale": 1.0, "locale": "ru", "title": "",
            "messages": [], "snippets": [], "field": {"hint": "", "before_cursor": ""}, "keyboard": True, "seed": 0}


def normalise_spec(spec: dict) -> dict:
    s = {**DEFAULTS, **spec}
    if s["template"] not in TEMPLATES:
        raise ValueError(f"unknown template {s['template']!r}; known: {sorted(TEMPLATES)}")
    if s["device"] not in ("fold7_cover", "fold7_inner"):
        raise ValueError(f"unknown device {s['device']!r}")
    if s["theme"] not in ("light", "dark"):
        raise ValueError(f"unknown theme {s['theme']!r}")
    if s["locale"] not in ("ru", "en", "he"):
        raise ValueError(f"unknown locale {s['locale']!r}")
    fs = float(s["font_scale"])
    if not 0.7 <= fs <= 1.6:
        raise ValueError(f"font_scale {fs} out of range")
    s["messages"] = [{"author": m.get("author") or "", "is_me": bool(m.get("is_me")), "text": m.get("text") or "",
                      **({"time": m["time"]} if m.get("time") else {})} for m in (s["messages"] or [])]
    s["snippets"] = [x for x in (s["snippets"] or []) if isinstance(x, str)]
    return s
