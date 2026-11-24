"""Template context and the frame every template uses (status bar, app, keyboard, gesture bar, bubble)."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import chrome as C
from .filler import UI, Filler

CONTENT_ROLES = ("title", "message", "author", "snippet", "field")
NON_CONTENT_ROLES = ("time", "date", "chrome", "status", "keyboard", "hint", "avatar", "badge", "filler", "murmur")


@dataclass
class Ctx:
    spec: dict
    r: random.Random
    dev: dict
    ui: dict
    fill: Filler
    theme: str
    accent: str
    locale: str
    wide: bool
    css: list = field(default_factory=list)

    @property
    def messages(self) -> list[dict]:
        return [m for m in (self.spec.get("messages") or []) if (m.get("text") or "").strip()]

    @property
    def snippets(self) -> list[str]:
        return [s for s in (self.spec.get("snippets") or []) if s and s.strip()]

    @property
    def title(self) -> str:
        return (self.spec.get("title") or "").strip()

    @property
    def field(self) -> dict:
        return self.spec.get("field") or {}

    @property
    def has_field(self) -> bool:
        return self.spec.get("field") is not None

    @property
    def keyboard(self) -> bool:
        return bool(self.spec.get("keyboard", True)) and self.has_field

    def times(self, n: int) -> list[str]:
        """n increasing clock times ending near the status-bar clock."""
        h, m = self.r.randint(8, 21), self.r.randint(0, 59)
        out = []
        for _ in range(n):
            out.append(f"{h}:{m:02d}")
            m += self.r.choice([0, 1, 1, 2, 3, 5, 12])
            h, m = (h + m // 60) % 24, m % 60
        return out

    def msg_times(self) -> list[str]:
        ts = self.times(len(self.messages) + 1)
        return [m.get("time") or ts[i] for i, m in enumerate(self.messages)]


def new_ctx(spec: dict) -> Ctx:
    seed = int(spec.get("seed", 0))
    r = random.Random(f"{spec.get('template')}|{seed}")
    dev = C.DEVICES[spec.get("device", "fold7_cover")]
    locale = spec.get("locale", "ru")
    avoid = list(spec.get("avoid") or [])
    ui_key = locale if locale in UI else "en"
    return Ctx(spec=spec, r=r, dev=dev, ui=UI[ui_key], fill=Filler(r, locale, avoid),
               theme=spec.get("theme", "light"), accent=spec.get("accent") or r.choice(C.ACCENTS), locale=locale,
               wide=spec.get("device") == "fold7_inner")


def frame(c: Ctx, app_html: str, extra_css: str = "", status_style: str = "", bubble_zone=None,
          keyboard_html: str | None = None) -> str:
    """status bar + app (flex column filling the rest) + optional keyboard + gesture bar + Murmur bubble."""
    clock = c.spec.get("clock") or c.fill.clock()
    kb = ""
    if c.keyboard:
        if c.locale == "he":
            default_kb = "he" if c.r.random() < 0.75 else "en"
        else:
            default_kb = c.locale if c.r.random() < 0.75 else ("en" if c.locale == "ru" else "ru")
        layout = c.spec.get("keyboard_layout") or default_kb
        kb = keyboard_html if keyboard_html is not None else C.keyboard(layout, c.r, wide=c.wide)
    bubble = ""
    if c.spec.get("bubble", c.r.random() < 0.5):
        z = bubble_zone or (90, int(c.dev["css_h"] * (0.45 if c.keyboard else 0.8)))
        bubble = C.murmur_bubble(c.r, c.dev["css_w"], c.dev["css_h"], z[0], z[1], c.locale)
    over = overlay(c)
    body = (f'<div style="{status_style}">{C.status_bar(c.r, clock)}</div>'
            f'<div class="grow col" style="position:relative;overflow:hidden">{app_html}{over}</div>'
            f'{kb}{C.gesture_bar()}{bubble}')
    default_font = "hebrew" if c.locale == "he" else ("roboto" if c.r.random() < 0.7 else "noto")
    font = c.spec.get("font") or default_font
    css = C.css_base(c.theme, c.accent, font, float(c.spec.get("font_scale", 1.0)), extra_css + "".join(c.css))
    if c.locale == "he":
        return C.page(css, body, c.title or c.spec.get("template", "screen"), direction="rtl", lang="he")
    return C.page(css, body, c.title or c.spec.get("template", "screen"))


OVERLAYS = {
    "ru": {"sent": "Сообщение отправлено", "undo": "Отменить", "copied": "Скопировано в буфер обмена",
           "translate": "Перевести эту страницу?", "translate_btn": "Перевести", "cookies": "Мы используем файлы cookie",
           "accept": "Принять", "notif_allow": "Разрешить приложению отправлять вам уведомления?", "allow": "Разрешить",
           "deny": "Не разрешать", "update": "Доступно обновление", "now": "сейчас"},
    "en": {"sent": "Message sent", "undo": "Undo", "copied": "Copied to clipboard", "translate": "Translate this page?",
           "translate_btn": "Translate", "cookies": "We use cookies", "accept": "Accept",
           "notif_allow": "Allow this app to send you notifications?", "allow": "Allow", "deny": "Don't allow",
           "update": "Update available", "now": "now"},
    "he": {"sent": "ההודעה נשלחה", "undo": "ביטול", "copied": "הועתק ללוח", "translate": "לתרגם את הדף הזה?",
           "translate_btn": "תרגם", "cookies": "אנחנו משתמשים בעוגיות", "accept": "אישור",
           "notif_allow": "לאפשר לאפליקציה לשלוח לך התראות?", "allow": "אפשר", "deny": "אל תאפשר",
           "update": "קיים עדכון חדש", "now": "עכשיו"},
}


def overlay(c: Ctx) -> str:
    """Transient UI seen on real screenshots (emulator probe): heads-up notification, snackbar, permission
    dialog, browser banners. spec["overlay"] = none|headsup|snackbar|dialog|banner, default random (~25 %)."""
    kind = c.spec.get("overlay")
    if kind is None:
        x = c.r.random()
        kind = "headsup" if x < 0.08 else "snackbar" if x < 0.15 else "dialog" if x < 0.18 else \
            "banner" if x < 0.25 and c.spec.get("template", "").startswith("browser") else "none"
    t = OVERLAYS[c.locale if c.locale in OVERLAYS else "en"]
    if kind == "headsup":
        who = (c.fill.names(1) or ["—"])[0]
        msg = (c.fill.msgs(1) or [""])[0]
        apps = ["דואר", "הודעות", "יומן", "בנק", "Mail", "Messages"] if c.locale == "he" else \
            ["Почта", "Mail", "Messages", "Сообщения", "Calendar", "Банк"]
        app = c.r.choice(apps)
        return (f'<div style="position:absolute;top:6px;left:8px;right:8px;z-index:40;background:var(--surface2);border-radius:20px;'
                f'padding:10px 14px;box-shadow:0 4px 14px rgba(0,0,0,.3)"><div class="row t2" style="gap:6px;font-size:{C.fs(12)}">'
                f'{C.icon("chat", 14)}<span data-role="filler">{C.e(app)} · {t["now"]}</span></div>'
                f'<div style="font-weight:600;font-size:{C.fs(14)}" data-role="filler">{C.e(who)}</div>'
                f'<div class="ell" style="font-size:{C.fs(14)}" data-role="filler">{C.e(msg)}</div></div>')
    if kind == "snackbar":
        txt = c.r.choice([t["sent"], t["copied"]])
        return (f'<div class="row" style="position:absolute;bottom:12px;left:12px;right:12px;z-index:40;background:#323232;color:#f1f1f1;'
                f'border-radius:6px;padding:12px 16px;font-size:{C.fs(14)}"><span class="grow" data-role="chrome">{C.e(txt)}</span>'
                f'<span style="color:#9fc4ff;font-weight:600" data-role="chrome">{C.e(t["undo"])}</span></div>')
    if kind == "dialog":
        return (f'<div style="position:absolute;inset:0;z-index:40;background:rgba(0,0,0,.45);display:flex;align-items:center;justify-content:center">'
                f'<div style="width:82%;background:var(--bg);border-radius:24px;padding:22px"><div style="font-size:{C.fs(18)};margin-bottom:18px" '
                f'data-role="chrome">{C.e(t["notif_allow"])}</div><div style="text-align:center;padding:10px;color:var(--accent);font-weight:600" '
                f'data-role="chrome">{C.e(t["allow"])}</div><div style="text-align:center;padding:10px;color:var(--accent);font-weight:600" '
                f'data-role="chrome">{C.e(t["deny"])}</div></div></div>')
    if kind == "banner":
        a, b = c.r.choice([(t["translate"], t["translate_btn"]), (t["cookies"], t["accept"])])
        return (f'<div class="row" style="position:absolute;top:0;left:0;right:0;z-index:40;background:var(--surface);padding:10px 14px;gap:10px;'
                f'box-shadow:0 2px 6px rgba(0,0,0,.2);font-size:{C.fs(14)}"><span class="grow" data-role="chrome">{C.e(a)}</span>'
                f'<span style="color:var(--accent);font-weight:600" data-role="chrome">{C.e(b)}</span></div>')
    return ""
