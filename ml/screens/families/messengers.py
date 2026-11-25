"""Messenger look-alikes: messenger_a (blue accent, patterned wallpaper, chat list on the inner display) and
messenger_b (green header, beige wallpaper), plus work_chat (channel feed, no bubbles, sidebar on inner)."""
from __future__ import annotations

from .. import chrome as C
from ..base import Ctx, frame

e = C.e
ICON_TICKS = {"sent": "check", "read": "dcheck"}


def _is_group(c: Ctx) -> bool:
    authors = {m.get("author") for m in c.messages if not m.get("is_me")}
    return len(authors) > 1


def _link_preview(c: Ctx, snippet: str, me: bool) -> str:
    dom = c.r.choice(["news", "blog", "media", "docs", "shop"]) + c.r.choice([".ru", ".com", ".io", ".org"])
    return (f'<div style="border-left:3px solid var(--accent);padding:2px 0 2px 8px;margin:4px 0">'
            f'<div style="color:var(--accent);font-weight:500;font-size:{C.fs(14)}" data-role="chrome">{e(dom)}</div>'
            f'<div style="font-size:{C.fs(14)}" data-role="snippet">{e(snippet)}</div></div>')


def _bubbles(c: Ctx, palette: dict, group: bool, tails: bool) -> str:
    out = []
    times = c.msg_times()
    n_fill = c.r.choice([0, 1, 2, 3]) if c.spec.get("filler", True) else 0
    fillers = c.fill.msgs(n_fill)
    sep = c.fill.date_sep()
    out.append(f'<div style="align-self:center;margin:6px 0;padding:3px 10px;border-radius:12px;background:{palette["sep"]};'
               f'color:{palette["septext"]};font-size:{C.fs(13)}" data-role="date">{e(sep)}</div>')
    items = [{"author": c.title or "", "is_me": c.r.random() < 0.5, "text": t, "_filler": True} for t in fillers]
    items += c.messages
    times = c.times(len(fillers)) + times
    snippets = list(c.snippets)
    preview_at = len(items) - 1 - c.r.randint(0, max(0, len(c.messages) - 1)) if snippets and c.r.random() < 0.35 else -1
    for i, m in enumerate(items):
        me = bool(m.get("is_me"))
        role = "filler" if m.get("_filler") else "message"
        bg = palette["out"] if me else palette["in"]
        author = ""
        if group and not me and m.get("author"):
            col = C.AV_COLORS[sum(map(ord, m["author"])) % len(C.AV_COLORS)]
            author = f'<div style="color:{col};font-weight:500;font-size:{C.fs(14)}" data-role="author">{e(m["author"])}</div>'
        tick = C.icon(ICON_TICKS["read" if c.r.random() < 0.8 else "sent"], 16) if me else ""
        edited = f'{e(c.ui["edited"])} ' if c.r.random() < 0.06 else ""
        meta = (f'<span style="float:right;margin:8px 0 -4px 10px;font-size:{C.fs(12)};color:{palette["meta_out" if me else "meta_in"]};'
                f'display:inline-flex;align-items:center;gap:2px" data-role="time">{edited}{e(times[i])}{tick}</span>')
        prev = _link_preview(c, snippets.pop(0), me) if i == preview_at and snippets else ""
        react = ""
        if c.r.random() < 0.1:
            react = (f'<div style="margin-top:4px"><span style="background:rgba(127,127,127,.18);border-radius:12px;padding:1px 8px;'
                     f'font-size:{C.fs(14)}" data-role="chrome">{c.r.choice(["👍", "❤️", "🔥", "😂"])} {c.r.randint(1, 3)}</span></div>')
        radius = "18px 18px 4px 18px" if me and tails else "18px 18px 18px 4px" if tails else "12px"
        av = ""
        if group and not me and c.r.random() < 0.9:
            av = f'<div style="align-self:flex-end;margin-right:6px">{C.avatar(m.get("author") or "?", c.r, 34)}</div>'
        out.append(
            f'<div class="row" style="justify-content:{"flex-end" if me else "flex-start"};margin:3px 8px;align-items:flex-end">{av}'
            f'<div style="max-width:{"78%" if not c.wide else "70%"};background:{bg};color:{palette["text_out" if me else "text_in"]};'
            f'border-radius:{radius};padding:6px 10px 6px 11px;box-shadow:0 1px 1px rgba(0,0,0,.12)">'
            f'{author}<div{" dir=\"rtl\"" if any("\u05d0" <= ch <= "\u05ea" for ch in m["text"]) else ""} style="white-space:pre-wrap;word-break:break-word" data-role="{role}">{e(m["text"])}{meta}</div>'
            f'{prev}{react}</div></div>')
    return "".join(out)


def _chat_list(c: Ctx, palette: dict, current: str) -> str:
    n = c.r.randint(7, 11)
    names = c.fill.names(n)
    msgs = c.fill.msgs(n)
    rows = []
    cur_i = c.r.randint(0, 2)
    for i in range(n):
        is_cur = i == cur_i and current
        name = current if is_cur else names[i % len(names)]
        if is_cur:
            last = c.messages[-1]["text"] if c.messages else (msgs[i % len(msgs)] if msgs else "")
            role_n, role_m = "title", ("message" if c.messages else "filler")
        else:
            last, role_n, role_m = msgs[i % len(msgs)] if msgs else "", "filler", "filler"
        badge = f'<span class="badge">{c.r.randint(1, 120)}</span>' if (not is_cur and c.r.random() < 0.35) else ""
        bg = f"background:{palette['sel']};" if is_cur else ""
        rows.append(
            f'<div class="row" style="gap:12px;padding:9px 12px;{bg}">{C.avatar(name, c.r, 50)}'
            f'<div class="grow col" style="gap:2px"><div class="row"><span class="grow ell" style="font-weight:500" data-role="{role_n}">{e(name)}</span>'
            f'<span class="t3" style="font-size:{C.fs(12)}" data-role="time">{e(c.fill.time_ago(i))}</span></div>'
            f'<div class="row"><span class="grow ell t2" style="font-size:{C.fs(15)}" data-role="{role_m}">{e(last)}</span>{badge}</div></div></div>')
    return "".join(rows)


PAL_A = {
    "light": {"wall": "linear-gradient(160deg,#cfe3b7,#a9cf9a 45%,#c9dfb3)", "in": "#ffffff", "out": "#e3fcd0",
              "text_in": "#111", "text_out": "#111", "meta_in": "#8b9198", "meta_out": "#4fae4e", "sep": "rgba(0,0,0,.18)",
              "septext": "#fff", "head": "var(--bar)", "headfg": "var(--text)", "sel": "rgba(42,139,214,.12)"},
    "dark": {"wall": "linear-gradient(160deg,#0f1b26,#162430 50%,#0d1720)", "in": "#1f2b37", "out": "#2b5278",
             "text_in": "#e9edf1", "text_out": "#f2f6fa", "meta_in": "#7d8b99", "meta_out": "#a9c6e4", "sep": "rgba(0,0,0,.35)",
             "septext": "#cfd8e0", "head": "#17212b", "headfg": "#f0f3f6", "sel": "rgba(80,140,200,.25)"},
}
PAL_B = {
    "light": {"wall": "#efeae2", "in": "#ffffff", "out": "#d9fdd3", "text_in": "#111b21", "text_out": "#111b21",
              "meta_in": "#667781", "meta_out": "#53bdeb", "sep": "#ffffff", "septext": "#54656f", "head": "#008069",
              "headfg": "#ffffff", "sel": "#f0f2f5"},
    "dark": {"wall": "#0b141a", "in": "#202c33", "out": "#005c4b", "text_in": "#e9edef", "text_out": "#e9edef",
             "meta_in": "#8696a0", "meta_out": "#53bdeb", "sep": "#182229", "septext": "#8696a0", "head": "#1f2c34",
             "headfg": "#e9edef", "sel": "#2a3942"},
}


def _messenger(c: Ctx, pal_all: dict, style: str) -> str:
    pal = pal_all[c.theme]
    group = _is_group(c)
    title = c.title or (c.messages[0].get("author") if c.messages else "") or c.fill.names(1)[0]
    if group:
        sub = f"{c.r.randint(3, 240)} {c.ui['members']}"
    else:
        sub = c.r.choice([c.ui["online"], c.ui["recently"], c.ui["typing"], f"{c.ui['recently']}"])
    head = (f'<div class="appbar" style="background:{pal["head"]};color:{pal["headfg"]}">'
            f'<div class="iconbtn">{C.icon("back")}</div>{C.avatar(title, c.r, 40)}'
            f'<div class="grow col" style="margin-left:4px"><div class="ttl ell" style="font-size:{C.fs(18)}" data-role="title">{e(title)}</div>'
            f'<div class="sub ell" data-role="chrome">{e(sub)}</div></div>'
            + (f'<div class="iconbtn">{C.icon("video")}</div>' if style == "b" else "")
            + f'<div class="iconbtn">{C.icon("phone")}</div><div class="iconbtn">{C.icon("dots")}</div></div>')
    pinned = ""
    snips = list(c.snippets)
    if snips and c.r.random() < 0.5:
        s = snips.pop(0)
        pinned = (f'<div class="row" style="background:{pal["head"] if style == "a" else "var(--bar)"};color:{pal["headfg"] if style == "a" else "var(--text)"};'
                  f'padding:6px 14px;gap:10px;border-top:1px solid var(--divider)"><div style="width:2px;align-self:stretch;background:var(--accent)"></div>'
                  f'<div class="grow col"><div style="color:var(--accent);font-size:{C.fs(13)};font-weight:500" data-role="chrome">{e(c.ui["pinned"])}</div>'
                  f'<div class="ell" style="font-size:{C.fs(14)}" data-role="snippet">{e(s)}</div></div></div>')
    c.spec = {**c.spec, "snippets": snips}
    msgs = _bubbles(c, pal, group, tails=(style == "a"))
    if style == "a":
        field = C.input_field(c.field, c.ui["message"], c.r, ("smile",), ("clip", "mic"), "inputbox")
        bar = f'<div class="inputbar" style="background:var(--bar);align-items:center">{field}</div>'
    else:
        field = C.input_field(c.field, c.ui["message"], c.r, ("smile",), ("clip", "camera"), "inputbox")
        bar = (f'<div class="inputbar" style="background:transparent;align-items:center">{field}'
               f'<div style="width:46px;height:46px;border-radius:23px;background:#00a884;color:#fff;display:flex;align-items:center;justify-content:center">{C.icon("mic")}</div></div>')
    chat = (f'{head}{pinned}<div class="grow col" style="justify-content:flex-end;overflow:hidden;background:{pal["wall"]};padding-bottom:4px">'
            f'{msgs}</div>{bar if c.has_field else ""}')
    if not c.wide:
        return frame(c, chat, status_style=f"--sbbg:{pal['head']};--sbfg:{pal['headfg']}")
    # inner display: chat list | conversation
    lst_head = (f'<div class="appbar" style="background:{pal["head"]};color:{pal["headfg"]}"><div class="iconbtn">{C.icon("menu")}</div>'
                f'<div class="ttl grow" data-role="chrome">{e(c.ui["chats"])}</div><div class="iconbtn">{C.icon("search")}</div></div>')
    left = (f'<div class="col" style="width:{c.r.choice([38, 40, 42])}%;border-right:1px solid var(--divider);overflow:hidden">{lst_head}'
            f'<div class="grow" style="overflow:hidden">{_chat_list(c, pal, title)}</div></div>')
    return frame(c, f'<div class="row grow" style="align-items:stretch;overflow:hidden">{left}<div class="col grow">{chat}</div></div>',
                 status_style=f"--sbbg:{pal['head']};--sbfg:{pal['headfg']}")


def messenger_a(c: Ctx) -> str:
    return _messenger(c, PAL_A, "a")


def messenger_b(c: Ctx) -> str:
    return _messenger(c, PAL_B, "b")


# ----------------------------------------------------------------------------- work chat

PAL_W = {"light": {"side": "#3f0e40", "sidefg": "#e9dfe9", "head": "#ffffff"},
         "dark": {"side": "#19171d", "sidefg": "#cfc8d2", "head": "#1a1d21"}}


def work_chat(c: Ctx) -> str:
    pal = PAL_W[c.theme]
    chan = c.title or "general"
    is_channel = not any(ch.isupper() for ch in chan[:1]) or chan.startswith("#")
    label = ("# " + chan.lstrip("#")) if is_channel else chan
    head = (f'<div class="appbar" style="border-bottom:1px solid var(--divider)"><div class="iconbtn">{C.icon("back")}</div>'
            f'<div class="grow col"><div class="ttl ell" style="font-size:{C.fs(18)};font-weight:700" data-role="title">{e(label)}</div>'
            f'<div class="sub ell" data-role="chrome">{c.r.randint(3, 90)} {e(c.ui["members"])}</div></div>'
            f'<div class="iconbtn">{C.icon("phone")}</div><div class="iconbtn">{C.icon("dots")}</div></div>')
    rows = []
    times = c.msg_times()
    sep = c.fill.date_sep()
    rows.append(f'<div class="row" style="margin:8px 0"><div class="divider grow"></div><span class="chip" style="border-radius:14px;'
                f'height:26px;font-size:{C.fs(13)};font-weight:600" data-role="date">{e(sep)}</span><div class="divider grow"></div></div>')
    snips = list(c.snippets)
    prev_author = None
    for i, m in enumerate(c.messages):
        author = c.ui["you"] if m.get("is_me") and not m.get("author") else (m.get("author") or "")
        cont = author == prev_author and c.r.random() < 0.7
        prev_author = author
        head_html = "" if cont else (
            f'<div class="row" style="gap:8px"><span style="font-weight:700" data-role="author">{e(author)}</span>'
            f'<span class="t3" style="font-size:{C.fs(12)}" data-role="time">{e(times[i])}</span></div>')
        av = '<div style="width:38px"></div>' if cont else (
            f'<div style="width:38px;height:38px;border-radius:8px;overflow:hidden;flex:none">{C.avatar(author, c.r, 38)}</div>')
        extra = ""
        if snips and c.r.random() < 0.4:
            extra = (f'<div style="margin-top:6px;border-left:4px solid var(--divider);padding-left:10px;font-size:{C.fs(15)}" '
                     f'data-role="snippet">{e(snips.pop(0))}</div>')
        if c.r.random() < 0.2:
            rep_lbl = "ответа" if c.locale == "ru" else "תגובות" if c.locale == "he" else "replies"
            extra += (f'<div style="margin-top:6px;color:var(--accent);font-size:{C.fs(14)};font-weight:600" data-role="chrome">'
                      f'{c.r.randint(2, 9)} {rep_lbl}</div>')
        rows.append(f'<div class="row" style="align-items:flex-start;gap:10px;padding:{"2" if cont else "8"}px 14px 2px">{av}'
                    f'<div class="grow col">{head_html}<div{" dir=\"rtl\"" if any("\u05d0" <= ch <= "\u05ea" for ch in m["text"]) else ""} style="white-space:pre-wrap;word-break:break-word" data-role="message">'
                    f'{e(m["text"])}</div>{extra}</div></div>')
    for s in snips:
        rows.append(f'<div style="margin:6px 14px 0 62px;padding:8px 10px;border:1px solid var(--divider);border-radius:8px;'
                    f'font-size:{C.fs(14)}" data-role="snippet">{e(s)}</div>')
    hint = c.field.get("hint") or (f'{c.ui["message"]} {label}')
    fld = dict(c.field, hint=hint)
    bar = (f'<div class="inputbar" style="border-top:1px solid var(--divider);flex-direction:column;align-items:stretch">'
           f'{C.input_field(fld, hint, c.r, (), (), "inputbox")}'
           f'<div class="row t2" style="gap:18px;padding:4px 6px">{C.icon("plus")}{C.icon("smile")}<span style="font-weight:700">@</span>'
           f'<span class="grow"></span>{C.icon("mic")}{C.icon("send")}</div></div>')
    body = (f'{head}<div class="grow col" style="justify-content:flex-end;overflow:hidden">{"".join(rows)}</div>'
            f'{bar if c.has_field else ""}')
    if not c.has_field and not c.wide:
        body += C.nav_bar([("home", c.ui["home"]), ("chat", c.ui["dms"]), ("bell", c.ui["activity"]),
                           ("dots", c.ui["more"])], 0)
    if not c.wide:
        return frame(c, body)
    chans = ["general", "random", "design", "standup", "support", "hiring", "office", "announcements"]
    c.r.shuffle(chans)
    items = "".join(
        f'<div style="padding:6px 16px;font-size:{C.fs(15)};{"background:rgba(255,255,255,.15);font-weight:700" if i == 1 else ""}" '
        f'data-role="filler"># {ch}</div>' for i, ch in enumerate(chans[:6]))
    dms = "".join(f'<div class="row" style="padding:6px 16px;gap:8px;font-size:{C.fs(15)}" data-role="filler">'
                  f'<span style="width:8px;height:8px;border-radius:4px;background:#2bac76"></span>{e(n)}</div>'
                  for n in c.fill.names(4))
    side = (f'<div class="col" style="width:32%;background:{pal["side"]};color:{pal["sidefg"]};overflow:hidden">'
            f'<div style="padding:14px 16px;font-size:{C.fs(18)};font-weight:700" data-role="chrome">'
            f'{e(c.r.choice(["Team", "Acme", "Studio", "Команда", "Отдел"]))}</div>{items}'
            f'<div style="padding:14px 16px 4px;opacity:.8;font-size:{C.fs(14)}" data-role="chrome">'
            f'{"Личные сообщения" if c.locale == "ru" else "הודעות ישירות" if c.locale == "he" else "Direct messages"}</div>{dms}</div>')
    return frame(c, f'<div class="row grow" style="align-items:stretch;overflow:hidden">{side}<div class="col grow">{body}</div></div>')
