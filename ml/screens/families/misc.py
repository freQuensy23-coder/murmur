"""AI chat, calendar / tasks and social feed look-alikes."""
from __future__ import annotations

from .. import chrome as C
from .. import filler as F
from ..base import Ctx, frame

e = C.e

TEMPLATE_APPS = {
    "ai_chat": ("com.openai.chatgpt", "ChatGPT", "default", "generic", "ai_chat"),
    "calendar_tasks": ("com.google.android.calendar", "Calendar", "default", "generic", "calendar"),
    "social_feed": ("com.vkontakte.android", "VK", "social", "message", "social"),
}


def ai_chat(c: Ctx) -> str:
    app_lbl = (c.spec.get("app") or {}).get("label") or ""
    if app_lbl and app_lbl not in ("", "default"):
        name = app_lbl
    else:
        name = c.r.choice(["ChatGPT", "Claude", "Gemini", "Perplexity", "Assistant", "Ассистент"])
    head = (f'<div class="appbar">{C.icon("menu")}<span class="grow" style="font-size:{C.fs(18)};font-weight:500;margin-left:8px" data-role="chrome">'
            f'{e(name)}</span>{C.icon("pen")}<span style="width:6px"></span>{C.icon("dots")}</div>')
    msgs = []
    for m in c.messages:
        rtl = ' dir="rtl"' if any("\u05d0" <= ch <= "\u05ea" for ch in m["text"]) else ""
        if m.get("is_me"):
            msgs.append(f'<div class="row" style="justify-content:flex-end;margin:10px 14px"><div{rtl} style="max-width:80%;background:var(--surface2);'
                        f'border-radius:20px;padding:10px 14px;white-space:pre-wrap;word-break:break-word" data-role="message">{e(m["text"])}</div></div>')
        else:
            paras = "".join(f'<p{rtl} style="margin:0 0 10px" data-role="message">{e(p)}</p>' for p in m["text"].split("\n") if p.strip())
            msgs.append(f'<div style="margin:10px 16px;line-height:1.5">{paras}<div class="row t3" style="gap:14px">'
                        f'{C.icon("dcheck", 18)}{C.icon("refresh", 18)}{C.icon("star", 18)}</div></div>')
    if c.snippets:
        msgs.append('<div class="row" style="gap:8px;margin:10px 14px;flex-wrap:wrap">' + "".join(
            f'<span class="chip" style="border-radius:16px;height:auto;padding:8px 12px;white-space:normal" data-role="snippet">{e(s)}</span>'
            for s in c.snippets) + "</div>")
    if not c.messages and not c.snippets:
        greet = "Чем могу помочь?" if c.locale == "ru" else "במה אוכל לעזור?" if c.locale == "he" else "What can I help with?"
        msgs.append(f'<div class="grow col" style="justify-content:center;align-items:center;font-size:{C.fs(24)}" data-role="chrome">{e(greet)}</div>')
    hint = c.field.get("hint") or c.ui["ask"]
    bar = (f'<div class="inputbar" style="flex-direction:column;align-items:stretch;border-radius:24px 24px 0 0;background:var(--surface);gap:4px">'
           f'{C.input_field(c.field, hint, c.r, (), (), "inputbox")}'
           f'<div class="row t2" style="gap:16px;padding:2px 8px">{C.icon("plus")}{C.icon("globe")}<span class="grow"></span>{C.icon("mic")}'
           f'<span style="width:34px;height:34px;border-radius:17px;background:var(--text);color:var(--bg);display:flex;align-items:center;justify-content:center">'
           f'{C.icon("send", 18)}</span></div></div>')
    title = (f'<div class="t3" style="text-align:center;font-size:{C.fs(13)};padding:2px" data-role="title">{e(c.title)}</div>' if c.title else "")
    chat = f'{head}{title}<div class="grow col" style="justify-content:flex-end;overflow:hidden">{"".join(msgs)}</div>{bar if c.has_field else ""}'
    if not c.wide:
        return frame(c, chat)
    hist = c.fill.pick(F.TASKS_RU + F.EVENTS_RU, F.TASKS_EN + F.EVENTS_EN, 9, pool_he=F.TASKS_HE + F.EVENTS_HE)
    side = (f'<div class="col" style="width:30%;background:var(--surface);padding:10px 8px;gap:4px;overflow:hidden">'
            f'<div class="row" style="gap:8px;padding:8px">{C.icon("pen")}<span data-role="chrome">{e(c.ui["new_chat"])}</span></div>'
            + "".join(f'<div class="ell" style="padding:8px;font-size:{C.fs(14)};{"background:var(--surface2);border-radius:8px" if i == 0 else ""}" data-role="filler">'
                      f'{e(h)}</div>' for i, h in enumerate(hist)) + "</div>")
    return frame(c, f'<div class="row grow" style="align-items:stretch;overflow:hidden">{side}<div class="col grow">{chat}</div></div>')


def calendar_tasks(c: Ctx) -> str:
    ru = c.locale == "ru"
    day = c.title or c.fill.date_sep()
    head = (f'<div class="appbar">{C.icon("menu")}<span class="grow" style="font-size:{C.fs(20)};margin-left:8px" data-role="title">{e(day)}</span>'
            f'{C.icon("search")}<span style="width:8px"></span>{C.icon("cal")}<span style="width:8px"></span>{C.avatar(c.fill.names(1)[0] or "A", c.r, 30)}</div>')
    wd = F.WEEKDAYS_RU if ru else F.WEEKDAYS_HE if c.locale == "he" else F.WEEKDAYS_EN
    d0 = c.r.randint(1, 22)
    week = '<div class="row" style="justify-content:space-around;padding:6px 0 10px;border-bottom:1px solid var(--divider)">' + "".join(
        f'<div class="col" style="align-items:center;gap:4px"><span class="t2" style="font-size:{C.fs(12)}" data-role="chrome">{w}</span>'
        f'<span style="width:32px;height:32px;border-radius:16px;display:flex;align-items:center;justify-content:center;'
        f'{"background:var(--accent);color:#fff" if i == 2 else ""}" data-role="chrome">{d0 + i}</span></div>' for i, w in enumerate(wd)) + "</div>"
    rows = []
    times = c.msg_times()
    for i, m in enumerate(c.messages):
        when = m.get("time") or times[i]
        who = m.get("author") or ""
        col = c.r.choice(C.AV_COLORS)
        rows.append(f'<div class="row" style="gap:12px;padding:6px 14px;align-items:flex-start"><span class="t2" style="width:52px;font-size:{C.fs(13)}" data-role="time">{e(when)}</span>'
                    f'<div class="grow" style="background:{col};color:#fff;border-radius:8px;padding:8px 10px">'
                    f'<div style="font-weight:500;white-space:pre-wrap;word-break:break-word" data-role="message">{e(m["text"])}</div>'
                    + (f'<div style="font-size:{C.fs(13)};opacity:.9" data-role="author">{e(who)}</div>' if who else "") + "</div></div>")
    ev = c.fill.pick(F.EVENTS_RU, F.EVENTS_EN, c.r.randint(1, 3), pool_he=F.EVENTS_HE)
    for x in ev:
        rows.append(f'<div class="row" style="gap:12px;padding:6px 14px"><span class="t2" style="width:52px;font-size:{C.fs(13)}" data-role="time">{c.fill.clock()}</span>'
                    f'<div class="grow" style="border-left:4px solid {c.r.choice(C.AV_COLORS)};padding:6px 10px;background:var(--surface);border-radius:4px" data-role="filler">{e(x)}</div></div>')
    tasks = []
    for s in c.snippets + c.fill.pick(F.TASKS_RU, F.TASKS_EN, c.r.randint(1, 3), pool_he=F.TASKS_HE):
        role = "snippet" if s in c.snippets else "filler"
        done = c.r.random() < 0.25 and role == "filler"
        tasks.append(f'<div class="row" style="gap:12px;padding:8px 16px"><span style="width:20px;height:20px;border-radius:10px;border:2px solid var(--text2);flex:none;'
                     f'{"background:var(--text2)" if done else ""}"></span><span style="{"text-decoration:line-through" if done else ""}" class="{"t3" if done else ""}" data-role="{role}">{e(s)}</span></div>')
    task_head = f'<div style="padding:10px 16px 2px;font-weight:600" data-role="chrome">{e(c.ui["tasks"])}</div>'
    newtask = ""
    if c.has_field:
        newtask = (f'<div class="inputbar" style="border-top:1px solid var(--divider)">'
                   f'{C.input_field(c.field, "Новая задача" if ru else "משימה חדשה" if c.locale == "he" else "Add a task", c.r, ("plus",), (), "inputbox")}</div>')
    body = f'{head}{week}<div class="grow" style="overflow:hidden">{"".join(rows)}{task_head}{"".join(tasks)}</div>{newtask}'
    if not c.has_field:
        body += f'<div class="fab" style="bottom:20px">{C.icon("plus")}</div>'
    return frame(c, body)


def social_feed(c: Ctx) -> str:
    head = (f'<div class="appbar"><div class="iconbtn">{C.icon("back")}</div><span class="grow ttl" style="font-size:{C.fs(18)}" data-role="chrome">'
            f'{"Запись" if c.locale == "ru" else "פוסט" if c.locale == "he" else "Post"}</span>{C.icon("dots")}</div>')
    posts = []
    first = c.messages[0] if c.messages else None
    if first:
        a = first.get("author") or c.title or "—"
        img = (f'<div style="height:{c.r.randint(140, 260)}px;margin-top:8px;background:linear-gradient({c.r.randint(0, 359)}deg,'
               f'{c.r.choice(C.AV_COLORS)},{c.r.choice(C.AV_COLORS)})"></div>' if c.r.random() < 0.5 else "")
        posts.append(f'<div style="padding:10px 14px"><div class="row" style="gap:10px">{C.avatar(a, c.r, 42)}<div class="col">'
                     f'<b data-role="author">{e(a)}</b><span class="t3" style="font-size:{C.fs(13)}" data-role="time">{e(c.fill.date_sep())}</span></div></div>'
                     + (f'<div style="font-weight:600;margin-top:8px" data-role="title">{e(c.title)}</div>' if c.title and c.title != a else "")
                     + f'<div style="margin-top:6px;white-space:pre-wrap;word-break:break-word" data-role="message">{e(first["text"])}</div></div>{img}'
                     f'<div class="row t2" style="gap:18px;padding:8px 14px;font-size:{C.fs(14)}" data-role="chrome">'
                     f'<span>❤ {c.r.randint(3, 900)}</span><span>💬 {c.r.randint(1, 80)}</span><span>↗ {c.r.randint(0, 40)}</span></div>'
                     f'<div class="divider"></div>')
    times = c.msg_times()
    for i, m in enumerate(c.messages[1:]):
        a = m.get("author") or c.ui["you"]
        posts.append(f'<div class="row" style="gap:10px;padding:8px 14px;align-items:flex-start">{C.avatar(a, c.r, 34)}<div class="grow col">'
                     f'<b style="font-size:{C.fs(14)}" data-role="author">{e(a)}</b><div style="white-space:pre-wrap;word-break:break-word" data-role="message">{e(m["text"])}</div>'
                     f'<span class="t3" style="font-size:{C.fs(12)}" data-role="time">{e(times[i + 1])} · {"Ответить" if c.locale == "ru" else "השב" if c.locale == "he" else "Reply"}</span></div></div>')
    for s in c.snippets:
        posts.append(f'<div class="t2" style="padding:6px 14px 6px 58px;font-size:{C.fs(14)}" data-role="snippet">{e(s)}</div>')
    bar = ""
    if c.has_field:
        bar = (f'<div class="inputbar" style="border-top:1px solid var(--divider)">'
               f'{C.input_field(c.field, "Комментарий" if c.locale == "ru" else "הוסף תגובה…" if c.locale == "he" else "Add a comment…", c.r, (), ("smile", "send"), "inputbox")}</div>')
    return frame(c, f'{head}<div class="grow" style="overflow:hidden">{"".join(posts)}</div>{bar}')


TEMPLATES = {"ai_chat": ai_chat, "calendar_tasks": calendar_tasks, "social_feed": social_feed}
