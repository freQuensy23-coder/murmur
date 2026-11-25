"""E-mail look-alikes: email_inbox (list; on the inner display list | open thread), email_thread (open
message with quick reply), email_compose (new mail / reply with quoted original)."""
from __future__ import annotations

from .. import chrome as C
from .. import filler as F
from ..base import Ctx, frame

e = C.e

TEMPLATE_APPS = {
    "email_inbox": ("com.google.android.gm", "Gmail", "email", "search", "email"),
    "email_thread": ("com.google.android.gm", "Gmail", "email", "email_body", "email"),
    "email_compose": ("com.google.android.gm", "Gmail", "email", "email_body", "email"),
}


def _addr(name: str, r) -> str:
    base = "".join(ch for ch in name.lower().split()[0] if ch.isascii() and ch.isalnum()) or r.choice(["info", "team", "hello"])
    return f"{base}@{r.choice(['mail.ru', 'yandex.ru', 'gmail.com', 'company.com', 'outlook.com'])}"


def _rows(c: Ctx) -> list[dict]:
    """Inbox rows: the teacher's messages (author = sender, title = subject of the newest) + filler rows."""
    rows = []
    msgs = c.messages
    for i, m in enumerate(reversed(msgs)):
        subj = c.title if i == 0 else (c.r.choice(["Re: ", "Fwd: ", ""]) + c.title if c.title else "")
        rows.append({"from": m.get("author") or c.ui["you"], "subject": subj, "text": m["text"], "content": True,
                     "unread": c.r.random() < 0.5})
    n = c.r.randint(5, 9)
    senders = c.fill.pick(F.SENDERS_RU + F.NAMES_RU, F.SENDERS_EN + F.NAMES_EN, n, pool_he=F.SENDERS_HE + F.NAMES_HE)
    subjects = c.fill.pick(F.SUBJ_RU, F.SUBJ_EN, n, pool_he=F.SUBJ_HE)
    snips = c.fill.pick(F.SNIP_RU, F.SNIP_EN, n, pool_he=F.SNIP_HE)
    for i in range(min(len(senders), len(subjects), len(snips))):
        rows.append({"from": senders[i], "subject": subjects[i], "text": snips[i] + "…", "content": False,
                     "unread": c.r.random() < 0.3})
    k = c.r.randint(0, min(2, len(rows) - len(msgs))) if len(rows) > len(msgs) else 0
    return rows[len(msgs):len(msgs) + k] + rows[:len(msgs)] + rows[len(msgs) + k:]


def _inbox_list(c: Ctx) -> str:
    out = []
    rows = _rows(c)
    for i, row in enumerate(rows):
        fr, su, tx = ("author", "title", "message") if row["content"] else ("filler", "filler", "filler")
        w = "700" if row["unread"] else "400"
        att = (f'<div style="margin-top:6px"><span class="chip" style="height:28px;border-radius:14px" data-role="filler">'
               f'{C.icon("attach2", 16)}&nbsp;{e(c.r.choice(["scan.pdf", "IMG_2041.jpg", "отчёт.docx", "notes.txt"]))}</span></div>'
               if c.r.random() < 0.12 else "")
        out.append(
            f'<div class="row" style="gap:14px;padding:10px 16px;align-items:flex-start">{C.avatar(row["from"], c.r, 40, 0.3)}'
            f'<div class="grow col" style="gap:1px"><div class="row"><span class="grow ell" style="font-weight:{w}" data-role="{fr}">{e(row["from"])}</span>'
            f'<span style="font-size:{C.fs(12)};font-weight:{w}" class="t2" data-role="time">{e(c.fill.time_ago(i))}</span></div>'
            f'<div class="ell" style="font-weight:{w};font-size:{C.fs(15)}" data-role="{su}">{e(row["subject"])}</div>'
            f'<div class="row"><span class="grow ell t2" style="font-size:{C.fs(15)}" data-role="{tx}">{e(row["text"])}</span>'
            f'<span class="t3" style="margin-left:8px">{C.icon("star", 20)}</span></div>{att}</div></div>')
    return "".join(out)


def _search_pill(c: Ctx, focused: bool) -> str:
    hint = c.field.get("hint") or ("Искать в почте" if c.locale == "ru" else "חיפוש בדואר" if c.locale == "he" else "Search in mail")
    if focused:
        inner = C.input_field({**c.field, "hint": hint}, hint, c.r, ("back",), ("mic",), "inputbox")
        return f'<div style="padding:8px 12px">{inner}</div>'
    return (f'<div style="padding:8px 12px"><div class="row" style="height:48px;border-radius:24px;background:var(--surface);padding:0 8px 0 14px;gap:12px">'
            f'{C.icon("menu")}<span class="grow t2" data-role="hint">{e(hint)}</span>{C.avatar(c.fill.names(1)[0] or "A", c.r, 32)}</div></div>')


def _inbox(c: Ctx, focused: bool) -> str:
    label = (f'<div style="padding:8px 18px 4px;font-size:{C.fs(13)};font-weight:500" class="t2" data-role="chrome">'
             f'{e(c.ui["primary"] if c.r.random() < 0.6 else c.ui["inbox"])}</div>')
    fab = (f'<div class="fab" style="bottom:{86 if not c.keyboard else 16}px;width:auto;padding:0 18px;gap:10px">'
           f'{C.icon("pen")}<span style="font-weight:500" data-role="chrome">{e(c.ui["compose"])}</span></div>')
    nav = "" if c.keyboard else C.nav_bar([("inbox", c.ui["mail"]), ("video", c.ui["meet"])], 0, {0: str(c.r.randint(1, 99))})
    return (f'{_search_pill(c, focused)}{label}<div class="grow" style="overflow:hidden">{_inbox_list(c)}</div>'
            f'{fab if not focused else ""}{nav}')


def _message_card(c: Ctx, m: dict, expanded: bool, t: str) -> str:
    name = m.get("author") or c.ui["you"]
    to = (("мне" if c.locale == "ru" else "אליי" if c.locale == "he" else "to me") if not m.get("is_me")
          else (("кому: " if c.locale == "ru" else "אל: " if c.locale == "he" else "to ") + c.fill.names(1)[0]))
    if expanded:
        body = f'<div style="padding:4px 16px 14px 70px;white-space:pre-wrap;word-break:break-word" data-role="message">{e(m["text"])}</div>'
    else:
        body = (f'<div class="ell t2" style="padding:0 16px 12px 70px;font-size:{C.fs(15)}" data-role="message">{e(m["text"])}</div>')
    return (f'<div style="border-bottom:1px solid var(--divider)"><div class="row" style="gap:14px;padding:12px 16px 4px">'
            f'{C.avatar(name, c.r, 40, 0.3)}<div class="grow col"><div class="row"><span class="ell" style="font-weight:500" data-role="author">'
            f'{e(name)}</span><span class="t3" style="margin-left:8px;font-size:{C.fs(12)}" data-role="time">{e(t)}</span></div>'
            f'<div class="t2 ell" style="font-size:{C.fs(13)}" data-role="chrome">{e(to)}</div></div>'
            f'<span class="t2">{C.icon("reply", 22)}</span><span class="t2">{C.icon("dots", 22)}</span></div>{body}</div>')


def _thread(c: Ctx, with_reply: bool) -> str:
    head = (f'<div class="appbar"><div class="iconbtn">{C.icon("back")}</div><span class="grow"></span>'
            f'<div class="iconbtn">{C.icon("inbox")}</div><div class="iconbtn">{C.icon("dots")}</div></div>')
    title = c.title or (c.fill.pick(F.SUBJ_RU, F.SUBJ_EN, 1, pool_he=F.SUBJ_HE) or [""])[0]
    subj = (f'<div class="row" style="padding:4px 16px 10px;gap:10px;align-items:flex-start"><div class="grow" style="font-size:{C.fs(22)}" data-role="title">'
            f'{e(title)}</div><span class="chip" style="height:24px;font-size:{C.fs(12)}" data-role="chrome">{e(c.ui["inbox"])}</span></div>')
    times = c.msg_times()
    cards = "".join(_message_card(c, m, i == len(c.messages) - 1, times[i]) for i, m in enumerate(c.messages))
    snips = "".join(f'<div style="margin:8px 16px 0 70px;padding:8px 12px;border-left:3px solid var(--divider);font-size:{C.fs(15)}" '
                    f'class="t2" data-role="snippet">{e(s)}</div>' for s in c.snippets)
    reply = ""
    if with_reply and c.has_field:
        reply = (f'<div class="inputbar" style="border-top:1px solid var(--divider)">'
                 f'{C.input_field(c.field, c.ui["reply"], c.r, (), ("send",), "inputbox")}</div>')
    elif not c.has_field:
        reply = (f'<div class="row" style="gap:10px;padding:10px 16px">'
                 + "".join(f'<span class="chip" style="height:40px;border-radius:20px;flex:1;justify-content:center" data-role="chrome">'
                           f'{C.icon(ic, 18)}&nbsp;{e(c.ui[k])}</span>' for ic, k in (("reply", "reply"), ("send", "forward")))
                 + "</div>")
    return (f'{head}{subj}<div class="grow col" style="overflow:hidden;justify-content:{"flex-end" if c.r.random() < 0.5 else "flex-start"}">'
            f'<div>{cards}{snips}</div></div>{reply}')


def email_inbox(c: Ctx) -> str:
    focused = c.has_field
    if not c.wide:
        return frame(c, _inbox(c, focused))
    left = f'<div class="col" style="width:44%;border-right:1px solid var(--divider);overflow:hidden">{_inbox(c, focused)}</div>'
    c2 = Ctx(**{**c.__dict__, "spec": {**c.spec, "field": None}})
    return frame(c, f'<div class="row grow" style="align-items:stretch;overflow:hidden">{left}<div class="col grow">{_thread(c2, False)}</div></div>')


def email_thread(c: Ctx) -> str:
    if not c.wide:
        return frame(c, _thread(c, True))
    c2 = Ctx(**{**c.__dict__, "spec": {**c.spec, "field": None}})
    left = f'<div class="col" style="width:42%;border-right:1px solid var(--divider);overflow:hidden">{_inbox(c2, False)}</div>'
    return frame(c, f'<div class="row grow" style="align-items:stretch;overflow:hidden">{left}<div class="col grow">{_thread(c, True)}</div></div>')


def email_compose(c: Ctx) -> str:
    title = c.ui["reply"] if c.messages else c.ui["compose"]
    head = (f'<div class="appbar"><div class="iconbtn">{C.icon("back")}</div><div class="ttl grow" data-role="chrome">{e(title)}</div>'
            f'<div class="iconbtn">{C.icon("clip")}</div><div class="iconbtn">{C.icon("send")}</div><div class="iconbtn">{C.icon("dots")}</div></div>')
    me = c.fill.names(1)[0] or "me"
    rec = (c.messages[-1].get("author") if c.messages and not c.messages[-1].get("is_me") else "") or c.fill.names(1)[0] or "team"
    subj = c.title or ""
    if c.messages and subj and not subj.lower().startswith("re"):
        subj = "Re: " + subj
    row = lambda k, v, role: (f'<div class="row" style="padding:12px 16px;gap:14px;border-bottom:1px solid var(--divider)">'
                              f'<span class="t2" style="min-width:52px" data-role="chrome">{e(k)}</span><span class="grow ell" data-role="{role}">{v}</span></div>')
    chip = (f'<span class="chip" style="border-radius:16px;height:32px;gap:6px;padding-left:2px">{C.avatar(rec, c.r, 26, 0)}'
            f'<span data-role="author">{e(rec)}</span></span>')
    to_focus = c.field.get("kind") == "address"   # the focused field is the recipient, not the body
    bc = c.field.get("before_cursor") or ""
    from_row = row(c.ui["from"], e(_addr(me, c.r)), "filler")
    if to_focus:
        to_html = ('<span data-focus="1">' + (f'<span data-role="field">{e(bc)}</span>' if bc else "")
                   + '<span class="cursor"></span></span>')
        fields = from_row + row(c.ui["to"], to_html, "field")
    else:
        fields = from_row + row(c.ui["to"], chip, "author")
    fields += row(c.ui["subject"], e(subj), "title") if subj else row(c.ui["subject"], "", "chrome")
    hint = c.field.get("hint") or ("Текст письма" if c.locale == "ru" else "כתיבת אימייל" if c.locale == "he" else "Compose email")
    if to_focus or not c.has_field:
        body = ""
    elif bc:
        body = f'<span style="white-space:pre-wrap" data-role="field">{e(bc)}</span><span class="cursor"></span>'
    else:
        body = f'<span class="cursor"></span><span class="hint" data-role="hint">{e(hint)}</span>'
    quoted = ""
    if c.messages:
        m = c.messages[-1]
        when = c.fill.date_sep()
        intro = (f'{when}, {m.get("author") or rec} пишет:' if c.locale == "ru" else
                 f'בתאריך {when}, {m.get("author") or rec} כתב/ה:' if c.locale == "he" else
                 f'On {when}, {m.get("author") or rec} wrote:')
        quoted = (f'<div class="t2" style="margin-top:22px;font-size:{C.fs(14)}" data-role="chrome">{e(intro)}</div>'
                  f'<div class="t2" style="border-left:2px solid var(--divider);padding-left:10px;margin-top:4px;white-space:pre-wrap" data-role="message">'
                  f'{e(m["text"])}</div>')
    snips = "".join(f'<div class="t2" style="margin-top:14px;font-size:{C.fs(14)}" data-role="snippet">{e(s)}</div>' for s in c.snippets)
    if body:   # the body EditText holds only the typed text; quote and snippets are separate views below it
        body = f'<div data-focus="1">{body}</div>'
    return frame(c, f'{head}{fields}<div class="grow" style="padding:14px 16px;overflow:hidden">{body}{quoted}{snips}</div>')


TEMPLATES = {"email_inbox": email_inbox, "email_thread": email_thread, "email_compose": email_compose}
