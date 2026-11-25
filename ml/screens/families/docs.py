"""Notes and document editor look-alikes.

notes: note editor; before_cursor is the note body up to the cursor, messages are the rest of the note
after the cursor (inserting in the middle of a note), snippets are tags / checklist items. On the inner
display: notes list | editor.
doc_editor: formatting toolbar, a page with the title as heading, messages as paragraphs (authored
messages become margin comments on the inner display), before_cursor in the paragraph being written.
"""
from __future__ import annotations

from .. import chrome as C
from .. import filler as F
from ..base import Ctx, frame

e = C.e

TEMPLATE_APPS = {
    "notes": ("com.google.android.keep", "Keep Notes", "notes", "note", "notes"),
    "doc_editor": ("com.google.android.apps.docs.editors.docs", "Docs", "notes", "note", "doc_editor"),
}

NOTE_BG = {"light": ["#ffffff", "#fff8b8", "#e2f6d3", "#d4e4ed", "#f6e2dd"], "dark": ["#202124", "#41331c", "#2a3a26", "#2d555e", "#5c2b29"]}


def _cursor_text(c: Ctx, hint: str) -> str:
    bc = c.field.get("before_cursor") or ""
    if bc:
        return f'<span style="white-space:pre-wrap;word-break:break-word" data-role="field">{e(bc)}</span><span class="cursor"></span>'
    if not c.has_field:
        return ""
    return f'<span class="cursor"></span><span class="hint" data-role="hint">{e(c.field.get("hint") or hint)}</span>'


def _note_editor(c: Ctx, bg: str) -> str:
    head = (f'<div class="appbar" style="background:{bg}"><div class="iconbtn">{C.icon("back")}</div><span class="grow"></span>'
            f'<div class="iconbtn">{C.icon("star")}</div><div class="iconbtn">{C.icon("bell")}</div><div class="iconbtn">{C.icon("inbox")}</div></div>')
    title = (f'<div style="font-size:{C.fs(22)};padding:4px 18px 8px;font-weight:400" data-role="title">{e(c.title)}</div>'
             if c.title else f'<div style="font-size:{C.fs(22)};padding:4px 18px 8px" class="t3" data-role="hint">'
                             f'{"Название" if c.locale == "ru" else "כותרת" if c.locale == "he" else "Title"}</div>')
    after = "".join(f'<div style="white-space:pre-wrap;word-break:break-word;margin-top:10px" data-role="message">{e(m["text"])}</div>'
                    for m in c.messages)
    tags = ""
    if c.snippets:
        if c.r.random() < 0.5:
            tags = "".join(f'<div class="row" style="gap:10px;margin-top:8px"><div style="width:18px;height:18px;border:2px solid var(--text2);'
                           f'border-radius:3px;flex:none"></div><span data-role="snippet">{e(s)}</span></div>' for s in c.snippets)
        else:
            tags = '<div class="row" style="gap:8px;flex-wrap:wrap;margin-top:14px">' + "".join(
                f'<span class="chip" style="border-radius:6px;background:rgba(127,127,127,.15);border:none" data-role="snippet">{e(s)}</span>'
                for s in c.snippets) + "</div>"
    edited = (f'{"Изменено" if c.locale == "ru" else "נערך" if c.locale == "he" else "Edited"} {c.fill.clock()}')
    foot = (f'<div class="row t2" style="padding:6px 12px;gap:16px;background:{bg}">{C.icon("plus")}{C.icon("pen")}'
            f'<span class="grow" style="text-align:center;font-size:{C.fs(12)}" data-role="chrome">{e(edited)}</span>{C.icon("undo")}{C.icon("dots")}</div>')
    text = _cursor_text(c, c.ui["write_note"]) + after
    if c.has_field:   # one EditText: the typed text and the rest of the note; checklist/chips are separate views
        text = f'<div data-focus="1">{text}</div>'
    body = f'<div class="grow" style="padding:0 18px;overflow:hidden">{text}{tags}</div>'
    return f'<div class="col grow" style="background:{bg}">{head}{title}{body}{foot}</div>'


def _note_list(c: Ctx) -> str:
    cards = []
    titles = c.fill.pick(F.TASKS_RU + F.EVENTS_RU, F.TASKS_EN + F.EVENTS_EN, 8, pool_he=F.TASKS_HE + F.EVENTS_HE)
    lines = c.fill.msgs(8)
    for i in range(min(len(titles), len(lines))):
        bg = c.r.choice(NOTE_BG[c.theme])
        cards.append(f'<div style="border:1px solid var(--divider);border-radius:12px;padding:12px;margin:6px 8px;background:{bg}">'
                     f'<div style="font-weight:500" data-role="filler">{e(titles[i])}</div>'
                     f'<div class="t2" style="font-size:{C.fs(14)};margin-top:4px" data-role="filler">{e(lines[i])}</div></div>')
    search = (f'<div style="padding:10px 10px 4px"><div class="row" style="height:46px;border-radius:23px;background:var(--surface);padding:0 14px;gap:12px">'
              f'{C.icon("menu")}<span class="grow t2" data-role="chrome">{"Поиск по заметкам" if c.locale == "ru" else "חיפוש בהערות" if c.locale == "he" else "Search your notes"}</span></div></div>')
    return f'{search}<div class="grow" style="overflow:hidden">{"".join(cards)}</div>'


def notes(c: Ctx) -> str:
    bg = c.r.choice(NOTE_BG[c.theme]) if c.r.random() < 0.4 else NOTE_BG[c.theme][0]
    if not c.wide:
        return frame(c, _note_editor(c, bg))
    left = f'<div class="col" style="width:40%;border-right:1px solid var(--divider);overflow:hidden">{_note_list(c)}</div>'
    return frame(c, f'<div class="row grow" style="align-items:stretch;overflow:hidden">{left}{_note_editor(c, bg)}</div>')


def doc_editor(c: Ctx) -> str:
    doc = c.title or ("Без названия" if c.locale == "ru" else "מסמך ללא כותרת" if c.locale == "he" else "Untitled document")
    head = (f'<div class="appbar"><div class="iconbtn">{C.icon("check")}</div><div class="grow ell t2" style="font-size:{C.fs(15)}" data-role="title">{e(doc)}</div>'
            f'<div class="iconbtn">{C.icon("undo")}</div><div class="iconbtn">{C.icon("chat")}</div><div class="iconbtn">{C.icon("dots")}</div></div>')
    tool = ('<div class="row t2" style="gap:18px;padding:8px 14px;border-top:1px solid var(--divider);border-bottom:1px solid var(--divider)">'
            + C.icon("bold") + f'<span style="font-style:italic;font-weight:600" data-role="chrome">I</span>'
            + f'<span style="text-decoration:underline" data-role="chrome">U</span>' + C.icon("list") + C.icon("plus")
            + f'<span class="grow"></span><span data-role="chrome" style="font-size:{C.fs(13)}">{"Обычный текст" if c.locale == "ru" else "Normal text"}</span></div>')
    paras, comments = [], []
    for m in c.messages:
        if m.get("author") and not m.get("is_me") and c.r.random() < 0.5:
            comments.append(m)
        else:
            paras.append(m)
    ph = "".join(f'<p style="margin:0 0 12px;white-space:pre-wrap;word-break:break-word" data-role="message">{e(m["text"])}</p>' for m in paras)
    sn = "".join(f'<li style="margin:0 0 6px 20px" data-role="snippet">{e(s)}</li>' for s in c.snippets)
    sn = f'<ul style="margin-bottom:12px">{sn}</ul>' if sn else ""
    cur = _cursor_text(c, "Введите текст" if c.locale == "ru" else "הקלד כאן" if c.locale == "he" else "Type here")
    pagew = "92%" if not c.wide else "62%"
    pg = (f'<div style="background:var(--bg);margin:12px auto;width:{pagew};padding:22px 20px;box-shadow:0 1px 4px rgba(0,0,0,.2);min-height:90%">'
          f'<h1 style="font-size:{C.fs(24)};font-weight:400;margin-bottom:14px" data-role="title">{e(doc)}</h1>{ph}{sn}<p>{cur}</p></div>')
    cm = ""
    if comments:
        cm = '<div class="col" style="width:30%;padding:12px 8px;gap:10px">' + "".join(
            f'<div style="background:var(--bg);border-radius:8px;padding:10px;box-shadow:0 1px 3px rgba(0,0,0,.2)">'
            f'<div class="row" style="gap:8px">{C.avatar(m["author"], c.r, 26, 0)}<span style="font-weight:500;font-size:{C.fs(14)}" data-role="author">{e(m["author"])}</span></div>'
            f'<div style="font-size:{C.fs(14)};margin-top:6px" data-role="message">{e(m["text"])}</div></div>' for m in comments) + "</div>"
        if not c.wide:   # cover: comments as a bottom card
            cm = cm.replace('style="width:30%;', 'style="position:absolute;bottom:0;left:0;right:0;')
    focus = ' data-focus="1"' if c.has_field else ""   # the document view is one editable node (assumed, see README)
    canvas = (f'<div class="grow row" style="align-items:flex-start;background:var(--surface);overflow:hidden;position:relative">'
              f'<div class="grow"{focus}>{pg}</div>{cm}</div>')
    return frame(c, f'{head}{tool}{canvas}')


TEMPLATES = {"notes": notes, "doc_editor": doc_editor}
