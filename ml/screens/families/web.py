"""Browser look-alikes: browser_article, browser_search (results page), browser_pr (code-review page),
browser_docs (documentation page). The focused field is the address bar when spec.field.kind is
"search"/"address", otherwise a comment / search box on the page."""
from __future__ import annotations

import re

from .. import chrome as C
from ..base import Ctx, frame

e = C.e

TEMPLATE_APPS = {
    "browser_article": ("com.android.chrome", "Chrome", "search", "generic", "browser"),
    "browser_search": ("com.android.chrome", "Chrome", "search", "search", "browser"),
    "browser_pr": ("com.android.chrome", "Chrome", "code", "code", "browser"),
    "browser_docs": ("com.sec.android.app.sbrowser", "Samsung Internet", "search", "generic", "browser"),
}

DOMAINS_RU = ["news.example.ru", "habr-like.ru", "tech-daily.ru", "gazeta-online.ru", "blog.example.ru"]
DOMAINS_EN = ["news.example.com", "techreview.example.com", "dailywire.example.org", "blog.example.io"]
DOMAINS_HE = ["news.example.co.il", "tech.example.co.il", "blog.example.co.il", "daily.example.co.il"]


def _slug(s: str) -> str:
    t = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return t[:40] or "page"


def _urlbar(c: Ctx, url: str, focus: bool, bottom: bool) -> str:
    if focus:
        f = dict(c.field)
        hint = f.get("hint") or ("Введите запрос или URL" if c.locale == "ru" else "חיפוש או הקלדת כתובת" if c.locale == "he" else "Search or type URL")
        box = C.input_field(f, hint, c.r, (), ("mic",), "inputbox")
        return f'<div class="row" style="padding:8px 10px;gap:8px;background:var(--bar)">{box}</div>'
    tabs = c.r.randint(2, 40)
    return (f'<div class="row" style="padding:8px 10px;gap:10px;background:var(--bar);{"border-top" if bottom else "border-bottom"}:1px solid var(--divider)">'
            f'<span class="t2">{C.icon("home", 22)}</span><div class="row grow" style="height:40px;border-radius:20px;background:var(--surface);padding:0 14px;gap:8px">'
            f'<span class="t2">{C.icon("lock", 16)}</span><span class="grow ell" style="font-size:{C.fs(15)}" data-role="chrome">{e(url)}</span></div>'
            f'<span style="border:2px solid var(--text);border-radius:5px;min-width:24px;height:24px;font-size:12px;font-weight:700;'
            f'display:flex;align-items:center;justify-content:center" data-role="chrome">{tabs if tabs < 100 else ":D"}</span>'
            f'<span class="t2">{C.icon("dots", 22)}</span></div>')


def _browser(c: Ctx, url: str, content: str, page_field: str = "", page_focused: bool = False) -> str:
    """page_field: an input bar below the page; page_focused: the focused field is already inside `content`."""
    focus = c.has_field and not page_focused and (c.field.get("kind") in ("search", "address") or not page_field)
    bottom = c.r.random() < 0.3 and not focus
    bar = _urlbar(c, url, focus, bottom)
    pf = page_field if (c.has_field and not focus) else ""
    main = f'<div class="grow col" style="overflow:hidden;position:relative">{content}</div>{pf}'
    return frame(c, (main + bar) if bottom else (bar + main))


def _comment_box(c: Ctx, hint: str) -> str:
    return (f'<div class="inputbar" style="border-top:1px solid var(--divider)">'
            f'{C.input_field(c.field, hint, c.r, (), ("send",), "inputbox")}</div>')


def browser_article(c: Ctx) -> str:
    dom = c.r.choice(DOMAINS_RU if c.locale == "ru" else DOMAINS_HE if c.locale == "he" else DOMAINS_EN)
    title = c.title or (c.messages[0]["text"][:60] if c.messages else "")
    url = f"{dom}/{c.r.choice(['news', 'articles', 'blog', 'p'])}/{_slug(title)}"
    site = (f'<div class="row" style="padding:10px 16px;gap:10px;border-bottom:1px solid var(--divider)">{C.icon("menu")}'
            f'<span style="font-weight:800;font-size:{C.fs(18)};letter-spacing:.5px" data-role="chrome">{e(dom.split(".")[0].upper())}</span>'
            f'<span class="grow"></span>{C.icon("search")}</div>')
    by = (f'<div class="t2" style="font-size:{C.fs(13)};margin:6px 0 12px" data-role="chrome">'
          f'{e(c.fill.names(1)[0] or "")} · {e(c.fill.date_sep())} · {c.r.randint(2, 15)} {"мин" if c.locale == "ru" else "דק׳ קריאה" if c.locale == "he" else "min read"}</div>')
    paras = "".join(f'<p style="margin:0 0 14px;font-size:{C.fs(17)};line-height:1.5" data-role="message">{e(m["text"])}</p>' for m in c.messages)
    rel = ""
    if c.snippets:
        rel = (f'<div style="font-weight:700;margin:16px 0 8px" data-role="chrome">{"Читайте также" if c.locale == "ru" else "קרא עוד" if c.locale == "he" else "Read more"}</div>'
               + "".join(f'<div style="color:var(--accent);margin-bottom:8px" data-role="snippet">{e(s)}</div>' for s in c.snippets))
    img = (f'<div style="height:{c.r.randint(120, 200)}px;border-radius:6px;margin-bottom:12px;'
           f'background:linear-gradient({c.r.randint(0, 359)}deg,{c.r.choice(C.AV_COLORS)},{c.r.choice(C.AV_COLORS)})"></div>'
           if c.r.random() < 0.5 else "")
    art = (f'{site}<div style="padding:14px 16px;overflow:hidden"><h1 style="font-size:{C.fs(25)};line-height:1.25" data-role="title">{e(title)}</h1>'
           f'{by}{img}{paras}{rel}</div>')
    return _browser(c, url, art, _comment_box(c, "Ваш комментарий" if c.locale == "ru" else c.ui["comment"]))


def browser_search(c: Ctx) -> str:
    q = c.title or (c.field.get("before_cursor") or "")
    eng = c.r.choice(["search.example", "find.example", "poisk.example"])
    url = f"{eng}.com/search?q={_slug(q)}"
    qbox = (f'<div class="row" style="margin:10px 12px;height:46px;border-radius:23px;box-shadow:0 1px 4px rgba(0,0,0,.25);padding:0 16px;gap:10px;background:var(--bg)">'
            f'<span class="grow ell" data-role="title">{e(q)}</span><span class="t2">{C.icon("mic", 22)}</span><span style="color:var(--accent)">{C.icon("search", 22)}</span></div>')
    tabs = (["Все", "Картинки", "Новости", "Видео", "Карты"] if c.locale == "ru"
            else ["הכל", "תמונות", "חדשות", "סרטונים", "מפות"] if c.locale == "he"
            else ["All", "Images", "News", "Videos", "Maps"])
    tabbar = ('<div class="row" style="gap:18px;padding:4px 16px 0;border-bottom:1px solid var(--divider);font-size:' + C.fs(14) + '">'
              + "".join(f'<span style="padding:8px 0;{"border-bottom:3px solid var(--accent);color:var(--accent)" if i == 0 else ""}" '
                        f'class="{"" if i == 0 else "t2"}" data-role="chrome">{e(t)}</span>' for i, t in enumerate(tabs)) + "</div>")
    res = []
    for m in c.messages:
        site = m.get("author") or c.r.choice(DOMAINS_RU + DOMAINS_EN)
        dom = site if "." in site else (_slug(site) + c.r.choice([".com", ".ru", ".org"]))
        head = m["text"].split(". ")[0][:70]
        res.append(f'<div style="padding:12px 16px;border-bottom:6px solid var(--surface)"><div class="row" style="gap:10px">'
                   f'{C.avatar(site, c.r, 28, 0)}<div class="col"><span style="font-size:{C.fs(14)}" data-role="author">{e(site)}</span>'
                   f'<span class="t3 ell" style="font-size:{C.fs(12)}" data-role="chrome">https://{e(dom)} › …</span></div></div>'
                   f'<div style="color:#1a5fd0;font-size:{C.fs(18)};margin:6px 0 4px" data-role="filler">{e(head)}</div>'
                   f'<div class="t2" style="font-size:{C.fs(15)}" data-role="message">{e(m["text"])}</div></div>')
    if c.snippets:
        res.append(f'<div style="padding:12px 16px"><div style="font-weight:500;margin-bottom:6px" data-role="chrome">'
                   f'{"Похожие запросы" if c.locale == "ru" else "אנשים שואלים גם" if c.locale == "he" else "People also ask"}</div>'
                   + "".join(f'<div class="row" style="padding:10px 0;border-top:1px solid var(--divider)"><span class="grow" data-role="snippet">{e(s)}</span>'
                             '</div>' for s in c.snippets) + "</div>")
    focus_on_page = c.has_field and c.field.get("kind") not in ("search", "address")
    if focus_on_page:   # the query box on the page is the focused field
        qbox = (f'<div style="margin:10px 12px">{C.input_field(c.field, q or ("Поиск" if c.locale == "ru" else "חיפוש" if c.locale == "he" else "Search"), c.r, (), ("mic",), "inputbox")}</div>')
    return _browser(c, url, f'{qbox}{tabbar}<div class="grow" style="overflow:hidden">{"".join(res)}</div>',
                    page_focused=focus_on_page)


def browser_pr(c: Ctx) -> str:
    org = c.r.choice(["acme", "murmur-dev", "platform-team", "open-tools", "infra"])
    repo = c.r.choice(["api", "web", "mobile", "backend", "core", "infra"])
    num = c.r.randint(12, 4999)
    url = f"code.example.com/{org}/{repo}/pull/{num}"
    title = c.title or "Update"
    st_open = c.r.random() < 0.75
    head = (f'<div style="padding:12px 16px"><div class="t2" style="font-size:{C.fs(14)}" data-role="chrome">{e(org)} / <b>{e(repo)}</b></div>'
            f'<div style="font-size:{C.fs(22)};margin-top:8px;line-height:1.3"><span data-role="title">{e(title)}</span> '
            f'<span class="t3" data-role="chrome">#{num}</span></div>'
            f'<div class="row" style="gap:8px;margin-top:10px;flex-wrap:wrap"><span class="chip" style="background:{"#1f883d" if st_open else "#8250df"};'
            f'color:#fff;border:none;border-radius:16px" data-role="chrome">{C.icon("merge", 16)}&nbsp;{"Open" if st_open else "Merged"}</span>'
            f'<span class="t2" style="font-size:{C.fs(14)}" data-role="chrome">{e(c.fill.names(1)[0] or "dev")} wants to merge {c.r.randint(1, 12)} commits into main</span></div></div>')
    tabs = ["Conversation", "Commits", "Checks", "Files changed"]
    tabbar = ('<div class="row" style="gap:16px;padding:0 16px;border-bottom:1px solid var(--divider);font-size:' + C.fs(14) + ';overflow:hidden">'
              + "".join(f'<span style="padding:10px 0;white-space:nowrap;{"border-bottom:2px solid #fd8c73;font-weight:600" if i == 0 else ""}" data-role="chrome">'
                        f'{e(t)} <span class="badge" style="background:var(--surface2);color:var(--text)">{c.r.randint(1, 30)}</span></span>'
                        for i, t in enumerate(tabs)) + "</div>")
    times = c.msg_times()
    cm = []
    for i, m in enumerate(c.messages):
        a = m.get("author") or c.ui["you"]
        cm.append(f'<div style="margin:12px 16px;border:1px solid var(--divider);border-radius:8px;overflow:hidden">'
                  f'<div class="row" style="gap:8px;padding:8px 12px;background:var(--surface);font-size:{C.fs(14)}">{C.avatar(a, c.r, 24, 0.5)}'
                  f'<b data-role="author">{e(a)}</b><span class="t2" data-role="time">commented {e(times[i])}</span></div>'
                  f'<div style="padding:10px 12px;white-space:pre-wrap;word-break:break-word" data-role="message">{e(m["text"])}</div></div>')
    for s in c.snippets:
        cm.append(f'<div class="row t2" style="gap:10px;margin:6px 16px 6px 28px;font-size:{C.fs(14)}">{C.icon("code", 18)}'
                  f'<span style="font-family:{C.MONO["roboto_mono"]}" data-role="snippet">{e(s)}</span></div>')
    box = _comment_box(c, "Leave a comment" if c.r.random() < 0.7 else c.ui["comment"])
    return _browser(c, url, f'{head}{tabbar}<div class="grow" style="overflow:hidden">{"".join(cm)}</div>', box)


def browser_docs(c: Ctx) -> str:
    prod = c.r.choice(["docs", "developer", "learn", "help"])
    title = c.title or "Getting started"
    url = f"{prod}.example.dev/guide/{_slug(title)}"
    nav = ""
    if c.wide:
        items = (["Введение", "Установка", "Настройка", "Справочник API", "Примеры", "FAQ"] if c.locale == "ru"
                 else ["Introduction", "Installation", "Configuration", "API reference", "Examples", "FAQ"])
        nav = ('<div class="col" style="width:26%;padding:14px 10px;gap:10px;border-right:1px solid var(--divider);font-size:' + C.fs(14) + '">'
               + "".join(f'<div class="t2" data-role="chrome">{e(t)}</div>' for t in items) + "</div>")
    paras = []
    for m in c.messages:
        if m.get("author") and c.r.random() < 0.4:
            paras.append(f'<h2 style="font-size:{C.fs(19)};margin:16px 0 8px" data-role="author">{e(m["author"])}</h2>')
        paras.append(f'<p style="margin:0 0 12px;line-height:1.5" data-role="message">{e(m["text"])}</p>')
    for s in c.snippets:
        paras.append(f'<pre style="background:var(--surface);padding:10px 12px;border-radius:6px;margin:0 0 12px;white-space:pre-wrap;'
                     f'font-family:{C.MONO["jetbrains"]};font-size:{C.fs(13)}" data-role="snippet">{e(s)}</pre>')
    top = (f'<div class="row" style="padding:10px 16px;gap:10px;border-bottom:1px solid var(--divider)">{C.icon("menu")}'
           f'<b data-role="chrome">{e(prod.capitalize())}</b><span class="grow"></span>{C.icon("search")}</div>')
    crumbs = f'<div class="t3" style="font-size:{C.fs(13)};margin-bottom:6px" data-role="chrome">{"Руководство" if c.locale == "ru" else "Guide"} › {e(title[:30])}</div>'
    body = (f'{top}<div class="row grow" style="align-items:flex-start;overflow:hidden">{nav}<div class="grow" style="padding:14px 16px">'
            f'{crumbs}<h1 style="font-size:{C.fs(26)};margin-bottom:12px" data-role="title">{e(title)}</h1>{"".join(paras)}</div></div>')
    return _browser(c, url, body, _comment_box(c, "Поиск по документации" if c.locale == "ru" else "Search docs"))


TEMPLATES = {"browser_article": browser_article, "browser_search": browser_search, "browser_pr": browser_pr,
             "browser_docs": browser_docs}
