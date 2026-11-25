"""Developer look-alikes: terminal (terminal emulator with an extra-keys row), code_editor (mobile code
editor with line numbers and tabs), issue_tracker (issue page with fields and comments)."""
from __future__ import annotations

import re

from .. import chrome as C
from ..base import Ctx, frame

e = C.e

TEMPLATE_APPS = {
    "terminal": ("com.termux", "Termux", "code", "code", "terminal"),
    "code_editor": ("com.example.codeeditor", "Code Editor", "code", "code", "code_editor"),
    "issue_tracker": ("com.atlassian.android.jira.core", "Jira", "work_chat", "message", "issue_tracker"),
}

TERM = {"light": ("#fdfdf8", "#1d1f21", "#2e7d32", "#1565c0"), "dark": ("#000000", "#e8e8e8", "#7ee787", "#79c0ff")}


def terminal(c: Ctx) -> str:
    bg, fg, green, blue = TERM[c.theme]
    mono = C.MONO[c.r.choice(list(C.MONO))]
    user = c.r.choice(["u0_a312", "dev", "alex", "me"])
    host = c.r.choice(["localhost", "pixel", "fold", "build-01"])
    cwd = c.r.choice(["~", "~/projects/app", "~/work/api", "~/src"])
    prompt = lambda: (f'<span style="color:{green}" data-role="chrome">{user}@{host}</span>:<span style="color:{blue}" data-role="chrome">{e(cwd)}</span>$ ')
    lines = []
    if c.r.random() < 0.5:
        lines.append(f'<div data-role="filler">Welcome to the terminal!</div><div>&nbsp;</div>')
    for m in c.messages:
        if m.get("is_me"):
            lines.append(f'<div>{prompt()}<span data-role="message">{e(m["text"])}</span></div>')
        else:
            for ln in m["text"].split("\n"):
                lines.append(f'<div data-role="message">{e(ln) or "&nbsp;"}</div>')
    for s in c.snippets:
        lines.append(f'<div data-role="snippet">{e(s)}</div>')
    bc = c.field.get("before_cursor") or ""
    cur = f'<span style="display:inline-block;width:.6em;height:1.15em;background:{fg};vertical-align:text-bottom"></span>'
    if c.has_field:
        lines.append(f'<div>{prompt()}<span data-role="field">{e(bc)}</span>{cur}</div>')
    title = ""
    if c.title:
        title = (f'<div class="row" style="padding:6px 12px;gap:8px;background:{"#e8e8e2" if c.theme == "light" else "#1b1b1b"};font-size:{C.fs(13)}">'
                 f'{C.icon("terminal", 18)}<span class="ell grow" data-role="title">{e(c.title)}</span>'
                 f'<span class="t3" data-role="chrome">[{c.r.randint(1, 3)}]</span></div>')
    fsz = c.r.choice([12, 13, 14])
    focus = ' data-focus="1"' if c.has_field else ""   # the terminal view is the IME target
    scr = (f'<div class="grow col"{focus} style="justify-content:flex-end;overflow:hidden;padding:4px 6px;background:{bg};color:{fg};'
           f'font-family:{mono};font-size:{C.fs(fsz)};line-height:1.3;white-space:pre-wrap;word-break:break-all">{"".join(lines)}</div>')
    keys = ["ESC", "/", "-", "HOME", "↑", "END", "PGUP"], ["TAB", "CTRL", "ALT", "←", "↓", "→", "PGDN"]
    extra = "".join(f'<div class="row" style="justify-content:space-around;height:36px;background:{bg};color:{fg};font-family:{mono};font-size:13px">'
                    + "".join(f'<span data-role="chrome">{e(k)}</span>' for k in row) + "</div>" for row in keys)
    return frame(c, f'{title}{scr}{extra if c.keyboard else ""}', status_style=f"--sbbg:{bg};--sbfg:{fg}",
                 extra_css=f"body{{background:{bg}}}")


KW = r"\b(def|class|return|import|from|if|else|for|while|const|let|function|val|fun|package|public|private|func|async|await|try|except)\b"


def _hl(line: str, pal: dict) -> str:
    s = e(line)
    if line.strip().startswith(("#", "//")):
        return f'<span style="color:{pal["cm"]}">{s}</span>'
    s = re.sub(r"(&quot;.*?&quot;|&#x27;.*?&#x27;)", lambda m: f'<span style="color:{pal["str"]}">{m.group(1)}</span>', s)
    s = re.sub(KW, lambda m: f'<span style="color:{pal["kw"]}">{m.group(1)}</span>', s)
    return s


CODE_PAL = {"light": {"bg": "#ffffff", "gut": "#f3f3f3", "num": "#9e9e9e", "kw": "#0033b3", "str": "#067d17", "cm": "#8c8c8c"},
            "dark": {"bg": "#1e1f22", "gut": "#1e1f22", "num": "#5c6370", "kw": "#cf8e6d", "str": "#6aab73", "cm": "#7a7e85"}}


def code_editor(c: Ctx) -> str:
    pal = CODE_PAL[c.theme]
    mono = C.MONO[c.r.choice(list(C.MONO))]
    fname = c.title or c.r.choice(["main.py", "App.kt", "index.ts", "server.go", "README.md"])
    others = [f for f in ["utils.py", "config.yaml", "Dockerfile", "build.gradle.kts", "api.ts"] if f != fname]
    tabs = ('<div class="row" style="overflow:hidden;font-size:' + C.fs(13) + ';border-bottom:1px solid var(--divider)">'
            + f'<span style="padding:9px 14px;border-bottom:2px solid var(--accent);white-space:nowrap" data-role="title">{e(fname)}</span>'
            + "".join(f'<span class="t2" style="padding:9px 14px;white-space:nowrap" data-role="chrome">{e(o)}</span>' for o in c.r.sample(others, 2))
            + "</div>")
    head = (f'<div class="appbar" style="min-height:50px">{C.icon("menu")}<span class="grow ell" style="font-size:{C.fs(16)}" data-role="chrome">'
            f'{e(c.r.choice(["my-app", "backend", "scripts", "notes"]))}</span>{C.icon("undo")}{C.icon("search")}{C.icon("dots")}</div>')
    src = []
    for m in c.messages:
        src.extend(m["text"].split("\n"))
        src.append("")
    for s in c.snippets:
        src.append(("# " if fname.endswith((".py", ".yaml")) else "// ") + s)
    bc = c.field.get("before_cursor") or ""
    start = c.r.randint(1, 120)
    rows = []
    for i, ln in enumerate(src):
        role = "snippet" if ln.startswith(("# ", "// ")) and ln[3:] in c.snippets else "message"
        rows.append(f'<div class="row" style="align-items:flex-start"><span style="width:3.2em;text-align:right;padding-right:10px;color:{pal["num"]};flex:none" '
                    f'data-role="chrome">{start + i}</span><span style="white-space:pre-wrap;word-break:break-all" data-role="{role}">{_hl(ln, pal) or "&nbsp;"}</span></div>')
    if c.has_field:
        rows.append(f'<div class="row" style="align-items:flex-start;background:rgba(127,127,127,.12)"><span style="width:3.2em;text-align:right;padding-right:10px;'
                    f'color:{pal["num"]};flex:none" data-role="chrome">{start + len(src)}</span><span style="white-space:pre-wrap" data-role="field">{e(bc)}</span>'
                    f'<span class="cursor"></span></div>')
    focus = ' data-focus="1"' if c.has_field else ""   # the editor view is one EditText
    ed = (f'<div class="grow"{focus} style="overflow:hidden;background:{pal["bg"]};font-family:{mono};font-size:{C.fs(c.r.choice([12, 13, 14]))};'
          f'line-height:1.45;padding-top:6px">{"".join(rows)}</div>')
    symbols = ('<div class="row" style="justify-content:space-around;height:34px;border-top:1px solid var(--divider);font-family:' + mono + '">'
               + "".join(f'<span data-role="chrome">{e(k)}</span>' for k in ["→", "{", "}", "(", ")", ";", "=", "\"", ":", "/"]) + "</div>")
    return frame(c, f'{head}{tabs}{ed}{symbols if c.keyboard else ""}')


def issue_tracker(c: Ctx) -> str:
    key = f'{c.r.choice(["MUR", "APP", "OPS", "WEB", "DATA"])}-{c.r.randint(10, 2999)}'
    title = c.title or "Bug"
    st = c.r.choice([c.ui["todo"], c.ui["in_progress"], c.ui["done"]])
    stc = {c.ui["todo"]: "#dfe1e6", c.ui["in_progress"]: "#cce0ff", c.ui["done"]: "#baf3db"}[st]
    head = (f'<div class="appbar"><div class="iconbtn">{C.icon("back")}</div><span class="grow t2" style="font-size:{C.fs(14)}" data-role="chrome">'
            f'{C.icon("flag", 16)} {key}</span><div class="iconbtn">{C.icon("bell")}</div><div class="iconbtn">{C.icon("dots")}</div></div>')
    desc_m = c.messages[0] if c.messages else None
    comments = c.messages[1:] if c.messages else []
    assignee = (c.messages[0].get("author") if c.messages and c.messages[0].get("author") else c.fill.names(1)[0]) or "—"
    fields = [(c.ui["status"], f'<span class="chip" style="background:{stc};color:#172b4d;border:none;font-weight:600" data-role="chrome">{e(st)}</span>'),
              (c.ui["assignee"], f'<span class="row" style="gap:6px">{C.avatar(assignee, c.r, 24, 0)}<span data-role="author">{e(assignee)}</span></span>'),
              (c.ui["priority"], f'<span data-role="chrome">{c.r.choice(["Высокий", "Средний", "Низкий"] if c.locale == "ru" else ["גבוהה", "בינונית", "נמוכה"] if c.locale == "he" else ["High", "Medium", "Low"])}</span>')]
    if c.snippets:
        fields.append((c.ui["labels"], " ".join(f'<span class="chip" style="height:24px" data-role="snippet">{e(s)}</span>' for s in c.snippets)))
    ftab = "".join(f'<div class="row" style="padding:6px 16px;gap:12px"><span class="t2" style="width:34%;font-size:{C.fs(14)}" data-role="chrome">{e(k)}</span>'
                   f'<span class="grow" style="font-size:{C.fs(14)}">{v}</span></div>' for k, v in fields)
    desc = ""
    if desc_m:
        desc = (f'<div style="padding:10px 16px 4px;font-weight:600" data-role="chrome">{e(c.ui["description"])}</div>'
                f'<div style="padding:0 16px 8px;white-space:pre-wrap;word-break:break-word" data-role="message">{e(desc_m["text"])}</div>')
    times = c.msg_times()
    cms = "".join(
        f'<div class="row" style="gap:10px;padding:8px 16px;align-items:flex-start">{C.avatar(m.get("author") or c.ui["you"], c.r, 32, 0.3)}'
        f'<div class="grow col"><div class="row" style="gap:8px"><b style="font-size:{C.fs(14)}" data-role="author">{e(m.get("author") or c.ui["you"])}</b>'
        f'<span class="t3" style="font-size:{C.fs(12)}" data-role="time">{e(times[i + 1])}</span></div>'
        f'<div style="white-space:pre-wrap;word-break:break-word" data-role="message">{e(m["text"])}</div></div></div>'
        for i, m in enumerate(comments))
    ctitle = f'<div style="padding:10px 16px 2px;font-weight:600" data-role="chrome">{e(c.ui["comments"])}</div>' if comments else ""
    body = (f'{head}<div class="grow" style="overflow:hidden"><div style="padding:4px 16px 8px;font-size:{C.fs(21)};font-weight:500" data-role="title">'
            f'{e(title)}</div>{ftab}{desc}{ctitle}{cms}</div>')
    if c.has_field:
        body += (f'<div class="inputbar" style="border-top:1px solid var(--divider)">'
                 f'{C.input_field(c.field, c.ui["comment"], c.r, (), ("send",), "inputbox")}</div>')
    return frame(c, body)


TEMPLATES = {"terminal": terminal, "code_editor": code_editor, "issue_tracker": issue_tracker}
