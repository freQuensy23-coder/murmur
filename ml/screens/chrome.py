"""Shared Android chrome for the templates: devices, themes, status bar, app bar, input bar, keyboard,
gesture bar, Murmur's bubble, icons and avatars. Generic look-alikes only: no logos or brand assets.

Units: CSS px == Android dp (the device scale factor maps dp to screenshot pixels). Text sizes use
`fs(n)` == n sp times the spec's font scale.
"""
from __future__ import annotations

import html
import random

# Display metrics. Pixel sizes: docs/research/android-platform.md (GSMArena / Google Fi spec pages).
# Density is ASSUMED (no Fold7 measurement yet): ~2.63 dp->px, chosen so that both CSS viewports are integers.
DEVICES = {
    "fold7_cover": {"w": 1080, "h": 2520, "css_w": 411, "css_h": 959},   # 2.628 px/dp
    "fold7_inner": {"w": 1968, "h": 2184, "css_w": 738, "css_h": 819},   # 2.667 px/dp
}

FONTS = {
    "roboto": "'Roboto Var', 'Noto Sans Hebrew', 'Rubik', 'Heebo', 'Arial', 'Noto Color Emoji'",
    "noto": "'Noto Sans Var', 'Noto Sans Hebrew', 'Rubik', 'Heebo', 'Arial', 'Noto Color Emoji'",
    "hebrew": "'Noto Sans Hebrew', 'Rubik', 'Heebo', 'Arial', 'Roboto Var', 'Noto Color Emoji'",
}
MONO = {"roboto_mono": "'Roboto Mono Var', 'Noto Sans Hebrew', 'Arial', 'Noto Color Emoji'",
        "jetbrains": "'JetBrains Mono Var', 'Noto Sans Hebrew', 'Arial', 'Noto Color Emoji'"}

e = html.escape


def fs(n: float) -> str:
    return f"calc(var(--fs) * {n}px)"


# ----------------------------------------------------------------------------- themes

THEMES = {
    "light": {"bg": "#ffffff", "surface": "#f3f4f6", "surface2": "#e8eaed", "text": "#1b1c1f", "text2": "#5f6368",
              "text3": "#80868b", "divider": "#e0e0e0", "bar": "#ffffff", "kb": "#e8eaed", "key": "#ffffff",
              "key2": "#c9ccd1", "keytext": "#1b1c1f", "nav": "#f6f7f9"},
    "dark": {"bg": "#121316", "surface": "#1e1f23", "surface2": "#2a2b30", "text": "#e6e7ea", "text2": "#a5a9b0",
             "text3": "#80858c", "divider": "#2e3035", "bar": "#17181b", "kb": "#1c1d20", "key": "#3a3b3f",
             "key2": "#2a2b2e", "keytext": "#e8e8ea", "nav": "#1a1b1e"},
}
ACCENTS = ["#2a8bd6", "#1d9e75", "#6750a4", "#e8710a", "#d93f63", "#0b8043", "#3367d6", "#8e44ad"]


def css_base(theme: str, accent: str, font: str, font_scale: float, extra: str = "") -> str:
    t = THEMES[theme]
    vars_ = ";".join(f"--{k}:{v}" for k, v in t.items())
    return f"""
@font-face{{font-family:'Roboto Var';src:url('fonts/Roboto[wdth,wght].ttf');font-weight:100 900}}
@font-face{{font-family:'Noto Sans Var';src:url('fonts/NotoSans[wdth,wght].ttf');font-weight:100 900}}
@font-face{{font-family:'Roboto Mono Var';src:url('fonts/RobotoMono[wght].ttf');font-weight:100 900}}
@font-face{{font-family:'JetBrains Mono Var';src:url('fonts/JetBrainsMono[wght].ttf');font-weight:100 900}}
@font-face{{font-family:'Noto Sans Hebrew';src:url('fonts/NotoSansHebrew[wdth,wght].ttf'),url('fonts/NotoSansHebrew.ttf'),url('ocr_fonts/NotoSansHebrew.ttf'),local('Noto Sans Hebrew'),local('Arial Hebrew');font-weight:100 900}}
@font-face{{font-family:'Rubik';src:url('fonts/Rubik.ttf'),url('ocr_fonts/Rubik.ttf'),local('Rubik');font-weight:100 900}}
@font-face{{font-family:'Heebo';src:url('fonts/Heebo.ttf'),url('ocr_fonts/Heebo.ttf'),local('Heebo');font-weight:100 900}}
@font-face{{font-family:'Assistant';src:url('fonts/Assistant.ttf'),url('ocr_fonts/Assistant.ttf'),local('Assistant');font-weight:100 900}}
@font-face{{font-family:'Noto Color Emoji';src:url('fonts/NotoColorEmoji-Regular.ttf')}}
:root{{{vars_};--accent:{accent};--fs:{font_scale}}}
*{{box-sizing:border-box;margin:0;padding:0;animation:none!important;transition:none!important;caret-color:transparent}}
html,body{{width:100%;height:100%;overflow:hidden;background:var(--bg);color:var(--text);
  font-family:{FONTS.get(font, FONTS['roboto'])};font-size:{fs(16)};line-height:1.35;-webkit-font-smoothing:antialiased}}
#screen{{position:absolute;inset:0;display:flex;flex-direction:column;overflow:hidden}}
.row{{display:flex;align-items:center}} .col{{display:flex;flex-direction:column}} .grow{{flex:1;min-width:0;min-height:0}}
.ell{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.ico{{width:24px;height:24px;flex:none;fill:currentColor}}
.t2{{color:var(--text2)}} .t3{{color:var(--text3)}}
.statusbar{{height:30px;flex:none;padding:0 18px;display:flex;align-items:center;font-size:13px;font-weight:500;
  background:var(--sbbg,var(--bar));color:var(--sbfg,var(--text))}}
.statusbar .sp{{flex:1}} .statusbar .ico{{width:15px;height:15px;margin-left:4px}}
.appbar{{flex:none;min-height:56px;display:flex;align-items:center;padding:0 8px 0 4px;gap:8px;background:var(--abbg,var(--bar));color:var(--abfg,var(--text))}}
.appbar .ttl{{font-size:{fs(20)};font-weight:500}}
.appbar .sub{{font-size:{fs(13)};opacity:.75}}
.iconbtn{{width:44px;height:44px;display:flex;align-items:center;justify-content:center;flex:none}}
.avatar{{border-radius:50%;flex:none;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:500;overflow:hidden}}
.badge{{min-width:20px;height:20px;border-radius:10px;padding:0 6px;background:var(--accent);color:#fff;font-size:{fs(12)};
  display:inline-flex;align-items:center;justify-content:center;font-weight:500}}
.divider{{height:1px;background:var(--divider);flex:none}}
.inputbar{{flex:none;display:flex;align-items:flex-end;gap:6px;padding:6px 8px;background:var(--ibbg,var(--bar))}}
.inputbox{{flex:1;min-width:0;min-height:42px;border-radius:21px;background:var(--surface);padding:10px 14px;font-size:{fs(16)};
  display:flex;align-items:center;max-height:150px;overflow:hidden}}
.inputbox .txt{{white-space:pre-wrap;word-break:break-word}}
.hint{{color:var(--text3)}}
.cursor{{display:inline-block;width:2px;height:1.2em;background:var(--accent);vertical-align:text-bottom;margin-left:1px}}
.keyboard{{flex:none;background:var(--kb);padding:4px 3px 2px;color:var(--keytext)}}
.kb-sugg{{height:40px;display:flex;align-items:center;font-size:{fs(15)};color:var(--keytext)}}
.kb-sugg div{{flex:1;text-align:center;border-right:1px solid var(--key2)}} .kb-sugg div:last-child{{border:none}}
.kb-row{{display:flex;justify-content:center;gap:5px;margin:5px 0}}
.key{{flex:1;height:44px;border-radius:7px;background:var(--key);display:flex;align-items:center;justify-content:center;
  font-size:21px;box-shadow:0 1px 0 rgba(0,0,0,.25);max-width:44px;position:relative}}
.key sup{{position:absolute;top:2px;right:4px;font-size:9px;opacity:.6}}
.key.w{{max-width:none;flex:1.5;background:var(--key2);font-size:14px}} .key.space{{max-width:none;flex:5;font-size:13px}}
.gesture{{height:22px;flex:none;display:flex;align-items:center;justify-content:center;background:var(--gbbg,var(--bar))}}
.gesture div{{width:110px;height:4px;border-radius:2px;background:var(--text);opacity:.55}}
.navbar{{flex:none;height:72px;display:flex;background:var(--nav);border-top:1px solid var(--divider)}}
.navbar>div{{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:{fs(12)};gap:3px;color:var(--text2)}}
.navbar>div.on{{color:var(--text);font-weight:500}}
.navbar>div.on .pill{{background:color-mix(in srgb,var(--accent) 25%,transparent)}}
.navbar .pill{{width:56px;height:30px;border-radius:15px;display:flex;align-items:center;justify-content:center}}
.bubble{{position:absolute;width:52px;height:52px;border-radius:26px;background:#5b3df5;display:flex;align-items:center;
  justify-content:center;box-shadow:0 3px 10px rgba(0,0,0,.35);z-index:50;color:#fff}}
.bubble.pill{{width:auto;padding:0 16px 0 12px;gap:6px;font-size:14px;font-weight:500}}
.fab{{position:absolute;right:16px;width:56px;height:56px;border-radius:16px;background:color-mix(in srgb,var(--accent) 30%,var(--bg));
  display:flex;align-items:center;justify-content:center;z-index:5;box-shadow:0 2px 6px rgba(0,0,0,.25)}}
.chip{{display:inline-flex;align-items:center;height:30px;padding:0 12px;border-radius:8px;border:1px solid var(--divider);font-size:{fs(14)};white-space:nowrap}}
[dir="rtl"] .statusbar{{direction:ltr}}
[dir="rtl"] .fab{{right:auto;left:16px}}
[dir="rtl"] .cursor{{margin-left:0;margin-right:1px}}
[dir="rtl"] .appbar{{padding:0 4px 0 8px}}
{extra}
"""


# ----------------------------------------------------------------------------- icons (simple generic shapes)

_P = {
    "back": "M20 11H7.8l5.6-5.6L12 4l-8 8 8 8 1.4-1.4L7.8 13H20z",
    "menu": "M3 6h18v2H3zm0 5h18v2H3zm0 5h18v2H3z",
    "search": "M15.5 14h-.8l-.3-.3A6.5 6.5 0 1 0 9.5 16a6.5 6.5 0 0 0 4.2-1.6l.3.3v.8l5 5 1.5-1.5zm-6 0A4.5 4.5 0 1 1 14 9.5 4.5 4.5 0 0 1 9.5 14z",
    "dots": "M12 8a2 2 0 1 0-2-2 2 2 0 0 0 2 2zm0 2a2 2 0 1 0 2 2 2 2 0 0 0-2-2zm0 6a2 2 0 1 0 2 2 2 2 0 0 0-2-2z",
    "mic": "M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.9V21h2v-3.1A7 7 0 0 0 19 11z",
    "send": "M2 21l21-9L2 3v7l15 2-15 2z",
    "clip": "M16.5 6v11.5a4 4 0 0 1-8 0V5a2.5 2.5 0 0 1 5 0v10.5a1 1 0 0 1-2 0V6H10v9.5a2.5 2.5 0 0 0 5 0V5a4 4 0 0 0-8 0v12.5a5.5 5.5 0 0 0 11 0V6z",
    "smile": "M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8zm-3.5-9A1.5 1.5 0 1 0 7 9.5 1.5 1.5 0 0 0 8.5 11zm7 0A1.5 1.5 0 1 0 14 9.5a1.5 1.5 0 0 0 1.5 1.5zM12 17.5a5.5 5.5 0 0 0 5.1-3.5H6.9a5.5 5.5 0 0 0 5.1 3.5z",
    "phone": "M6.6 10.8a15 15 0 0 0 6.6 6.6l2.2-2.2a1 1 0 0 1 1-.2 11.4 11.4 0 0 0 3.6.6 1 1 0 0 1 1 1V20a1 1 0 0 1-1 1A17 17 0 0 1 3 4a1 1 0 0 1 1-1h3.5a1 1 0 0 1 1 1 11.4 11.4 0 0 0 .6 3.6 1 1 0 0 1-.3 1z",
    "video": "M17 10.5V7a1 1 0 0 0-1-1H4a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-3.5l4 4v-11z",
    "pen": "M3 17.2V21h3.8l11-11-3.8-3.8zM20.7 7a1 1 0 0 0 0-1.4l-2.3-2.3a1 1 0 0 0-1.4 0l-1.8 1.8 3.8 3.8z",
    "star": "M12 17.3l6.2 3.7-1.6-7 5.4-4.7-7.2-.6L12 2 9.2 8.7 2 9.3l5.4 4.7-1.6 7z",
    "home": "M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z",
    "chat": "M20 2H4a2 2 0 0 0-2 2v18l4-4h14a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2z",
    "people": "M16 11a3 3 0 1 0-3-3 3 3 0 0 0 3 3zm-8 0a3 3 0 1 0-3-3 3 3 0 0 0 3 3zm0 2c-2.3 0-7 1.2-7 3.5V19h14v-2.5C15 14.2 10.3 13 8 13zm8 0h-1a4.2 4.2 0 0 1 2 3.5V19h6v-2.5c0-2.3-4.7-3.5-7-3.5z",
    "gear": "M19.4 13a7.5 7.5 0 0 0 0-2l2.1-1.6-2-3.5-2.5 1a7.4 7.4 0 0 0-1.7-1L15 3h-4l-.4 2.7a7.4 7.4 0 0 0-1.7 1l-2.5-1-2 3.5L6.6 11a7.5 7.5 0 0 0 0 2l-2.1 1.6 2 3.5 2.5-1a7.4 7.4 0 0 0 1.7 1L11 21h4l.4-2.7a7.4 7.4 0 0 0 1.7-1l2.5 1 2-3.5zM13 15.5a3.5 3.5 0 1 1 3.5-3.5 3.5 3.5 0 0 1-3.5 3.5z",
    "inbox": "M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2zm0 12h-4a3 3 0 0 1-6 0H5V5h14z",
    "check": "M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z",
    "dcheck": "M18 7l-1.4-1.4-6.3 6.3 1.4 1.4zm4.2-1.4L11.7 16.2 7.5 12l-1.4 1.4 5.6 5.6 12-12zM.4 13.4 6 19l1.4-1.4L1.8 12z",
    "plus": "M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6z",
    "reply": "M10 9V5l-7 7 7 7v-4.1c5 0 8.5 1.6 11 5.1-1-5-4-10-11-11z",
    "folder": "M10 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-8z",
    "bell": "M12 22a2 2 0 0 0 2-2h-4a2 2 0 0 0 2 2zm6-6v-5a6 6 0 0 0-5-5.9V4a1 1 0 0 0-2 0v1.1A6 6 0 0 0 6 11v5l-2 2v1h16v-1z",
    "cal": "M19 4h-1V2h-2v2H8V2H6v2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2zm0 16H5V9h14z",
    "lock": "M18 8h-1V6A5 5 0 0 0 7 6v2H6a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V10a2 2 0 0 0-2-2zm-6 9a2 2 0 1 1 2-2 2 2 0 0 1-2 2zm3.1-9H8.9V6a3.1 3.1 0 0 1 6.2 0z",
    "tabs": "M4 4h12v12H4zm4 4v12h12V8h-2v10H10V8z",
    "code": "M9.4 16.6 4.8 12l4.6-4.6L8 6l-6 6 6 6zm5.2 0 4.6-4.6-4.6-4.6L16 6l6 6-6 6z",
    "bolt": "M11 21h-1l1-7H7.5c-.6 0-.6-.3-.4-.7L13 3h1l-1 7h3.5c.5 0 .6.3.4.7z",
    "sig": "M2 22h20V2z",
    "wifi": "M12 21l3.6-4.5a6 6 0 0 0-7.2 0zm0-18A17 17 0 0 0 1 7l2 2.5a14 14 0 0 1 18 0L23 7a17 17 0 0 0-11-4zm0 6a11 11 0 0 0-7 2.5l2 2.5a8 8 0 0 1 10 0l2-2.5A11 11 0 0 0 12 9z",
    "bksp": "M22 3H7c-.7 0-1.2.3-1.6.9L0 12l5.4 8.1c.4.5.9.9 1.6.9h15a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2zm-3 12.6L17.6 17 14 13.4 10.4 17 9 15.6l3.6-3.6L9 8.4 10.4 7l3.6 3.6L17.6 7 19 8.4 15.4 12z",
    "enter": "M19 7v4H5.8l3.6-3.6L8 6l-6 6 6 6 1.4-1.4L5.8 13H21V7z",
    "bat": "M15.7 4H14V2h-4v2H8.3A1.3 1.3 0 0 0 7 5.3v15.4A1.3 1.3 0 0 0 8.3 22h7.4a1.3 1.3 0 0 0 1.3-1.3V5.3A1.3 1.3 0 0 0 15.7 4z",
    "attach2": "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zm-1 7V3.5L18.5 9z",
    "camera": "M9 2 7.2 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-3.2L15 2zm3 15a5 5 0 1 1 5-5 5 5 0 0 1-5 5z",
    "refresh": "M17.6 6.4A8 8 0 1 0 19.7 14h-2.1a6 6 0 1 1-1.4-6.2L13 11h7V4z",
    "terminal": "M20 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2zm-9.5 9.5-4 3.5L5 15.5 8 13 5 10.5 6.5 9l4 3.5zM18 17h-6v-2h6z",
    "sparkle": "M12 2l2.4 7.6L22 12l-7.6 2.4L12 22l-2.4-7.6L2 12l7.6-2.4z",
    "flag": "M14.4 6 14 4H5v17h2v-7h5.6l.4 2h7V6z",
    "circle": "M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z",
    "ring": "M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z",
    "merge": "M17 20.4l1.4-1.4-3.4-3.4-1.4 1.4zM7.5 8H11v5.6L5.6 19 7 20.4l6-6V8h3.5L12 3.5z",
    "bold": "M15.6 10.8A3.9 3.9 0 0 0 13 4H7v16h7a4 4 0 0 0 1.6-9.2zM10 6.5h3a1.5 1.5 0 0 1 0 3h-3zm3.5 11H10v-3h3.5a1.5 1.5 0 0 1 0 3z",
    "list": "M3 13h2v-2H3zm0 4h2v-2H3zm0-8h2V7H3zm4 4h14v-2H7zm0 4h14v-2H7zM7 7v2h14V7z",
    "undo": "M12.5 8A10.5 10.5 0 0 0 5.6 10.6L2 7v9h9l-3.6-3.6A8 8 0 0 1 20.2 16l2.4-.8A10.5 10.5 0 0 0 12.5 8z",
    "globe": "M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm6.9 6H16a15.6 15.6 0 0 0-1.4-3.6A8 8 0 0 1 18.9 8zM12 4a14 14 0 0 1 1.9 4h-3.8A14 14 0 0 1 12 4zM4.3 14a8.2 8.2 0 0 1 0-4h3.4a16.5 16.5 0 0 0 0 4zm.8 2H8a15.6 15.6 0 0 0 1.4 3.6A8 8 0 0 1 5.1 16zM8 8H5.1a8 8 0 0 1 4.3-3.6A15.6 15.6 0 0 0 8 8zm4 12a14 14 0 0 1-1.9-4h3.8A14 14 0 0 1 12 20zm2.3-6H9.7a14.7 14.7 0 0 1 0-4h4.6a14.7 14.7 0 0 1 0 4zm.3 5.6A15.6 15.6 0 0 0 16 16h2.9a8 8 0 0 1-4.3 3.6zM16.3 14a16.5 16.5 0 0 0 0-4h3.4a8.2 8.2 0 0 1 0 4z",
}


def icon(name: str, size: int = 24, cls: str = "") -> str:
    return (f'<svg class="ico {cls}" style="width:{size}px;height:{size}px" viewBox="0 0 24 24">'
            f'<path d="{_P[name]}"/></svg>')


AV_COLORS = ["#e57373", "#f06292", "#ba68c8", "#7986cb", "#4fc3f7", "#4db6ac", "#81c784", "#ffb74d", "#a1887f", "#90a4ae"]


def initials(name: str) -> str:
    parts = [p for p in name.replace("@", " ").split() if p[:1].isalpha()]
    return "".join(p[0].upper() for p in parts[:2]) or "?"


def avatar(name: str, r: random.Random, size: int = 40, photo_p: float = 0.45) -> str:
    """Initials avatar (OCR sees the letters, as on real phones) or a photo-like gradient blob (no text)."""
    c = AV_COLORS[sum(map(ord, name)) % len(AV_COLORS)]
    if r.random() < photo_p:
        c2 = r.choice(AV_COLORS)
        ang = r.randint(0, 359)
        return (f'<div class="avatar" style="width:{size}px;height:{size}px;background:linear-gradient({ang}deg,{c},{c2})">'
                f'<div style="width:{size*0.45}px;height:{size*0.45}px;border-radius:50%;background:rgba(255,255,255,.35);'
                f'margin-top:{size*0.1}px"></div></div>')
    return (f'<div class="avatar" data-role="avatar" style="width:{size}px;height:{size}px;background:{c};'
            f'font-size:{size*0.4}px">{e(initials(name))}</div>')


# ----------------------------------------------------------------------------- bars


def status_bar(r: random.Random, clock: str, dark_bar: bool | None = None) -> str:
    pct = r.randint(8, 100)
    icons_left = "".join(icon(n, 15) for n in r.sample(["chat", "bell", "inbox", "camera"], r.randint(0, 3)))
    net = r.choice(["5G", "LTE", "4G+", "", ""])
    return (f'<div class="statusbar" data-role="status" data-exclude="status_bar"><span>{e(clock)}</span>'
            f'<span style="margin-left:6px;display:flex;opacity:.8">{icons_left}</span><span class="sp"></span>'
            f'{icon("wifi", 15) if not net or r.random() < 0.6 else ""}<span style="margin-left:4px;font-size:11px">{net}</span>'
            f'{icon("sig", 15)}<span style="margin-left:5px">{pct}%</span>{icon("bat", 15)}</div>')


def gesture_bar() -> str:
    return '<div class="gesture"><div></div></div>'


KB_RU = ["йцукенгшщзх", "фывапролджэ", "ячсмитьбю"]
KB_EN = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
KB_HE = ["קראטוןםפ", "שדגכעיחלךף", "זסבהנמצתץ"]
SUGG = {"ru": [["и", "в", "не"], ["Привет", "Да", "Спасибо"], ["что", "это", "как"], ["сейчас", "завтра", "ок"]],
        "en": [["I", "the", "and"], ["Thanks", "Yes", "OK"], ["it", "is", "to"], ["sure", "today", "on my way"]],
        "he": [["אני", "את", "זה"], ["שלום", "כן", "תודה"], ["מה", "על", "גם"], ["עכשיו", "מחר", "סבבה"]]}


def keyboard(layout: str, r: random.Random, wide: bool = False, numbers: bool | None = None) -> str:
    """On-screen keyboard (Gboard / Samsung Keyboard look-alike) with a suggestion strip."""
    rows = KB_RU if layout == "ru" else KB_HE if layout == "he" else KB_EN
    if numbers is None:
        numbers = r.random() < 0.5
    sugg = r.choice(SUGG["ru" if layout == "ru" else "he" if layout == "he" else "en"])
    h = []
    h.append('<div class="kb-sugg">' + "".join(f"<div>{e(s)}</div>" for s in sugg) + "</div>")
    if numbers:
        h.append('<div class="kb-row" dir="ltr">' + "".join(f'<div class="key" style="height:36px;font-size:17px">{d}</div>'
                                                            for d in "1234567890") + "</div>")
    for i, row in enumerate(rows):
        keys = "".join(f'<div class="key">{c}{"" if numbers else f"<sup>{(j + 1) % 10}</sup>" if i == 0 and j < 10 else ""}</div>'
                       for j, c in enumerate(row))
        if i == 2:
            keys = f'<div class="key w">⇧</div>{keys}<div class="key w">{icon("bksp", 20)}</div>'
        h.append(f'<div class="kb-row" dir="ltr">{keys}</div>' if layout == "he" else f'<div class="kb-row">{keys}</div>')
    lang = "Русский" if layout == "ru" else "עברית" if layout == "he" else "English"
    h.append(f'<div class="kb-row"{" dir=\"ltr\"" if layout == "he" else ""}><div class="key w">!#1</div><div class="key">,</div><div class="key w">'
             f'{icon("globe", 18)}</div><div class="key space">{lang}</div><div class="key">.</div>'
             f'<div class="key w">{icon("enter", 20)}</div></div>')
    style = "padding-left:18%;padding-right:18%" if wide and r.random() < 0.5 else ""
    return f'<div class="keyboard" data-role="keyboard" data-exclude="keyboard" style="{style}">{"".join(h)}</div>'


def input_field(field: dict, hint_default: str, r: random.Random, left_icons=("smile",), right_icons=("clip", "mic"),
                cls: str = "inputbox") -> str:
    bc = field.get("before_cursor") or ""
    hint = field.get("hint") or hint_default
    if bc:
        inner = f'<span class="txt" data-role="field">{e(bc)}</span><span class="cursor"></span>'
    else:
        inner = f'<span class="cursor" style="margin-left:0;margin-right:2px"></span><span class="hint ell" data-role="hint">{e(hint)}</span>'
    li = "".join(icon(n, 24, "t2") for n in left_icons)
    ri = "".join(icon(n, 24, "t2") for n in right_icons)
    # data-focus: the focused editable node, whose bounds become ctx.field.box (getBoundsInScreen on the phone).
    # "y-parent": the text area between the icon buttons, as tall as the pill (a real Messages compose EditText
    # on the emulator: [316,2310][796,2436], icons outside it, the pill's full height).
    return (f'<div class="{cls}"><span style="margin-right:8px;display:flex">{li}</span>'
            f'<div class="grow" style="overflow:hidden" data-focus="y-parent">{inner}</div><span style="display:flex;gap:10px">{ri}</span></div>')


def murmur_bubble(r: random.Random, css_w: int, css_h: int, top_min: int, top_max: int, locale: str) -> str:
    """Murmur's floating bubble (always an exclusion rect). Sometimes the expanded 'listening' pill."""
    y = r.randint(top_min, max(top_min, top_max))
    pill = r.random() < 0.3
    side = "right" if r.random() < 0.75 else "left"
    x = 8 if side == "left" else None
    pos = f"top:{y}px;" + (f"left:{x}px;" if x is not None else "right:8px;")
    if pill:
        label = "Слушаю…" if locale == "ru" else "מקשיב…" if locale == "he" else "Listening…"
        return (f'<div class="bubble pill" data-exclude="murmur_bubble" data-role="murmur" style="{pos}height:44px">'
                f'{icon("mic", 20)}<span>{label}</span></div>')
    return f'<div class="bubble" data-exclude="murmur_bubble" data-role="murmur" style="{pos}">{icon("mic", 26)}</div>'


def nav_bar(items: list[tuple[str, str]], on: int, badges: dict | None = None) -> str:
    badges = badges or {}
    cells = []
    for i, (ic, label) in enumerate(items):
        b = f'<span class="badge" style="position:absolute;margin:-26px 0 0 26px;height:16px;min-width:16px;font-size:10px">{badges[i]}</span>' if i in badges else ""
        cells.append(f'<div class="{"on" if i == on else ""}"><div class="pill">{icon(ic)}</div>{b}<span>{e(label)}</span></div>')
    return f'<div class="navbar" data-role="chrome">{"".join(cells)}</div>'


def page(css: str, body: str, title: str = "screen", direction: str = "ltr", lang: str = "") -> str:
    attrs = ""
    if direction == "rtl":
        attrs += ' dir="rtl"'
    if lang:
        attrs += f' lang="{e(lang)}"'
    return (f'<!doctype html><html{attrs}><head><meta charset="utf-8"><title>{e(title)}</title><style>{css}</style></head>'
            f'<body><div id="screen">{body}</div></body></html>')
