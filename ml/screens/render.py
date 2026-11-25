"""Render a screen spec to a phone-sized PNG with the ground truth of the visible text.

    from ml.screens import render
    out = render(spec)   # {"png": bytes, "size": [w, h], "visible": [{"role", "text", "box"}], "exclude_rects": [[x0, y0, x1, y1]],
                         #  "exclude_kinds", "field_box": [x0, y0, x1, y1] of the focused field or None, "spec"}

    python -m ml.screens.render SPEC.json OUT.png            # also writes OUT.json (everything but the PNG)
    python -m ml.screens.render --batch SPECS.jsonl OUTDIR   # one browser for all specs

One Node process (driver.js, playwright-core + the system Chrome) is kept per Renderer and reused.
Deterministic: the spec's seed drives every random choice; fonts are local; animations are off.
"""
from __future__ import annotations

import argparse
import atexit
import itertools
import json
import math
import os
import subprocess
import sys
import threading
from pathlib import Path

from .base import new_ctx
from .templates import TEMPLATES, normalise_spec

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRATCH = Path(os.environ.get("MURMUR_SCREENS_SCRATCH", REPO / ".cache" / "ml" / "screens"))
FONTS = Path(os.environ.get("MURMUR_SCREENS_FONTS", SCRATCH / "fonts"))
OCR_FONTS = Path(os.environ.get("MURMUR_OCR_FONTS", REPO / ".cache" / "ml" / "ocr" / "fonts"))
CHROME = os.environ.get("MURMUR_CHROME", "/usr/local/bin/google-chrome")   # driver.js reads the same variable


def build_html(spec: dict) -> tuple[str, dict]:
    spec = normalise_spec(spec)
    c = new_ctx(spec)
    return TEMPLATES[spec["template"]](c), c.dev


def _merge(items: list[dict]) -> list[dict]:
    """Merge fragments of the same role that sit on the same visual line and nearly touch."""
    items = sorted(items, key=lambda d: (d["box"][1], d["box"][0]))
    out: list[dict] = []
    for it in items:
        b = it["box"]
        h = b[3] - b[1]
        for o in reversed(out[-6:]):
            ob = o["box"]
            oh = ob[3] - ob[1]
            ov = min(b[3], ob[3]) - max(b[1], ob[1])
            gap = b[0] - ob[2]
            if o["role"] == it["role"] and ov >= 0.6 * min(h, oh) and -2 <= gap <= 0.6 * max(h, oh):
                sep = " " if (gap > 0.12 * max(h, oh) or it.get("lead") or o.get("trail")) else ""
                o["text"] = o["text"] + sep + it["text"]
                o["trail"] = it.get("trail", False)
                o["box"] = [min(ob[0], b[0]), min(ob[1], b[1]), max(ob[2], b[2]), max(ob[3], b[3])]
                break
        else:
            out.append({"role": it["role"], "text": it["text"], "box": list(b), "trail": it.get("trail", False)})
    return [{"role": o["role"], "text": o["text"], "box": o["box"]} for o in out]


def _px(box, s: float, w: int, h: int) -> list[int]:
    return [max(0, int(math.floor(box[0] * s))), max(0, int(math.floor(box[1] * s))),
            min(w, int(math.ceil(box[2] * s))), min(h, int(math.ceil(box[3] * s)))]


class Renderer:
    """Keeps one Chrome (through driver.js) alive for many renders. Thread-safe (requests are serialised)."""

    def __init__(self, workdir: Path | None = None):
        self.workdir = Path(workdir or SCRATCH / "work" / f"{os.getpid()}-{id(self):x}")
        self.workdir.mkdir(parents=True, exist_ok=True)
        link = self.workdir / "fonts"
        if not link.exists():
            link.symlink_to(FONTS, target_is_directory=True)
        olink = self.workdir / "ocr_fonts"
        if not olink.exists() and OCR_FONTS.exists():
            olink.symlink_to(OCR_FONTS, target_is_directory=True)
        self.proc = None
        self.lock = threading.Lock()
        self.ids = itertools.count()
        atexit.register(self.close)

    def _start(self):
        env = dict(os.environ, MURMUR_CHROME=CHROME)
        self.proc = subprocess.Popen(["node", str(HERE / "driver.js")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, encoding="utf-8", env=env, bufsize=1)

    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=20)
            except Exception:
                self.proc.kill()
        self.proc = None

    def render(self, spec: dict, out_png: str | Path | None = None) -> dict:
        html, dev = build_html(spec)
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            i = next(self.ids)
            hp = self.workdir / f"{i % 4}.html"
            hp.write_text(html, encoding="utf-8")
            png = Path(out_png) if out_png else self.workdir / f"{i % 4}.png"
            req = {"id": i, "html": str(hp), "w": dev["w"], "h": dev["h"], "css_w": dev["css_w"], "css_h": dev["css_h"],
                   "out": str(png.resolve()), "locale": spec.get("locale", "ru")}
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        if not line:
            self.close()
            raise RuntimeError("renderer process died")
        res = json.loads(line)
        if not res.get("ok"):
            raise RuntimeError(res.get("error"))
        s = res["dsf"]
        w, h = dev["w"], dev["h"]
        vis = []
        for it in res["visible"]["items"]:
            b = _px(it["box"], s, w, h)
            if b[2] > b[0] and b[3] > b[1]:
                vis.append({"role": it["role"], "text": it["text"], "box": b, "lead": bool(it.get("lead")),
                            "trail": bool(it.get("trail"))})
        ex = [{"kind": x["kind"], "box": _px(x["box"], s, w, h)} for x in res["visible"]["exclude"]]
        foc = res["visible"].get("focus") or []
        if len(foc) > 1:
            raise RuntimeError(f"template {spec.get('template')!r} marked {len(foc)} focused fields")
        fbox = _px(foc[0], s, w, h) if foc and foc[0] else None
        data = png.read_bytes()
        return {"png": data, "size": [w, h], "visible": _merge(vis), "exclude_rects": [x["box"] for x in ex],
                "exclude_kinds": [x["kind"] for x in ex], "field_box": fbox, "spec": normalise_spec(spec)}


_DEFAULT: Renderer | None = None


def render(spec: dict) -> dict:
    """CONTEXT_V2 interface (module-level Renderer, created on first use)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Renderer()
    return _DEFAULT.render(spec)


def meta_json(res: dict) -> str:
    return json.dumps({k: v for k, v in res.items() if k != "png"}, ensure_ascii=False, indent=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ml.screens.render")
    ap.add_argument("spec", help="SPEC.json, or SPECS.jsonl with --batch")
    ap.add_argument("out", help="OUT.png, or OUTDIR with --batch")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--html", action="store_true", help="also save the HTML next to the PNG")
    a = ap.parse_args(argv)
    r = Renderer()
    if not a.batch:
        spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
        res = r.render(spec, a.out)
        Path(a.out).with_suffix(".json").write_text(meta_json(res), encoding="utf-8")
        if a.html:
            Path(a.out).with_suffix(".html").write_text(build_html(spec)[0], encoding="utf-8")
        print(json.dumps({"png": a.out, "size": res["size"], "visible": len(res["visible"])}))
    else:
        od = Path(a.out)
        od.mkdir(parents=True, exist_ok=True)
        n = 0
        for k, line in enumerate(Path(a.spec).read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            spec = json.loads(line)
            name = spec.get("id") or f"{k:05d}"
            res = r.render(spec, od / f"{name}.png")
            (od / f"{name}.json").write_text(meta_json(res), encoding="utf-8")
            if a.html:
                (od / f"{name}.html").write_text(build_html(spec)[0], encoding="utf-8")
            n += 1
        print(json.dumps({"rendered": n, "dir": str(od)}))
    r.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
