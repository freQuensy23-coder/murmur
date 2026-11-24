"""spec -> ctx v2 (docs/ml/CONTEXT_V2.md): render -> OCR with the render's exclude_rects -> app/field/screen.
The only way training and eval contexts are produced.

    from ml.screens.build_ctx import build_ctx, build_ctx_detailed
    ctx = build_ctx(spec)

    python -m ml.screens.build_ctx SPEC.json                      # ctx v2 on stdout
    python -m ml.screens.build_ctx --batch SPECS.jsonl OUT.jsonl [--workers 4] [--png-dir DIR]
      OUT.jsonl: one {"id", "ctx", "visible", "exclude_rects", "meta"} per spec, in input order.
    python -m ml.screens.build_ctx --check FILE.jsonl [--loose]
      exit 1 if any record's build_ctx meta (top-level "meta", "meta.render" of an eval case or
      "meta.screen.render" of a data record) was made by another OCR / renderer / Chrome version (see stale()).

ctx.field.box = bounds of the template's focused node ([data-focus]) in screenshot pixels, clipped to the
visible area: what AccessibilityNodeInfo.getBoundsInScreen gives the phone. prompts/render.py uses it to
anchor "nearest to the field" and to drop OCR lines inside the field.

Spec extras beyond CONTEXT_V2's render spec (all optional): "app" {package, label, category} (default: the
template's look-alike app), "field": null = no focused node (ctx.field absent), field.kind, "lang"
("ru"|"en"|"mixed" -> language.asr), "privacy" ("full" | anything else -> no screen, reason set),
"avoid" (strings filler must not contain). Use ml.screens.specs.from_teacher() to make specs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

from ml.ocr import pipeline as ocr
from ml.ocr.config import DEFAULT as OCR_DEFAULT, PIPELINE_VERSION, source_hash as ocr_source_hash

from . import RENDERER_VERSION, source_hash as screens_source_hash
from .render import CHROME, Renderer
from .templates import TEMPLATE_APPS, normalise_spec

_R: Renderer | None = None


def _renderer() -> Renderer:
    global _R
    if _R is None:
        _R = Renderer()
    return _R


def app_of(spec: dict) -> dict:
    if spec.get("app"):
        a = spec["app"]
        return {"package": a.get("package", ""), "label": a.get("label", ""), "category": a.get("category", "default")}
    pkg, label, cat, _, _ = TEMPLATE_APPS[spec["template"]]
    return {"package": pkg, "label": label, "category": cat}


def field_of(spec: dict, box: list[int] | None = None) -> dict | None:
    """Accessibility side of the focused field. box: the render's field_box (focused node bounds)."""
    f = spec.get("field")
    if f is None:
        return None
    kind = f.get("kind") or TEMPLATE_APPS[spec["template"]][3]
    out = {"kind": kind, "hint": f.get("hint") or "", "before_cursor": f.get("before_cursor") or ""}
    if box:
        out["box"] = [int(v) for v in box]
    if f.get("password"):
        out["password"] = True
    return out


@lru_cache(maxsize=1)
def chrome_version() -> str:
    try:
        out = subprocess.run([CHROME, "--version"], capture_output=True, text=True, timeout=30).stdout
        return out.strip().split()[-1] if out.strip() else "unknown"
    except Exception:   # noqa: BLE001
        return "unknown"


def versions() -> dict:
    """What a context depends on besides its spec. Recorded in every meta; stale() compares against it."""
    return {"ocr": PIPELINE_VERSION, "ocr_src": ocr_source_hash(), "screens": RENDERER_VERSION,
            "screens_src": screens_source_hash(), "chrome": chrome_version(), "det_limit": OCR_DEFAULT.det_limit}


STRICT_KEYS = ("ocr_src", "screens_src", "chrome")


def stale(meta: dict, strict: bool = True) -> list[str]:
    """Why a context built with this build_ctx meta is not what the current pipeline produces ([] = current).
    Loose: hand-bumped versions and det_limit only. Strict (default): also the source hashes and the Chrome
    build (a forgotten bump, or a browser update that moves glyphs). Records without the keys are stale."""
    cur = versions()
    keys = ["ocr", "screens", "det_limit"] + (list(STRICT_KEYS) if strict else [])
    return [f"{k}: {meta.get(k)!r} != {cur[k]!r}" for k in keys if meta.get(k) != cur[k]]


def build_ctx_detailed(spec: dict, renderer: Renderer | None = None, ocr_cfg=OCR_DEFAULT, png_out=None) -> dict:
    spec = normalise_spec(spec)
    t0 = time.perf_counter()
    res = (renderer or _renderer()).render(spec, png_out)
    t1 = time.perf_counter()
    privacy = spec.get("privacy", "full")
    field = field_of(spec, res.get("field_box"))
    screen, reason = None, None
    if privacy != "full":
        reason = spec.get("privacy_reason") or privacy
    elif field and field.get("password"):
        reason = "password_field"
    else:
        o = ocr.run(res["png"], res["exclude_rects"], ocr_cfg)
        if o["lines"]:
            screen = o
    t2 = time.perf_counter()
    lang = spec.get("lang", "ru")
    ctx = {"v": 2, "app": app_of(spec)}
    if field:
        ctx["field"] = field
    if screen:
        ctx["screen"] = {"size": screen["size"], "lines": screen["lines"]}
    ctx["language"] = {"asr": "he" if lang in ("he", "he_en", "he_ru") else ("en" if lang == "en" else "ru")}
    ctx["privacy"] = {"level": privacy, "reason": reason}
    meta = {"template": spec["template"], "device": spec["device"], "theme": spec["theme"], "font_scale": spec["font_scale"],
            "locale": spec["locale"], "keyboard": bool(spec.get("keyboard")) and spec.get("field") is not None,
            "seed": spec["seed"], **versions(), "det_limit": ocr_cfg.det_limit,
            "ms": {"render": round((t1 - t0) * 1e3), "ocr": round((t2 - t1) * 1e3)}}
    return {"ctx": ctx, "visible": res["visible"], "exclude_rects": res["exclude_rects"],
            "exclude_kinds": res["exclude_kinds"], "meta": meta, "png": res["png"]}


def build_ctx(spec: dict) -> dict:
    """CONTEXT_V2 interface: spec -> ctx v2."""
    return build_ctx_detailed(spec)["ctx"]


# ----------------------------------------------------------------------------- batch

_W: dict = {}


def _init(threads: int, det_limit: int):
    _W["r"] = Renderer()
    _W["cfg"] = OCR_DEFAULT.with_(threads=threads, det_limit=det_limit)


def _one(args):
    i, spec, png_dir = args
    sid = spec.get("id") or f"{i:06d}"
    png = Path(png_dir) / f"{sid}.png" if png_dir else None
    try:
        d = build_ctx_detailed(spec, _W["r"], _W["cfg"], png)
        d.pop("png")
        return i, {"id": sid, **d}
    except Exception as ex:   # one bad spec must not kill a batch
        return i, {"id": sid, "error": f"{type(ex).__name__}: {ex}"}


def batch(specs: list[dict], out_path: str, workers: int = 4, threads: int = 2, png_dir: str | None = None,
          det_limit: int = OCR_DEFAULT.det_limit) -> dict:
    import multiprocessing as mp
    if png_dir:
        Path(png_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res: dict[int, dict] = {}
    jobs = [(i, s, png_dir) for i, s in enumerate(specs)]
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(threads, det_limit)) as pool:
        for i, d in pool.imap_unordered(_one, jobs, chunksize=1):
            res[i] = d
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(len(specs)):
            f.write(json.dumps(res[i], ensure_ascii=False) + "\n")
    errs = sum(1 for d in res.values() if "error" in d)
    return {"n": len(specs), "errors": errs, "wall_s": round(time.time() - t0, 1), "workers": workers, "threads": threads}


def _meta_of(rec: dict) -> dict | None:
    """build_ctx meta inside a record: top-level meta (build_ctx --batch output), meta.render (prompt eval
    sets) or meta.screen.render (ml/data records). None = the record has no screen built by build_ctx."""
    m = rec.get("meta") or {}
    for cand in (m, m.get("render"), (m.get("screen") or {}).get("render")):
        if isinstance(cand, dict) and "template" in cand and "ocr" in cand:
            return cand
    return None


def check_file(path: str, strict: bool = True) -> int:
    n = bad = noscreen = 0
    why: dict[str, int] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        n += 1
        m = _meta_of(rec)
        if m is None:
            noscreen += 1
            continue
        reasons = stale(m, strict)
        if reasons:
            bad += 1
            for r in reasons:
                k = r.split(":")[0] + ": " + r.split(": ", 1)[1]
                why[k] = why.get(k, 0) + 1
    print(json.dumps({"records": n, "without_build_ctx_meta": noscreen, "stale": bad, "current": versions(),
                      "reasons": dict(sorted(why.items(), key=lambda kv: -kv[1])[:20])}, ensure_ascii=False, indent=1))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ml.screens.build_ctx")
    ap.add_argument("spec")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("MURMUR_SCREENS_WORKERS", "4")))
    ap.add_argument("--threads", type=int, default=2, help="onnxruntime threads per worker")
    ap.add_argument("--png-dir")
    ap.add_argument("--det-limit", type=int, default=OCR_DEFAULT.det_limit)
    ap.add_argument("--check", action="store_true", help="report records built by another pipeline version")
    ap.add_argument("--loose", action="store_true", help="with --check: compare hand-bumped versions only")
    a = ap.parse_args(argv)
    if a.check:
        return check_file(a.spec, strict=not a.loose)
    if a.batch:
        specs = [json.loads(x) for x in Path(a.spec).read_text(encoding="utf-8").splitlines() if x.strip()]
        print(json.dumps(batch(specs, a.out, a.workers, a.threads, a.png_dir, a.det_limit)))
    else:
        spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
        d = build_ctx_detailed(spec, png_out=a.out)
        sys.stdout.write(json.dumps(d["ctx"], ensure_ascii=False, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
