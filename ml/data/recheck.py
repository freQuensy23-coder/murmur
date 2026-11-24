"""Re-check stored v2 records against CONTEXT_V2 category rules with the CURRENT prompts/render.py.

The <context> block is rendered at load time, so a change of the renderer (budget, order, trimming) can hide a
line an example depends on. This re-applies the rules to ctx.screen.lines and to the rendered block, without
OCR or teacher calls, and writes the surviving records:

  python -m ml.data.recheck --run .cache/ml/data/synth_v2 [--out DIR]   # default: report only

Rules (record meta written by ml/data/synth/build.py):
  context_resolution / context_names   meta.entities shown exactly (lines and block)
  near_miss                            meta.distractors shown, meta.entities absent (case-folded)
  context_counterpart                  meta.check.forbid[0] and close variants absent (lines and block)
  injection                            meta.injection.text visible (fuzzy: OCR noise)
  continuation                         field.before_cursor rendered
  no screen (any category)             privacy.level = prompts/render.py NO_SCREEN_LEVEL[reason]; no password_field (ADR-070)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from ml.common.render_bridge import rendered_context
from ml.common.schema import write_jsonl
from ml.data.synth import screen as SCR


def rule_errors(rec: dict) -> list[str]:
    ctx, meta, cat = rec["ctx"], rec["meta"], rec["category"]
    if ctx.get("v") != 2:
        return []
    blk = rendered_context(ctx)
    out = []
    if not ctx.get("screen"):  # ADR-070: the level the phone sends for the no-screen reason (shared table)
        pv = ctx.get("privacy") or {}
        try:
            if pv.get("level") != SCR.no_screen_level(pv.get("reason")):
                out.append(f"privacy:level_mismatch:{pv.get('reason')}:{pv.get('level')}")
        except ValueError:
            out.append(f"privacy:reason_not_generated:{pv.get('reason')}")

    def shown(s):
        return SCR.on_screen(ctx, s) and SCR.exact_in_text(blk, s)

    if cat in ("context_resolution", "context_names"):
        for e in meta.get("entities") or []:
            if not shown(e):
                out.append(f"{cat}:not_shown:{e}")
    elif cat == "near_miss":
        for d in meta.get("distractors") or []:
            if not shown(d):
                out.append(f"near_miss:distractor_not_shown:{d}")
        for e in meta.get("entities") or []:
            x = e.split()[-1] if " " in e and (e[:1].isupper() or "\u05d0" <= e[:1] <= "\u05ea") and not re.search(r"[a-z][A-Z]", e) else e
            if re.search(r"(?<!\w)" + re.escape(SCR._cf(x)) + r"(?!\w)", SCR._cf(" ".join(SCR.lines_of(ctx)) + " " + blk)):
                out.append(f"near_miss:said_shown:{x}")
    elif cat == "context_counterpart":
        t = ((meta.get("check") or {}).get("forbid") or [""])[0]
        if t and (SCR.close_on_screen(ctx, t) or SCR.close_in_text(blk, t)):
            out.append(f"counterpart:term_visible:{t}")
    elif cat == "injection":
        inj = (meta.get("injection") or {}).get("text", "")
        if inj and not (SCR.fuzzy_on_screen(ctx, inj) and SCR.fuzzy_on_screen({"screen": {"lines": [{"text": blk}]}}, inj)):
            out.append("injection:not_visible")
    elif cat == "continuation":
        if "before the cursor" not in blk:
            out.append("continuation:before_cursor_not_rendered")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None, help="write surviving {train,dev,test}.jsonl here")
    a = ap.parse_args(argv)
    run = Path(a.run)
    bad, n, kept = Counter(), 0, {}
    for split in ("train", "dev", "test"):
        f = run / f"{split}.jsonl"
        recs = [json.loads(l) for l in open(f, encoding="utf-8")] if f.exists() else []
        kept[split] = []
        for r in recs:
            n += 1
            errs = rule_errors(r)
            for e in errs:
                bad[e.split(":")[0] + ":" + e.split(":")[1]] += 1
            if not errs:
                kept[split].append(r)
    if a.out:
        for split, rs in kept.items():
            write_jsonl(Path(a.out) / f"{split}.jsonl", rs)
    print(json.dumps({"records": n, "kept": sum(len(v) for v in kept.values()), "failures": dict(bad.most_common())},
                     ensure_ascii=False, indent=1))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
