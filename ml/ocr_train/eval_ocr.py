"""Evaluate exported 3-language (`ru + en + he`) ONNX recognizer on:

1. Held-out validation crops (`he`, `he_en`, `ru_he`, `ru`, `ru_en`, `en`) using exact
   `ml/ocr/pipeline.py` preprocessing & greedy CTC decoding, reporting both visual-order CER
   and logical-order CER + exact match + CPU latency.
2. Golden regression suite (`ml/ocr/golden/001..026.png`), printing side-by-side CER and exact
   recall vs stock `cyrillic_PP-OCRv5_mobile_rec` (using pinned stock outputs in `001..026.json`
   and/or live stock ONNX).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .charset import DEFAULT_DICT_PATH, REPO_ROOT, Charset, load_charset, visual_to_logical
from .train_rec import HE_CATS, REC_BUCKETS, REC_HEIGHT, RU_EN_CATS, _load_manifest, _resize_bgr_f32, levenshtein

GOLDEN_DIR = REPO_ROOT / "ml" / "ocr" / "golden"


def _make_ort_session(onnx_path: Path, threads: int = 4):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])


def _rec_width_exact(w: int, h: int) -> tuple[int, int]:
    rw = int(math.ceil(REC_HEIGHT * w / max(1, h)))
    for b in REC_BUCKETS:
        if rw <= b:
            return rw, b
    return REC_BUCKETS[-1], REC_BUCKETS[-1]


def _prep_crop_tensor(img_path: Path) -> np.ndarray:
    """Exact `ml/ocr/pipeline.py:rec_input` preprocessing on a crop image."""
    rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    bgr = rgb[:, :, ::-1]
    h, w = bgr.shape[:2]
    rw, bw = _rec_width_exact(w, h)
    r = _resize_bgr_f32(bgr, rw, REC_HEIGHT) / np.float32(255.0)
    r = (r - np.float32(0.5)) / np.float32(0.5)
    t = np.zeros((1, 3, REC_HEIGHT, bw), dtype=np.float32)
    t[0, :, :, :rw] = r.transpose(2, 0, 1)
    return t


def eval_val_crops(
    onnx_path: Path,
    crops_dir: Path,
    dict_path: Path = DEFAULT_DICT_PATH,
    split: str = "val",
    limit: int | None = None,
    threads: int = 4,
) -> dict:
    cs = load_charset(dict_path)
    sess = _make_ort_session(onnx_path, threads=threads)
    n_out = sess.get_outputs()[0].shape[-1]
    if n_out != cs.num_classes:
        raise ValueError(f"ONNX outputs {n_out} classes, dict has {cs.num_classes}")

    rows = _load_manifest(crops_dir, split)
    if limit is not None:
        rows = rows[:limit]

    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0, 0])  # vis_err, vis_ch, vis_ex, log_err, log_ch, n
    confs: list[float] = []
    worst: list[dict] = []

    t0 = time.perf_counter()
    for r in rows:
        x = _prep_crop_tensor(crops_dir / r["image"])
        probs = sess.run(None, {"x": x})[0][0]
        hyp_vis, conf = cs.decode_probs(probs)
        confs.append(conf)

        gt_vis = r["visual"]
        gt_log = r.get("text", gt_vis)
        bdir = r.get("base_dir")
        hyp_log = visual_to_logical(hyp_vis, base_dir=bdir)

        ve = levenshtein(gt_vis, hyp_vis)
        le = levenshtein(gt_log, hyp_log)
        st = by_cat[r.get("cat", "unknown")]
        st[0] += ve
        st[1] += max(1, len(gt_vis))
        st[2] += int(hyp_vis == gt_vis)
        st[3] += le
        st[4] += max(1, len(gt_log))
        st[5] += 1

        if ve > 0:
            worst.append({
                "cat": r.get("cat"),
                "cer": round(ve / max(1, len(gt_vis)), 3),
                "gt_text": gt_log,
                "hyp_text": hyp_log,
                "gt_vis": gt_vis,
                "hyp_vis": hyp_vis,
                "conf": round(conf, 3),
            })

    elapsed = time.perf_counter() - t0
    worst.sort(key=lambda w: -w["cer"])

    def _agg(cats: set[str] | None) -> dict:
        keys = [c for c in by_cat if (cats is None or c in cats)]
        ve = sum(by_cat[c][0] for c in keys)
        vc = sum(by_cat[c][1] for c in keys)
        vx = sum(by_cat[c][2] for c in keys)
        le = sum(by_cat[c][3] for c in keys)
        lc = sum(by_cat[c][4] for c in keys)
        nn = sum(by_cat[c][5] for c in keys)
        return {
            "n": nn,
            "vis_cer": round(ve / max(1, vc), 5),
            "vis_exact": round(vx / max(1, nn), 5),
            "log_cer": round(le / max(1, lc), 5),
        }

    per_cat = {
        c: {
            "n": v[5],
            "vis_cer": round(v[0] / max(1, v[1]), 5),
            "vis_exact": round(v[2] / max(1, v[5]), 5),
            "log_cer": round(v[3] / max(1, v[4]), 5),
        }
        for c, v in sorted(by_cat.items())
    }
    return {
        "onnx": str(onnx_path),
        "n": len(rows),
        "ms_per_crop": round(elapsed * 1e3 / max(1, len(rows)), 2),
        "mean_conf": round(float(np.mean(confs)) if confs else 0.0, 4),
        "overall": _agg(None),
        "hebrew_all": _agg(HE_CATS),
        "ru_en_all": _agg(RU_EN_CATS),
        "by_cat": per_cat,
        "worst_samples": worst[:15],
    }


def eval_golden_regression(
    onnx_path: Path,
    yml_or_dict_path: Path,
    golden_dir: Path = GOLDEN_DIR,
    stock_onnx_path: Path | None = None,
    threads: int = 4,
) -> dict:
    """Run full `ml/ocr/pipeline.py` on `ml/ocr/golden/001..026.png` and compare vs stock."""
    from ml.ocr import metrics as M
    from ml.ocr import pipeline as P
    from ml.ocr.config import DEFAULT, MODEL_DIR

    det_path = MODEL_DIR / DEFAULT.det_model
    if not det_path.exists():
        raise FileNotFoundError(
            f"Detector model missing at {det_path}. Run `bash ml/ocr/fetch_assets.sh` first."
        )

    new_cfg = DEFAULT.with_(
        rec_model=str(onnx_path.resolve()),
        rec_dict=str(yml_or_dict_path.resolve()),
        threads=threads,
    )
    stock_cfg = None
    if stock_onnx_path is not None and stock_onnx_path.exists():
        stock_cfg = DEFAULT.with_(rec_model=str(stock_onnx_path.resolve()), threads=threads)

    pngs = sorted(golden_dir.glob("[0-9][0-9][0-9].png"))
    if not pngs:
        raise FileNotFoundError(f"No golden PNGs found in {golden_dir}")

    new_results = []
    stock_results = []
    per_screen = []
    rec_ms_total = 0.0

    for png in pngs:
        doc = json.loads(png.with_suffix(".json").read_text(encoding="utf-8"))
        excl = doc["exclude_rects"]
        vis = doc["visible"]

        new_det = P.run_detailed(png, exclude_rects=excl, cfg=new_cfg)
        rec_ms_total += new_det["ms"]["rec"]
        new_cmp = M.compare(vis, new_det["result"]["lines"], exclude_rects=excl)
        new_results.append(new_cmp)

        if stock_cfg is not None:
            stock_lines = P.run(png, exclude_rects=excl, cfg=stock_cfg)["lines"]
        else:
            # Use exact pinned stock OCR lines stored in the golden JSON
            stock_lines = doc["ocr"]["lines"]
        stock_cmp = M.compare(vis, stock_lines, exclude_rects=excl)
        stock_results.append(stock_cmp)

        per_screen.append({
            "screen": png.name,
            "template": doc["spec"].get("template", ""),
            "locale": doc["spec"].get("locale", ""),
            "stock_cer": round(stock_cmp["cer"], 4),
            "new_cer": round(new_cmp["cer"], 4),
            "delta_cer": round(new_cmp["cer"] - stock_cmp["cer"], 4),
            "stock_exact": f"{stock_cmp['exact_items']}/{stock_cmp['gt_items']}",
            "new_exact": f"{new_cmp['exact_items']}/{new_cmp['gt_items']}",
        })

    new_agg = M.aggregate(new_results)
    stock_agg = M.aggregate(stock_results)

    return {
        "screens": len(pngs),
        "avg_rec_ms_per_screen": round(rec_ms_total / max(1, len(pngs)), 1),
        "stock": {
            "cer": round(stock_agg["cer"], 5),
            "content_cer": round(stock_agg["content_cer"], 5),
            "item_recall_exact": round(stock_agg["item_recall_exact"], 5),
            "missed_share": round(stock_agg["missed_share"], 5),
        },
        "new": {
            "cer": round(new_agg["cer"], 5),
            "content_cer": round(new_agg["content_cer"], 5),
            "item_recall_exact": round(new_agg["item_recall_exact"], 5),
            "missed_share": round(new_agg["missed_share"], 5),
        },
        "delta": {
            "cer": round(new_agg["cer"] - stock_agg["cer"], 5),
            "content_cer": round(new_agg["content_cer"] - stock_agg["content_cer"], 5),
            "item_recall_exact": round(new_agg["item_recall_exact"] - stock_agg["item_recall_exact"], 5),
        },
        "per_screen": per_screen,
    }


def _print_report(val_res: dict | None, golden_res: dict | None) -> None:
    if val_res is not None:
        print("\n=== Held-Out Validation Crops (ONNX Runtime CPU) ===")
        print(
            f"N={val_res['n']:,} | {val_res['ms_per_crop']:.2f} ms/crop | mean_conf={val_res['mean_conf']:.4f}\n"
            f"  Overall  : vis_CER={val_res['overall']['vis_cer']*100:.2f}%  "
            f"log_CER={val_res['overall']['log_cer']*100:.2f}%  exact={val_res['overall']['vis_exact']*100:.2f}%\n"
            f"  Hebrew   : vis_CER={val_res['hebrew_all']['vis_cer']*100:.2f}%  "
            f"log_CER={val_res['hebrew_all']['log_cer']*100:.2f}%  exact={val_res['hebrew_all']['vis_exact']*100:.2f}%\n"
            f"  RU + EN  : vis_CER={val_res['ru_en_all']['vis_cer']*100:.2f}%  "
            f"log_CER={val_res['ru_en_all']['log_cer']*100:.2f}%  exact={val_res['ru_en_all']['vis_exact']*100:.2f}%"
        )
        print("  Per-category:")
        for cat, st in val_res["by_cat"].items():
            print(
                f"    {cat:8s} n={st['n']:5d}  vis_CER={st['vis_cer']*100:5.2f}%  "
                f"log_CER={st['log_cer']*100:5.2f}%  exact={st['vis_exact']*100:5.2f}%"
            )

    if golden_res is not None:
        print("\n=== Golden Regression Suite (ml/ocr/golden/001..026.png, RU+EN) ===")
        st = golden_res["stock"]
        nw = golden_res["new"]
        dl = golden_res["delta"]
        print(f"Screens: {golden_res['screens']} | avg rec latency: {golden_res['avg_rec_ms_per_screen']:.1f} ms/screen")
        print(f"  {'Metric':20s} {'Stock':>10s} {'New (3-lang)':>14s} {'Delta':>10s}")
        print(f"  {'CER':20s} {st['cer']*100:9.2f}% {nw['cer']*100:13.2f}% {dl['cer']*100:+9.2f}%")
        print(f"  {'Content CER':20s} {st['content_cer']*100:9.2f}% {nw['content_cer']*100:13.2f}% {dl['content_cer']*100:+9.2f}%")
        print(f"  {'Exact Item Recall':20s} {st['item_recall_exact']*100:9.2f}% {nw['item_recall_exact']*100:13.2f}% {dl['item_recall_exact']*100:+9.2f}%")
        print("\n  Per-screen breakdown (top deltas):")
        for sc in sorted(golden_res["per_screen"], key=lambda x: -x["delta_cer"])[:10]:
            print(
                f"    {sc['screen']} ({sc['template']:16s} {sc['locale']:2s}) "
                f"stock={sc['stock_cer']*100:5.2f}%  new={sc['new_cer']*100:5.2f}%  "
                f"delta={sc['delta_cer']*100:+5.2f}%  exact={sc['new_exact']}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate exported 3-language OCR ONNX model.")
    ap.add_argument("--onnx", type=Path, required=True, help="Path to exported inference.onnx")
    ap.add_argument("--dict", type=Path, default=DEFAULT_DICT_PATH, help="Path to dict_ru_en_he.txt or inference.yml")
    ap.add_argument("--crops-dir", type=Path, default=None, help="Validation crops dir for he/he_en/ru_he/ru/en eval")
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--golden-dir", type=Path, default=GOLDEN_DIR, help="Golden regression dir (001..026.png)")
    ap.add_argument("--skip-golden", action="store_true")
    ap.add_argument("--stock-onnx", type=Path, default=None, help="Optional live stock ONNX (else compares vs pinned golden JSON)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    val_res = None
    if args.crops_dir is not None:
        val_res = eval_val_crops(args.onnx, args.crops_dir, dict_path=args.dict, limit=args.val_limit, threads=args.threads)

    golden_res = None
    if not args.skip_golden and args.golden_dir.exists():
        golden_res = eval_golden_regression(
            args.onnx, args.dict, golden_dir=args.golden_dir, stock_onnx_path=args.stock_onnx, threads=args.threads
        )

    _print_report(val_res, golden_res)

    payload = {"val_crops": val_res, "golden_regression": golden_res}
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
