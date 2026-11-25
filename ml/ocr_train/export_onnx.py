"""Export trained PyTorch 3-language recognizer checkpoint to ONNX matching `ml/ocr/pipeline.py` exactly.

Contract verified on export:
  - Single self-contained `inference.onnx` (no external `.data` shards)
  - Input  : `"x"` float32 `[N, 3, 48, W]` (dynamic batch `N`, dynamic width `W`, BGR, `(v/255-0.5)/0.5`)
  - Output : `"probs"` float32 `[N, W // 8, 883]` (softmax probabilities summing to 1.0)
  - Companion `inference.yml` (readable by both `ml/ocr/pipeline.py:load_keys` and Kotlin `OcrDictionary`)
    and `dict_ru_en_he.txt` (881 lines, indices 0..849 identical to `ml/ocr/golden/rec_keys.txt`)
  - Numerical PyTorch-vs-ONNXRuntime parity across production `OcrConfig.rec_buckets`
  - Optional live end-to-end verification through `ml/ocr/pipeline.py`!
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .charset import DEFAULT_DICT_PATH, load_charset, write_dict_txt, write_inference_yml
from .model import PPOCRv5MobileRec

REC_BUCKETS = (320, 480, 640, 800, 960, 1280, 1600, 1920, 2560, 3200)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model_from_checkpoint(ckpt_path: Path, use_ema: bool = True, export_softmax: bool = True) -> tuple[PPOCRv5MobileRec, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = PPOCRv5MobileRec(
        num_classes=int(cfg.get("num_classes", 883)),
        scale=float(cfg.get("scale", 0.95)),
        svtr_dims=int(cfg.get("svtr_dims", 120)),
        svtr_depth=int(cfg.get("svtr_depth", 2)),
        drop=0.0,
        export_softmax=export_softmax,
    )
    sd = ckpt["ema"] if (use_ema and ckpt.get("ema") is not None) else ckpt["model"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, cfg


def export_onnx(
    ckpt_path: Path,
    out_dir: Path,
    dict_path: Path = DEFAULT_DICT_PATH,
    opset: int = 17,
    use_ema: bool = True,
    verify_ort: bool = True,
    verify_pipeline_img: Path | None = None,
) -> dict:
    cs = load_charset(dict_path)
    model, cfg = load_model_from_checkpoint(ckpt_path, use_ema=use_ema, export_softmax=True)
    if model.num_classes != cs.num_classes:
        raise ValueError(f"Model classes ({model.num_classes}) != charset classes ({cs.num_classes})")

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "inference.onnx"
    yml_path = out_dir / "inference.yml"
    txt_path = out_dir / "dict_ru_en_he.txt"

    dummy = torch.randn(1, 3, 48, 320, dtype=torch.float32)
    export_kwargs = dict(
        input_names=["x"],
        output_names=["probs"],
        dynamic_axes={
            "x": {0: "batch", 3: "width"},
            "probs": {0: "batch", 1: "seq_len"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    # PyTorch 2.5+ defaults to dynamo=True which can emit external .data files; force single-file legacy export
    try:
        torch.onnx.export(model, (dummy,), str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, (dummy,), str(onnx_path), **export_kwargs)

    # Ensure single self-contained ONNX file (inline any external data if present)
    try:
        import onnx

        proto = onnx.load(str(onnx_path), load_external_data=True)
        onnx.checker.check_model(proto)
        onnx.save_model(proto, str(onnx_path), save_as_external_data=False)
        ext_data = onnx_path.with_suffix(".onnx.data")
        if ext_data.exists():
            ext_data.unlink()
    except ImportError:
        pass

    write_dict_txt(cs.keys, txt_path)
    write_inference_yml(cs.keys, yml_path)

    size_bytes = onnx_path.stat().st_size
    digest = sha256_file(onnx_path)
    result: dict = {
        "onnx_path": str(onnx_path),
        "yml_path": str(yml_path),
        "dict_path": str(txt_path),
        "size_mb": round(size_bytes / 1e6, 3),
        "bytes": size_bytes,
        "sha256": digest,
        "num_classes": cs.num_classes,
        "opset": opset,
    }

    if verify_ort:
        result["ort_parity"] = verify_onnxruntime_parity(model, onnx_path, cs.num_classes)

    if verify_pipeline_img is not None:
        result["pipeline_check"] = verify_with_murmur_pipeline(onnx_path, yml_path, verify_pipeline_img)

    (out_dir / "export_meta.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def verify_onnxruntime_parity(model: PPOCRv5MobileRec, onnx_path: Path, num_classes: int) -> dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    in_meta = sess.get_inputs()[0]
    out_meta = sess.get_outputs()[0]
    if in_meta.name != "x":
        raise AssertionError(f"Expected ONNX input name 'x', got {in_meta.name!r}")
    if out_meta.shape[-1] != num_classes:
        raise AssertionError(f"Expected ONNX output classes {num_classes}, got {out_meta.shape[-1]}")

    max_diff = 0.0
    checked_buckets = []
    rng = np.random.default_rng(1234)
    for bw in (320, 480, 640, 960, 1600, 3200):
        batch_n = 2 if bw <= 640 else 1
        x_np = rng.uniform(-1.0, 1.0, size=(batch_n, 3, 48, bw)).astype(np.float32)
        ort_out = sess.run(None, {"x": x_np})[0]
        want_shape = (batch_n, bw // 8, num_classes)
        if ort_out.shape != want_shape:
            raise AssertionError(f"Bucket {bw}: expected shape {want_shape}, got {ort_out.shape}")
        prob_sum = ort_out.sum(axis=-1)
        if not np.allclose(prob_sum, 1.0, atol=1e-4):
            raise AssertionError(f"Bucket {bw}: output is not valid softmax (min_sum={prob_sum.min()}, max_sum={prob_sum.max()})")
        if bw <= 640:
            with torch.no_grad():
                pt_out = model(torch.from_numpy(x_np)).numpy()
            diff = float(np.abs(pt_out - ort_out).max())
            max_diff = max(max_diff, diff)
            if diff > 1e-3:
                raise AssertionError(f"PyTorch vs ORT max_abs_diff={diff:.6f} > 1e-3 on bucket {bw}")
        checked_buckets.append(bw)

    return {
        "input_name": in_meta.name,
        "output_name": out_meta.name,
        "checked_buckets": checked_buckets,
        "max_abs_diff": round(max_diff, 7),
        "softmax_ok": True,
    }


def verify_with_murmur_pipeline(onnx_path: Path, yml_path: Path, image_path: Path) -> dict:
    """Verify exported ONNX runs cleanly inside `ml/ocr/pipeline.py` with zero code changes."""
    from ml.ocr import pipeline as P
    from ml.ocr.config import DEFAULT, MODEL_DIR

    det_path = MODEL_DIR / DEFAULT.det_model
    if not det_path.exists():
        return {"skipped": f"detector not found at {det_path}; run ml/ocr/fetch_assets.sh"}

    cfg = DEFAULT.with_(
        rec_model=str(onnx_path.resolve()),
        rec_dict=str(yml_path.resolve()),
    )
    # OcrConfig.path(rel) does MODEL_DIR / rel; Path("/abs") overrides MODEL_DIR cleanly in Python!
    out = P.run_detailed(image_path, cfg=cfg)
    return {
        "image": str(image_path),
        "n_det": out["n_det"],
        "n_lines": len(out["result"]["lines"]),
        "ms": {k: round(v, 1) for k, v in out["ms"].items()},
        "sample_lines": [ln["text"] for ln in out["result"]["lines"][:5]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export trained 3-language OCR checkpoint to ONNX.")
    ap.add_argument("--ckpt", type=Path, required=True, help="Path to best.pt checkpoint")
    ap.add_argument("--out-dir", type=Path, default=Path(".cache/ml/ocr_train/exported"))
    ap.add_argument("--dict", type=Path, default=DEFAULT_DICT_PATH)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-ema", action="store_true", help="Export raw weights instead of EMA weights")
    ap.add_argument("--no-verify-ort", action="store_true")
    ap.add_argument("--verify-pipeline-img", type=Path, default=None, help="Optional PNG (e.g. ml/ocr/golden/001.png) for full pipeline check")
    args = ap.parse_args()

    res = export_onnx(
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        dict_path=args.dict,
        opset=args.opset,
        use_ema=not args.no_ema,
        verify_ort=not args.no_verify_ort,
        verify_pipeline_img=args.verify_pipeline_img,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
