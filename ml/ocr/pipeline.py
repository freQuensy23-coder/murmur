"""Reference OCR context pipeline (CONTEXT_V2): PP-OCRv6_tiny_det + cyrillic_PP-OCRv5_mobile_rec on onnxruntime CPU.

    from ml.ocr.pipeline import run
    out = run("screen.png", exclude_rects=[(0, 0, 1080, 110)])   # -> {"size": [w, h], "lines": [...]}

Only numpy + onnxruntime do the work; OpenCV is used for two operations with fixed, documented semantics
(float32 bilinear resize with half-pixel centres, 8-connected components). README.md is the porting spec.
"""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import numpy as np

from . import postprocess as pp
from .config import DEFAULT, OcrConfig

_LOCK = threading.Lock()
_CACHE: dict = {}


# ----------------------------------------------------------------------------- image I/O


def load_rgb(image) -> np.ndarray:
    """Path / bytes / PIL image / HxWx3 uint8 RGB array -> HxWx3 uint8 RGB (alpha dropped, no colour mgmt)."""
    if isinstance(image, np.ndarray):
        a = image
    else:
        from PIL import Image
        import io
        if isinstance(image, (str, Path)):
            im = Image.open(image)
        elif isinstance(image, (bytes, bytearray)):
            im = Image.open(io.BytesIO(image))
        else:
            im = image
        a = np.asarray(im.convert("RGB"))
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"expected HxWx3 image, got {a.shape}")
    return np.ascontiguousarray(a[:, :, :3], dtype=np.uint8)


def resize_bilinear(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Bilinear resize computed in float32, half-pixel centres, edge clamp (== cv2.INTER_LINEAR on float32)."""
    import cv2
    return cv2.resize(img.astype(np.float32, copy=False), (w, h), interpolation=cv2.INTER_LINEAR)


# ----------------------------------------------------------------------------- sessions


class Models:
    def __init__(self, cfg: OcrConfig):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = cfg.threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        prov = ["CPUExecutionProvider"]
        self.det = ort.InferenceSession(str(cfg.path(cfg.det_model)), so, providers=prov)
        self.rec = ort.InferenceSession(str(cfg.path(cfg.rec_model)), so, providers=prov)
        keys = load_keys(cfg)
        # CTC classes: 0 = blank, 1..N = dictionary, N+1 = space (PaddleOCR use_space_char=True)
        self.chars = ["", *keys, " "]
        n_out = self.rec.get_outputs()[0].shape[-1]
        if n_out != len(self.chars):
            raise ValueError(f"dictionary has {len(self.chars)} classes, model outputs {n_out}")


def load_keys(cfg: OcrConfig = DEFAULT) -> list[str]:
    """Recogniser dictionary: PostProcess.character_dict of the model's inference.yml (YAML-parsed, so quoted
    entries such as "'" come out right), or a plain one-entry-per-line text file."""
    p = cfg.path(cfg.rec_dict)
    if p.suffix in (".yml", ".yaml"):
        import yaml
        keys = [str(k) for k in yaml.safe_load(p.read_text(encoding="utf-8"))["PostProcess"]["character_dict"]]
    else:
        keys = [k for k in p.read_text(encoding="utf-8").split("\n") if k != ""]
    return keys


def models(cfg: OcrConfig = DEFAULT) -> Models:
    key = (cfg.det_model, cfg.rec_model, cfg.rec_dict, cfg.threads)
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = Models(cfg)
        return _CACHE[key]


# ----------------------------------------------------------------------------- detector


def det_size(w: int, h: int, cfg: OcrConfig) -> tuple[int, int]:
    """Resized detector input (w, h): scale so the longest side <= det_limit (never upscale), then round each
    side half-up to a multiple of det_multiple (minimum one multiple)."""
    r = 1.0
    if cfg.det_limit and max(w, h) > cfg.det_limit:
        r = cfg.det_limit / max(w, h)
    m = cfg.det_multiple
    return max(m, int(math.floor(w * r / m + 0.5)) * m), max(m, int(math.floor(h * r / m + 0.5)) * m)


def det_input(rgb: np.ndarray, cfg: OcrConfig) -> tuple[np.ndarray, int, int]:
    h, w = rgb.shape[:2]
    tw, th = det_size(w, h, cfg)
    bgr = rgb[:, :, ::-1]
    x = resize_bilinear(bgr, tw, th) / np.float32(255.0)
    x = (x - np.array(cfg.det_mean, np.float32)) / np.array(cfg.det_std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32), tw, th


def prob_map(m: Models, x: np.ndarray, cfg: OcrConfig) -> np.ndarray:
    """Probability map (H, W) for the (1, 3, H, W) input, optionally in full-width horizontal tiles."""
    _, _, H, W = x.shape
    if not cfg.det_tile or H <= cfg.det_tile:
        return m.det.run(None, {"x": x})[0][0, 0]
    T, ov = cfg.det_tile, cfg.det_tile_overlap
    out = np.zeros((H, W), np.float32)
    y = 0
    while True:
        y1 = min(H, y + T)
        y0 = max(0, y1 - T)
        p = m.det.run(None, {"x": np.ascontiguousarray(x[:, :, y0:y1])})[0][0, 0]
        # each tile owns rows from the middle of the overlap above to the middle of the overlap below
        own0 = 0 if y0 == 0 else (y0 + ov // 2)
        own1 = H if y1 == H else (y1 - ov // 2)
        out[own0:own1] = p[own0 - y0:own1 - y0]
        if y1 == H:
            break
        y = y1 - ov
    return out


def db_boxes(prob: np.ndarray, src_w: int, src_h: int, cfg: OcrConfig) -> list[tuple[list[int], float]]:
    """DB post-processing with axis-aligned boxes: threshold -> 8-connected components -> box score ->
    unclip -> scale to the source image. Returns [(box, det_score)] sorted by (y0, x0)."""
    import cv2
    H, W = prob.shape
    binary = (prob > cfg.db_thresh).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_32S)
    sx, sy = src_w / W, src_h / H
    # integral image for O(1) box means
    ii = np.zeros((H + 1, W + 1), np.float64)
    ii[1:, 1:] = prob.astype(np.float64).cumsum(0).cumsum(1)
    out = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i, :4])
        if min(w, h) < cfg.db_min_side:
            continue
        s = (ii[y + h, x + w] - ii[y, x + w] - ii[y + h, x] + ii[y, x]) / (w * h)
        if s < cfg.db_box_thresh:
            continue
        d = w * h * cfg.db_unclip_ratio / (2.0 * (w + h))
        bx0, by0, bx1, by1 = x - d, y - d, x + w + d, y + h + d
        if min(bx1 - bx0, by1 - by0) < cfg.db_min_side_unclipped:
            continue
        box = [int(math.floor(max(0.0, bx0) * sx + 0.5)), int(math.floor(max(0.0, by0) * sy + 0.5)),
               int(math.floor(min(float(W), bx1) * sx + 0.5)), int(math.floor(min(float(H), by1) * sy + 0.5))]
        box = [min(max(box[0], 0), src_w), min(max(box[1], 0), src_h),
               min(max(box[2], 0), src_w), min(max(box[3], 0), src_h)]
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue
        out.append((box, round(float(s), 4)))
    out.sort(key=lambda t: (t[0][1], t[0][0], t[0][3], t[0][2]))
    return out


# ----------------------------------------------------------------------------- recogniser


def rec_width(w: int, h: int, cfg: OcrConfig) -> tuple[int, int]:
    """(resized width, padded bucket width) for a crop of w x h."""
    rw = int(math.ceil(cfg.rec_height * w / h))
    for b in cfg.rec_buckets:
        if rw <= b:
            return rw, b
    return cfg.rec_buckets[-1], cfg.rec_buckets[-1]


def rec_input(rgb: np.ndarray, box, cfg: OcrConfig) -> tuple[np.ndarray, int]:
    x0, y0, x1, y1 = box
    crop = rgb[y0:y1, x0:x1, ::-1]
    rw, bw = rec_width(x1 - x0, y1 - y0, cfg)
    r = resize_bilinear(crop, rw, cfg.rec_height) / np.float32(255.0)
    r = (r - np.float32(0.5)) / np.float32(0.5)
    t = np.zeros((3, cfg.rec_height, bw), np.float32)
    t[:, :, :rw] = r.transpose(2, 0, 1)
    return t, bw


def ctc_decode(probs: np.ndarray, chars: list[str]) -> tuple[str, float]:
    """Greedy CTC: argmax per step (first maximum on ties), merge repeats, drop blanks (index 0).
    conf = mean of the max probability of the kept steps (0.0 for an empty result)."""
    idx = probs.argmax(-1)
    mx = probs.max(-1)
    text, confs, prev = [], [], -1
    for t, k in enumerate(idx.tolist()):
        if k != prev and k != 0:
            text.append(chars[k])
            confs.append(float(mx[t]))
        prev = k
    return "".join(text), (sum(confs) / len(confs) if confs else 0.0)


def recognise(m: Models, rgb: np.ndarray, boxes: list[list[int]], cfg: OcrConfig) -> list[tuple[str, float]]:
    items = [rec_input(rgb, b, cfg) for b in boxes]
    res: list = [None] * len(boxes)
    # batches of the same bucket width, in box order inside each bucket
    by_w: dict[int, list[int]] = {}
    for i, (_, bw) in enumerate(items):
        by_w.setdefault(bw, []).append(i)
    for bw in sorted(by_w):
        ids = by_w[bw]
        for k in range(0, len(ids), cfg.rec_batch):
            chunk = ids[k:k + cfg.rec_batch]
            x = np.stack([items[i][0] for i in chunk])
            p = m.rec.run(None, {"x": x})[0]
            for j, i in enumerate(chunk):
                res[i] = ctc_decode(p[j], m.chars)
    return res


# ----------------------------------------------------------------------------- the pipeline


def run_detailed(image, exclude_rects=(), cfg: OcrConfig = DEFAULT) -> dict:
    """Pipeline output plus intermediate stages and timings (for measurements; not part of the contract)."""
    t0 = time.perf_counter()
    rgb = load_rgb(image)
    h, w = rgb.shape[:2]
    m = models(cfg)
    x, tw, th = det_input(rgb, cfg)
    prob = prob_map(m, x, cfg)
    t1 = time.perf_counter()
    dets = db_boxes(prob, w, h, cfg)
    rects = [tuple(int(v) for v in r) for r in exclude_rects]
    boxes = [b for b, _ in dets if not pp.excluded(b, rects, cfg.exclude_min_overlap)]
    t2 = time.perf_counter()
    recs = recognise(m, rgb, boxes, cfg)
    t3 = time.perf_counter()
    raw = [{"text": pp.norm_text(t), "conf": c, "box": b} for b, (t, c) in zip(boxes, recs)]
    lines = [ln for ln in raw if pp.keep(ln, cfg)]
    dominant = pp.screen_script([ln["text"] for ln in lines], cfg.screen_script_ratio)
    for ln in lines:
        ln["text"] = pp.visual_to_logical(pp.repair_homoglyphs(ln["text"], dominant))
    lines = pp.dedup_geometric(lines, cfg.dedup_iou)
    lines = pp.reading_order(lines, w, h, cfg)
    lines = pp.dedup_text(lines)
    out = {"size": [w, h], "lines": [{"text": ln["text"], "conf": round(ln["conf"], cfg.conf_digits),
                                      "box": list(ln["box"])} for ln in lines]}
    t4 = time.perf_counter()
    return {"result": out, "raw": raw, "n_det": len(dets), "det_input": [tw, th],
            "ms": {"det": (t1 - t0) * 1e3, "rec": (t3 - t2) * 1e3, "total": (t4 - t0) * 1e3}}


def run(image, exclude_rects=(), cfg: OcrConfig = DEFAULT) -> dict:
    """CONTEXT_V2 interface: image -> {"size": [w, h], "lines": [{"text", "conf", "box"}]}."""
    return run_detailed(image, exclude_rects, cfg)["result"]
