"""Train the 3-language (ru + en + he) PP-OCRv5 Mobile Recognizer in PyTorch.

Designed for 1x RTX 5090 (32 GB, bf16):
  - Exact preprocessing parity with `ml/ocr/pipeline.py` (BGR, `(v/255 - 0.5)/0.5`, right zero-padded
    to `OcrConfig.rec_buckets` widths `[320, 480, 640, 800, 960, 1280, ...]`).
  - Bucketed batch sampling so every batch uses minimal padding and matches production bucket widths.
  - Visual-order CTC loss (`nn.CTCLoss(blank=0, zero_infinity=True)`) computed in FP32 for stability.
  - Optional pretrained weight transfer (`--init-onnx`) AND optional online teacher logit distillation
    (`--teacher-onnx`) from stock `cyrillic_PP-OCRv5_mobile_rec_onnx/inference.onnx` on ru/en samples
    (mapping stock indices 0..850 -> 0..850 and space 851 -> 882).
  - Live per-script validation CER & exact-match logging (`he`, `he_en`, `ru_he`, `ru`, `ru_en`, `en`).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from .charset import DEFAULT_DICT_PATH, EXPECTED_BASE_LEN, Charset, load_charset
from .model import PPOCRv5MobileRec, load_checkpoint_weights, transfer_from_onnx

REC_HEIGHT = 48
REC_BUCKETS = (320, 480, 640, 800, 960, 1280, 1600, 1920, 2560, 3200)
HE_CATS = {"he", "he_en", "ru_he"}
RU_EN_CATS = {"ru", "ru_en", "en"}


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def pick_bucket(rw: int, max_width: int = 1280) -> tuple[int, int]:
    """Return `(clamped_rw, bucket_width)` using `OcrConfig.rec_buckets`."""
    cap = max_width
    if rw > cap:
        rw = cap
    for b in REC_BUCKETS:
        if b > cap:
            break
        if rw <= b:
            return rw, b
    return min(rw, cap), cap


def _resize_bgr_f32(bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    """Bilinear resize in float32 matching `ml/ocr/pipeline.py:resize_bilinear`."""
    try:
        import cv2  # type: ignore

        return cv2.resize(bgr.astype(np.float32, copy=False), (w, h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        im = Image.fromarray(bgr.astype(np.uint8, copy=False))
        return np.asarray(im.resize((w, h), resample=Image.Resampling.BILINEAR), dtype=np.float32)


def _load_manifest(crops_dir: Path, split: str) -> list[dict]:
    meta = crops_dir / f"{split}_meta.jsonl"
    if meta.exists():
        rows = []
        with meta.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    lst = crops_dir / f"{split}_list.txt"
    if lst.exists():
        rows = []
        with lst.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.rstrip("\n")
                if not line or "\t" not in line:
                    continue
                img_rel, vis = line.split("\t", 1)
                rows.append({"id": idx, "image": img_rel, "visual": vis, "text": vis, "cat": "unknown", "width": 320})
        return rows
    raise FileNotFoundError(f"Neither {meta} nor {lst} exists in {crops_dir}")


class OcrCropDataset(Dataset):
    def __init__(
        self,
        crops_dir: Path,
        rows: list[dict],
        cs: Charset,
        max_width: int = 1280,
        augment: bool = False,
    ):
        self.crops_dir = crops_dir
        self.cs = cs
        self.max_width = max_width
        self.augment = augment

        # Filter to CTC-valid samples (target length strictly < max time steps T = bucket_w // 8)
        self.samples: list[dict] = []
        self.buckets: list[int] = []
        for r in rows:
            vis = r["visual"]
            if not vis or not cs.can_encode(vis):
                continue
            ids = cs.encode(vis)
            w = int(r.get("width", 320))
            _, bw = pick_bucket(w, max_width=max_width)
            # Ensure CTC time steps T = bw // 8 can comfortably emit ids + repeats
            min_needed_t = len(ids) + sum(1 for a, b in zip(ids, ids[1:]) if a == b) + 2
            while bw // 8 < min_needed_t and bw < max_width:
                next_b = [b for b in REC_BUCKETS if b > bw and b <= max_width]
                if not next_b:
                    break
                bw = next_b[0]
            if bw // 8 < min_needed_t:
                continue
            self.samples.append({
                "image": r["image"],
                "visual": vis,
                "ids": ids,
                "cat": r.get("cat", "unknown"),
                "bucket": bw,
            })
            self.buckets.append(bw)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        p = self.crops_dir / s["image"]
        rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        bgr = rgb[:, :, ::-1]
        h, w = bgr.shape[:2]

        # Compute target width preserving aspect ratio for height = 48
        rw = max(8, int(math.ceil(REC_HEIGHT * w / max(1, h))))
        if self.augment:
            # Mild horizontal stretch (±6%) matching font-width variations across devices
            if random.random() < 0.45:
                rw = max(8, int(round(rw * random.uniform(0.94, 1.06))))
        rw, bw = pick_bucket(rw, max_width=self.max_width)

        if h != REC_HEIGHT or w != rw:
            arr = _resize_bgr_f32(bgr, rw, REC_HEIGHT)
        else:
            arr = bgr.astype(np.float32)

        if self.augment:
            # Mild brightness/contrast jitter (±8%)
            if random.random() < 0.40:
                alpha = random.uniform(0.92, 1.08)
                beta = random.uniform(-10.0, 10.0)
                arr = np.clip(arr * alpha + beta, 0.0, 255.0)
            # Subtle per-channel tint shift (±4)
            if random.random() < 0.25:
                shift = np.random.uniform(-4.0, 4.0, size=(1, 1, 3)).astype(np.float32)
                arr = np.clip(arr + shift, 0.0, 255.0)

        # Exact `ml/ocr/pipeline.py:rec_input` normalization: (v / 255 - 0.5) / 0.5, CHW
        arr = (arr / np.float32(255.0) - np.float32(0.5)) / np.float32(0.5)
        chw = np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)
        return {
            "chw": chw,
            "rw": rw,
            "bucket": bw,
            "ids": s["ids"],
            "visual": s["visual"],
            "cat": s["cat"],
        }


class BucketBatchSampler(Sampler[list[int]]):
    """Groups crops into production `OcrConfig.rec_buckets` widths for fast training with zero excess pad."""

    def __init__(self, buckets: list[int], batch_size: int, shuffle: bool = True, drop_last: bool = False, seed: int = 42):
        self.buckets = buckets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        self.by_bucket: dict[int, list[int]] = defaultdict(list)
        for i, b in enumerate(buckets):
            self.by_bucket[b].append(i)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for b, idxs in sorted(self.by_bucket.items()):
            idxs = list(idxs)
            if self.shuffle:
                rng.shuffle(idxs)
            # Scale batch size gently down for very wide buckets (>=960) to keep VRAM flat
            bs = self.batch_size if b <= 640 else max(16, self.batch_size // 2)
            for k in range(0, len(idxs), bs):
                chunk = idxs[k : k + bs]
                if self.drop_last and len(chunk) < bs:
                    continue
                batches.append(chunk)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        total = 0
        for b, idxs in self.by_bucket.items():
            bs = self.batch_size if b <= 640 else max(16, self.batch_size // 2)
            total += len(idxs) // bs if self.drop_last else math.ceil(len(idxs) / bs)
        return total


def collate_bucket(batch: list[dict]) -> dict:
    bw = max(item["bucket"] for item in batch)
    n = len(batch)
    x = np.zeros((n, 3, REC_HEIGHT, bw), dtype=np.float32)  # right zero-padded = mid-grey
    target_list: list[int] = []
    target_lengths: list[int] = []
    visuals: list[str] = []
    cats: list[str] = []

    for i, item in enumerate(batch):
        rw = min(item["rw"], bw)
        x[i, :, :, :rw] = item["chw"][:, :, :rw]
        target_list.extend(item["ids"])
        target_lengths.append(len(item["ids"]))
        visuals.append(item["visual"])
        cats.append(item["cat"])

    return {
        "x": torch.from_numpy(x),
        "targets": torch.tensor(target_list, dtype=torch.long),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "visuals": visuals,
        "cats": cats,
    }


# ----------------------------------------------------------------------------- optional ONNX teacher KD


class OnnxTeacher:
    """Runs stock 852-class `cyrillic_PP-OCRv5_mobile_rec` ONNX to distill soft probs onto ru/en crops."""

    def __init__(self, onnx_path: Path, num_student_classes: int, base_dict_len: int = EXPECTED_BASE_LEN):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        avail = ort.get_available_providers()
        provs = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=provs)
        self.in_name = self.sess.get_inputs()[0].name
        self.num_student_classes = num_student_classes
        self.base_dict_len = base_dict_len
        self.stock_classes = base_dict_len + 2  # 852

    def kl_loss(self, x_cpu: np.ndarray, student_log_probs: torch.Tensor, cats: list[str]) -> torch.Tensor:
        """Compute KL(student || teacher) on ru/ru_en/en samples in the batch."""
        idxs = [i for i, c in enumerate(cats) if c in RU_EN_CATS]
        if not idxs:
            return student_log_probs.new_zeros(())
        sub_x = np.ascontiguousarray(x_cpu[idxs], dtype=np.float32)
        t_probs_np = self.sess.run(None, {self.in_name: sub_x})[0]  # [M, T, 852]
        if t_probs_np.shape[-1] != self.stock_classes:
            return student_log_probs.new_zeros(())

        device = student_log_probs.device
        t_probs = torch.from_numpy(t_probs_np).to(device=device, dtype=torch.float32)
        # Scatter stock [0..850, 851] -> student [0..850, 882]
        mapped = torch.zeros(
            (t_probs.shape[0], t_probs.shape[1], self.num_student_classes),
            device=device,
            dtype=torch.float32,
        )
        mapped[:, :, : self.base_dict_len + 1] = t_probs[:, :, : self.base_dict_len + 1]
        mapped[:, :, self.num_student_classes - 1] = t_probs[:, :, self.stock_classes - 1]
        mapped = mapped.clamp(min=1e-7)
        mapped = mapped / mapped.sum(dim=-1, keepdim=True)

        s_lp = student_log_probs[idxs].float()
        return F.kl_div(s_lp, mapped, reduction="batchmean")


# ----------------------------------------------------------------------------- evaluation & EMA


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cs: Charset,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    show_examples: int = 4,
) -> dict:
    model.eval()
    tot_err = tot_chars = tot_exact = tot_n = 0
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # err, chars, exact, n
    examples: list[dict] = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        preds = logits.argmax(dim=-1).cpu().tolist()
        for pred_ids, gt_vis, cat in zip(preds, batch["visuals"], batch["cats"]):
            hyp = cs.decode_ids(pred_ids)
            err = levenshtein(gt_vis, hyp)
            n_ch = max(1, len(gt_vis))
            ex = int(hyp == gt_vis)
            tot_err += err
            tot_chars += n_ch
            tot_exact += ex
            tot_n += 1
            st = by_cat[cat]
            st[0] += err
            st[1] += n_ch
            st[2] += ex
            st[3] += 1
            if len(examples) < show_examples and (err > 0 or len(examples) < 2):
                examples.append({"cat": cat, "gt_vis": gt_vis, "hyp_vis": hyp, "err": err})

    def _agg(cats: set[str]) -> tuple[float, float, int]:
        e = sum(by_cat[c][0] for c in cats if c in by_cat)
        ch = sum(by_cat[c][1] for c in cats if c in by_cat)
        ex = sum(by_cat[c][2] for c in cats if c in by_cat)
        nn_ = sum(by_cat[c][3] for c in cats if c in by_cat)
        return (e / max(1, ch), ex / max(1, nn_), nn_)

    he_cer, he_acc, he_n = _agg(HE_CATS)
    ru_cer, ru_acc, ru_n = _agg(RU_EN_CATS)
    all_cer = tot_err / max(1, tot_chars)
    all_acc = tot_exact / max(1, tot_n)
    bal_cer = 0.5 * he_cer + 0.5 * ru_cer if (he_n > 0 and ru_n > 0) else all_cer

    per_cat = {
        c: {
            "cer": round(v[0] / max(1, v[1]), 4),
            "exact": round(v[2] / max(1, v[3]), 4),
            "n": v[3],
        }
        for c, v in sorted(by_cat.items())
    }
    return {
        "cer": round(all_cer, 5),
        "exact": round(all_acc, 5),
        "balanced_cer": round(bal_cer, 5),
        "he_cer": round(he_cer, 5),
        "he_exact": round(he_acc, 5),
        "ru_en_cer": round(ru_cer, 5),
        "ru_en_exact": round(ru_acc, 5),
        "n": tot_n,
        "by_cat": per_cat,
        "examples": examples,
    }


def _cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * min(1.0, prog)))


def _resolve_device_and_amp(device_arg: str, amp_arg: str) -> tuple[torch.device, torch.dtype | None]:
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_arg)

    if amp_arg == "none" or device.type != "cuda":
        return device, None
    if amp_arg == "bf16":
        return device, torch.bfloat16
    if amp_arg == "fp16":
        return device, torch.float16
    # auto on CUDA
    return device, (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)


def train(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cs = load_charset(args.dict)
    device, amp_dtype = _resolve_device_and_amp(args.device, args.amp)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_rows = _load_manifest(args.crops_dir, "train")
    val_rows = _load_manifest(args.crops_dir, "val")
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    if args.limit_val:
        val_rows = val_rows[: args.limit_val]

    train_ds = OcrCropDataset(args.crops_dir, train_rows, cs, max_width=args.max_width, augment=True)
    val_ds = OcrCropDataset(args.crops_dir, val_rows, cs, max_width=args.max_width, augment=False)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Empty dataset: train={len(train_ds)}, val={len(val_ds)}")

    train_sampler = BucketBatchSampler(
        train_ds.buckets, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size, seed=args.seed
    )
    val_sampler = BucketBatchSampler(val_ds.buckets, batch_size=args.batch_size, shuffle=False, drop_last=False, seed=args.seed)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=pin,
        persistent_workers=args.workers > 0,
        collate_fn=collate_bucket,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        num_workers=max(0, min(4, args.workers)),
        pin_memory=pin,
        collate_fn=collate_bucket,
    )

    model = PPOCRv5MobileRec(
        num_classes=cs.num_classes,
        scale=args.scale,
        svtr_dims=args.svtr_dims,
        svtr_depth=args.svtr_depth,
        drop=args.dropout,
        export_softmax=False,
    )
    print(
        f"[train_rec] Model: PPOCRv5MobileRec(scale={args.scale}, svtr={args.svtr_dims}x{args.svtr_depth}, "
        f"classes={cs.num_classes}, params={model.count_parameters():,}, ~{model.count_parameters() * 4 / 1e6:.2f} MB FP32)",
        flush=True,
    )
    print(
        f"[train_rec] Device={device}, AMP={amp_dtype}, train={len(train_ds):,}, val={len(val_ds):,}, "
        f"bs={args.batch_size}, epochs={args.epochs}, lr={args.lr}",
        flush=True,
    )

    if args.init_onnx is not None and args.init_onnx.exists():
        info = transfer_from_onnx(model, args.init_onnx)
        print(f"[train_rec] Transferred ONNX weights: {json.dumps(info)}", flush=True)
    elif args.init_ckpt is not None and args.init_ckpt.exists():
        info = load_checkpoint_weights(model, args.init_ckpt)
        print(f"[train_rec] Loaded init checkpoint: {json.dumps(info)}", flush=True)

    model.to(device)
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    teacher = None
    if args.teacher_onnx is not None and args.teacher_onnx.exists() and args.distill_weight > 0:
        teacher = OnnxTeacher(args.teacher_onnx, num_student_classes=cs.num_classes)
        print(
            f"[train_rec] Online teacher KD enabled from {args.teacher_onnx} (weight={args.distill_weight})",
            flush=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))
    ctc_loss_fn = nn.CTCLoss(blank=cs.blank_id, reduction="mean", zero_infinity=True)

    start_epoch = 0
    best_score = float("inf")
    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if ema is not None and "ema" in ckpt:
            ema.ema.load_state_dict(ckpt["ema"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_score = float(ckpt.get("best_score", float("inf")))
        print(f"[train_rec] Resumed from {args.resume} (epoch={start_epoch}, best={best_score:.5f})", flush=True)

    if args.compile and hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model)  # type: ignore

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.out_dir / "history.jsonl"

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    global_step = start_epoch * steps_per_epoch

    best_metrics: dict = {}
    t_start = time.perf_counter()

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        ep_t0 = time.perf_counter()
        running_loss = 0.0
        running_ctc = 0.0
        running_kd = 0.0
        n_batches = 0

        for step_in_ep, batch in enumerate(train_loader):
            lr = _cosine_lr(global_step, total_steps, warmup_steps, args.lr, args.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            x_cpu = batch["x"].numpy() if teacher is not None else None
            x = batch["x"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x)  # [N, T, C]

            # Always run log_softmax + CTCLoss in FP32 to prevent overflow/NaN
            log_probs_ntc = F.log_softmax(logits.float(), dim=-1)
            log_probs_tnc = log_probs_ntc.transpose(0, 1)  # [T, N, C]
            input_lengths = torch.full((x.shape[0],), log_probs_tnc.shape[0], dtype=torch.long, device=device)
            loss_ctc = ctc_loss_fn(log_probs_tnc, targets, input_lengths, target_lengths)

            if teacher is not None and x_cpu is not None:
                loss_kd = teacher.kl_loss(x_cpu, log_probs_ntc, batch["cats"])
                loss = loss_ctc + args.distill_weight * loss_kd
            else:
                loss_kd = loss_ctc.new_zeros(())
                loss = loss_ctc

            if not torch.isfinite(loss):
                print(f"[train_rec] WARN non-finite loss at ep={epoch + 1} step={step_in_ep}; skipping batch", flush=True)
                global_step += 1
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            if ema is not None:
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                ema.update(raw_model)

            running_loss += float(loss.item())
            running_ctc += float(loss_ctc.item())
            running_kd += float(loss_kd.item())
            n_batches += 1
            global_step += 1

            if (step_in_ep + 1) % args.log_every == 0 or (step_in_ep + 1) == steps_per_epoch:
                dt = max(1e-3, time.perf_counter() - ep_t0)
                print(
                    f"[ep {epoch + 1:02d}/{args.epochs:02d}] step {step_in_ep + 1:04d}/{steps_per_epoch:04d} | "
                    f"loss={running_loss / n_batches:.4f} (ctc={running_ctc / n_batches:.4f}"
                    + (f", kd={running_kd / n_batches:.4f}" if teacher is not None else "")
                    + f") | lr={lr:.2e} | {dt:.1f}s",
                    flush=True,
                )

        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        eval_model = ema.ema if (ema is not None and epoch >= 1) else raw_model
        val_metrics = evaluate(eval_model, val_loader, cs, device, amp_dtype)
        ep_elapsed = time.perf_counter() - ep_t0

        print(
            f"==> [ep {epoch + 1:02d}/{args.epochs:02d}] val CER={val_metrics['cer']:.4f} exact={val_metrics['exact']:.4f} | "
            f"HE CER={val_metrics['he_cer']:.4f} exact={val_metrics['he_exact']:.4f} | "
            f"RU+EN CER={val_metrics['ru_en_cer']:.4f} exact={val_metrics['ru_en_exact']:.4f} | "
            f"bal={val_metrics['balanced_cer']:.4f} ({ep_elapsed:.1f}s)",
            flush=True,
        )
        print(f"    per-cat: {json.dumps(val_metrics['by_cat'], ensure_ascii=False)}", flush=True)

        record = {
            "epoch": epoch + 1,
            "train_loss": round(running_loss / max(1, n_batches), 5),
            "val": val_metrics,
            "elapsed_s": round(ep_elapsed, 1),
        }
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        ckpt_payload = {
            "epoch": epoch + 1,
            "model": raw_model.state_dict(),
            "ema": ema.ema.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict(),
            "best_score": min(best_score, val_metrics["balanced_cer"]),
            "val": val_metrics,
            "config": {
                "num_classes": cs.num_classes,
                "scale": args.scale,
                "svtr_dims": args.svtr_dims,
                "svtr_depth": args.svtr_depth,
                "dict_len": len(cs.keys),
            },
        }
        torch.save(ckpt_payload, args.out_dir / "last.pt")
        if val_metrics["balanced_cer"] < best_score:
            best_score = val_metrics["balanced_cer"]
            best_metrics = val_metrics
            torch.save(ckpt_payload, args.out_dir / "best.pt")
            print(f"    [*] Saved new best.pt (balanced_cer={best_score:.5f})", flush=True)

    summary = {
        "total_elapsed_s": round(time.perf_counter() - t_start, 1),
        "best_balanced_cer": round(best_score, 5),
        "best_val": best_metrics,
        "best_ckpt": str(args.out_dir / "best.pt"),
        "last_ckpt": str(args.out_dir / "last.pt"),
    }
    (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train 3-language (ru+en+he) PP-OCRv5 Mobile Recognizer.")
    ap.add_argument("--crops-dir", type=Path, default=Path(".cache/ml/ocr_train/crops"))
    ap.add_argument("--out-dir", type=Path, default=Path(".cache/ml/ocr_train/runs/ru_en_he_v1"))
    ap.add_argument("--dict", type=Path, default=DEFAULT_DICT_PATH)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--scale", type=float, default=0.95)
    ap.add_argument("--svtr-dims", type=int, default=120)
    ap.add_argument("--svtr-depth", type=int, default=2)
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "none"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--init-onnx", type=Path, default=None, help="Optional stock cyrillic_PP-OCRv5_mobile_rec inference.onnx for weight transfer")
    ap.add_argument("--init-ckpt", type=Path, default=None, help="Optional PyTorch .pt checkpoint to initialize from")
    ap.add_argument("--teacher-onnx", type=Path, default=None, help="Optional stock ONNX teacher for ru/en KL distillation")
    ap.add_argument("--distill-weight", type=float, default=0.25)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-val", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="Fast 2-epoch sanity training run")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.batch_size = min(args.batch_size, 32)
        args.workers = 0
        args.log_every = 10
        args.ema_decay = 0.0
    summary = train(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
