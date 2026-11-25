"""PP-OCRv5 Mobile Recognizer (SVTR_LCNet: PPLCNetV3 + EncoderWithSVTR + CTCHead) in pure PyTorch.

Exact contract with `ml/ocr/pipeline.py` and `core/ocr/OcrPipeline.kt`:
  - Input  : float32 BGR tensor `[N, 3, 48, W]`, normalized `(v / 255 - 0.5) / 0.5`, right zero-padded,
             where `W` is any multiple of 8 (production buckets: 320, 480, 640, ..., 3200).
  - Output : float32 `[N, W // 8, num_classes]` (logits in training; softmax probabilities on export),
             where `num_classes = 883` (0=blank, 1..850=base dict, 851..881=Hebrew, 882=space).
  - Size   : ~2.1M parameters (~8.4 MB FP32 ONNX, matching stock `cyrillic_PP-OCRv5_mobile_rec` 8.0 MB).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .charset import EXPECTED_BASE_LEN, EXPECTED_TOTAL_LEN


def make_divisible(v: float, divisor: int = 8, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | None = None,
        groups: int = 1,
        act: str = "hswish",
    ):
        super().__init__()
        if padding is None:
            if isinstance(kernel_size, tuple):
                padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            else:
                padding = kernel_size // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        if act == "hswish":
            self.act: nn.Module = nn.Hardswish(inplace=True)
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "none":
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unknown act: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """Lightweight Squeeze-Excitation matching PPLCNetV3 (ReLU -> Hardsigmoid)."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, mid, kernel_size=1)
        self.fc2 = nn.Conv2d(mid, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = F.hardsigmoid(self.fc2(s), inplace=True)
        return x * s


class DepthwiseSeparable(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int,
        stride: int | tuple[int, int] = 1,
        use_se: bool = False,
    ):
        super().__init__()
        self.dw = ConvBNAct(in_c, in_c, kernel_size=kernel_size, stride=stride, groups=in_c, act="hswish")
        self.se = SEBlock(in_c) if use_se else nn.Identity()
        self.pw = ConvBNAct(in_c, out_c, kernel_size=1, stride=1, padding=0, act="hswish")
        s_tuple = (stride, stride) if isinstance(stride, int) else stride
        self.use_res = (in_c == out_c) and (s_tuple == (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw(self.se(self.dw(x)))
        return x + y if self.use_res else y


class PPLCNetV3Backbone(nn.Module):
    """PPLCNetV3 recognition backbone: `[N, 3, 48, W]` -> `[N, out_channels, 1, W // 8]`.

    Height progression : 48 -> 24 -> 24 -> 12 -> 6 -> 3 -> 2 -> 1
    Width progression  :  W -> W/2 -> W/2 -> W/2 -> W/4 -> W/4 -> W/4 -> W/8
    """

    def __init__(self, scale: float = 0.95):
        super().__init__()
        c = lambda ch: make_divisible(ch * scale, 8)
        self.conv1 = ConvBNAct(3, c(16), kernel_size=3, stride=(2, 2), padding=1)

        # (k, in_c, out_c, stride, use_se)
        cfg = [
            # blocks2: 24 x W/2
            (3, 16, 32, (1, 1), False),
            # blocks3: 12 x W/2
            (3, 32, 64, (2, 1), False),
            (3, 64, 64, (1, 1), False),
            # blocks4: 6 x W/4
            (3, 64, 128, (2, 2), False),
            (3, 128, 128, (1, 1), False),
            # blocks5: 3 x W/4
            (3, 128, 256, (1, 1), False),
            (5, 256, 256, (2, 1), False),
            (5, 256, 256, (1, 1), False),
            (5, 256, 256, (1, 1), False),
            (5, 256, 256, (1, 1), False),
            # blocks6: 2 x W/4
            (5, 256, 512, (2, 1), True),
            (5, 512, 512, (1, 1), True),
            (5, 512, 512, (1, 1), True),
            (5, 512, 512, (1, 1), True),
        ]
        self.blocks = nn.Sequential(
            *[DepthwiseSeparable(c(ic), c(oc), k, s, se) for (k, ic, oc, s, se) in cfg]
        )
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.out_channels = c(512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.pool(x)  # [N, out_channels, 1, W // 8]
        return x


class SVTRBlock(nn.Module):
    """Standard Pre-LN Transformer block with ONNX-clean explicit scaled dot-product attention."""

    def __init__(self, dim: int = 120, num_heads: int = 8, mlp_ratio: float = 2.0, drop: float = 0.05):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.mlp_drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, D]
        n, t, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(n, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each [N, T, H, Dh]
        q = q.transpose(1, 2)  # [N, H, T, Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(n, t, d)
        x = x + self.proj(out)

        m = self.norm2(x)
        m = self.fc2(self.mlp_drop(F.gelu(self.fc1(m))))
        x = x + self.mlp_drop(m)
        return x


class EncoderWithSVTR(nn.Module):
    """PP-OCRv5 mobile SVTR sequence neck (`dims=120, depth=2, kernel_size=[1,3]`)."""

    def __init__(
        self,
        in_channels: int,
        dims: int = 120,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        drop: float = 0.05,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), groups=in_channels)
        self.conv2 = ConvBNAct(in_channels, dims, kernel_size=(1, 1), padding=(0, 0))
        self.svtr_blocks = nn.ModuleList(
            [SVTRBlock(dim=dims, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dims)
        self.conv3 = ConvBNAct(dims, dims, kernel_size=(1, 1), padding=(0, 0))
        self.conv4 = ConvBNAct(in_channels + dims, dims, kernel_size=(1, 3), padding=(0, 1))
        self.out_channels = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, 1, T]
        shortcut = x
        h = self.conv2(self.conv1(x))  # [N, dims, 1, T]
        seq = h.squeeze(2).transpose(1, 2)  # [N, T, dims]
        for blk in self.svtr_blocks:
            seq = blk(seq)
        seq = self.norm(seq)
        h = seq.transpose(1, 2).unsqueeze(2)  # [N, dims, 1, T]
        h = self.conv3(h)
        h = torch.cat([shortcut, h], dim=1)  # [N, C + dims, 1, T]
        h = self.conv4(h)  # [N, dims, 1, T]
        return h.squeeze(2).transpose(1, 2)  # [N, T, dims]


class PPOCRv5MobileRec(nn.Module):
    """Complete 3-language PP-OCRv5 Mobile Recognizer (~2.1M params, ~8.4 MB FP32 ONNX)."""

    def __init__(
        self,
        num_classes: int = EXPECTED_TOTAL_LEN + 2,  # 883
        scale: float = 0.95,
        svtr_dims: int = 120,
        svtr_depth: int = 2,
        svtr_heads: int = 8,
        drop: float = 0.05,
        export_softmax: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.svtr_dims = svtr_dims
        self.svtr_depth = svtr_depth
        self.export_softmax = export_softmax

        self.backbone = PPLCNetV3Backbone(scale=scale)
        self.neck = EncoderWithSVTR(
            in_channels=self.backbone.out_channels,
            dims=svtr_dims,
            depth=svtr_depth,
            num_heads=svtr_heads,
            drop=drop,
        )
        self.head = nn.Linear(self.neck.out_channels, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        seq = self.neck(feat)
        logits = self.head(seq)  # [N, W // 8, num_classes]
        if self.export_softmax:
            return F.softmax(logits, dim=-1)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------------- weight transfer helpers


def expand_ctc_head_weights(
    src_weight: torch.Tensor,
    src_bias: torch.Tensor | None,
    dst_weight: torch.Tensor,
    dst_bias: torch.Tensor | None,
    base_dict_len: int = EXPECTED_BASE_LEN,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Transfer 100% of stock Cyrillic/Latin+blank+space CTC head rows into an expanded Hebrew head.

    Stock layout (852 classes):
      0        -> blank
      1..850   -> base dictionary (rec_keys.txt)
      851      -> space (" ")

    Expanded layout (883 classes):
      0        -> blank         (copied from stock 0)
      1..850   -> base dict     (copied from stock 1..850)
      851..881 -> Hebrew chars  (freshly initialized matching stock weight scale)
      882      -> space (" ")   (copied from stock 851)
    """
    src_n = src_weight.shape[0]
    dst_n = dst_weight.shape[0]
    expected_src = base_dict_len + 2  # 852
    if src_n != expected_src:
        raise ValueError(f"Expected stock head with {expected_src} classes, got {src_n}")
    if dst_n < expected_src:
        raise ValueError(f"Destination head ({dst_n}) smaller than source ({src_n})")
    if src_weight.shape[1] != dst_weight.shape[1]:
        raise ValueError(f"Head hidden dim mismatch: src={src_weight.shape[1]} vs dst={dst_weight.shape[1]}")

    new_w = dst_weight.detach().clone()
    # Match new Hebrew class rows to stock weight std so logits start well-calibrated
    std = float(src_weight.float().std().item()) or 0.02
    nn.init.normal_(new_w, mean=0.0, std=std * 0.5)
    new_w[: base_dict_len + 1].copy_(src_weight[: base_dict_len + 1])
    new_w[dst_n - 1].copy_(src_weight[src_n - 1])  # space token

    new_b = None
    if dst_bias is not None:
        new_b = dst_bias.detach().clone()
        new_b.zero_()
        if src_bias is not None:
            new_b[: base_dict_len + 1].copy_(src_bias[: base_dict_len + 1])
            new_b[dst_n - 1].copy_(src_bias[src_n - 1])
    return new_w, new_b


def transfer_from_onnx(
    model: PPOCRv5MobileRec,
    onnx_path: Path,
    base_dict_len: int = EXPECTED_BASE_LEN,
) -> dict:
    """Transfer compatible pretrained weights (including 852->883 CTC head) from stock PP-OCRv5 ONNX.

    Because Paddle's exported inference ONNX fuses Conv+BN and may reorder or slightly rename
    initializers across Paddle2ONNX versions, this loader matches:
      1. The CTC head weight/bias (`852 x D` or `D x 852`) -> exact 0..850 + space(851->882) transfer.
      2. Shape-compatible 4D convolution kernels in topological graph order (with paired BN left
         at identity when weights are fused).
    Returns a summary dict of transferred tensors & parameter share.
    """
    import onnx
    from onnx import numpy_helper

    proto = onnx.load(str(onnx_path))
    inits = {t.name: numpy_helper.to_array(t) for t in proto.graph.initializer}

    # Walk graph nodes in execution order to preserve layer ordering
    ordered_2d: list[tuple[str, np.ndarray]] = []
    ordered_4d: list[tuple[str, np.ndarray]] = []
    ordered_1d: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for node in proto.graph.node:
        for inp in node.input:
            if inp in inits and inp not in seen:
                seen.add(inp)
                arr = inits[inp]
                if arr.ndim == 4:
                    ordered_4d.append((inp, arr))
                elif arr.ndim == 2:
                    ordered_2d.append((inp, arr))
                elif arr.ndim == 1:
                    ordered_1d.append((inp, arr))

    transferred_params = 0
    transferred_tensors = 0
    head_transferred = False
    src_classes = base_dict_len + 2  # 852
    head_dim = model.head.in_features

    # 1. Transfer CTC head (check both [D, 852] MatMul and [852, D] Gemm layouts)
    src_w = None
    src_b = None
    for name, arr in reversed(ordered_2d):
        if arr.shape == (src_classes, head_dim):
            src_w = torch.from_numpy(arr.copy()).float()
            break
        if arr.shape == (head_dim, src_classes):
            src_w = torch.from_numpy(arr.T.copy()).float()
            break
    for name, arr in reversed(ordered_1d):
        if arr.shape == (src_classes,):
            src_b = torch.from_numpy(arr.copy()).float()
            break

    if src_w is not None:
        new_w, new_b = expand_ctc_head_weights(
            src_w, src_b, model.head.weight, model.head.bias, base_dict_len=base_dict_len
        )
        with torch.no_grad():
            model.head.weight.copy_(new_w)
            if new_b is not None and model.head.bias is not None:
                model.head.bias.copy_(new_b)
        head_transferred = True
        transferred_tensors += 1 + (1 if src_b is not None else 0)
        transferred_params += int(src_w.numel()) + (int(src_b.numel()) if src_b is not None else 0)

    # 2. Greedily match Conv2d kernels in topological order when shapes match identically
    onnx_idx = 0
    matched_convs = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        want_shape = tuple(mod.weight.shape)
        # Look ahead a small window in ONNX 4D initializers to skip extra rep/SE convs
        found_j = None
        for j in range(onnx_idx, min(len(ordered_4d), onnx_idx + 8)):
            if ordered_4d[j][1].shape == want_shape:
                found_j = j
                break
        if found_j is not None:
            arr = ordered_4d[found_j][1]
            with torch.no_grad():
                mod.weight.copy_(torch.from_numpy(arr.copy()).float())
            onnx_idx = found_j + 1
            matched_convs += 1
            transferred_tensors += 1
            transferred_params += int(arr.size)

    return {
        "onnx_path": str(onnx_path),
        "head_transferred": head_transferred,
        "matched_convs": matched_convs,
        "transferred_tensors": transferred_tensors,
        "transferred_params": transferred_params,
        "total_params": model.count_parameters(),
        "transferred_ratio": round(transferred_params / max(1, model.count_parameters()), 4),
    }


def load_checkpoint_weights(
    model: PPOCRv5MobileRec,
    ckpt_path: Path,
    base_dict_len: int = EXPECTED_BASE_LEN,
) -> dict:
    """Load PyTorch checkpoint, automatically expanding an 852-class head to 883 if needed."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state

    dst_sd = model.state_dict()
    loaded = 0
    head_expanded = False
    for k, v in sd.items():
        if k not in dst_sd:
            continue
        if dst_sd[k].shape == v.shape:
            dst_sd[k].copy_(v)
            loaded += 1
        elif k == "head.weight" and v.shape[0] == base_dict_len + 2 and v.shape[1] == dst_sd[k].shape[1]:
            sb = sd.get("head.bias")
            nw, nb = expand_ctc_head_weights(v, sb, dst_sd["head.weight"], dst_sd.get("head.bias"), base_dict_len)
            dst_sd["head.weight"].copy_(nw)
            if nb is not None and "head.bias" in dst_sd:
                dst_sd["head.bias"].copy_(nb)
            head_expanded = True
            loaded += 1
    model.load_state_dict(dst_sd)
    return {"ckpt_path": str(ckpt_path), "loaded_tensors": loaded, "head_expanded": head_expanded}
