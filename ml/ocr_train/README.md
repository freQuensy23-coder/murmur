# `ml/ocr_train` — 3-Language (`ru + en + he`) PP-OCRv5 Mobile Recognizer Pipeline

Self-contained training, ONNX export, and validation pipeline producing **one ~8.9 MB ONNX recognizer** (`PPOCRv5MobileRec`: `PPLCNetV3` + `EncoderWithSVTR` + `CTCHead`) that reads **Russian + English + Hebrew** (including mixed code-switching lines like `עשיתי push ל-GitHub` and `Привет שלום`) in a single pass.

## Contract Parity (`ml/ocr/pipeline.py` & Android `OcrPipeline.kt`)

| Property | Exact Value |
|---|---|
| Input tensor | `"x"`, `float32`, `[N, 3, 48, W]` (dynamic `N`, dynamic `W` in `rec_buckets` `320..3200`) |
| Preprocessing | BGR channel order, float32 bilinear resize to `48×rw`, `(v/255 - 0.5)/0.5`, right zero-padded to bucket `W` |
| Output tensor | `"probs"`, `float32`, `[N, W/8, 883]` (softmax probabilities) |
| Dictionary | `dict_ru_en_he.txt` / `inference.yml`: **881 entries** (`0..849` identical to `ml/ocr/golden/rec_keys.txt`, `850..880` = 27 Hebrew letters `א-ת` incl. sofits `ךםןףץ` + `׳ ״ ₪ ־`) |
| CTC classes | `0` = blank, `1..850` = original Cyrillic/Latin/symbols, `851..881` = Hebrew, `882` = space (`" "`) |
| Visual-order CTC | Labels converted to visual L→R order via `python-bidi` (`get_display`); `OcrPostProcess` restores logical order at inference |

---

## Prerequisites

On Ubuntu/Debian (both CPU prep machine `openclaw-dp` and RTX 5090 pod), ensure HarfBuzz + FriBidi are installed so Pillow `raqm` shapes Hebrew RTL properly (`render_crops.py` verifies this at startup and fails loudly if missing):

```bash
sudo apt-get update && sudo apt-get install -y libfribidi0 libharfbuzz0b libraqm0
pip install -r ml/ocr_train/requirements.txt
```

Fetch stock OCR assets (detector + stock Cyrillic rec for weight transfer / distillation / golden regression) and OFL Hebrew+Android fonts (automatically reuses `.cache/ml/ocr/fonts/` if already present):

```bash
bash ml/ocr/fetch_assets.sh
bash ml/ocr_train/fetch_fonts.sh
python3 -m ml.ocr_train.render_crops --check-only
```

---

## Turnkey Runbook

### Step 1 — Build Corpus & Render Crops (CPU — ~4–8 min on 16 cores)

Generates 400K balanced lines (`40% he`, `15% he+en`, `35% ru/ru+en`, `5% en/code`, `5% ru+he`) with visual-order CTC labels and renders realistic `48×W` Android UI crops across light/dark/Telegram/WhatsApp/code themes:

```bash
# 1a. Build 400K balanced 3-language corpus (HF wikipedia he/ru/en + ivrit-ai + Israeli UI/code-switch)
python3 -m ml.ocr_train.build_corpus \
  --out-dir .cache/ml/ocr_train/corpus \
  --num-lines 400000

# 1b. Render 48xW Android UI crops (auto-discovers .cache/ml/ocr/fonts/ + .cache/ml/ocr_train/fonts/)
python3 -m ml.ocr_train.render_crops \
  --corpus-dir .cache/ml/ocr_train/corpus \
  --out-dir .cache/ml/ocr_train/crops \
  --workers 16
```

*(If prepping on `openclaw-dp` and syncing to the RTX 5090 pod, rsync `.cache/ml/ocr_train/crops` and `.cache/ml/ocr` to the pod.)*

### Step 2 — Train on 1x RTX 5090 (32 GB, bf16 — ~35–55 min)

```bash
python3 -m ml.ocr_train.train_rec \
  --crops-dir .cache/ml/ocr_train/crops \
  --out-dir .cache/ml/ocr_train/runs/ru_en_he_v1 \
  --epochs 35 \
  --batch-size 256 \
  --lr 1.5e-3 \
  --amp bf16 \
  --workers 8 \
  --init-onnx .cache/ml/ocr/cyrillic_PP-OCRv5_mobile_rec_onnx/inference.onnx \
  --teacher-onnx .cache/ml/ocr/cyrillic_PP-OCRv5_mobile_rec_onnx/inference.onnx \
  --distill-weight 0.25
```

- Logs live validation CER & exact-match on both **Hebrew (`he`, `he_en`, `ru_he`)** and **RU+EN (`ru`, `ru_en`, `en`)** every epoch.
- Saves `best.pt` (lowest balanced `(he_cer + ru_en_cer) / 2`) and `last.pt` (resume anytime via `--resume`).
- Drop `--teacher-onnx` if you want ~20% faster step time without CPU/GPU teacher distillation.

### Step 3 — Export ONNX + Verify Parity & Golden Regression (~1 min)

```bash
# 3a. Export single-file inference.onnx + inference.yml + dict_ru_en_he.txt + verify ORT parity & pipeline
python3 -m ml.ocr_train.export_onnx \
  --ckpt .cache/ml/ocr_train/runs/ru_en_he_v1/best.pt \
  --out-dir .cache/ml/ocr_train/exported/ru_en_he_v1 \
  --verify-pipeline-img ml/ocr/golden/001.png

# 3b. Evaluate held-out he/he+en/ru+he val crops AND golden 001..026.png regression vs stock
python3 -m ml.ocr_train.eval_ocr \
  --onnx .cache/ml/ocr_train/exported/ru_en_he_v1/inference.onnx \
  --dict .cache/ml/ocr_train/exported/ru_en_he_v1/inference.yml \
  --crops-dir .cache/ml/ocr_train/crops \
  --golden-dir ml/ocr/golden \
  --out-json .cache/ml/ocr_train/exported/ru_en_he_v1/eval_report.json
```

---

## Fast Local Smoke Test (CPU, ~15 seconds)

```bash
python3 -m ml.ocr_train.build_corpus --smoke --out-dir /tmp/ocr_smoke/corpus
python3 -m ml.ocr_train.render_crops --corpus-dir /tmp/ocr_smoke/corpus --out-dir /tmp/ocr_smoke/crops --limit-train 128 --limit-val 32 --workers 4
python3 -m ml.ocr_train.train_rec --smoke --crops-dir /tmp/ocr_smoke/crops --out-dir /tmp/ocr_smoke/run --device cpu
pytest -q ml/ocr_train/tests
```
