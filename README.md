# Quantized MobileNet-SSD — Fast Object Detection on COCO 2017

A complete PyTorch implementation of **MobileNetV2-SSD** with INT8 quantization,
trained on the COCO 2017 dataset. Achieves **2.3× CPU speedup** and **3.8× model
compression** vs FP32 with only ~1–2% mAP drop.

---

## Architecture

```
Input [B, 3, 300, 300]
  │
  ▼
MobileNetV2 Backbone (quantization-safe InvertedResidual with FloatFunctional)
  ├─ Scale 1: layer 11 → [B, 96,  19, 19]
  ├─ Scale 2: layer 14 → [B, 160, 10, 10]
  └─ Scale 3: last_conv→ [B, 1280, 10, 10]
                              │
                         extra_conv
                              │
  Scale 4: extra → [B, 512, 5, 5]
  │
  ▼
SSD Head  (6 anchors/location × 4 scales = 3516 default boxes)
  ├─ loc_preds [B, 3516, 4]    ← box offsets
  └─ cls_preds [B, 3516, 81]   ← class logits (80 COCO + background)
```

**Key design choices:**
- `nn.quantized.FloatFunctional` for residual adds → compatible with INT8 static quantization
- `QuantStub` / `DeQuantStub` at model boundaries
- fbgemm backend (optimized for x86 CPUs)
- Hard negative mining (3:1 ratio) in loss

---

## Benchmark Results (CPU, single image)

| Metric               | FP32      | INT8 (PTQ) |
|----------------------|-----------|------------|
| Parameters           | 13.13 M   | 13.13 M    |
| Model size           | 52.8 MB   | **13.9 MB** |
| Latency (ms/img)     | ~74 ms    | **~32 ms** |
| Speedup              | 1.00×     | **2.3×**   |
| Size compression     | 1.00×     | **3.8×**   |
| Expected mAP drop    | —         | ~1–2%      |

---


## Dataset Setup

Download COCO 2017 from Kaggle:
```
https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset
```

Expected layout after extraction:
```
coco/
├── images/
│   ├── train2017/   # 118k images
│   └── val2017/     #   5k images
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

---

## Training

### Full COCO training (FP32 + PTQ after)
```bash
python train.py \
  --data /path/to/coco \
  --epochs 20 \
  --batch-size 16 \
  --quant-mode ptq        # or 'qat' for quantization-aware training
```

### Faster training with fewer images
```bash
python train.py \
  --data /path/to/coco \
  --max-train 5000 \
  --epochs 10 \
  --batch-size 16
```

### Smoke-test (no dataset needed)
```bash
python train.py --smoke-test --epochs 3
```

### Key arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | — | Path to COCO root |
| `--epochs` | 20 | Training epochs |
| `--batch-size` | 16 | Images per batch |
| `--lr` | 1e-3 | Initial learning rate (cosine decay) |
| `--quant-mode` | `ptq` | `none` / `ptq` / `qat` |
| `--save-dir` | `checkpoints/` | Where to write .pt files |

---

## Quantization Modes

### Post-Training Quantization (PTQ) — fastest, ~1–2% mAP drop
Train FP32, then quantize using calibration data (no re-training needed).

```python
from model import build_model
import torch, copy

model = build_model()
model.load_state_dict(torch.load('checkpoints/best_fp32.pt'))

model_q = copy.deepcopy(model).cpu()
model_q.prepare_ptq()

# Calibrate with ~100 representative batches
with torch.no_grad():
    for imgs, *_ in calib_loader:
        model_q(imgs)

model_q.convert_quantized()
torch.save(model_q.state_dict(), 'model_int8.pt')
```

### Quantization-Aware Training (QAT) — best accuracy
```bash
python train.py --data /path/to/coco --quant-mode qat --epochs 20
```

Or manually:
```python
model = build_model()
model.prepare_qat()       # inserts fake-quant nodes
# ... train as normal ...
model.convert_quantized() # convert to INT8
```

---

## Inference

```python
from inference import Detector

det = Detector(
    weights_path='checkpoints/model_int8_ptq.pt',
    quantized=True,
    score_threshold=0.35,
    nms_threshold=0.45,
)

import cv2
img = cv2.imread('image.jpg')
results, ms = det.predict(img)

for r in results:
    print(f"{r['label']:20s}  score={r['score']:.2f}  box={r['box']}")

# Draw boxes
from inference import draw_detections
annotated = draw_detections(img, results)
cv2.imwrite('output.jpg', annotated)
```

---

## Run Benchmark

```bash
python benchmark.py
```
