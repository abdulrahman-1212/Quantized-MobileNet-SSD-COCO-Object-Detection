"""
Benchmark: FP32 vs INT8-PTQ MobileNet-SSD
Demonstrates PTQ workflow end-to-end with synthetic calibration data.
"""

import os, copy, time, tempfile
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import build_model, NUM_CLASSES


def measure_latency(model, n_runs=80, warmup=10):
    model.eval()
    x = torch.randn(1, 3, 300, 300)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(x)
    return (time.perf_counter() - t0) / n_runs * 1000


def model_size_mb(model):
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(model.state_dict(), f.name)
        mb = os.path.getsize(f.name) / 1e6
    os.unlink(f.name)
    return mb


def make_calib_loader(n=64, n_anchors=3516, batch=8):
    imgs  = torch.randn(n, 3, 300, 300)
    loc_t = torch.randn(n, n_anchors, 4)
    cls_t = torch.zeros(n, n_anchors, dtype=torch.long)
    return DataLoader(TensorDataset(imgs, loc_t, cls_t), batch_size=batch)


def ptq_pipeline(fp32_model, calib_loader):
    """Run PTQ: prepare → calibrate → convert."""
    model_q = copy.deepcopy(fp32_model).cpu().eval()
    model_q.prepare_ptq()
    print("   Calibrating …", end=" ", flush=True)
    with torch.no_grad():
        for imgs, *_ in calib_loader:
            model_q(imgs)
    model_q.convert_quantized()
    print("done")
    return model_q


def main():

    print(" Quantized MobileNet-SSD  ─  Benchmark")

    # FP32 
    print("\n Building FP32 model …")
    fp32 = build_model().eval()
    n_params = sum(p.numel() for p in fp32.parameters()) / 1e6
    fp32_size = model_size_mb(fp32)
    print(f"   Parameters : {n_params:.2f} M")
    print(f"   Model size : {fp32_size:.1f} MB")

    print("   Measuring FP32 latency …")
    fp32_ms = measure_latency(fp32)
    print(f"   Latency    : {fp32_ms:.2f} ms / image")

    #  PTQ 
    print("\n  Post-Training Quantization (PTQ → INT8) …")
    calib = make_calib_loader()
    int8  = ptq_pipeline(fp32, calib)

    int8_size = model_size_mb(int8)
    print(f"   Model size : {int8_size:.1f} MB")

    print("   Measuring INT8 latency …")
    int8_ms = measure_latency(int8)
    print(f"   Latency    : {int8_ms:.2f} ms / image")

    #  Summary 
    speedup     = fp32_ms  / max(int8_ms,  0.001)
    compression = fp32_size / max(int8_size, 0.001)

    print(f"  {'Metric':<28} {'FP32':>10} {'INT8 (PTQ)':>12}")
    print("-" * 62)
    print(f"  {'Parameters (M)':<28} {n_params:>10.2f} {'same':>12}")
    print(f"  {'Model size (MB)':<28} {fp32_size:>10.1f} {int8_size:>12.1f}")
    print(f"  {'Latency (ms/img, CPU)':<28} {fp32_ms:>10.2f} {int8_ms:>12.2f}")
    print(f"  {'Speedup':<28} {'1.00×':>10} {speedup:>11.2f}×")
    print(f"  {'Size compression':<28} {'1.00×':>10} {compression:>11.2f}×")
    print("\nExpected accuracy impact: ~1-2% mAP drop vs FP32 with PTQ.")
    print("For minimal drop, use QAT (train.py --quant-mode qat).\n")


if __name__ == '__main__':
    main()
