"""
Training script for Quantized MobileNet-SSD on COCO 2017.

Usage:
    python train.py --data /path/to/coco --epochs 20 --batch-size 16

For quick smoke-test (no real COCO data required):
    python train.py --smoke-test
"""

import os
import sys
import time
import argparse
import copy
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from model   import build_model, NUM_CLASSES
from dataset import AnchorGenerator, build_dataloaders, COCODetectionDataset
from loss    import SSDLoss


# Helpers

def get_anchors():
    gen = AnchorGenerator()
    return gen.generate()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, (imgs, loc_t, cls_t) in enumerate(loader):
        imgs  = imgs.to(device)
        loc_t = loc_t.to(device)
        cls_t = cls_t.to(device)

        optimizer.zero_grad()
        loc_p, cls_p = model(imgs)
        loss = criterion(loc_p, cls_p, loc_t, cls_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()

        if step % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch} | step {step}/{len(loader)} | "
                  f"loss={loss.item():.4f} | {elapsed:.1f}s")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for imgs, loc_t, cls_t in loader:
        imgs  = imgs.to(device)
        loc_t = loc_t.to(device)
        cls_t = cls_t.to(device)
        loc_p, cls_p = model(imgs)
        loss = criterion(loc_p, cls_p, loc_t, cls_t)
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)



# Smoke-test dataset (random tensors, no files needed)

class SmokeDataset(torch.utils.data.Dataset):
    def __init__(self, n=64, n_anchors=2766):
        self.n = n
        self.n_anchors = n_anchors

    def __len__(self): return self.n

    def __getitem__(self, _):
        img   = torch.randn(3, 300, 300)
        loc_t = torch.randn(self.n_anchors, 4)
        cls_t = torch.randint(0, NUM_CLASSES, (self.n_anchors,))
        return img, loc_t, cls_t


# Quantization helpers

def calibrate_ptq(model, loader, device, n_batches=10):
    """Run calibration data through a PTQ-prepared model."""
    model.eval()
    with torch.no_grad():
        for i, (imgs, *_) in enumerate(loader):
            if i >= n_batches:
                break
            model(imgs.to(device))


def quantize_model(model, train_loader, device, mode='ptq'):
    """
    mode = 'ptq' : Post-training static quantization
    mode = 'qat' : Quantization-aware training (call before training)
    """
    print(f"\n[Quantization] mode={mode}")
    model_q = copy.deepcopy(model).cpu()  # quantization runs on CPU

    if mode == 'ptq':
        model_q.prepare_ptq()
        calibrate_ptq(model_q, train_loader, device='cpu')
        model_q.convert_quantized()
        print("[PTQ] Conversion complete")

    elif mode == 'qat':
        model_q.prepare_qat()
        print("[QAT] Model prepared – train normally then call convert_to_quantized()")

    return model_q


def compare_model_size(fp32_model, q_model, path='/tmp'):
    """Save both models and compare file sizes."""
    fp_path = os.path.join(path, 'model_fp32.pt')
    q_path  = os.path.join(path, 'model_int8.pt')
    torch.save(fp32_model.state_dict(), fp_path)
    torch.save(q_model.state_dict(),    q_path)
    fp_mb = os.path.getsize(fp_path) / 1e6
    q_mb  = os.path.getsize(q_path)  / 1e6
    print(f"\n[Size] FP32: {fp_mb:.1f} MB  |  INT8: {q_mb:.1f} MB  "
          f"(compression {fp_mb/max(q_mb,0.001):.1f}×)")


# Main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       type=str, default='',     help='Path to COCO root dir')
    parser.add_argument('--epochs',     type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--workers',    type=int, default=2)
    parser.add_argument('--quant-mode', type=str, default='ptq',  choices=['none','ptq','qat'])
    parser.add_argument('--save-dir',   type=str, default='checkpoints')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Run a quick sanity-check without COCO data')
    parser.add_argument('--max-train',  type=int, default=None,
                        help='Limit number of training images (for fast runs)')
    parser.add_argument('--max-val',    type=int, default=500)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Anchors
    anchors = get_anchors().to(device)
    print(f"[Anchors] {anchors.shape[0]} default boxes")

    # Data 
    if args.smoke_test:
        print("\n[Smoke-test] Using random synthetic data (300 batches)")
        n_anchors = anchors.shape[0]
        smoke_ds = SmokeDataset(n=args.batch_size * 10, n_anchors=n_anchors)
        train_loader = torch.utils.data.DataLoader(
            smoke_ds, batch_size=args.batch_size, shuffle=True
        )
        val_loader = train_loader
        args.epochs = min(args.epochs, 2)
    else:
        if not args.data:
            print("ERROR: --data is required (path to COCO root). "
                  "Use --smoke-test for a quick run without data.")
            sys.exit(1)
        anchors_cpu = get_anchors()  # dataset runs on CPU
        train_loader, val_loader = build_dataloaders(
            args.data,
            anchors=anchors_cpu,
            batch_size=args.batch_size,
            num_workers=args.workers,
            max_train=args.max_train,
            max_val=args.max_val,
        )

    #  Model 
    model = build_model(num_classes=NUM_CLASSES).to(device)
    criterion = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # QAT: prepare before training
    if args.quant_mode == 'qat':
        model = model.cpu()
        model.prepare_qat()
        model = model.to(device)

    #  Training loop 
    best_val_loss = float('inf')
    history = {'train': [], 'val': []}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss   = validate(model, val_loader, criterion, device)
        scheduler.step()

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        print(f"\nEpoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.save_dir, 'best_fp32.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best model → {ckpt_path}")

    #  Post-training quantization 
    if args.quant_mode == 'ptq':
        model.load_state_dict(
            torch.load(os.path.join(args.save_dir, 'best_fp32.pt'))
        )
        model_q = quantize_model(model, train_loader, device, mode='ptq')
        q_path = os.path.join(args.save_dir, 'model_int8_ptq.pt')
        torch.save(model_q.state_dict(), q_path)
        print(f"[PTQ] Quantized model saved → {q_path}")
        compare_model_size(model, model_q, path=args.save_dir)

    elif args.quant_mode == 'qat':
        model.cpu()
        model.convert_quantized()
        q_path = os.path.join(args.save_dir, 'model_int8_qat.pt')
        torch.save(model.state_dict(), q_path)
        print(f"[QAT] Quantized model saved → {q_path}")
        compare_model_size(model, model, path=args.save_dir)

    # Save final FP32
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_fp32.pt'))
    print("\nTraining complete!")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Checkpoints   : {args.save_dir}/")
    return history


if __name__ == '__main__':
    main()
