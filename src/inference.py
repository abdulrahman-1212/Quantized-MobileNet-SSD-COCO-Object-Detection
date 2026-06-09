"""
Inference utilities: decode predictions, apply NMS, draw results.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import time

from model   import build_model, COCO_CLASSES, NUM_CLASSES
from dataset import AnchorGenerator, decode_boxes, cxcywh_to_xyxy


# NMS

def nms(boxes, scores, iou_threshold=0.45):
    """Pure-PyTorch single-class NMS. Returns kept indices."""
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ix1 = x1[rest].clamp(min=x1[i].item())
        iy1 = y1[rest].clamp(min=y1[i].item())
        ix2 = x2[rest].clamp(max=x2[i].item())
        iy2 = y2[rest].clamp(max=y2[i].item())
        iw  = (ix2 - ix1).clamp(min=0)
        ih  = (iy2 - iy1).clamp(min=0)
        inter = iw * ih
        ovr   = inter / (area[i] + area[rest] - inter).clamp(min=1e-6)
        order = rest[ovr <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long)



# Detector wrapper

class Detector:
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, weights_path, quantized=False, device='cpu',
                 score_threshold=0.35, nms_threshold=0.45, top_k=200):
        self.device = torch.device(device)
        self.score_threshold = score_threshold
        self.nms_threshold   = nms_threshold
        self.top_k           = top_k

        self.model = build_model(num_classes=NUM_CLASSES, pretrained=False)
        if quantized:
            self.model.prepare_ptq()  # loads structure
            self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            self.model.convert_to_quantized()
        else:
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
            self.model.to(self.device)
        self.model.eval()

        anchors_cxcywh = AnchorGenerator().generate()
        self.anchors_cxcywh = anchors_cxcywh
        self.anchors_xyxy   = cxcywh_to_xyxy(anchors_cxcywh)

    def preprocess(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (300, 300))
        t   = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        t   = (t - self.MEAN) / self.STD
        return t.unsqueeze(0)

    @torch.no_grad()
    def predict(self, img_bgr):
        """
        Returns list of dicts: {label, score, box:[x1,y1,x2,y2] in pixel coords}
        """
        h, w = img_bgr.shape[:2]
        inp  = self.preprocess(img_bgr).to(self.device)

        t0 = time.perf_counter()
        loc_p, cls_p = self.model(inp)       # [1, A, 4], [1, A, C]
        infer_ms = (time.perf_counter() - t0) * 1000

        loc_p = loc_p.squeeze(0)             # [A, 4]
        cls_p = cls_p.squeeze(0)             # [A, C]

        scores = F.softmax(cls_p, dim=-1)    # [A, C]

        boxes_cxcywh = decode_boxes(loc_p, self.anchors_cxcywh)
        boxes_xyxy   = cxcywh_to_xyxy(boxes_cxcywh).clamp(0, 1)  # [A, 4]

        results = []
        for cls_idx in range(1, NUM_CLASSES):
            cls_scores = scores[:, cls_idx]
            mask = cls_scores > self.score_threshold
            if not mask.any():
                continue
            b = boxes_xyxy[mask]
            s = cls_scores[mask]
            keep = nms(b, s, self.nms_threshold)
            keep = keep[:self.top_k]
            for ki in keep:
                x1, y1, x2, y2 = b[ki].tolist()
                results.append({
                    'label': COCO_CLASSES[cls_idx],
                    'score': float(s[ki]),
                    'box':   [int(x1*w), int(y1*h), int(x2*w), int(y2*h)],
                })

        return results, infer_ms


# Visualization

COLORS = np.random.default_rng(42).integers(60, 240, (NUM_CLASSES, 3)).tolist()

def draw_detections(img_bgr, detections):
    img = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det['box']
        cls_idx = COCO_CLASSES.index(det['label']) if det['label'] in COCO_CLASSES else 0
        color = tuple(COLORS[cls_idx])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det['label']} {det['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


# Demo (no weights file needed – random init)

if __name__ == '__main__':
    import tempfile

    print("Building model + saving random weights for demo...")
    m = build_model(pretrained=False)
    m.eval()

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(m.state_dict(), f.name)
        wpath = f.name

    det = Detector(wpath, quantized=False, device='cpu',
                   score_threshold=0.01)  # low threshold to see output

    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    results, ms = det.predict(dummy_img)

    print(f"\nInference time : {ms:.1f} ms")
    print(f"Detections     : {len(results)}")
    if results:
        for r in results[:5]:
            print(f"  {r['label']:20s}  score={r['score']:.3f}  box={r['box']}")
