"""
COCO 2017 Dataset loader + Anchor generator for MobileNet-SSD
"""

import os
import json
import math
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


 
# Anchor Generator

class AnchorGenerator:
    """
    Generates default (prior) boxes for 4 feature map scales.
    Follows the original SSD paper.
    """

    def __init__(
        self,
        image_size=300,
        feature_maps=(19, 10, 10, 5),
        min_sizes=(60, 105, 150, 195),
        max_sizes=(105, 150, 195, 240),
        aspect_ratios=((2, 3), (2, 3), (2, 3), (2, 3)),
        steps=(16, 30, 30, 60),
        clip=True,
    ):
        self.image_size = image_size
        self.feature_maps = feature_maps
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.aspect_ratios = aspect_ratios
        self.steps = steps
        self.clip = clip

    def generate(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            step = self.steps[k]
            for i in range(f):
                for j in range(f):
                    cx = (j + 0.5) * step / self.image_size
                    cy = (i + 0.5) * step / self.image_size
                    s  = self.min_sizes[k] / self.image_size
                    anchors.append([cx, cy, s, s])

                    s2 = math.sqrt(s * self.max_sizes[k] / self.image_size)
                    anchors.append([cx, cy, s2, s2])

                    for ar in self.aspect_ratios[k]:
                        anchors.append([cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)])
                        anchors.append([cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)])

        anchors = torch.tensor(anchors, dtype=torch.float32)
        if self.clip:
            anchors.clamp_(0, 1)
        return anchors  # [N, 4]  cx,cy,w,h


# Box utilities

def cxcywh_to_xyxy(boxes):
    """Convert [cx,cy,w,h] → [x1,y1,x2,y2]."""
    x1 = boxes[..., 0] - boxes[..., 2] / 2
    y1 = boxes[..., 1] - boxes[..., 3] / 2
    x2 = boxes[..., 0] + boxes[..., 2] / 2
    y2 = boxes[..., 1] + boxes[..., 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_to_cxcywh(boxes):
    """Convert [x1,y1,x2,y2] → [cx,cy,w,h]."""
    cx = (boxes[..., 0] + boxes[..., 2]) / 2
    cy = (boxes[..., 1] + boxes[..., 3]) / 2
    w  =  boxes[..., 2] - boxes[..., 0]
    h  =  boxes[..., 3] - boxes[..., 1]
    return torch.stack([cx, cy, w, h], dim=-1)


def iou(boxes_a, boxes_b):
    """Compute pairwise IoU between [N,4] and [M,4] (xyxy format)."""
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(0)
    inter_h = (inter_y2 - inter_y1).clamp(0)
    inter   = inter_w * inter_h

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def encode_boxes(gt_boxes, anchors, variances=(0.1, 0.2)):
    """Encode ground-truth boxes relative to anchors (SSD offset encoding)."""
    # both in cxcywh
    loc = torch.zeros_like(anchors)
    loc[:, :2] = (gt_boxes[:, :2] - anchors[:, :2]) / (variances[0] * anchors[:, 2:])
    loc[:, 2:] = torch.log(gt_boxes[:, 2:] / anchors[:, 2:]) / variances[1]
    return loc


def decode_boxes(loc_pred, anchors, variances=(0.1, 0.2)):
    """Decode predicted offsets back to absolute cxcywh boxes."""
    boxes = torch.zeros_like(loc_pred)
    boxes[:, :2] = loc_pred[:, :2] * variances[0] * anchors[:, 2:] + anchors[:, :2]
    boxes[:, 2:] = torch.exp(loc_pred[:, 2:] * variances[1]) * anchors[:, 2:]
    return boxes


# Collate + Target Builder


class TargetEncoder:
    """Matches GT boxes to anchors and encodes regression targets."""

    def __init__(self, anchors, iou_threshold=0.5):
        self.anchors_cxcywh = anchors
        self.anchors_xyxy   = cxcywh_to_xyxy(anchors)
        self.iou_threshold  = iou_threshold

    def encode(self, gt_boxes_xyxy, gt_labels):
        """
        Args:
            gt_boxes_xyxy: [N, 4] normalized xyxy
            gt_labels:     [N]    int class ids (1-indexed, 0=background)
        Returns:
            loc_targets:   [A, 4]
            cls_targets:   [A]    (0 = background)
        """
        n_anchors = self.anchors_xyxy.size(0)
        cls_targets = torch.zeros(n_anchors, dtype=torch.long)
        loc_targets = torch.zeros(n_anchors, 4)

        if len(gt_boxes_xyxy) == 0:
            return loc_targets, cls_targets

        ious = iou(self.anchors_xyxy, gt_boxes_xyxy)  # [A, N]
        best_gt_iou, best_gt_idx = ious.max(dim=1)    # [A]

        # Force each GT to be matched to at least one anchor
        best_anchor_per_gt = ious.argmax(dim=0)        # [N]
        best_gt_iou[best_anchor_per_gt] = 1.0
        best_gt_idx[best_anchor_per_gt] = torch.arange(len(gt_boxes_xyxy))

        matched_boxes  = gt_boxes_xyxy[best_gt_idx]    # [A, 4] xyxy
        matched_labels = gt_labels[best_gt_idx]        # [A]

        pos_mask = best_gt_iou >= self.iou_threshold
        cls_targets[pos_mask]  = matched_labels[pos_mask]
        cls_targets[~pos_mask] = 0

        matched_cxcywh = xyxy_to_cxcywh(matched_boxes)
        loc_targets    = encode_boxes(matched_cxcywh, self.anchors_cxcywh)

        return loc_targets, cls_targets


# COCO Dataset

class COCODetectionDataset(Dataset):
    """
    Loads COCO 2017 images + annotations.

    Expected directory layout (Kaggle download):
        root/
          images/
            train2017/   *.jpg
            val2017/     *.jpg
          annotations/
            instances_train2017.json
            instances_val2017.json
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, root, split='train', image_size=300,
                 anchors=None, iou_threshold=0.5, max_images=None):
        self.root       = root
        self.split      = split
        self.image_size = image_size

        ann_file = os.path.join(
            root, 'annotations', f'instances_{split}2017.json'
        )
        with open(ann_file) as f:
            data = json.load(f)

        # Build category id → contiguous class index
        # COCO ids are not contiguous (1-90 with gaps)
        cats = sorted(data['categories'], key=lambda c: c['id'])
        self.cat_id_to_idx = {c['id']: i + 1 for i, c in enumerate(cats)}  # 1-indexed

        # Build image id → annotations index
        self.img_info = {img['id']: img for img in data['images']}
        self.ann_by_img = {}
        for ann in data['annotations']:
            if ann.get('iscrowd', 0):
                continue
            iid = ann['image_id']
            self.ann_by_img.setdefault(iid, []).append(ann)

        self.img_ids = list(self.img_info.keys())
        if max_images:
            self.img_ids = self.img_ids[:max_images]

        self.image_dir = os.path.join(root, 'images', f'{split}2017')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

        self.target_encoder = None
        if anchors is not None:
            self.target_encoder = TargetEncoder(anchors, iou_threshold)

        print(f"[COCO] {split}: {len(self.img_ids)} images loaded")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info   = self.img_info[img_id]
        path   = os.path.join(self.image_dir, info['file_name'])

        # Load & resize
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img = cv2.resize(img, (self.image_size, self.image_size))
        img_tensor = self.transform(img)

        # Annotations
        anns = self.ann_by_img.get(img_id, [])
        boxes, labels = [], []
        for ann in anns:
            x, y, bw, bh = ann['bbox']
            x1 = max(x / w, 0); y1 = max(y / h, 0)
            x2 = min((x + bw) / w, 1); y2 = min((y + bh) / h, 1)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.cat_id_to_idx.get(ann['category_id'], 0))

        boxes  = torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.long)    if labels else torch.zeros((0,), dtype=torch.long)

        if self.target_encoder is not None:
            loc_t, cls_t = self.target_encoder.encode(boxes, labels)
            return img_tensor, loc_t, cls_t

        return img_tensor, boxes, labels


def collate_fn(batch):
    """Collate for variable-length annotations."""
    imgs, boxes_list, labels_list = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(boxes_list), list(labels_list)


def build_dataloaders(root, anchors, batch_size=16, num_workers=2,
                      image_size=300, max_train=None, max_val=None):
    train_ds = COCODetectionDataset(
        root, 'train', image_size, anchors=anchors, max_images=max_train
    )
    val_ds = COCODetectionDataset(
        root, 'val', image_size, anchors=anchors, max_images=max_val
    )
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_dl, val_dl
