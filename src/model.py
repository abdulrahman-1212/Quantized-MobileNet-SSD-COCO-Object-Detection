"""
Quantized MobileNetV2-SSD for Fast Object Detection on COCO 2017.

The key challenge: torchvision's MobileNetV2 InvertedResidual blocks use
Python-level `x + self.conv(x)`, which does NOT work after INT8 conversion
(aten::add.out not in QuantizedCPU backend).

Solution: rebuild the InvertedResidual with nn.quantized.FloatFunctional
so the add is quantization-aware.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import QuantStub, DeQuantStub


#  COCO class names (81 including background)

COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
NUM_CLASSES = len(COCO_CLASSES)   # 81


#  Quantization-compatible building blocks


def _conv_bn_relu(in_ch, out_ch, kernel=3, stride=1, groups=1):
    padding = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                  padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    """
    MobileNetV2 bottleneck with FloatFunctional for the residual add.
    This makes the module compatible with torch.ao.quantization.
    """

    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden = in_ch * expand_ratio
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            layers.append(_conv_bn_relu(in_ch, hidden, kernel=1))
        layers += [
            _conv_bn_relu(hidden, hidden, kernel=3, stride=stride, groups=hidden),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

        if self.use_res:
            self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res:
            return self.skip_add.add(x, self.conv(x))
        return self.conv(x)


#  MobileNetV2 Backbone (quantization-compatible)

# MobileNetV2 configuration: (t=expand_ratio, c=out_channels, n=repeats, s=stride)
MV2_CONFIG = [
    (1,  16, 1, 1),
    (6,  24, 2, 2),
    (6,  32, 3, 2),
    (6,  64, 4, 2),
    (6,  96, 3, 1),   # ← tap here (layer 11 cumulative)
    (6, 160, 3, 2),   # ← tap here (last of these)
    (6, 320, 1, 1),
]


class MobileNetV2Backbone(nn.Module):
    """
    Quantization-safe MobileNetV2 feature extractor.
    Returns feature maps at three scales.
    """

    def __init__(self):
        super().__init__()
        self.first_conv = _conv_bn_relu(3, 32, kernel=3, stride=2)

        blocks = []
        in_ch = 32
        self._tap_indices = []   # will hold indices of tap layers

        cumulative = 0
        for t, c, n, s in MV2_CONFIG:
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(InvertedResidual(in_ch, c, stride, t))
                in_ch = c
                cumulative += 1
            # tap after 5th group (96ch) and 6th group (160ch)
            if c in (96, 160):
                self._tap_indices.append(cumulative - 1)

        self.blocks = nn.ModuleList(blocks)

        self.last_conv = _conv_bn_relu(320, 1280, kernel=1)
        self._tap_indices.append('last')   # 1280-channel output

    def forward(self, x):
        x = self.first_conv(x)
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self._tap_indices:
                features.append(x)
        x = self.last_conv(x)
        features.append(x)
        return features   # [f96, f160, f1280]


#  SSD Detection Head

class SSDHead(nn.Module):
    def __init__(self, in_channels_list, num_anchors_list, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.loc_heads = nn.ModuleList()
        self.cls_heads = nn.ModuleList()
        for in_ch, n_a in zip(in_channels_list, num_anchors_list):
            self.loc_heads.append(nn.Conv2d(in_ch, n_a * 4,           3, padding=1))
            self.cls_heads.append(nn.Conv2d(in_ch, n_a * num_classes,  3, padding=1))

    def forward(self, feats):
        locs, clss = [], []
        for f, lh, ch in zip(feats, self.loc_heads, self.cls_heads):
            B = f.size(0)
            locs.append(lh(f).permute(0,2,3,1).contiguous().view(B,-1,4))
            clss.append(ch(f).permute(0,2,3,1).contiguous().view(B,-1,self.num_classes))
        return torch.cat(locs, 1), torch.cat(clss, 1)


#  Full MobileNet-SSD Model

class MobileNetV2SSD(nn.Module):
    """
    MobileNetV2 backbone + 4-scale SSD head.

    Feature maps:
      Scale 1: after group-5 (96 ch,  ~19×19 @ 300px input)
      Scale 2: after group-6 (160 ch, ~10×10)
      Scale 3: last_conv     (1280 ch,  ~5×5)
      Scale 4: extra_conv    (512 ch,   ~3×3)

    Total default boxes: 6 anchors × (19²+10²+5²+3²) = 3516

    Quantization:
      PTQ: model.prepare_ptq() → run calibration → model.convert_quantized()
      QAT: model.prepare_qat() → train as normal → model.convert_quantized()
    """

    FEATURE_CHANNELS = [96, 160, 1280, 512]
    NUM_ANCHORS       = [6,    6,    6,   6]

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        self.quant   = QuantStub()
        self.dequant = DeQuantStub()

        self.backbone = MobileNetV2Backbone()

        self.extra_conv = nn.Sequential(
            nn.Conv2d(1280, 256, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU6(inplace=True),
        )

        self.ssd_head = SSDHead(
            self.FEATURE_CHANNELS, self.NUM_ANCHORS, num_classes
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.ssd_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.extra_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.quant(x)
        feats = self.backbone(x)            # [f96, f160, f1280]
        feats.append(self.extra_conv(feats[-1]))  # f512
        loc_p, cls_p = self.ssd_head(feats)
        return self.dequant(loc_p), self.dequant(cls_p)

    #  Quantization API 

    def prepare_ptq(self, backend='fbgemm'):
        """Prepare for post-training static quantization."""
        self.eval()
        self.qconfig = torch.ao.quantization.get_default_qconfig(backend)
        torch.ao.quantization.prepare(self, inplace=True)
        return self

    def prepare_qat(self, backend='fbgemm'):
        """Prepare for quantization-aware training."""
        self.train()
        self.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
        torch.ao.quantization.prepare_qat(self, inplace=True)
        return self

    def convert_quantized(self):
        """Convert to fully quantized INT8 model."""
        self.eval()
        torch.ao.quantization.convert(self, inplace=True)
        return self


def build_model(num_classes=NUM_CLASSES):
    return MobileNetV2SSD(num_classes=num_classes)


#  Sanity check 

if __name__ == '__main__':
    import copy

    print("Building FP32 model …")
    m = build_model()
    m.eval()
    x = torch.randn(1, 3, 300, 300)
    loc, cls = m(x)
    total = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"FP32 forward OK  |  params={total:.2f}M  "
          f"loc={tuple(loc.shape)}  cls={tuple(cls.shape)}")

    print("Applying PTQ …")
    m2 = copy.deepcopy(m)
    m2.prepare_ptq()
    with torch.no_grad():
        for _ in range(5):
            m2(torch.randn(2, 3, 300, 300))
    m2.convert_quantized()
    loc2, cls2 = m2(x)
    print(f"INT8 forward OK   loc={tuple(loc2.shape)}  cls={tuple(cls2.shape)}")
