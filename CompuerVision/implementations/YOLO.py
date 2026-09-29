import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2

CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

IMG_SIZE = 448
S = 7                 # grid size (S x S cells)
C = len(CLASSES)      # number of classes
# anchor (w, h) relative to the whole image: roughly square, wide, tall
ANCHORS = torch.tensor([[0.3, 0.3], [0.6, 0.3], [0.3, 0.6]])
B = len(ANCHORS)

image_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize([IMG_SIZE, IMG_SIZE]),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # pretrained resnet stats
])


def anchor_iou(w, h):
    """IoU between a (w, h) box and every anchor, assuming shared centers. Returns shape (B,)."""
    inter = torch.minimum(ANCHORS[:, 0], torch.tensor(w)) * torch.minimum(ANCHORS[:, 1], torch.tensor(h))
    union = ANCHORS[:, 0] * ANCHORS[:, 1] + w * h - inter
    return inter / union


def getlabel(target):
    """
    Turns the VOC annotation dict into a tensor of shape (S, S, B, 5 + C).
    Last dim: [obj, x, y, w, h, one-hot class]
      obj  = 1 if this anchor in this cell is responsible for an object
      x, y = box center relative to the cell (0..1)
      w, h = box size relative to the whole image (0..1)
    """
    ann = target["annotation"]
    img_w = float(ann["size"]["width"])
    img_h = float(ann["size"]["height"])

    label = torch.zeros(S, S, B, 5 + C)

    for obj in ann["object"]:
        box = obj["bndbox"]
        xmin, ymin = float(box["xmin"]), float(box["ymin"])
        xmax, ymax = float(box["xmax"]), float(box["ymax"])

        # pixel corners -> relative center / size (0..1 of the original image, so resizing is free)
        cx = (xmin + xmax) / 2 / img_w
        cy = (ymin + ymax) / 2 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        # which cell holds the center (clamp so cx == 1.0 doesn't overflow)
        col = min(int(cx * S), S - 1)
        row = min(int(cy * S), S - 1)
        x_cell = cx * S - col
        y_cell = cy * S - row

        # best anchor first; fall back to the next-best if it is already taken in this cell
        for a in anchor_iou(w, h).argsort(descending=True).tolist():
            if label[row, col, a, 0] == 0:
                label[row, col, a, 0] = 1
                label[row, col, a, 1:5] = torch.tensor([x_cell, y_cell, w, h])
                label[row, col, a, 5 + CLASS_TO_IDX[obj["name"]]] = 1
                break
        # if all anchors in the cell are taken the object is dropped

    return label


def voc_transform(img, target):
    return image_transform(img), getlabel(target)


dataset = torchvision.datasets.VOCDetection(
    root="CompuerVision/implementations/data", year="2007", image_set="train",
    download=True, transforms=voc_transform,
)
dataset_test = torchvision.datasets.VOCDetection(
    root="CompuerVision/implementations/data", year="2007", image_set="test",
    download=True, transforms=voc_transform,
)

dataloader_train = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=2)


class YOLO(nn.Module):
    """resnet18 backbone (448 -> 14x14x512) + stride-2 conv (-> 7x7) + 1x1 conv to B*(5+C) channels."""

    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.head = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, B * (5 + C), kernel_size=1),
        )

    def forward(self, x):
        out = self.head(self.backbone(x))                     # (N, B*(5+C), S, S)
        n = out.shape[0]
        out = out.view(n, B, 5 + C, S, S).permute(0, 3, 4, 1, 2)
        return out                                            # raw logits, (N, S, S, B, 5+C)


def yolo_loss(out, target, lambda_coord=5.0, lambda_noobj=0.5):
    """out: raw logits (N,S,S,B,5+C); target: from getlabel (N,S,S,B,5+C)."""
    n = out.shape[0]
    obj = target[..., 0] == 1

    # objectness: every anchor, but empty ones are down-weighted
    obj_bce = nn.functional.binary_cross_entropy_with_logits(out[..., 0], target[..., 0], reduction="none")
    obj_loss = obj_bce[obj].sum() + lambda_noobj * obj_bce[~obj].sum()

    if obj.any():
        pred_box = torch.sigmoid(out[..., 1:5])[obj]          # x, y, w, h all in 0..1
        true_box = target[..., 1:5][obj]
        xy_loss = nn.functional.mse_loss(pred_box[:, :2], true_box[:, :2], reduction="sum")
        wh_loss = nn.functional.mse_loss(pred_box[:, 2:].clamp(min=1e-6).sqrt(), true_box[:, 2:].sqrt(), reduction="sum")
        cls_loss = nn.functional.cross_entropy(out[..., 5:][obj], target[..., 5:][obj].argmax(-1), reduction="sum")
    else:
        xy_loss = wh_loss = cls_loss = out.sum() * 0

    return (lambda_coord * (xy_loss + wh_loss) + obj_loss + cls_loss) / n


def decode(out, conf_thresh=0.3, iou_thresh=0.5):
    """
    Raw logits (N,S,S,B,5+C) -> per image (boxes[K,4] xyxy in 0..1, scores[K], labels[K]) after per-class NMS.
    Works on ground-truth tensors too if you pass logits; for a label tensor use decode_label.
    """
    probs = torch.cat([torch.sigmoid(out[..., :5]), out[..., 5:].softmax(-1)], dim=-1)
    return _decode_probs(probs, conf_thresh, iou_thresh)


def decode_label(label, conf_thresh=0.5, iou_thresh=0.5):
    """Same as decode but for an already-probability tensor like getlabel's output (sanity check of the encoding)."""
    return _decode_probs(label, conf_thresh, iou_thresh)


def _decode_probs(p, conf_thresh, iou_thresh):
    n = p.shape[0]
    rows = torch.arange(S, device=p.device).view(1, S, 1, 1).expand(n, S, S, B)
    cols = torch.arange(S, device=p.device).view(1, 1, S, 1).expand(n, S, S, B)

    cx = (cols + p[..., 1]) / S
    cy = (rows + p[..., 2]) / S
    w, h = p[..., 3], p[..., 4]
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)   # (N,S,S,B,4)

    cls_score, cls_idx = p[..., 5:].max(-1)
    scores = p[..., 0] * cls_score

    results = []
    for i in range(n):
        keep = scores[i] > conf_thresh
        b, s, l = boxes[i][keep], scores[i][keep], cls_idx[i][keep]
        kept = torchvision.ops.batched_nms(b, s, l, iou_thresh)
        results.append((b[kept], s[kept], l[kept]))
    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        total = 0.0
        for imgs, labels in dataloader_train:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = yolo_loss(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch {epoch}: train loss {total / len(dataloader_train):.4f}")

        model.eval()
        with torch.no_grad():
            total = 0.0
            for imgs, labels in dataloader_test:
                imgs, labels = imgs.to(device), labels.to(device)
                total += yolo_loss(model(imgs), labels).item()
        print(f"epoch {epoch}: test loss {total / len(dataloader_test):.4f}")

    torch.save(model.state_dict(), "CompuerVision/implementations/yolo.pt")

    # inference: boxes are xyxy in 0..1 of the image
    imgs, _ = next(iter(dataloader_test))
    with torch.no_grad():
        detections = decode(model(imgs.to(device)))
    boxes, scores, cls = detections[0]
    for b, s, c in zip(boxes.tolist(), scores.tolist(), cls.tolist()):
        print(CLASSES[c], round(s, 2), [round(v, 2) for v in b])
