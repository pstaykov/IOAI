"""
Pretrained DETR, ViT, YOLO and CLIP through their library APIs.

pip install transformers timm ultralytics     (timm is needed by DETR's resnet backbone)
Usage: python pretrained_apis.py [image path or url]
"""
import sys
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor, AutoModelForImageClassification, AutoModelForObjectDetection,
    CLIPModel, CLIPProcessor,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_IMAGE = "http://images.cocodataset.org/val2017/000000039769.jpg"  # two cats on a couch


def load_image(src=DEFAULT_IMAGE):
    if src.startswith("http"):
        return Image.open(BytesIO(requests.get(src, timeout=30).content)).convert("RGB")
    return Image.open(src).convert("RGB")


# ---------------------------------------------------------------- ViT: image classification
def vit_classify(image, top_k=5, name="google/vit-base-patch16-224"):
    processor = AutoImageProcessor.from_pretrained(name)          # resize 224, normalize
    model = AutoModelForImageClassification.from_pretrained(name).to(device).eval()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits                           # (1, 1000)
    probs = logits.softmax(-1)[0]
    scores, idx = probs.topk(top_k)
    return [(model.config.id2label[i.item()], s.item()) for s, i in zip(scores, idx)]


# ---------------------------------------------------------------- DETR: object detection
def detr_detect(image, threshold=0.7, name="facebook/detr-resnet-50"):
    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForObjectDetection.from_pretrained(name).to(device).eval()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)                                 # 100 object queries, no NMS needed
    # boxes come back as xyxy pixels of the original image (target_sizes is (h, w))
    result = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=[image.size[::-1]]
    )[0]
    return [
        (model.config.id2label[l.item()], s.item(), [round(v, 1) for v in b.tolist()])
        for s, l, b in zip(result["scores"], result["labels"], result["boxes"])
    ]


# ---------------------------------------------------------------- YOLO: object detection (ultralytics)
def yolo_detect(image, conf=0.25, weights="yolov8n.pt"):
    from ultralytics import YOLO
    model = YOLO(weights)                                         # downloads weights on first use
    result = model.predict(image, conf=conf, verbose=False)[0]    # NMS is done inside
    return [
        (result.names[int(c)], s.item(), [round(v, 1) for v in b.tolist()])
        for c, s, b in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy)
    ]


# ---------------------------------------------------------------- CLIP: zero-shot classification
def clip_zero_shot(image, labels, name="openai/clip-vit-base-patch32"):
    processor = CLIPProcessor.from_pretrained(name)
    model = CLIPModel.from_pretrained(name).to(device).eval()
    texts = [f"a photo of a {l}" for l in labels]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image                 # scaled cosine similarity, (1, n_labels)
    probs = logits.softmax(-1)[0]
    return sorted(zip(labels, probs.tolist()), key=lambda t: -t[1])


if __name__ == "__main__":
    image = load_image(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE)

    print("ViT :", vit_classify(image))
    print("DETR:", detr_detect(image))
    print("YOLO:", yolo_detect(image))
    print("CLIP:", clip_zero_shot(image, ["cat", "dog", "car", "person", "remote control"]))
