from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from config import DEVICE
from PIL import Image
import torch

# Detector settings (GroundingDINO)
DETECTION_PROMPT = 'a tool. an object.'
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.20
NMS_IOU_THRESHOLD = 0.5
MAX_DETECTIONS_PER_IMAGE = 2

# GroundingDINO
GD_MODEL_ID = 'IDEA-Research/grounding-dino-tiny'
gd_processor = AutoProcessor.from_pretrained(GD_MODEL_ID)
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(GD_MODEL_ID).to(DEVICE).eval()

#TODO: Replace with torchvision
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0

#TODO: Replace with torchvision
def _nms(detections, iou_thr):
    sorted_dets = sorted(detections, key=lambda d: d['score'], reverse=True)
    kept = []
    for d in sorted_dets:
        if all(_iou(d['bbox'], k['bbox']) < iou_thr for k in kept):
            kept.append(d)
    return kept


@torch.no_grad()
def detect_with_dino(image: Image.Image):
    inputs = gd_processor(images=image, text=DETECTION_PROMPT, return_tensors='pt').to(DEVICE)
    outputs = gd_model(**inputs)
    results = gd_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]]
    )[0]
    out = []
    for box, score, label in zip(results['boxes'], results['scores'], results['labels']):
        x1, y1, x2, y2 = (int(round(v)) for v in box.tolist())
        W, H = image.size
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        if x2 <= x1 or y2 <= y1: continue
        out.append({
            'bbox': (x1, y1, x2, y2), 'label': label, 'score': float(score),
            'crop': image.crop((x1, y1, x2, y2)),
        })
    out = _nms(out, NMS_IOU_THRESHOLD)
    out = sorted(out, key=lambda d: d['score'], reverse=True)[:MAX_DETECTIONS_PER_IMAGE]
    return out