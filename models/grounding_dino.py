import torch
import torchvision.ops as tv_ops
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from config import DEVICE

# Detector settings (GroundingDINO)
DETECTION_PROMPT = 'a tool. an object.'
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.20
NMS_IOU_THRESHOLD = 0.5
MAX_DETECTIONS_PER_IMAGE = 2

# GroundingDINO-tiny: already the smallest available checkpoint (~172 M params)
GD_MODEL_ID = 'IDEA-Research/grounding-dino-tiny'
gd_processor = AutoProcessor.from_pretrained(GD_MODEL_ID)
_gd_model_fp32 = AutoModelForZeroShotObjectDetection.from_pretrained(GD_MODEL_ID).eval()

if DEVICE.type == 'cpu':
    # Dynamic quantization converts Linear layers to int8 at call time
    gd_model = torch.quantization.quantize_dynamic(
        _gd_model_fp32,
        qconfig_spec={torch.nn.Linear},
        dtype=torch.qint8,
    )
    # Quantized models must stay on CPU - do not .to(DEVICE) after this
else:
    gd_model = _gd_model_fp32.to(DEVICE)

del _gd_model_fp32  # release the fp32 copy


def _nms(detections: list[dict], iou_thr: float) -> list[dict]:
    """Non-maximum suppression via torchvision.ops.nms (C++/CUDA kernel).

    Replaces the pure-Python O(n^2) loop. torchvision.ops.nms expects
    boxes in (x1, y1, x2, y2) float format and scores as a 1-D tensor.
    """
    if not detections:
        return []
    boxes = torch.tensor([d['bbox'] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d['score'] for d in detections], dtype=torch.float32)
    # nms returns indices of kept boxes sorted by descending score.
    keep_idx = tv_ops.nms(boxes, scores, iou_thr)
    return [detections[i] for i in keep_idx.tolist()]


@torch.no_grad()
def detect_with_dino(image: Image.Image) -> list[dict]:
    """Detect objects in *image* using GroundingDINO-tiny.

    Returns a list of dicts with keys: bbox, label, score, crop.
    The list is NMS-filtered and capped at MAX_DETECTIONS_PER_IMAGE,
    sorted descending by confidence score.
    """
    # On CPU the quantized model lives on CPU; on GPU inputs go to DEVICE.
    inputs = gd_processor(images=image, text=DETECTION_PROMPT, return_tensors='pt')
    if DEVICE.type != 'cpu':
        inputs = inputs.to(DEVICE)

    outputs = gd_model(**inputs)
    results = gd_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )[0]

    W, H = image.size
    out = []
    for box, score, label in zip(results['boxes'], results['scores'], results['labels']):
        x1, y1, x2, y2 = (int(round(v)) for v in box.tolist())
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        out.append({
            'bbox': (x1, y1, x2, y2),
            'label': label,
            'score': float(score),
            'crop': image.crop((x1, y1, x2, y2)),
        })

    out = _nms(out, NMS_IOU_THRESHOLD)
    out = sorted(out, key=lambda d: d['score'], reverse=True)[:MAX_DETECTIONS_PER_IMAGE]
    return out
