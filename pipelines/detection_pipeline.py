from pathlib import Path

from PIL import Image

from models.crop import Crop
from models.dinov2 import embed_dino
from models.grounding_dino import detect_with_dino

FALLBACK_TO_WHOLE_IMAGE = True

def process_image(path: Path) -> tuple[list[Crop], Image.Image]:
    img = Image.open(path).convert('RGB')
    detections = detect_with_dino(img)
    if not detections and FALLBACK_TO_WHOLE_IMAGE:
        detections = [{
            'bbox': (0, 0, img.size[0], img.size[1]),
            'label': 'whole_image_fallback', 'score': 0.0, 'crop': img,
        }]
    crops = []
    for i, det in enumerate(detections):
        crops.append(Crop(
            source_path=path, crop_idx=i, bbox=det['bbox'],
            detector_label=det['label'], detector_score=det['score'],
            image=det['crop'],
            emb_dino=embed_dino(det['crop'])
        ))
    return crops, img