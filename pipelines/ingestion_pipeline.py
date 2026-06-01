from PIL import Image, ImageDraw
from pathlib import Path
from pipelines.detection_pipeline import process_image
from services.crop_service import register_crop
import time

PHOTOS_FOLDER = './DLWC_AA/things_photos'

LOCATION_BY_FILE = {
    'bosch_gbh':       'workshop tool cabinet',
    'kanister':        'garage shelf',
    'klucz':           'workshop drawer',
    'mlotek':          'workshop pegboard',
}

def location_for(path: Path) -> str:
    name = path.stem.lower()
    for prefix, loc in LOCATION_BY_FILE.items():
        if name.startswith(prefix):
            return loc
    return 'unsorted'


def draw_detections(img: Image.Image, crops):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for c in crops:
        x1, y1, x2, y2 = c.bbox
        color = 'yellow' if c.detector_label == 'whole_image_fallback' else 'red'
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    return out


def run_ingestion_pipeline(things, plotResults=False) -> dict:
    t_start = time.time()

    if things.initialized:
        return {
            "status": "ok",
            "message": "Things registry already initialized. No action taken.",
        }

    image_paths = sorted(
        p for p in Path(PHOTOS_FOLDER).iterdir()
        if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    )

    total_crops = 0
    fallbacks = 0

    for path in image_paths:
        crops, img = process_image(path)
        default_loc = location_for(path)

        for c in crops:
            if c.detector_label == 'whole_image_fallback':
                fallbacks += 1
            register_crop(c, location=default_loc, things=things)
            total_crops += 1

    elapsed = round(time.time() - t_start, 1)
    things.initialized = True
    return {
        "photos_scanned": len(image_paths),
        "objects_registered": total_crops,
        "fallbacks": fallbacks,
        "elapsed_seconds": elapsed,
    }