from PIL import Image,ImageDraw
from matplotlib import pyplot as plt
from pathlib import Path
from pipelines.detection_pipeline import process_image
from IPython.display import clear_output
from services.crop_service import register_crop

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


annotated = {}
def run_ingestion_pipeline(things,plotResults):
    print("Starting ingestion pipeline...")

    image_paths = sorted(p for p in Path(PHOTOS_FOLDER).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    print(f'Found {len(image_paths)} photos.')

    for path in image_paths:
        crops, img = process_image(path)

        default_loc = location_for(path)

        for c in crops:
            clear_output(wait=True)

            if plotResults:
                plt.figure(figsize=(5, 5))
                plt.imshow(c.image)
                plt.title(f"Plik: {path.name} | Obiekt nr: {c.crop_idx}")
                plt.axis('off')
                plt.show()

            print("-" * 50)
            print(f"Default localization '{default_loc}'")


            final_loc = default_loc

            cid = register_crop(c, location=final_loc,things=things)



            print("\n=== Save with success ===")
            print(f"Location: {final_loc}")
            print(f"ID in wektor database: {cid[:8]}...")
        annotated[path] = draw_detections(img, crops)