from models.crop import Crop
from pathlib import Path
import hashlib
from datetime import datetime, timezone

def register_crop(crop: Crop, location: str, things, custom_label: str = None) -> str:
    """Add a crop to the vector DB. Returns the assigned ID.
    ID is deterministic (source_path + crop_idx) so re-running ingestion
    updates existing rows instead of creating duplicates."""
    file_bytes = Path(crop.source_path).read_bytes()
    crop_id = hashlib.md5(file_bytes + str(crop.crop_idx).encode()).hexdigest()


    things.add(
        ids=[crop_id],
        embeddings=[crop.emb_dino.tolist()],
        metadatas=[{
            'location': location,
            'source_path': str(crop.source_path),
            'crop_idx': int(crop.crop_idx),
            'bbox': f'{crop.bbox[0]},{crop.bbox[1]},{crop.bbox[2]},{crop.bbox[3]}',
            'created_at': datetime.now(timezone.utc).isoformat(),
        }],
    )
    return crop_id