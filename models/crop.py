import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Crop:
    source_path: Path
    crop_idx: int
    bbox: tuple
    detector_label: str
    detector_score: float
    image: Image.Image
    emb_dino: np.ndarray

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)