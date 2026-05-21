from torchvision import transforms
from config import DINOV2_VARIANT,DEVICE
import torch
from PIL import Image, ImageDraw
import numpy as np

# DINOv2 (visual retrieval)
dino = torch.hub.load('facebookresearch/dinov2', DINOV2_VARIANT).to(DEVICE).eval()
_dino_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def embed_dino(image: Image.Image) -> np.ndarray:
    x = _dino_tf(image.convert('RGB')).unsqueeze(0).to(DEVICE)
    feat = dino(x).squeeze(0).cpu().numpy().astype(np.float32)
    n = np.linalg.norm(feat)
    return feat / n if n else feat