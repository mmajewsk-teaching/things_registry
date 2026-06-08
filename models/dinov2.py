import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import DEVICE, DINOV2_VARIANT

# DINOv2 ViT-S/14 (21 M params, 384-dim) - downgraded from ViT-L/14 (307 M,
# 1024-dim). Embedding dimension is now 384
_dino_fp32 = torch.hub.load(
    str(torch.hub.get_dir()) + '/facebookresearch_dinov2_main',
    DINOV2_VARIANT,
    source='local',
).eval()

if DEVICE.type == 'cpu':
    # Dynamic quantization of the transformer's Linear layers for
    # speedup on CPU with negligible retrieval-quality regression for
    # image-similarity tasks 
    dino = torch.quantization.quantize_dynamic(
        _dino_fp32,
        qconfig_spec={torch.nn.Linear},
        dtype=torch.qint8,
    )
    # Quantized model stays on CPU.
else:
    dino = _dino_fp32.to(DEVICE)

del _dino_fp32  # release fp32 copy

_dino_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def embed_dino(image: Image.Image) -> np.ndarray:
    """Embed *image* with DINOv2 ViT-S/14 and return an L2-normalised float32 vector.

    Output shape: (384,)
    """
    x = _dino_tf(image.convert('RGB')).unsqueeze(0)
    if DEVICE.type != 'cpu':
        x = x.to(DEVICE)
    feat = dino(x).squeeze(0).cpu().numpy().astype(np.float32)
    n = np.linalg.norm(feat)
    return feat / n if n else feat
