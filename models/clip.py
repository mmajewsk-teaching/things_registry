import open_clip
from config import DEVICE
from PIL import Image
import numpy as np
import torch

# CLIP (open-vocab category)
CLIP_MODEL = 'ViT-B-32'
CLIP_PRETRAINED = 'openai'
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
clip_model = clip_model.to(DEVICE).eval()

@torch.no_grad()
def embed_clip_image(image: Image.Image) -> np.ndarray:
    x = clip_preprocess(image.convert('RGB')).unsqueeze(0).to(DEVICE)
    feat = clip_model.encode_image(x).squeeze(0).cpu().numpy().astype(np.float32)
    n = np.linalg.norm(feat)
    return feat / n if n else feat


@torch.no_grad()
def embed_clip_text_batch(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Embed a list of texts. Used to build the open-vocab label bank."""
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        tokens = clip_tokenizer(chunk).to(DEVICE)
        feat = clip_model.encode_text(tokens).cpu().numpy().astype(np.float32)
        feat = feat / np.linalg.norm(feat, axis=1, keepdims=True).clip(min=1e-12)
        out.append(feat)
    return np.concatenate(out, axis=0)

@torch.no_grad()
def embed_clip_text(text: str) -> np.ndarray:
    return embed_clip_text_batch([text])[0]