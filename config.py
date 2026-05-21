import torch

DINOV2_VARIANT = 'dinov2_vitl14'  # 1024-dim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')