import os
import torch

# CPU optimization: use all available logical cores for intra-op parallelism.
# torch defaults to 1 thread on many systems; setting this explicitly gives a
# meaningful throughput improvement for the transformer forward passes.
_num_threads = os.cpu_count() or 1
torch.set_num_threads(_num_threads)
torch.set_num_interop_threads(max(1, _num_threads // 2))

# DINOv2 variant downgraded from vitl14 (307 M params, 1024-dim) to vits14
# (21 M params, 384-dim). Embedding dimension shrinks accordingly; any
# pgvector column or stored embeddings must match this new dimension.
# Speedup on CPU
DINOV2_VARIANT = 'dinov2_vits14'  # ViT-Small/14 - 384-dim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE} | CPU threads: {_num_threads}')
