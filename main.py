from pipelines.ingestion_pipeline import run_ingestion_pipeline
from db.postgres import create_things
from services.vocabulary_service import initialize_vocabulary

# def list_things(things) -> list[dict]:
#     """List everything currently stored."""
#     if things.count() == 0:
#         return []
#     res = things.get(include=['metadatas'])
#     print(f"Things ma {things.count()} obiektow")
#     return [{'id': i, 'metadata': m} for i, m in zip(res['ids'], res['metadatas'])]
#
# # def clear_db(things):
# #     """Wipe the things collection. Useful for re-running cleanly."""
# #     things.clear()
# #     print('Cleared.')
#
# print("\nKrok1 - Inicjalizacja things")
# # things = create_things()
# clear_db(things)
#
# print("\nKrok2 - Inicjalizacja slownika")
# initialize_vocabulary()
#
# print("\nKrok3 - Ingestion Pipeline")
# run_ingestion_pipeline(things)
#
# print(list_things(things))

from dataclasses import dataclass
from pathlib import Path
import uuid

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Where photos to register/query live

# Where photos to register/query live







# Three-tier query response thresholds (cosine sim against stored embeddings)

                                  # < SAME_CATEGORY: unknown


##paRT 4
import json
import difflib





# def extend_vocabulary(new_label: str):
#     global OPEN_VOCAB_LABELS, OPEN_VOCAB_PROMPTS, OPEN_VOCAB_MATRIX
#
#     if new_label in OPEN_VOCAB_LABELS:
#         return
#
#     print(f"Uczę się nowego słowa: '{new_label}' i zapisuję do bazy w JSON...")
#     OPEN_VOCAB_LABELS.append(new_label)
#
#     # Aktualizacja macierzy CLIP
#     prompt = f"a photo of a {new_label}"
#     new_emb = embed_clip_text(prompt)
#     OPEN_VOCAB_MATRIX = np.vstack([OPEN_VOCAB_MATRIX, new_emb])
#
#     # Zapis całej listy z powrotem do pliku JSON
#     with open(ALL_VOCAB_FILE, 'w', encoding='utf-8') as f:
#         json.dump(OPEN_VOCAB_LABELS, f, ensure_ascii=False, indent=2)
#PART 5

    
##PART 6








# print('Detection pipeline ready.')
#PART 7


#PART 8











# print('DB API ready.')



#PART 9

from collections import Counter


    
## PAR 10 


from IPython.display import clear_output
import difflib
from pathlib import Path
from datetime import datetime,timezone









# clear_output(wait=True)

## Paart 11 - Display results and list DB contents
# n = len(annotated)
# if n:
#     cols = min(2, n); rows = (n + cols - 1) // cols
#     fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
#     axes = np.array(axes).reshape(-1)
#     for ax, (path, img) in zip(axes, annotated.items()):
#         ax.imshow(img); ax.set_title(path.name); ax.axis('off')
#     for ax in axes[len(annotated):]: ax.axis('off')
#     plt.tight_layout(); plt.show()
#
#
# rows = list_things()
# print(f'{len(rows)} entries:\n')
# for r in rows:
#     m = r['metadata']
#     src = Path(m['source_path']).name
#     print(f'  {r["id"][:8]}  cat="{m["category"]}" ({m["category_score"]:.2f})  '
#           f'loc="{m["location"]}"  src={src}#{m["crop_idx"]}')



