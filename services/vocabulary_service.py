import ast

from models.clip import embed_clip_text
import numpy as np
from models.clip import embed_clip_text_batch
import json

# DRIVE_FOLDER = './DLWC_AA'

# Open-vocab category recognition
CATEGORY_CONFIDENCE_THRESHOLD = 0.22  # below this -> 'unknown'

# OPEN_VOCAB_MATRIX = None
# OPEN_VOCAB_PROMPTS = None
# OPEN_VOCAB_LABELS = None

# def get_labels():
#     return OPEN_VOCAB_LABELS

# def initialize_vocabulary():
#     global OPEN_VOCAB_MATRIX,OPEN_VOCAB_PROMPTS,OPEN_VOCAB_LABELS
#
#     # Ścieżka do Twojego głównego pliku ze słownikiem na Dysku
#     ALL_VOCAB_FILE = f'{DRIVE_FOLDER}/all_vocab.json'
#
#     # Wczytywanie etykiet bezpośrednio z pliku JSON
#     try:
#         with open(ALL_VOCAB_FILE, 'r', encoding='utf-8') as f:
#             OPEN_VOCAB_LABELS = json.load(f)
#         print(f"Wczytano {len(OPEN_VOCAB_LABELS)} etykiet z pliku {ALL_VOCAB_FILE}.")
#     except FileNotFoundError:
#         print(f"BŁĄD: Nie znaleziono pliku {ALL_VOCAB_FILE}.")
#         # Awaryjna lista, żeby notatnik się nie zawiesił
#         OPEN_VOCAB_LABELS = ['unknown object']
#
#     print("Wektoryzacja słownika (to potrwa chwilę)...")
#     OPEN_VOCAB_PROMPTS = [f'a photo of a {x}' for x in OPEN_VOCAB_LABELS]
#     OPEN_VOCAB_MATRIX = embed_clip_text_batch(OPEN_VOCAB_PROMPTS)
#     print(f"Kształt macierzy słownika: {OPEN_VOCAB_MATRIX.shape}")

def initialize_vocabulary_from_json(vocab_collection):
    """One-time initialization:JSON → Postgres vocabulary table"""

    ALL_VOCAB_FILE = f'./DLWC_AA/all_vocab.json'

    try:
        with open(ALL_VOCAB_FILE, 'r', encoding='utf-8') as f:
            labels = json.load(f)

        print(f"Loaded {len(labels)} labels from JSON.")

    except FileNotFoundError:
        print(f"Error: Could not find {ALL_VOCAB_FILE}")
        labels = ['unknown object']

    print("Generating embeddings...")

    prompts = [f"a photo of a {x}" for x in labels]
    embeddings = embed_clip_text_batch(prompts)

    print("Saving vocabulary to database...")

    vocab_collection.add_many(labels, embeddings)

    print("Initialization of vocabulary complete")


def load_vocab_cache_from_db(vocab_collection):
    """Load vocabulary from database"""
    rows = vocab_collection.get_all()
    # print("DEBUG ROW TYPE:", type(rows[0][1]))
    # print("DEBUG SAMPLE VALUE:", rows[0][1])

    labels = [r[0] for r in rows]

    matrix = np.array([
        # np.array(r[1], dtype=np.float32)
        np.array(ast.literal_eval(r[1]), dtype=np.float32)
        for r in rows
    ])

    prompts = [
        f"a photo of a {x}"
        for x in labels
    ]

    return {
        "labels": labels,
        "matrix": matrix,
        "prompts": prompts,
    }

#VERSION BEFORE INTRODUCTION OF DB
# def classify_open_vocab(crop_emb_clip: np.ndarray, top_n: int = 5):
#     global OPEN_VOCAB_MATRIX,OPEN_VOCAB_LABELS
#
#     sims = OPEN_VOCAB_MATRIX @ crop_emb_clip
#     order = np.argsort(-sims)
#     topn = [(OPEN_VOCAB_LABELS[i], float(sims[i])) for i in order[:top_n]]
#     best_name, best_sim = topn[0]
#     label = best_name if best_sim >= CATEGORY_CONFIDENCE_THRESHOLD else 'unknown'
#     return label, best_sim, topn

def classify_open_vocab(crop_emb_clip: np.ndarray,vocab_cache,top_n: int = 5):
    matrix = vocab_cache["matrix"]
    labels = vocab_cache["labels"]

    sims = matrix @ crop_emb_clip
    order = np.argsort(-sims)
    topn = [(labels[i], float(sims[i])) for i in order[:top_n]]
    best_name, best_sim = topn[0]
    label = (best_name if best_sim >= CATEGORY_CONFIDENCE_THRESHOLD else "unknown")

    return label, best_sim, topn

#VERSION BEFORE INTRODUCTION OF DB
# def extend_vocabulary(new_label: str):
#     """Dynamically adds a new custom label to the open vocabulary matrix if it's present and not already in vocabulary."""
#     global OPEN_VOCAB_LABELS,OPEN_VOCAB_MATRIX
#
#     if new_label is None:
#         return
#
#     if new_label in OPEN_VOCAB_LABELS:
#         print(f"Label '{new_label}' already exists in the vocabulary.")
#         return
#
#     print(f"Adding new label: '{new_label}'")
#     OPEN_VOCAB_LABELS.append(new_label)
#
#     # Generate the embedding for the new label
#     prompt = f"a photo of a {new_label}"
#     new_emb = embed_clip_text(prompt)
#
#     # Append the new embedding to the existing matrix
#     OPEN_VOCAB_MATRIX = np.vstack([OPEN_VOCAB_MATRIX, new_emb])
#     print("Vocabulary updated successfully. The system will now recognize this item.")

def extend_vocabulary(new_label: str,vocab_collection,vocab_cache):
    """Add new label to vocabulary DB and update runtime cache."""

    if new_label is None: return

    labels = vocab_cache["labels"]
    matrix = vocab_cache["matrix"]
    prompts = vocab_cache["prompts"]

    if new_label in labels:
        print(f"Label '{new_label}' already exists.")
        return

    print(f"Adding new label: '{new_label}'")

    # Generate CLIP text embedding
    prompt = f"a photo of a {new_label}"

    new_emb = embed_clip_text(prompt)

    # Persist to DB
    vocab_collection.add(label=new_label,embedding=new_emb)

    # Update runtime cache
    labels.append(new_label)

    prompts.append(prompt)

    vocab_cache["matrix"] = np.vstack([matrix,new_emb])

    print("Vocabulary updated successfully.")