from collections import Counter

from models.crop import Crop

SAME_OBJECT_THRESHOLD = 0.90      # >= this: same specimen
SAME_CATEGORY_THRESHOLD = 0.70    # >= this but < SAME_OBJECT: same category, different specimen

def query_crop(crop: Crop, things, top_k: int = 5) -> list[dict]:
    """Query the vector DB for top-K most similar entries.
    Returns list of dicts: {id, distance, similarity, metadata}.
    Empty list if DB is empty."""
    if things.count() == 0:
        return []
    res = things.query(
        query_embeddings=[crop.emb_dino.tolist()],
        n_results=min(top_k, things.count()),
        # include=['metadatas', 'distances'],
    )
    out = []
    for i in range(len(res['ids'][0])):
        dist = float(res['distances'][0][i])
        out.append({
            'id': res['ids'][0][i],
            'distance': dist,
            'similarity': 1.0 - dist,  # cosine distance -> cosine similarity
            'metadata': res['metadatas'][0][i],
        })
    return out
def respond_to_query_location(location: str, things) -> dict:
    """Query the vector DB for entries at a given location.
    Returns dict with 'message' and 'matches' (list of dicts with id, metadata, location)."""
    res = things.query_loc(location=location)
    matches = []
    for i in range(len(res['ids'][0])):
        matches.append({
            'id': res['ids'][0][i],
            'metadata': res['metadatas'][0][i],
            'location': res['locations'][0][i],
        })
    message = f'Found {len(matches)} things at location "{location}".'
    return {
        'message': message,
        'matches': matches,
    }

def respond_to_query(crop: Crop, things,top_k: int = 5) -> dict:
    matches = query_crop(crop, things,top_k=top_k)
    response = {
        'message': '',
        'matches': matches,
    }
    if not matches:
        response['message'] = (
            'Register it to start the registry.'
        )
        return response

    top = matches[0]
    top_sim = top['similarity']
    top_loc = top['metadata'].get('location', '?')

    if top_sim >= SAME_OBJECT_THRESHOLD:
        response['message'] = (
            f'This appears to be your specific thing'
            f'(similarity {top_sim:.3f}). It belongs at: {top_loc}.'
        )
    elif top_sim >= SAME_CATEGORY_THRESHOLD:
        # Aggregate locations of similar items above the same-category threshold
        relevant = [m for m in matches if m['similarity'] >= SAME_CATEGORY_THRESHOLD]
        loc_counts = Counter(m['metadata'].get('location', '?') for m in relevant)
        loc_dominant = loc_counts.most_common(1)[0][0]
       
        response['message'] = (
            f'top similarity {top_sim:.3f} '
            f'Items in this group are typically kept at: {loc_dominant}. '
        )
    else:
        response['tier'] = 'unknown'
        response['message'] = (
            f'I do not recognize this object (best similarity only {top_sim:.3f}). '
            f'Top open-vocab guesses: '
            # + ', '.join(f'{n} ({s:.2f})' for n, s in crop.category_topk[:3])
            + '. Want to register it?'
        )
    return response

# 13. Query: pick a registered crop and ask the registry about it
# QUERY_INDEX_IN_REGISTERED = 0
#
# if not registered:
#     print('No registered crops to query. Run cell 10 first.')
# else:
#     _, qcrop, qloc = registered[QUERY_INDEX_IN_REGISTERED]
#     print(f'Query: {qcrop.source_path.name}#{qcrop.crop_idx}  (registered location: "{qloc}")')
#     print()
#     response = respond_to_query(qcrop, top_k=5)
#     print(f'>>> tier:    {response["tier"]}')
#     print(f'>>> message: {response["message"]}')
#     print()
#     print('Top matches:')
#     for m in response['matches']:
#         meta = m['metadata']
#         print(f'  sim={m["similarity"]:.3f}  cat="{meta["category"]}"  '
#               f'loc="{meta["location"]}"  src={Path(meta["source_path"]).name}#{meta["crop_idx"]}')
#
#     # Show the query plus its top matches visually
#     fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))
#     axes[0].imshow(qcrop.image)
#     axes[0].set_title(f'QUERY\n{qcrop.category}\n{qcrop.source_path.stem}', fontsize=9)
#     axes[0].axis('off')
#     for ax, m in zip(axes[1:], response['matches']):
#         meta = m['metadata']
#         # Reload the source image to display the matched crop
#         try:
#             mimg = Image.open(meta['source_path']).convert('RGB')
#             x1, y1, x2, y2 = (int(v) for v in meta['bbox'].split(','))
#             ax.imshow(mimg.crop((x1, y1, x2, y2)))
#         except Exception:
#             ax.text(0.5, 0.5, '(image not found)', ha='center', va='center')
#         ax.set_title(f'sim={m["similarity"]:.3f}\n{meta["category"]}\n{meta["location"]}', fontsize=9)
#         ax.axis('off')
#     plt.tight_layout(); plt.show()