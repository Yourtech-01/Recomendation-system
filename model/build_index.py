"""
model/build_index.py
Builds a Faiss IVF-Flat index from pre-computed item embeddings.

IVF-Flat: inverted file index with exact distance within each cell.
- Fast at scale (millions of items) with minimal recall loss
- nlist=100 clusters, nprobe=10 cells searched at query time

Run after training: python -m model.build_index
Saves: model/artefacts/faiss.index
"""

import pathlib
import numpy as np

ARTEFACT_DIR = pathlib.Path("model/artefacts")


def build_index():
    try:
        import faiss
    except ImportError:
        print("Install faiss: pip install faiss-cpu")
        return

    embeddings = np.load(ARTEFACT_DIR / "item_embeddings.npy").astype(np.float32)
    n, d = embeddings.shape
    print(f"[index] Building Faiss IVF-Flat index: {n} items, {d} dims")

    # L2-normalise (embeddings already normalised from training, but be safe)
    faiss.normalize_L2(embeddings)

    # IVF-Flat index: good balance of speed vs recall for < 10M items
    nlist    = min(100, n // 10)    # number of Voronoi cells
    quantiser = faiss.IndexFlatIP(d)   # inner product on L2-normalised = cosine
    index    = faiss.IndexIVFFlat(quantiser, d, nlist, faiss.METRIC_INNER_PRODUCT)

    # Train the index (needed for IVF)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = 10   # cells to search at query time — higher = more accurate

    faiss.write_index(index, str(ARTEFACT_DIR / "faiss.index"))
    print(f"[index] Index built: {index.ntotal} vectors, nlist={nlist}, nprobe=10")
    print(f"[index] Saved to {ARTEFACT_DIR / 'faiss.index'}")


if __name__ == "__main__":
    build_index()
