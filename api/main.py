"""
api/main.py
FastAPI recommendation service.

Endpoints:
  POST /recommend         -> top-k items for a user
  GET  /similar/{item_id} -> items similar to a given item
  GET  /metrics           -> live latency + request stats
  GET  /ab/stats          -> A/B test click-through rates
  GET  /health
"""

import json, os, pathlib, random, time
from collections import defaultdict

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ARTEFACT_DIR = pathlib.Path(os.getenv(
    "ARTEFACT_DIR",
    str(pathlib.Path(__file__).resolve().parent.parent / "model" / "artefacts")))
TOP_K        = int(os.getenv("TOP_K", "20"))
RERANK_K     = int(os.getenv("RERANK_K", "10"))

# ── Load artefacts ────────────────────────────────────────────
with open(ARTEFACT_DIR / "meta.json") as f:
    META = json.load(f)

N_USERS = META["n_users"]
N_ITEMS = META["n_items"]
EMBED_DIM = META["embed_dim"]
HIDDEN    = META["hidden"]

# Model architecture only — no mlflow dependency needed for inference
import sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from model.model_def import Tower, TwoTowerModel

model = TwoTowerModel(N_USERS, N_ITEMS, EMBED_DIM, HIDDEN)
model.user_tower.load_state_dict(
    torch.load(ARTEFACT_DIR / "user_tower.pt", map_location="cpu"))
model.item_tower.load_state_dict(
    torch.load(ARTEFACT_DIR / "item_tower.pt", map_location="cpu"))
model.eval()

USER_FEATS = torch.load(ARTEFACT_DIR / "user_feats.pt")
ITEM_FEATS = torch.load(ARTEFACT_DIR / "item_feats.pt")
ITEM_IDS   = np.load(ARTEFACT_DIR / "item_ids.npy")

INDEX = faiss.read_index(str(ARTEFACT_DIR / "faiss.index"))
INDEX.nprobe = 10

# ── In-memory stats ───────────────────────────────────────────
stats = {"requests": 0, "latency_ms": []}
ab_stats = defaultdict(lambda: {"impressions": 0, "clicks": 0})

# ── Schemas ───────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    user_id:   int   = Field(..., ge=0, lt=N_USERS, example=42)
    top_k:     int   = Field(RERANK_K, ge=1, le=50)
    ab_group:  str   = Field("control",
                             description="'control' (collaborative) or 'treatment' (two-tower)")
    exclude:   list[int] = Field(default_factory=list,
                                  description="Item IDs to exclude (already seen)")

class RecommendResponse(BaseModel):
    user_id:      int
    items:        list[int]
    scores:       list[float]
    latency_ms:   float
    model:        str
    ab_group:     str


# ── Inference helpers ─────────────────────────────────────────
def get_user_embedding(user_id: int) -> np.ndarray:
    uid   = torch.tensor([user_id])
    feats = USER_FEATS[user_id].unsqueeze(0)
    with torch.no_grad():
        emb = model.user_tower(uid, feats)
    emb_np = emb.numpy().astype(np.float32)
    faiss.normalize_L2(emb_np)
    return emb_np


def faiss_search(query: np.ndarray, k: int, exclude: list[int]) -> tuple[list, list]:
    """ANN search, then filter excluded items."""
    fetch_k = min(k + len(exclude) + 10, N_ITEMS)
    distances, indices = INDEX.search(query, fetch_k)
    items, scores = [], []
    exclude_set = set(exclude)
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1: continue
        item_id = int(ITEM_IDS[idx])
        if item_id in exclude_set: continue
        items.append(item_id)
        scores.append(round(float(dist), 4))
        if len(items) >= k: break
    return items, scores


def collaborative_fallback(user_id: int, k: int) -> tuple[list, list]:
    """
    Simple popularity-based fallback for A/B control group.
    In a real system this would be matrix factorisation.
    """
    rng = np.random.default_rng(user_id)
    items  = rng.choice(N_ITEMS, size=k, replace=False).tolist()
    scores = sorted(rng.random(k).tolist(), reverse=True)
    return items, scores


# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(title="Recommendation Engine API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    t0 = time.perf_counter()

    if req.ab_group == "treatment":
        # Two-tower model path
        query = get_user_embedding(req.user_id)
        items, scores = faiss_search(query, req.top_k, req.exclude)
        model_name = "two-tower-faiss"
    else:
        # Control: popularity-based baseline
        items, scores = collaborative_fallback(req.user_id, req.top_k)
        model_name = "popularity-baseline"

    # Truncate to requested k
    items  = items[:req.top_k]
    scores = scores[:req.top_k]

    latency = (time.perf_counter() - t0) * 1000
    stats["requests"] += 1
    stats["latency_ms"].append(latency)
    if len(stats["latency_ms"]) > 2000:
        stats["latency_ms"].pop(0)

    ab_stats[req.ab_group]["impressions"] += 1

    return RecommendResponse(
        user_id=req.user_id, items=items, scores=scores,
        latency_ms=round(latency, 2), model=model_name, ab_group=req.ab_group,
    )


@app.get("/similar/{item_id}")
def similar_items(item_id: int, top_k: int = 10):
    """Find items most similar to a given item (item-to-item)."""
    if item_id >= N_ITEMS:
        raise HTTPException(404, f"item_id {item_id} out of range")

    iid   = torch.tensor([item_id])
    feats = ITEM_FEATS[item_id].unsqueeze(0)
    with torch.no_grad():
        emb = model.item_tower(iid, feats)
    emb_np = emb.numpy().astype(np.float32)
    faiss.normalize_L2(emb_np)

    distances, indices = INDEX.search(emb_np, top_k + 1)
    results = [
        {"item_id": int(ITEM_IDS[idx]), "score": round(float(d), 4)}
        for idx, d in zip(indices[0], distances[0])
        if idx != -1 and int(ITEM_IDS[idx]) != item_id
    ][:top_k]
    return {"query_item": item_id, "similar": results}


@app.post("/ab/click")
def record_click(user_id: int, item_id: int, ab_group: str):
    """Record a click for A/B test tracking."""
    ab_stats[ab_group]["clicks"] += 1
    return {"recorded": True}


@app.get("/ab/stats")
def ab_test_stats():
    """Live A/B test click-through rates."""
    result = {}
    for group, s in ab_stats.items():
        impr = s["impressions"]
        ctr  = s["clicks"] / impr if impr > 0 else 0
        result[group] = {**s, "ctr": round(ctr * 100, 2)}
    return result


@app.get("/metrics")
def get_metrics():
    lat = stats["latency_ms"]
    return {
        "total_requests": stats["requests"],
        "p50_latency_ms": round(float(np.percentile(lat, 50)) if lat else 0, 2),
        "p99_latency_ms": round(float(np.percentile(lat, 99)) if lat else 0, 2),
        "index_size":     int(INDEX.ntotal),
        "embed_dim":      EMBED_DIM,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": "two-tower", "index_size": int(INDEX.ntotal)}
