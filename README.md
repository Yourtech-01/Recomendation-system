# Personalized recommendation engine

> Two-tower neural network · Faiss ANN · FastAPI · A/B testing · Streamlit

**Live demo:** _add your Streamlit URL here after deploying_
**API docs:** _add your Render URL here after deploying_ `/docs`

See [README_DEPLOY.md](README_DEPLOY.md) for the full deployment guide.

---

## The problem

Generic recommendations show users items they've already seen or don't care
about, killing click-through rates. This project builds a **personalised
two-tower model** that learns a 64-dimensional embedding for every user and
item, then serves recommendations at scale using approximate
nearest-neighbour search.

## Results

Measured against **real held-out interactions** (a 90/10 split of the
synthetic interaction data — see "Evaluation methodology" below), not an
arbitrary random set:

| Metric | Two-tower model | Random baseline | Lift |
|---|---|---|---|
| NDCG@10 | 0.0181 | 0.0008 | **~23x** |
| NDCG@20 | 0.0253 | 0.0018 | **~14x** |
| Hit Rate@20 | 7.0% | 0.6% | **~12x** |
| P99 inference latency | **< 10 ms** | — | — |

**Read this before quoting these numbers anywhere:** the absolute Recall/NDCG
values are low in absolute terms — that's expected and honest, not a bug.
With 5,000 items and only ~4 held-out interactions per user on average, even
a strong model will have modest absolute recall; the meaningful number here
is the **lift over a random baseline** (10-20x), which isolates what the
model actually learned. See the "Evaluation methodology" section for why
this matters and how to get stronger absolute numbers with more/real data.

## Architecture

```
User features (ID + history)     Item features (ID + content)
          │                                  │
    [User tower]                       [Item tower]
     64-dim emb                         64-dim emb
          │                                  │
          └──── dot product similarity ──────┘
                    InfoNCE loss
                    (in-batch negatives)
                         │
               [Faiss IVF-Flat index]
               all item embeddings
                         │
               [Re-ranker: top-k filter]
                         │
              [FastAPI /recommend]  ←── A/B layer
                         │
              [Streamlit demo UI]
```

The API and UI are deployed as two independent services:

| Service | Tech | Hosted on |
|---|---|---|
| `api/` | FastAPI, Faiss, PyTorch (inference only) | Render |
| `ui/`  | Streamlit | Streamlit Community Cloud |

## Tech stack

| Layer | Technology |
|---|---|
| Model | PyTorch two-tower, InfoNCE loss |
| Retrieval | Faiss IVF-Flat (cosine similarity) |
| Serving | FastAPI async |
| Experiment tracking (local/dev only) | MLflow |
| UI | Streamlit with live A/B stats |
| Infra | Docker Compose (local), Render + Streamlit Cloud (live) |

## Key engineering decisions

**Why two-tower over matrix factorisation?**
Two-tower handles cold-start: a new user or item gets a reasonable embedding
from side features alone (age, activity, content tags) without needing
interaction history. Classic MF fails for new entities.

**Why Faiss IVF-Flat over brute force?**
Brute-force exact search over 5k items is fast enough, but IVF-Flat scales
to millions with minimal recall loss at much lower latency. Starting with
the scalable approach is deliberate, even at this dataset size.

**InfoNCE with in-batch negatives**
Every item in the batch that isn't the positive becomes a negative
automatically — no separate negative sampling step. This is the same
technique used in large-scale systems (YouTube DNN, Pinterest PinSage). More
efficient than pairwise BPR loss.

**A/B testing built in from day one**
The `/recommend` endpoint accepts an `ab_group` parameter. The `/ab/stats`
endpoint tracks impressions and clicks per group, so CTR lift over baseline
is measurable live in the demo, not just claimed in a README.

**Model architecture is decoupled from training (`model/model_def.py`)**
The `Tower`/`TwoTowerModel` classes live in their own module, separate from
`train.py`. This means the deployed API only imports the architecture
definition, not the training script — so it doesn't need `mlflow` or any
other training-only dependency at inference time. Smaller, faster, more
reliable production image.

## Evaluation methodology

`eval/evaluate.py` computes Recall@K, Precision@K, NDCG@K, and Hit Rate@K
against **real held-out interactions**, not a synthetic proxy. Specifically:

1. `model/train.py` splits the generated interactions 90/10 before training
   — the model never sees the held-out 10%.
2. The held-out interactions per user are saved to
   `model/artefacts/held_out.json`.
3. `eval/evaluate.py` loads that file and scores the model's ranked
   recommendations against it, alongside a random-recommendation baseline
   for the same users, so the reported lift is directly attributable to the
   model rather than to how the ground truth was defined.

This matters because it's easy to get impressive-looking numbers from an
evaluation script whose "ground truth" doesn't actually reflect what the
model was trained on — that produces numbers that don't survive a follow-up
question in an interview. This project's eval is set up so the numbers are
reproducible and defensible.

**To get materially stronger absolute numbers:** train on real interaction
data (more signal per user than 4 synthetic held-out clicks) and/or increase
`N_INTERACTIONS`/`EPOCHS` in `model/train.py`.

```bash
python -m eval.evaluate
```

## Quick start

```bash
# 1. Install and train
pip install -r requirements.txt
python -m model.train           # trains + saves held_out.json for eval
python -m model.build_index     # builds Faiss index

# 2. Run offline eval (Recall@K, NDCG@K, HitRate@K vs random baseline)
python -m eval.evaluate

# 3. Start API + UI + MLflow
docker-compose up --build

# 4. Open the demo
open http://localhost:8501      # Streamlit UI
open http://localhost:8000/docs # FastAPI auto-docs
open http://localhost:5000      # MLflow experiment tracker
```

## Deploying it live

See [README_DEPLOY.md](README_DEPLOY.md) for the full Render + Streamlit
Community Cloud walkthrough. Model weights are committed under
`model/artefacts/` (~4.6MB total), so the live API needs no training or
ingestion step at boot — it serves immediately.

## Project structure

```
├── model/
│   ├── model_def.py      # Tower/TwoTowerModel architecture + constants (no mlflow dep)
│   ├── train.py           # Data generation, 90/10 split, training loop, MLflow logging
│   ├── build_index.py     # Faiss IVF-Flat index builder
│   └── artefacts/         # Committed: user_tower.pt, item_tower.pt, faiss.index,
│                           # item_embeddings.npy, item_ids.npy, meta.json, held_out.json
├── api/
│   ├── main.py             # FastAPI: /recommend, /similar, /ab/stats, /metrics, /health
│   └── requirements.txt    # Lean inference-only deps (torch cpu, faiss-cpu, fastapi)
├── eval/
│   └── evaluate.py         # Recall@K/NDCG@K/HitRate@K vs random baseline, on real held-out data
├── ui/
│   ├── app.py               # Streamlit: live recs + item similarity + A/B dashboard
│   └── requirements.txt     # Lean UI-only deps (streamlit, requests, plotly)
├── tests/
│   └── test_rec_engine.py  # pytest: model shape, L2-norm, loss, metrics
├── requirements.txt         # Full dev requirements (training + eval + serving + tests)
├── Dockerfile                # Local docker-compose image (dev only)
├── docker-compose.yml         # API + UI + MLflow in one command (local dev)
├── render.yaml                 # Render deployment blueprint (production API)
└── README_DEPLOY.md            # Step-by-step live deployment guide
```

## Running tests

```bash
pytest tests/ -v
# Covers: embedding shape, L2 normalisation, loss decreasing, metric correctness
```

## Resume bullets

- Built a two-tower recommendation model (PyTorch, InfoNCE loss with
  in-batch negatives) evaluated against real held-out interactions —
  10-20x NDCG lift over a random baseline
- Deployed a Faiss IVF-Flat ANN index serving 5,000 items at sub-10ms
  inference latency via FastAPI
- Built an A/B testing framework (treatment vs. baseline) with live
  CTR tracking exposed through a Streamlit dashboard
- Designed a train/inference dependency split (`model_def.py`) so the
  production API image doesn't require training-only dependencies

## Author

Happy Kumar — M.Tech Data Science, Chandigarh University
[GitHub](https://github.com/Yourtech-01)
