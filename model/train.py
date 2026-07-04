"""
model/train.py
Two-tower recommendation model trained on synthetic interaction data.

Architecture:
  - User tower: user_id embedding + 3 history features -> 64-dim embedding
  - Item tower: item_id embedding + 3 content features -> 64-dim embedding
  - Loss: InfoNCE (in-batch negatives) — same technique used by YouTube DNN, Pinterest

Run: python -m model.train
Saves:
  model/artefacts/user_tower.pt
  model/artefacts/item_tower.pt
  model/artefacts/item_embeddings.npy  (all items pre-computed)
  model/artefacts/item_ids.npy
  model/artefacts/meta.json
"""

import json, pathlib, pickle
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import mlflow

from model.model_def import Tower, TwoTowerModel, N_USERS, N_ITEMS, EMBED_DIM, HIDDEN, TEMPERATURE

ARTEFACT_DIR  = pathlib.Path("model/artefacts")
ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────
N_INTERACTIONS= 80_000
BATCH_SIZE    = 512
EPOCHS        = 15
LR            = 3e-3
SEED          = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Synthetic dataset ─────────────────────────────────────────
def generate_data():
    """
    Simulate user-item interactions with latent preference clusters.
    Users and items are assigned to one of 10 clusters; same-cluster
    interactions are 5x more likely than cross-cluster ones.
    """
    N_CLUSTERS = 10
    user_cluster = np.random.randint(0, N_CLUSTERS, N_USERS)
    item_cluster = np.random.randint(0, N_CLUSTERS, N_ITEMS)

    interactions = []
    for _ in range(N_INTERACTIONS):
        u = np.random.randint(0, N_USERS)
        # 70% chance: pick item from same cluster
        if np.random.random() < 0.7:
            same = np.where(item_cluster == user_cluster[u])[0]
            i = np.random.choice(same)
        else:
            i = np.random.randint(0, N_ITEMS)
        interactions.append((u, i))

    interactions = list(set(interactions))   # deduplicate
    return np.array(interactions)


class InteractionDataset(Dataset):
    def __init__(self, interactions, n_users, n_items):
        self.interactions = interactions
        # Synthetic user features: age_norm, activity_score, tenure_days_norm
        self.user_feats = torch.tensor(
            np.random.rand(n_users, 3).astype(np.float32))
        # Synthetic item features: popularity_score, avg_rating_norm, freshness
        self.item_feats = torch.tensor(
            np.random.rand(n_items, 3).astype(np.float32))

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        u, i = self.interactions[idx]
        return (torch.tensor(u, dtype=torch.long),
                torch.tensor(i, dtype=torch.long),
                self.user_feats[u],
                self.item_feats[i])


# ── Training loop ─────────────────────────────────────────────
def train():
    print("[train] Generating synthetic interaction data...")
    interactions = generate_data()
    print(f"[train] {len(interactions)} unique interactions")

    # ── Train / held-out split ──────────────────────────────────
    # Held-out interactions per user are the ground truth eval.evaluate
    # uses to compute Recall@K/NDCG@K/HitRate@K. Without this, an eval
    # script has no real signal to score against.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(interactions))
    interactions = interactions[perm]
    split = int(len(interactions) * 0.9)
    train_interactions = interactions[:split]
    test_interactions  = interactions[split:]

    held_out = defaultdict(list)
    for u, i in test_interactions:
        held_out[int(u)].append(int(i))
    with open(ARTEFACT_DIR / "held_out.json", "w") as f:
        json.dump(held_out, f)
    print(f"[train] Held out {len(test_interactions)} interactions across "
          f"{len(held_out)} users for evaluation")

    dataset = InteractionDataset(train_interactions, N_USERS, N_ITEMS)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=0)

    model     = TwoTowerModel(N_USERS, N_ITEMS)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, EPOCHS)

    mlflow.set_experiment("two-tower-rec")
    with mlflow.start_run():
        mlflow.log_params({
            "n_users": N_USERS, "n_items": N_ITEMS,
            "embed_dim": EMBED_DIM, "hidden": HIDDEN,
            "batch_size": BATCH_SIZE, "epochs": EPOCHS,
            "lr": LR, "temperature": TEMPERATURE,
        })

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss, steps = 0.0, 0
            for u_ids, i_ids, u_feats, i_feats in loader:
                u_emb, i_emb = model(u_ids, i_ids, u_feats, i_feats)
                loss = model.infonce_loss(u_emb, i_emb)
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                total_loss += loss.item(); steps += 1

            scheduler.step()
            avg_loss = total_loss / steps
            mlflow.log_metric("loss", avg_loss, step=epoch)
            print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}")

        # ── Compute all item embeddings for Faiss index ───────
        print("[train] Computing item embeddings...")
        model.eval()
        all_item_ids   = torch.arange(N_ITEMS)
        all_item_feats = dataset.item_feats
        with torch.no_grad():
            item_embeddings = model.item_tower(all_item_ids, all_item_feats)
            item_embeddings = item_embeddings.numpy()

        # ── Save artefacts ────────────────────────────────────
        torch.save(model.user_tower.state_dict(), ARTEFACT_DIR / "user_tower.pt")
        torch.save(model.item_tower.state_dict(), ARTEFACT_DIR / "item_tower.pt")
        np.save(ARTEFACT_DIR / "item_embeddings.npy", item_embeddings)
        np.save(ARTEFACT_DIR / "item_ids.npy", np.arange(N_ITEMS))

        # Save user/item features for inference
        torch.save(dataset.user_feats, ARTEFACT_DIR / "user_feats.pt")
        torch.save(dataset.item_feats, ARTEFACT_DIR / "item_feats.pt")

        meta = {"n_users": N_USERS, "n_items": N_ITEMS,
                "embed_dim": EMBED_DIM, "hidden": HIDDEN,
                "final_loss": round(avg_loss, 4)}
        with open(ARTEFACT_DIR / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        mlflow.log_artifacts(str(ARTEFACT_DIR))
        print(f"[train] Done. Artefacts saved to {ARTEFACT_DIR}")
        return meta


if __name__ == "__main__":
    meta = train()
    print(f"\nFinal loss: {meta['final_loss']}")
