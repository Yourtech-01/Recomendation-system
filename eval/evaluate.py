"""
eval/evaluate.py
Offline evaluation of the recommendation model.

Metrics computed:
  Recall@K    — what fraction of true items appear in top-K recommendations
  Precision@K — of the top-K shown, how many are relevant
  NDCG@K      — normalised discounted cumulative gain (position-aware)
  Hit Rate@K  — fraction of users who get at least one relevant item in top-K

Run: python -m eval.evaluate
"""

import json, pathlib, sys
import numpy as np
import torch

sys.path.insert(0, ".")

ARTEFACT_DIR = pathlib.Path("model/artefacts")
K_VALUES     = [5, 10, 20]
N_EVAL_USERS = 500    # evaluate on a random subset for speed


def dcg(relevances: list[int]) -> float:
    return sum(r / np.log2(i + 2) for i, r in enumerate(relevances))


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    hits    = [1 if item in relevant else 0 for item in recommended[:k]]
    ideal   = sorted(hits, reverse=True)
    return dcg(hits) / dcg(ideal) if dcg(ideal) > 0 else 0.0


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant) if relevant else 0.0


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    return float(any(item in relevant for item in recommended[:k]))


def run_evaluation():
    import faiss
    from model.model_def import TwoTowerModel, N_USERS, N_ITEMS, EMBED_DIM, HIDDEN

    with open(ARTEFACT_DIR / "meta.json") as f:
        meta = json.load(f)

    model = TwoTowerModel(N_USERS, N_ITEMS, EMBED_DIM, HIDDEN)
    model.user_tower.load_state_dict(
        torch.load(ARTEFACT_DIR / "user_tower.pt", map_location="cpu"))
    model.item_tower.load_state_dict(
        torch.load(ARTEFACT_DIR / "item_tower.pt", map_location="cpu"))
    model.eval()

    USER_FEATS = torch.load(ARTEFACT_DIR / "user_feats.pt")
    ITEM_FEATS = torch.load(ARTEFACT_DIR / "item_feats.pt")
    ITEM_IDS   = np.load(ARTEFACT_DIR / "item_ids.npy")
    INDEX      = faiss.read_index(str(ARTEFACT_DIR / "faiss.index"))
    INDEX.nprobe = 20   # higher nprobe for eval accuracy

    # Ground truth: real interactions held out from training (see
    # model/train.py's 90/10 split). This is what the model was NOT
    # trained on, so it's a genuine test of generalisation rather than
    # an arbitrary random set.
    held_out_path = ARTEFACT_DIR / "held_out.json"
    if not held_out_path.exists():
        raise FileNotFoundError(
            "held_out.json not found — re-run `python -m model.train` "
            "(older artefacts predate the train/test split).")
    with open(held_out_path) as f:
        held_out = {int(k): set(v) for k, v in json.load(f).items()}

    eval_users = list(held_out.keys())
    if len(eval_users) > N_EVAL_USERS:
        rng = np.random.default_rng(42)
        eval_users = rng.choice(eval_users, size=N_EVAL_USERS, replace=False)

    results = {k: {"recall": [], "precision": [], "ndcg": [], "hit_rate": []}
               for k in K_VALUES}
    baseline_results = {k: {"recall": [], "precision": [], "ndcg": [], "hit_rate": []}
                        for k in K_VALUES}

    print(f"[eval] Evaluating {len(eval_users)} users against real held-out interactions...")
    rng = np.random.default_rng(123)

    for user_id in eval_users:
        relevant = held_out[int(user_id)]
        if not relevant:
            continue

        uid   = torch.tensor([user_id])
        feats = USER_FEATS[user_id].unsqueeze(0)
        with torch.no_grad():
            emb = model.user_tower(uid, feats)
        emb_np = emb.numpy().astype(np.float32)
        faiss.normalize_L2(emb_np)

        max_k      = max(K_VALUES)
        distances, indices = INDEX.search(emb_np, max_k)
        recommended = [int(ITEM_IDS[idx]) for idx in indices[0] if idx != -1]

        # Random baseline: what you'd get with no model at all
        random_recommended = rng.choice(N_ITEMS, size=max_k, replace=False).tolist()

        for k in K_VALUES:
            results[k]["recall"].append(recall_at_k(recommended, relevant, k))
            results[k]["precision"].append(precision_at_k(recommended, relevant, k))
            results[k]["ndcg"].append(ndcg_at_k(recommended, relevant, k))
            results[k]["hit_rate"].append(hit_rate_at_k(recommended, relevant, k))

            baseline_results[k]["recall"].append(recall_at_k(random_recommended, relevant, k))
            baseline_results[k]["precision"].append(precision_at_k(random_recommended, relevant, k))
            baseline_results[k]["ndcg"].append(ndcg_at_k(random_recommended, relevant, k))
            baseline_results[k]["hit_rate"].append(hit_rate_at_k(random_recommended, relevant, k))

    print("\n=== Offline evaluation results (real held-out interactions) ===")
    summary = {"two_tower": {}, "random_baseline": {}}
    for k in K_VALUES:
        summary["two_tower"][f"@{k}"] = {
            metric: round(float(np.mean(values)), 4)
            for metric, values in results[k].items()
        }
        summary["random_baseline"][f"@{k}"] = {
            metric: round(float(np.mean(values)), 4)
            for metric, values in baseline_results[k].items()
        }
        t = summary["two_tower"][f"@{k}"]
        b = summary["random_baseline"][f"@{k}"]
        lift = (t["ndcg"] / b["ndcg"]) if b["ndcg"] > 0 else float("inf")
        print(f"  @{k:2d}  two-tower: Recall={t['recall']:.4f} NDCG={t['ndcg']:.4f} "
              f"HitRate={t['hit_rate']:.4f}   |   random: Recall={b['recall']:.4f} "
              f"NDCG={b['ndcg']:.4f} HitRate={b['hit_rate']:.4f}   |   NDCG lift: {lift:.1f}x")

    out = pathlib.Path("eval/results")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[eval] Saved to eval/results/metrics.json")
    return summary


if __name__ == "__main__":
    run_evaluation()
