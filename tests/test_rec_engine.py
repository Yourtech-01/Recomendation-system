"""
tests/test_rec_engine.py
Run: pytest tests/ -v
"""

import sys, pathlib
import numpy as np
import torch
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


# ── Model unit tests ──────────────────────────────────────────
class TestTwoTowerModel:
    def setup_method(self):
        from model.model_def import TwoTowerModel
        self.model = TwoTowerModel(n_users=100, n_items=200,
                                   embed_dim=32, hidden=64)

    def test_output_shape(self):
        u_ids   = torch.randint(0, 100, (16,))
        i_ids   = torch.randint(0, 200, (16,))
        u_feats = torch.rand(16, 3)
        i_feats = torch.rand(16, 3)
        u_emb, i_emb = self.model(u_ids, i_ids, u_feats, i_feats)
        assert u_emb.shape == (16, 32)
        assert i_emb.shape == (16, 32)

    def test_embeddings_l2_normalised(self):
        u_ids   = torch.randint(0, 100, (8,))
        u_feats = torch.rand(8, 3)
        i_ids   = torch.randint(0, 200, (8,))
        i_feats = torch.rand(8, 3)
        u_emb, i_emb = self.model(u_ids, i_ids, u_feats, i_feats)
        u_norms = torch.norm(u_emb, dim=-1)
        i_norms = torch.norm(i_emb, dim=-1)
        assert torch.allclose(u_norms, torch.ones(8), atol=1e-5), "User embeddings not L2-normalised"
        assert torch.allclose(i_norms, torch.ones(8), atol=1e-5), "Item embeddings not L2-normalised"

    def test_infonce_loss_positive(self):
        u_ids   = torch.randint(0, 100, (32,))
        i_ids   = torch.randint(0, 200, (32,))
        u_feats = torch.rand(32, 3)
        i_feats = torch.rand(32, 3)
        u_emb, i_emb = self.model(u_ids, i_ids, u_feats, i_feats)
        loss = self.model.infonce_loss(u_emb, i_emb)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_loss_decreases_after_step(self):
        """One gradient step should reduce the loss."""
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        u_ids   = torch.randint(0, 100, (32,))
        i_ids   = torch.randint(0, 200, (32,))
        u_feats = torch.rand(32, 3)
        i_feats = torch.rand(32, 3)
        u_emb, i_emb = self.model(u_ids, i_ids, u_feats, i_feats)
        loss1 = self.model.infonce_loss(u_emb, i_emb)
        opt.zero_grad(); loss1.backward(); opt.step()
        u_emb2, i_emb2 = self.model(u_ids, i_ids, u_feats, i_feats)
        loss2 = self.model.infonce_loss(u_emb2, i_emb2)
        assert loss2.item() < loss1.item(), "Loss should decrease after one step"


# ── Eval metric tests ─────────────────────────────────────────
class TestMetrics:
    def setup_method(self):
        from eval.evaluate import recall_at_k, precision_at_k, ndcg_at_k, hit_rate_at_k
        self.recall     = recall_at_k
        self.precision  = precision_at_k
        self.ndcg       = ndcg_at_k
        self.hit_rate   = hit_rate_at_k

    def test_perfect_recall(self):
        recommended = [1, 2, 3, 4, 5]
        relevant    = {1, 2, 3}
        assert self.recall(recommended, relevant, k=5) == 1.0

    def test_zero_recall(self):
        assert self.recall([1, 2, 3], {4, 5, 6}, k=3) == 0.0

    def test_ndcg_perfect_order(self):
        recommended = [1, 2, 3]
        relevant    = {1, 2, 3}
        assert self.ndcg(recommended, relevant, k=3) == pytest.approx(1.0, abs=1e-4)

    def test_hit_rate_hit(self):
        assert self.hit_rate([10, 20, 30], {20}, k=3) == 1.0

    def test_hit_rate_miss(self):
        assert self.hit_rate([10, 20, 30], {99}, k=3) == 0.0


# ── Artefacts test ────────────────────────────────────────────
def test_artefacts_present():
    base = pathlib.Path("model/artefacts")
    for fname in ["user_tower.pt", "item_tower.pt",
                  "item_embeddings.npy", "faiss.index", "meta.json"]:
        assert (base / fname).exists(), f"Missing: {fname} — run model/train.py first"
