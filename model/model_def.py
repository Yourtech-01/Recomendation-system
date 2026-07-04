"""
model/model_def.py
Shared two-tower model architecture and dataset constants.

Kept separate from train.py so that the API (api/main.py) and eval script
(eval/evaluate.py) can import the model class + constants without pulling in
mlflow, which is only needed during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Dataset / architecture constants ─────────────────────────────────────────
N_USERS      = 2_000
N_ITEMS      = 5_000
EMBED_DIM    = 64
HIDDEN       = 128
TEMPERATURE  = 0.07     # InfoNCE temperature


class Tower(nn.Module):
    """Single tower: ID embedding + side features -> L2-normalised embedding."""
    def __init__(self, n_ids: int, n_feats: int, embed_dim: int, hidden: int):
        super().__init__()
        self.id_embed = nn.Embedding(n_ids, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + n_feats, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, embed_dim),
        )
        nn.init.xavier_uniform_(self.id_embed.weight)

    def forward(self, ids: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.id_embed(ids), feats], dim=-1)
        out = self.mlp(x)
        return F.normalize(out, dim=-1)   # L2 normalise for cosine similarity


class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=EMBED_DIM, hidden=HIDDEN):
        super().__init__()
        self.user_tower = Tower(n_users, 3, embed_dim, hidden)
        self.item_tower = Tower(n_items, 3, embed_dim, hidden)
        self.temperature = nn.Parameter(torch.tensor(TEMPERATURE))

    def forward(self, user_ids, item_ids, user_feats, item_feats):
        u_emb = self.user_tower(user_ids, user_feats)    # (B, D)
        i_emb = self.item_tower(item_ids, item_feats)    # (B, D)
        return u_emb, i_emb

    def infonce_loss(self, u_emb, i_emb):
        """
        In-batch negatives InfoNCE loss.
        Every item in the batch that isn't the positive is a negative.
        Logits shape: (B, B) — diagonal = positive pairs.
        """
        logits = torch.matmul(u_emb, i_emb.T) / self.temperature.clamp(min=1e-4)
        labels = torch.arange(len(u_emb), device=u_emb.device)
        loss_u = F.cross_entropy(logits, labels)
        loss_i = F.cross_entropy(logits.T, labels)
        return (loss_u + loss_i) / 2
