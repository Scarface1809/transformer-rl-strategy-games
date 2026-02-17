import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, num_tiles, num_nations, d_model=64):
        super().__init__()

        # --- Embeddings ---
        self.tile_pos_emb = nn.Embedding(num_tiles, d_model)
        self.terrain_emb = nn.Embedding(2, d_model)  # CLEAR=1, DIFFICULT=2

        self.nation_emb = nn.Embedding(num_nations, d_model)
        self.piece_tile_emb = nn.Embedding(num_tiles, d_model)

        # --- Fusion ---
        self.tile_fc = nn.Linear(2 * d_model, d_model)
        self.piece_fc = nn.Linear(2 * d_model, d_model)

        # --- Policy ---
        self.tile_policy = nn.Linear(d_model, 1)      # per-tile logit
        self.end_turn_logit = nn.Parameter(torch.tensor(0.0))

        # --- Value ---
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(
        self,
        tile_idxs,          # (num_tiles,)
        terrain_types,      # (num_tiles,)
        nation_idxs,        # (num_units,)
        piece_tile_idxs,    # (num_units,)
        active_nation,      # scalar LongTensor
    ):
        device = tile_idxs.device
        num_tiles = tile_idxs.size(0)

        # ---- Tile encodings ----
        tile_pos = self.tile_pos_emb(tile_idxs)
        terr = self.terrain_emb(terrain_types - 1)  # TerrainType value 1->0, 2->1
        tile_encs = self.tile_fc(torch.cat([tile_pos, terr], dim=-1))

        # ---- Piece encodings ----
        if nation_idxs.numel() > 0:
            n_emb = self.nation_emb(nation_idxs)
            t_emb = self.piece_tile_emb(piece_tile_idxs)
            piece_encs = self.piece_fc(torch.cat([n_emb, t_emb], dim=-1))
        else:
            piece_encs = torch.zeros(0, tile_encs.size(-1), device=device)

        # ---- Aggregate pieces into tiles ----
        tile_piece_sum = torch.zeros_like(tile_encs)
        for i in range(piece_encs.size(0)):
            tile_piece_sum[piece_tile_idxs[i]] += piece_encs[i]

        # ---- Active nation conditioning ----
        active_n = self.nation_emb(active_nation).unsqueeze(0)
        tile_state = tile_encs + tile_piece_sum + active_n

        # ---- Policy ----
        tile_logits = self.tile_policy(tile_state).squeeze(-1)
        policy_logits = torch.cat([tile_logits, self.end_turn_logit.expand(1)], dim=0)

        # ---- Value ----
        pooled = tile_state.mean(dim=0)
        value = self.value_head(pooled).squeeze(-1)
        return policy_logits, value
