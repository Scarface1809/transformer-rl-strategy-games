import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """
    Policy + value model for SimpleHispaniaEnv.

    Inputs
    ------
    tile_idxs       : (num_tiles,)   LongTensor
    terrain_types   : (num_tiles,)   LongTensor  (TerrainType.value, 1-based)
    nation_idxs     : (num_units,)   LongTensor
    piece_tile_idxs : (num_units,)   LongTensor
    active_nation   : ()             scalar LongTensor
    phase_id        : ()             scalar LongTensor  (Phase.value, 1-based)

    Outputs
    -------
    tile_logits     : (num_tiles,)
    unit_logits     : (num_units,)
    value           : ()             scalar
    end_turn_logit  : nn.Parameter  (shared for END_TURN and END_PHASE sentinels)
    """

    def __init__(self, num_tiles: int, num_nations: int, d_model: int = 64):
        super().__init__()

        # --- Embeddings ---
        self.tile_pos_emb  = nn.Embedding(num_tiles, d_model)
        self.terrain_emb   = nn.Embedding(4, d_model)       # terrain values 1-4 → indices 0-3
        self.nation_emb    = nn.Embedding(num_nations, d_model)
        self.piece_tile_emb = nn.Embedding(num_tiles, d_model)
        self.phase_emb     = nn.Embedding(8, d_model)       # up to 8 phases; value 1-based → idx 0-based

        # --- Fusion ---
        self.tile_fc  = nn.Linear(2 * d_model, d_model)
        self.piece_fc = nn.Linear(2 * d_model, d_model)

        # --- Policy heads ---
        self.tile_policy      = nn.Linear(d_model, 1)   # per-tile logit (move target / placement tile)
        self.unit_policy      = nn.Linear(d_model, 1)   # per-unit logit (movement phase)
        # Single learnable scalar for both "end turn" and "end growth" actions
        self.end_turn_logit   = nn.Parameter(torch.tensor(0.0))

        # --- Value head ---
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        tile_idxs,        # (T,)
        terrain_types,    # (T,)  1-based
        nation_idxs,      # (U,)
        piece_tile_idxs,  # (U,)
        active_nation,    # scalar
        phase_id,         # scalar  (Phase.value, 1-based)
    ):
        device = tile_idxs.device

        # ---- Tile encodings ----
        tile_pos  = self.tile_pos_emb(tile_idxs)
        terr      = self.terrain_emb((terrain_types - 1).clamp(min=0, max=3))
        tile_encs = self.tile_fc(torch.cat([tile_pos, terr], dim=-1))   # (T, d)

        # ---- Piece encodings ----
        if nation_idxs.numel() > 0:
            n_emb      = self.nation_emb(nation_idxs)
            t_emb      = self.piece_tile_emb(piece_tile_idxs)
            piece_encs = self.piece_fc(torch.cat([n_emb, t_emb], dim=-1))  # (U, d)
        else:
            piece_encs = torch.zeros(0, tile_encs.size(-1), device=device)

        # ---- Aggregate pieces into tiles ----
        tile_piece_sum = torch.zeros_like(tile_encs)
        if piece_encs.numel() > 0:
            for i in range(piece_encs.size(0)):
                tile_piece_sum[piece_tile_idxs[i]] += piece_encs[i]

        # ---- Active nation + phase conditioning ----
        active_n    = self.nation_emb(active_nation).unsqueeze(0)          # (1, d)
        phase_enc   = self.phase_emb((phase_id - 1).clamp(min=0, max=7)).unsqueeze(0)  # (1, d)
        tile_state  = tile_encs + tile_piece_sum + active_n + phase_enc    # (T, d)

        # ---- Policy logits ----
        tile_logits = self.tile_policy(tile_state).squeeze(-1)             # (T,)
        if piece_encs.numel() > 0:
            unit_logits = self.unit_policy(piece_encs).squeeze(-1)        # (U,)
        else:
            unit_logits = torch.empty(0, device=device)

        # ---- Value ----
        pooled = tile_state.mean(dim=0)
        value  = self.value_head(pooled).squeeze(-1)

        return tile_logits, unit_logits, value