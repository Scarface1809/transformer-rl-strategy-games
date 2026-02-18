import torch
import torch.nn as nn

class SimpleTransformerModel(nn.Module):
    def __init__(self, num_tiles, num_nations, d_model=64, n_heads=4, n_layers=2):
        super().__init__()

        # ---- Embeddings ----
        self.tile_pos_emb = nn.Embedding(num_tiles, d_model)
        self.terrain_emb = nn.Embedding(2, d_model)
        self.nation_emb = nn.Embedding(num_nations, d_model)
        self.unit_tile_emb = nn.Embedding(num_tiles, d_model)

        # ---- Encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        # ---- Decoder ----
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)

        # ---- Heads ----
        self.tile_policy = nn.Linear(d_model, 1)
        self.unit_policy = nn.Linear(d_model, 1)
        self.end_turn_logit = nn.Parameter(torch.tensor(0.0))

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(
        self,
        tile_idxs,
        terrain_types,
        nation_idxs,
        piece_tile_idxs,
        active_nation
    ):
        device = tile_idxs.device

        # ---- Tile tokens ----
        tile_tokens = (
            self.tile_pos_emb(tile_idxs)
            + self.terrain_emb(terrain_types - 1)
            + self.nation_emb(active_nation)
        )  # (T, D)

        tile_tokens = tile_tokens.unsqueeze(0)  # (1, T, D)

        # ---- Unit tokens ----
        if nation_idxs.numel() > 0:
            unit_tokens = (
                self.nation_emb(nation_idxs)
                + self.unit_tile_emb(piece_tile_idxs)
            )
            unit_tokens = unit_tokens.unsqueeze(0)
            tokens = torch.cat([tile_tokens, unit_tokens], dim=1)
        else:
            tokens = tile_tokens

        # ---- Encode ----
        memory = self.encoder(tokens)

        # ---- Decode (tiles attend to full memory) ----
        tile_context = self.decoder(tile_tokens, memory).squeeze(0)

        # ---- Policy ----
        tile_logits = self.tile_policy(tile_context).squeeze(-1)
        if tokens.size(1) > tile_tokens.size(1):
            unit_context = memory.squeeze(0)[tile_tokens.size(1):]
            unit_logits = self.unit_policy(unit_context).squeeze(-1)
        else:
            unit_logits = torch.empty(0, device=device)

        # ---- Value ----
        value = self.value_head(tile_context.mean(dim=0)).squeeze(-1)

        return tile_logits, unit_logits, value
