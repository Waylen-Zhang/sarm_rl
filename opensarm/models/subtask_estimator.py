import torch
import torch.nn as nn

class SubtaskTransformer(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 vis_emb_dim: int = 512,
                 text_emb_dim: int = 512,
                 state_dim: int = 7,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 num_cameras: int = 1):
        super().__init__()
        self.d_model = d_model
        self.num_cameras = num_cameras

        # Projections
        self.lang_proj = nn.Linear(text_emb_dim, d_model)
        self.visual_proj = nn.Linear(vis_emb_dim, d_model)
        self.state_proj  = nn.Linear(state_dim, d_model)

        # Encoder
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, n_layers)

        # Learned bias on first visual frame (avoid absolute time leakage)
        self.first_pos = nn.Parameter(torch.zeros(1, d_model))

        # Shared fusion backbone (per timestep → d_model)
        fused_in = d_model * (num_cameras + 3)
        self.fusion_backbone = nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, d_model),
            nn.ReLU(),
        )

        # Scheme-specific heads (both regress in [0,1], but separate params help)
        self.heads = nn.ModuleDict({
            "sparse": nn.Linear(d_model, 1),
            "dense":  nn.Linear(d_model, 1),
        })

    def _prep_lang(self, lang_emb: torch.Tensor, B: int, T: int, D: int) -> torch.Tensor:
        """
        Accepts lang_emb of shape:
          - (B, text_emb_dim): one prompt per clip → broadcast to all T
          - (B, T, text_emb_dim): per-timestep text (dense)
        Returns (B, 1, T, D)
        """
        if lang_emb.dim() == 3:
            # (B, T, E) -> (B, T, D) -> (B, 1, T, D)
            return self.lang_proj(lang_emb).unsqueeze(1)
        else:
            # (B, E) -> (B, 1, 1, D) -> (B, 1, T, D)
            return self.lang_proj(lang_emb).unsqueeze(1).unsqueeze(2).expand(B, 1, T, D)

    def _stage_to_dmodel(self, stage_prior: torch.Tensor) -> torch.Tensor:
        """
        Deterministic projection of one-hot to d_model by pad/truncate.
        stage_prior: (B,1,T,C) -> (B,1,T,d_model)
        """
        B, one, T, C = stage_prior.shape
        D = self.d_model
        if D == C:
            return stage_prior
        elif D > C:
            pad = torch.zeros(B, one, T, D - C, device=stage_prior.device, dtype=stage_prior.dtype)
            return torch.cat([stage_prior, pad], dim=-1)
        else:
            return stage_prior[..., :D]
        
    def forward(self,
                img_seq: torch.Tensor,      # (B, N, T, vis_emb_dim)
                lang_emb: torch.Tensor,     # (B, E) or (B, T, E)
                state: torch.Tensor,        # (B, T, state_dim)
                lengths: torch.Tensor,      # (B,)
                stage_prior: torch.Tensor,  # (B,1,T,C) one-hot (from gen_stage_emb)
                scheme: str = "sparse"      # "sparse" or "dense"
                ) -> torch.Tensor:
        assert scheme in self.heads, f"Unknown scheme '{scheme}'. Use one of {list(self.heads.keys())}."
        B, N, T, _ = img_seq.shape
        D = self.d_model
        device = img_seq.device

        # Project inputs
        vis_proj   = self.visual_proj(img_seq)               # (B, N, T, D)
        state_proj = self.state_proj(state).unsqueeze(1)     # (B, 1, T, D)
        lang_proj  = self._prep_lang(lang_emb, B, T, D)      # (B, 1, T, D)
        stage_emb  = self._stage_to_dmodel(stage_prior)      # (B,1,T,D)

        # Concatenate and add positional encoding
        x = torch.cat([vis_proj, lang_proj, state_proj, stage_emb], dim=1)  # (B,N+3,T,D)
        x[:, :N, 0, :] += self.first_pos
        x_tokens = x.view(B, (N + 3) * T, D)
        L = x_tokens.size(1)

        # Create padding mask
        base_mask = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)
        mask = base_mask.unsqueeze(1).expand(B, N + 3, T).reshape(B, (N + 3) * T)
        causal_mask = torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool),
            diagonal=1
        )  # (L, L)
        
        # Transformer encoder with padding mask
        h = self.transformer(x_tokens, 
                             mask=causal_mask,
                             src_key_padding_mask=mask,
                             is_causal=True)                            # (B, (N+3)*T, D)         
        h = h.view(B, N + 3, T, D)
        h_flat = h.view(B, T, (N + 3) * D)
        fused = self.fusion_backbone(h_flat)                            # (B,T,D)

        # Scheme-specific regression head
        r = torch.sigmoid(self.heads[scheme](fused)).squeeze(-1)        # (B, T)
        return r
