"""BBox modulation adapter: gate + per-layer shift/scale (zero-init outputs)."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def zero_init_linear(m: nn.Linear) -> None:
    nn.init.zeros_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class BBoxModulationAdapter(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        adapter_dim: int,
        inner_dim: int,
        num_injection_layers: int,
        mlp_gate_final_bias: float = -1.0,
        use_bbox_index_embedding: bool = False,
        max_boxes: int = 32,
    ):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_injection_layers = num_injection_layers
        self.feat_proj = nn.Linear(in_channels, adapter_dim)

        hidden = adapter_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(adapter_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.gate_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, mlp_gate_final_bias)

        self.use_bbox_index_embedding = use_bbox_index_embedding
        if use_bbox_index_embedding:
            self.box_emb = nn.Embedding(max_boxes, adapter_dim)

        self.shift_heads = nn.ModuleList()
        self.scale_heads = nn.ModuleList()
        for _ in range(num_injection_layers):
            ls = nn.Linear(adapter_dim, inner_dim)
            sc = nn.Linear(adapter_dim, inner_dim)
            zero_init_linear(ls)
            zero_init_linear(sc)
            self.shift_heads.append(ls)
            self.scale_heads.append(sc)

    def forward(
        self,
        pooled_latent_feats: torch.Tensor,
        box_valid: torch.Tensor,
        box_indices: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        pooled_latent_feats: (B, K, C)
        box_valid: (B, K) bool
        Returns:
          gate: (B, K)
          layers: list length L of (delta_shift, delta_scale) each (B, K, inner_dim) — raw, before gate
        """
        B, K, C = pooled_latent_feats.shape
        x = self.feat_proj(pooled_latent_feats)
        if self.use_bbox_index_embedding and box_indices is not None:
            x = x + self.box_emb(box_indices.clamp(0, self.box_emb.num_embeddings - 1))

        h = self.gate_mlp(x)
        gate = torch.sigmoid(self.gate_head(h).squeeze(-1))
        gate = gate * box_valid.float()

        layers_out: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for li in range(self.num_injection_layers):
            ds = self.shift_heads[li](x)
            dsc = self.scale_heads[li](x)
            g = gate.unsqueeze(-1)
            ds = ds * g
            dsc = dsc * g
            layers_out.append((ds, dsc))

        return gate, layers_out
