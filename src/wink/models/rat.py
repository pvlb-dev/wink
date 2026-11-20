"""RAT denoiser: tardits' bidirectional fast-weight block on image patches.

The stock tardits ``Rattention`` rejects a bidirectional window longer than its
numerically safe scan chunk. That chunk is derived from the trained ``|gamma|``,
so a 64-token CIFAR run can start with a 64-token window and later fail when
gamma decays. This module keeps the same mathematics but splits a long
bidirectional window into chunks and carries the fast-weight state across them.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tardits.models.base import ModelConfig as TarditsModelConfig
from tardits.models.rat import RATBlock, Rattention
from tardits.modules.core import cd_conjugate, cd_mul

from wink.models.base import ImageModelConfig, ImageTransformerBase


class ChunkedBidirectionalRattention(Rattention):
    """Bidirectional RAT attention with a chunked, state-carrying scan.

    For ``T <= chunk`` this delegates to tardits. Above the chunk it computes the
    forward and backward decayed sums in chunks, carrying ``S_last`` between
    chunks, then applies the usual ``S_fwd + S_bwd - A_self`` combination. This
    keeps the float32 exponent bounded while preserving the exact kernel.
    """

    def _chunked_scan(
        self, kv_outer: torch.Tensor, chunk: int
    ) -> torch.Tensor:
        outs = []
        state = None
        total = kv_outer.shape[2]
        for start in range(0, total, chunk):
            local = kv_outer[:, :, start : start + chunk]
            S = self._causal_scan(
                local, force_scan=True, initial_state=state
            )
            state = S[:, :, -1].clone()
            outs.append(S)
        return torch.cat(outs, dim=2)

    def forward(
        self,
        x: torch.Tensor,
        return_state: bool = False,
        initial_state: torch.Tensor | None = None,
        force_scan: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, C, hyper_dim = x.shape
        chunk = self._scan_chunk_size()
        if not self.config.bidirectional:
            return super().forward(
                x,
                return_state=return_state,
                initial_state=initial_state,
                force_scan=force_scan,
            )
        if initial_state is not None or return_state:
            raise ValueError("bidirectional RAT does not expose a recurrent state")
        if T <= chunk:
            return super().forward(x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_head, self.head_size, hyper_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size, hyper_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size, hyper_dim).transpose(1, 2)

        k_conj = cd_conjugate(k, level=self.level)
        v_exp = v.unsqueeze(-2)
        k_exp = k_conj.unsqueeze(-3)
        kv_outer = cd_mul(v_exp, k_exp, level=self.level)

        S_fwd = self._chunked_scan(kv_outer, chunk)
        S_bwd = self._chunked_scan(kv_outer.flip(2), chunk).flip(2)
        S = S_fwd + S_bwd - kv_outer

        q_exp = q.unsqueeze(-3)
        out_raw = (
            cd_mul(S, q_exp, level=self.level).sum(dim=-2) * self.scale
        )
        out = out_raw.transpose(1, 2).contiguous().view(B, T, C, hyper_dim)
        return self.dropout(self.out_proj(out))


class RATDiT(ImageTransformerBase):
    """Complex diffusion transformer with bidirectional linear fast-weight memory.

    The block is tardits' ``RATBlock`` with ``ChunkedBidirectionalRattention`` in
    place of the stock attention, so 64-token CIFAR windows survive gamma decay.
    """

    arch = "rat"

    def _build_blocks(self, config: ImageModelConfig) -> nn.ModuleList:
        block_config = TarditsModelConfig(
            level=config.level,
            block_size=self.num_patches,
            vocab_size=1,
            n_head=config.n_head,
            n_embd=config.n_embd,
            n_layer=1,
            dropout=config.dropout,
            bidirectional=True,
        )
        blocks = []
        for _ in range(config.n_layer):
            block = RATBlock(block_config)
            block.sa = ChunkedBidirectionalRattention(block_config)
            blocks.append(block)
        return nn.ModuleList(blocks)
