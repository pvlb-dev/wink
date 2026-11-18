"""RAT denoiser: tardits' bidirectional fast-weight block on image patches."""

from __future__ import annotations

import torch.nn as nn

from tardits.models.base import ModelConfig as TarditsModelConfig
from tardits.models.rat import RATBlock

from wink.models.base import ImageModelConfig, ImageTransformerBase


class RATDiT(ImageTransformerBase):
    """Complex diffusion transformer with bidirectional linear fast-weight memory.

    The block is tardits' ``RATBlock``; its ``Rattention`` is configured with
    ``bidirectional=True`` so every patch sees every other patch, decayed by the
    learned complex rotor over distance. The block's ``block_size`` is the number
    of image patches, which keeps the window inside the supported scan chunk.
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
        return nn.ModuleList(
            [RATBlock(block_config) for _ in range(config.n_layer)]
        )
