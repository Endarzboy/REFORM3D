# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


# ============================================================================
# Attention Backend: PyTorch SDPA (FlashAttention v2) with Vanilla fallback
# ============================================================================
# PyTorch 2.0+ has native scaled_dot_product_attention (SDPA) which automatically
# selects the optimal backend: FlashAttention, Memory-Efficient, or Math.

SDPA_AVAILABLE = hasattr(F, "scaled_dot_product_attention")  # PyTorch 2.0+

if SDPA_AVAILABLE:
    logger.info("Attention backend: PyTorch SDPA (FlashAttention v2)")
else:
    logger.warning("Attention backend: Vanilla (PyTorch < 2.0, may be slow)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, return_attn=False) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn_drop = self.attn_drop(attn)

        x = (attn_drop @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            return x, attn

        return x


class MemEffAttention(Attention):
    """Memory-efficient attention using PyTorch SDPA (FlashAttention v2).
    
    Backend selection (automatic):
    1. PyTorch SDPA with FlashAttention (PyTorch 2.0+, recommended)
    2. Vanilla attention (fallback for older PyTorch)
    """
    
    def forward(self, x: Tensor, return_attn=False) -> Tensor:
        # Return attention weights requires vanilla implementation
        if return_attn:
            return super().forward(x, return_attn=True)

        B, N, C = x.shape
        
        # ====== Backend 1: PyTorch SDPA (FlashAttention v2, PyTorch 2.0+) ======
        if SDPA_AVAILABLE:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # SDPA automatically selects FlashAttention, Memory-Efficient, or Math backend
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
            )
            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        
        # ====== Backend 2: Vanilla attention (fallback) ======
        return super().forward(x)
