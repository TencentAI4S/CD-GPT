# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/19 20:59
import torch
import torch.nn as nn

from .layer import precompute_freqs_cis, Block


def apc(x):
    """Perform average product correct, used for contact prediction.
    Args:
        x: [*, seq_len, seq_len]
    """
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)
    avg = a1 * a2
    avg.div_(a12 + 1e-6)  # in-place to reduce memory
    normalized = x - avg
    return normalized


def symmetrize(x):
    """Make layer symmetric in final two dimensions, used for contact prediction."""
    return x + x.transpose(-1, -2)


class ResiduePairPredictionHead(nn.Module):
    """Head for residule pair level classification tasks."""
    
    def __init__(
            self,
            dim: int,
            num_classes: int = 2,
            bias=True
    ):
        super().__init__()
        self.dim = dim
        # self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fusion = nn.Linear(dim, 64, bias=bias)
        self.linear = nn.Linear(64, num_classes, bias=bias)
        self.activation = nn.Softmax(dim=-1)
        self.eps = 1e-6

    def forward(self, attentions, causal=True, seq_mask=None):
        """
        Args:
            attentions: [bs, num_layers, num_heads, seq_len, seq_len]
            seq_mask: [bs, seq_len]
        """
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)
        attentions = self.relu(self.fusion(attentions.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        attentions = apc(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1)
        output = self.activation(self.linear(attentions).squeeze(3))  # [bs, seq_len, seq_len, 2]
        return output


class TokenPredictionHead(nn.Module):
    """Head for token level classification tasks."""

    def __init__(self, dim, num_classes=2, dropout=0.0, num_heads=None, max_len=None, eps=None):
        super().__init__()
        self.num_layers = 1
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_proj = nn.Linear(dim, num_classes)
        self.embedding_dim = dim
        self.num_heads = num_heads
        self.max_len = max_len
        self.eps = eps
        self.attn_mask = None
        self.transformer = nn.ModuleList([
            Block(self.embedding_dim, self.num_heads, self.max_len, eps=self.eps) for _ in
            range(self.num_layers)
        ])

    def _make_rope_mask(self, device, dtype=torch.int64):
        return precompute_freqs_cis(
            seq_len=self.max_len,
            n_elem=self.embedding_dim // self.num_heads,
            dtype=dtype,
            device=device
        )

    def forward(self, features):
        # [B, T, C]
        x = features
        self.rope_cache = self._make_rope_mask(x.device, x.dtype)
        rope = self.rope_cache[:x.shape[1]]
        for layer in self.transformer:
            x, _, _ = layer(x, rope, attn_mask=self.attn_mask, need_attn=True)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class SequencePredictionHead(nn.Module):
    """Head for sequence level classification tasks."""

    def __init__(self, dim, num_classes=2, dropout=0.0):
        super().__init__()
        self.num_layers = 1
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.out_proj = nn.Linear(dim, num_classes)

    def forward(self, features):
        # [B, T, C]
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
