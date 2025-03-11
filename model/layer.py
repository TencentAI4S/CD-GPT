# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/17 11:20
import logging
import math
import numbers
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from apex.normalization.fused_layer_norm import mixed_dtype_fused_rms_norm_affine
    logging.warning("using apex fused rms norm")
except ImportError:
    mixed_dtype_fused_rms_norm_affine = None

try:
    from xformers.ops import swiglu, unbind
    logging.warning(f"using xformer swiglu")
except:
    swiglu = None

KVCache = Tuple[torch.Tensor, torch.Tensor]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization

    Ref：
        1.Root Mean Square Layer Normalization
    """

    def __init__(self,
                 normalized_shape: int,
                 eps: float = 1e-6,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.eps = eps
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(*normalized_shape, **factory_kwargs))

    def _norm(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        return hidden_states * torch.rsqrt(variance + self.eps)

    def forward(self, input):
        if mixed_dtype_fused_rms_norm_affine is None or torch.jit.is_tracing() or torch.jit.is_scripting() or not input.is_cuda:
            input = self._norm(input).type_as(input)
            return self.weight * input
        else:
            return mixed_dtype_fused_rms_norm_affine(input, self.weight, self.normalized_shape, self.eps)

    # def extra_repr(self):
    #     return "{normalized_shape}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def precompute_freqs_cis(seq_len: int,
                         n_elem: int,
                         theta: int = 10000,
                         dtype: torch.dtype = torch.float32,
                         device: torch.device = torch.device("cpu"),
                         compile=True) -> torch.Tensor:
    """build rope cache"""
    freqs = 1.0 / (theta ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))
    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)
    freqs = torch.outer(seq_idx, freqs).float()
    low_precison_dtypes = (torch.float16, torch.bfloat16, torch.int8)
    # TODO: if pytorch2.0 compile suport complex64, delete it
    if compile:
        freqs_cis = torch.stack([torch.cos(freqs),
                                 torch.sin(freqs)], dim=-1)
        if dtype in low_precison_dtypes:
            freqs_cis = freqs_cis.half()
    else:
        scalar_dtype = torch.float32 if dtype in low_precison_dtypes else dtype
        complex_dtype = (
            torch.complex32 if dtype in low_precison_dtypes else torch.complex64
        )
        freqs_cis = torch.polar(torch.ones_like(freqs).to(scalar_dtype),
                                freqs.to(scalar_dtype)).to(complex_dtype)

    return freqs_cis


def apply_rotary_emb(
        x: torch.Tensor,
        freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: [bs, seq_len, hidden_dim], xq or xk
        rope_cache: [bs, seq_len, 2]
    """
    T = x.shape[-3]
    freqs_cis = freqs_cis[:T]
    # TODO: wait pytorch2.0 support torch.complex32
    if freqs_cis.dtype in (torch.complex32, torch.complex64):
        # cast because `view_as_complex` does not support 16 bit tensors
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.view(1, xc.size(1), 1, xc.size(3))
        x_out = torch.view_as_real(xc * freqs_cis).flatten(3)

        return x_out.type_as(x)
    else:
        xc = x.reshape(*x.shape[:-1], -1, 2).float()  # [*, seq_len, num_heads, head_dim // 2, 2]
        freqs_cis = freqs_cis.view(xc.size(-4), 1, xc.size(-2), 2)  # [seq_len, 1, head_dim // 2, 2]
        out = torch.stack([
            xc[..., 0] * freqs_cis[..., 0] - xc[..., 1] * freqs_cis[..., 1],
            xc[..., 1] * freqs_cis[..., 0] + xc[..., 0] * freqs_cis[..., 1]
        ], dim=-1).flatten(start_dim=-2)
        out = out.type_as(x)
        return out


class SwiGLU(nn.Module):
    """swiglu feed forward network

    Args:
        dim: input embedding dim
        hidden_dim: inner hidden dim, also named ffn_dim in other project
        multiple_of: emsure hidden dim are divided
        ffn_dim_multiplier: config param in llama2, default none for compact with llama
        bias: linear layer bias
        _pack_weights: pack fc linear and than split, set true for faster training

    Note that MLP is also called swiglu operator in some papers, you call speed up by installing xformers
    """

    def __init__(
            self,
            dim: int,
            hidden_dim: int = None,
            multiple_of: int = 256,
            ffn_dim_multiplier: Optional[float] = None,
            _pack_weights=True,
            bias=False
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.pack_fc = _pack_weights
        if self.pack_fc:
            self.c_fc = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        else:
            self.c_fc1 = nn.Linear(dim, hidden_dim, bias=bias)
            self.c_fc2 = nn.Linear(dim, hidden_dim, bias=bias)

        self.c_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def native_impl(self, x):
        if self.pack_fc:
            x1, x2 = torch.chunk(self.c_fc(x), 2, dim=2)
            x = F.silu(x1) * x2
        else:
            x = F.silu(self.c_fc1(x)) * self.c_fc2(x)

        x = self.c_proj(x)
        return x

    def swiglu_impl(self, x):
        if self.pack_fc:
            fcw = self.c_fc.weight
            fc1w, fc2w = unbind(
                fcw.view([2, fcw.shape[0] // 2, fcw.shape[1]]),
                dim=0,
            )
            fcb = self.c_fc.bias
            if fcb is not None:
                fc1b, fc2b = unbind(fcb.view([2, fcb.shape[0] // 2]), dim=0)
            else:
                fc1b, fc2b = None, None
            x = swiglu(x,
                       fc1w, fc1b,
                       fc2w, fc2b,
                       self.c_proj.weight, self.c_proj.bias)
        else:
            x = swiglu(x,
                       self.c_fc1.weight, self.c_fc1.bias,
                       self.c_fc2.weight, self.c_fc2.bias,
                       self.c_proj.weight, self.c_proj.bias)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if swiglu is not None:
            x = self.swiglu_impl(x)
        else:
            x = self.native_impl(x)
        return x


class CasualSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, max_len, bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_len = max_len
        # self.rot_emb = RotaryEmbedding(dim=self.head_dim)
        if (self.head_dim * num_heads) != self.dim:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.c_attn = nn.Linear(self.dim, 3 * self.dim, bias=bias)
        self.c_proj = nn.Linear(self.dim, self.dim, bias=bias)

    def _project_qkv(self,
                     x: torch.Tensor,
                     rope: torch.Tensor
                     ):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.dim, dim=2)
        head_dim = C // self.num_heads
        k = k.view(B, T, self.num_heads, head_dim)  # (B,T, nh, hs)
        q = q.view(B, T, self.num_heads, head_dim)
        v = v.view(B, T, self.num_heads, head_dim)

        k = apply_rotary_emb(k, rope).transpose(1, 2)  # [B, nh, T, hs]
        q = apply_rotary_emb(q, rope).transpose(1, 2)

        v = v.transpose(1, 2)

        # [bs, num_head, seq_len, head_dim]
        return q, k, v

    def _scaled_dot_product_attention(self, q, k, v, attn_mask=None, dropout_p=0.0, need_attn=False):
        if hasattr(F, "scaled_dot_product_attention") and not need_attn:
            scores = None
            is_causal = attn_mask is None
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        else:
            scores = (q @ k.transpose(2, 3)) / math.sqrt(self.head_dim)
            T = q.size(2)
            # causal mask to ensure that attention is only applied to the left in the input sequence
            if attn_mask is None:
                attn_mask = torch.tril(torch.ones(T, T, dtype=scores.dtype, device=scores.device)).view(1, 1, T, T)
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            y = scores @ v

        return y, scores

    def forward(self,
                x: torch.Tensor,
                rope: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                pos_ids: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None,
                need_attn: bool = False,
                ):
        q, k, v = self._project_qkv(x, rope)
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            # check if reached token limit
            if pos_ids[-1] >= self.max_len:
                pos_ids = torch.tensor(self.max_len - 1, device=pos_ids.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, shifts=-1, dims=2)
                cache_v = torch.roll(cache_v, shifts=-1, dims=2)
            k = cache_k.index_copy(2, pos_ids, k)
            v = cache_v.index_copy(2, pos_ids, v)
            kv_cache = k, v
        y, scores = self._scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, need_attn=need_attn)
        y = y.transpose(1, 2).contiguous().view(*x.shape)
        y = self.c_proj(y)

        return y, kv_cache, scores


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 max_len,
                 multiple_of=256,
                 bias=False,
                 eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.attn = CasualSelfAttention(dim, num_heads, max_len, bias=bias)
        self.mlp = SwiGLU(dim,
                          hidden_dim=4 * dim,
                          multiple_of=multiple_of,
                          _pack_weights=False)
        self.rms_1 = RMSNorm(dim, eps=eps)
        self.rms_2 = RMSNorm(dim, eps=eps)

    def forward(self,
                x: torch.Tensor,
                rope: torch.Tensor,
                pos_ids: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None,
                need_attn=False):
        residue = x
        hidden, kv_cache, scores = self.attn(self.rms_1(x), rope,
                                             attn_mask=attn_mask,
                                             pos_ids=pos_ids,
                                             kv_cache=kv_cache,
                                             need_attn=need_attn)
        x = residue + hidden
        x = x + self.mlp(self.rms_2(x))

        return x, kv_cache, scores
