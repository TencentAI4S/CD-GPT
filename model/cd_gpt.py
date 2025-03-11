# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/17 0:18
import math
from typing import Optional, Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from config.utils import configurable
from tokenizer import SentencePieceTokenizer
from .generation import GenerationOutput, sample
from .layer import RMSNorm, precompute_freqs_cis, Block
from .output_head import SequencePredictionHead, ResiduePairPredictionHead, TokenPredictionHead


class CDGPT(nn.Module):
    CONFIG = {
        "cdgpt-1b": dict(num_layers=12, num_heads=24, embedding_dim=2304),
        "cdgpt-7b": dict(num_layers=32, num_heads=32, embedding_dim=4096)
    }

    @classmethod
    def from_config(cls, cfg):
        model_type = cfg.model.type
        if model_type:
            mcfg = cls.CONFIG[model_type]
            num_layers, num_heads, embedding_dim = mcfg['num_layers'], mcfg['num_heads'], mcfg['embedding_dim']
        else:
            num_layers = cfg.model.num_layers
            num_heads = cfg.model.num_heads
            embedding_dim = cfg.model.num_hiddens
        pad_id = SentencePieceTokenizer(cfg).pad_id
        return {
            "vocab_size": cfg.tokenizer.vocab_size,
            "max_len": cfg.model.max_len,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "pad_id": pad_id
        }

    @configurable
    def __init__(self,
                 vocab_size: int,
                 max_len: int = 1024,
                 embedding_dim=2304,
                 num_layers: int = 12,
                 num_heads: int = 24,
                 bias=False,
                 eps=1e-5,
                 pad_id=None,
                 include_head=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(self.vocab_size, self.embedding_dim),
                h=nn.ModuleList([
                    Block(self.embedding_dim, self.num_heads, self.max_len, eps=self.eps) for _ in
                    range(self.num_layers)
                ]),
                ln_f=RMSNorm(self.embedding_dim, eps=self.eps),
            )
        )
        self.Block = Block
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=bias) if include_head else None
        self.rope_cache = None
        self.kv_caches = []
        self.pad_id = pad_id
        self.apply(self._init_weights)
        self.activation_checkpoint = False
        self.activation_checkpoint_func = checkpoint
        n_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def enable_activation_checkpoint(self, enabled=True):
        self.activation_checkpoint = enabled

    def finetune_vocab(self):
        h = self.transformer.h
        for moudle in h:
            moudle.requires_grad_(False)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))

    def _make_casual_mask(self, device):
        """
        Args:
            input_ids: [bs, seq_len]
        """
        ones = torch.ones((self.max_len, self.max_len), dtype=torch.bool, device=device)
        return torch.tril(ones)[None, None]

    def _make_rope_mask(self, device, dtype=torch.int64):
        return precompute_freqs_cis(
            seq_len=self.max_len,
            n_elem=self.embedding_dim // self.num_heads,
            dtype=dtype,
            device=device
        )

    def _forward_embedding_impl(self, input_ids):
        x = self.transformer.wte(input_ids)  # [bs, seq_len, hidden_dim]
        return x

    def _forward_head_impl(self, x):
        if self.lm_head is not None:
            x = self.lm_head(x)  # (b, t, vocab_size)
        return x

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                pos_ids: Optional[torch.Tensor] = None):
        """
        Args:
            input_ids: [bs, seq_len], input token indics
            attention_mask: [bs, 1, seq_len, seq_len], attention mask, when it's none,
                default casual mask
            pos_ids: [seq_len or 1], use it when inference generating new token id or
                keep it none when training.
        """
        bs, seq_len = input_ids.shape
        device = input_ids.device
        dtype = input_ids.dtype
        assert (
                seq_len <= self.max_len
        ), f"Cannot forward sequence of length {seq_len}, max length is only {self.max_len}"
        if self.rope_cache is None:
            self.rope_cache = self._make_rope_mask(device, dtype)  # [max_len, ...]

        if pos_ids is not None:
            rope = self.rope_cache.index_select(0, pos_ids)
            if attention_mask is None:
                attention_mask = self._make_casual_mask(device)
            attention_mask = attention_mask.index_select(2, pos_ids)
            attention_mask = attention_mask[:, :, :, :self.max_len]
        else:
            rope = self.rope_cache[:seq_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :, :seq_len, :seq_len]

        x = self._forward_embedding_impl(input_ids)
        if pos_ids is None:
            for block in self.transformer.h:
                if self.activation_checkpoint:
                    x, _, _ = self.activation_checkpoint_func(block, x, rope, attention_mask)
                else:
                    x, _, _ = block(x, rope, attn_mask=attention_mask)
        else:
            if not self.kv_caches:
                head_dim = self.embedding_dim // self.num_heads
                cache_shape = (bs, self.num_heads, self.max_len, head_dim)
                # prelocate memory
                self.kv_caches = [
                    (torch.zeros(cache_shape, device=x.device, dtype=x.dtype),
                     torch.zeros(cache_shape, device=x.device, dtype=x.dtype))
                    for _ in range(self.num_layers)
                ]
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i], _ = block(x, rope,
                                                attn_mask=attention_mask,
                                                pos_ids=pos_ids,
                                                kv_cache=self.kv_caches[i])
        x = self.transformer.ln_f(x)
        x = self._forward_head_impl(x)
        return x

    def get_embedding_pooling(self, input_ids):
        x = self._forward_embedding_impl(input_ids)
        x = x.mean(dim=0)
        return x

    def reset_cache(self):
        self.kv_caches.clear()

    @torch.no_grad()
    def generate(self,
                 token_ids,
                 max_new_tokens,
                 *,
                 top_k: int = 0,
                 top_p: float = 0.,
                 temperature: float = 1.0,
                 output_score: bool = True,
                 stop_ids: Any = None):
        if token_ids.dim() == 2 or isinstance(token_ids, list):
            return [self.generate(t,
                                  max_new_tokens,
                                  top_k=top_k,
                                  top_p=top_p,
                                  temperature=temperature,
                                  output_score=output_score,
                                  stop_ids=stop_ids) for t in token_ids]
        seq_len = token_ids.size(0)
        assert seq_len < self.max_len, f"input token is too long"
        device, dtype = token_ids.device, token_ids.dtype
        max_len = min(self.max_len, seq_len + max_new_tokens)
        # create an empty tensor of the expected final shape and fill in the current tokens
        empty = torch.empty(max_len, dtype=dtype, device=device)
        empty[:seq_len] = token_ids
        token_ids = empty
        scores = [] if output_score else None
        input_pos = torch.arange(0, seq_len, device=device)
        for cur_pos in range(seq_len, max_len):
            x = token_ids.index_select(0, input_pos)[None]
            logits = self(x, pos_ids=input_pos)[:, -1]
            idx_next = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)[0]
            input_pos = input_pos[-1:] + 1
            # concatenate the new generation
            token_ids = token_ids.index_copy(0, input_pos, idx_next)

            if output_score:
                scores.append(logits.softmax(dim=-1)[0, idx_next])

            if stop_ids is not None and idx_next.item() in stop_ids:
                break

        self.reset_cache()
        return GenerationOutput(sequences=token_ids[:input_pos + 1], scores=scores)


class CDGPTSequencePrediction(CDGPT):

    @classmethod
    def from_config(cls, cfg):
        pad_id = cfg.tokenizer.pad_id
        num_classes = cfg.model.num_classes
        return {
            "num_classes": num_classes,
            "pad_id": pad_id,
            **super().from_config(cfg)
        }

    @configurable
    def __init__(self,
                 num_classes: int,
                 vocab_size: int,
                 max_len: int = 2048,
                 embedding_dim=2304,
                 num_layers: int = 12,
                 num_heads: int = 24,
                 bias=False,
                 eps=1e-5,
                 pad_id=2,
                 dropout=0.0):
        super().__init__(vocab_size, max_len, embedding_dim, num_layers, num_heads, bias, eps, include_head=False)
        self.num_classes = num_classes
        self.pad_id = pad_id
        self.dropout = dropout
        self.cls_head = SequencePredictionHead(self.embedding_dim, self.num_classes, self.dropout)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                pos_ids: Optional[torch.Tensor] = None):
        hiddens = super().forward(input_ids, attention_mask, pos_ids)
        result = {}
        if self.pad_id is None:
            sequence_lengths = -1  # last token for classification or regression
        else:
            sequence_lengths = torch.ne(input_ids, self.pad_id).sum(-1) - 1
        batch_size = hiddens.shape[0]
        hiddens = hiddens[torch.arange(batch_size, device=hiddens.device), sequence_lengths]
        res = self.cls_head(hiddens)
        result["output"] = res
        return result


class CDGPTTokenPrediction(CDGPT):

    @classmethod
    def from_config(cls, cfg):
        pad_id = cfg.tokenizer.pad_id
        num_classes = cfg.model.num_classes
        return {
            "num_classes": num_classes,
            "pad_id": pad_id,
            **super().from_config(cfg)
        }

    @configurable
    def __init__(self,
                 num_classes,
                 vocab_size: int,
                 max_len: int = 2048,
                 embedding_dim=2304,
                 num_layers: int = 12,
                 num_heads: int = 24,
                 bias=False,
                 eps=1e-5,
                 pad_id=2,
                 dropout=0.0):
        super().__init__(vocab_size=vocab_size,
                         max_len=max_len,
                         embedding_dim=embedding_dim,
                         num_layers=num_layers,
                         num_heads=num_heads,
                         bias=bias,
                         eps=eps,
                         include_head=True)
        self.num_classes = num_classes
        self.pad_id = pad_id
        self.cls_head = TokenPredictionHead(self.embedding_dim, self.num_classes, dropout, num_heads, max_len, eps)

    def forward(self, token_ids, pos_ids=None, attention_mask=None):
        bs, seq_len = token_ids.shape
        device = token_ids.device
        dtype = token_ids.dtype
        assert (
                seq_len <= self.max_len
        ), f"Cannot forward sequence of length {seq_len}, max length is only {self.max_len}"

        if self.rope_cache is None:
            self.rope_cache = self._make_rope_mask(device, dtype)  # [max_len, ...]

        rope = self.rope_cache[:seq_len]
        if attention_mask is not None:
            attention_mask = self.attention_mask[:, :, :seq_len, :seq_len]

        x = self._forward_embedding_impl(token_ids)

        for block in self.transformer.h:
            if self.activation_checkpoint:
                x, _, _ = self.activation_checkpoint_func(block, x, rope, attention_mask, None, None, True)
            else:
                x, _, _ = block(x, rope, attn_mask=attention_mask, need_attn=True)

        x = self.transformer.ln_f(x)
        result = {}
        result["output"] = self.cls_head(x)
        return result


class CDGPTResiduePairPrediction(CDGPT):

    @classmethod
    def from_config(cls, cfg):
        pad_id = cfg.tokenizer.pad_id
        num_classes = cfg.model.num_classes
        return {
            "num_classes": num_classes,
            "pad_id": pad_id,
            **super().from_config(cfg)
        }

    @configurable
    def __init__(self,
                 num_classes,
                 vocab_size: int,
                 max_len: int = 2048,
                 embedding_dim=2304,
                 num_layers: int = 12,
                 num_heads: int = 24,
                 bias=True,
                 eps=1e-5,
                 pad_id=2,
                 ):
        super().__init__(vocab_size=vocab_size,
                         max_len=max_len,
                         embedding_dim=embedding_dim,
                         num_layers=num_layers,
                         num_heads=num_heads,
                         bias=bias,
                         eps=eps,
                         include_head=True,
                         pad_id=pad_id)
        self.num_classes = num_classes
        self.contact_head = ResiduePairPredictionHead(num_heads * num_layers, self.num_classes, bias)

    def forward(self, token_ids, pos_ids=None, attention_mask=None):
        bs, seq_len = token_ids.shape
        device = token_ids.device
        dtype = token_ids.dtype
        assert (
                seq_len <= self.max_len
        ), f"Cannot forward sequence of length {seq_len}, max length is only {self.max_len}"

        if self.rope_cache is None:
            self.rope_cache = self._make_rope_mask(device, dtype)  # [max_len, ...]

        rope = self.rope_cache[:seq_len]
        if attention_mask is not None:
            attention_mask = self.attention_mask[:, :, :seq_len, :seq_len]

        x = self._forward_embedding_impl(token_ids)
        attn_weights = []
        for block in self.transformer.h:
            if self.activation_checkpoint:
                x, _, attn = self.activation_checkpoint_func(block, x, rope, attention_mask, None, None, True)
            else:
                x, _, attn = block(x, rope, attn_mask=attention_mask, need_attn=True)
            attn_weights.append(attn)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        result = {}
        # stack attentions 
        attentions = torch.stack(attn_weights, 1)
        contact = self.contact_head(attentions)
        result["output"] = contact
        result["logits"] = logits

        return result
