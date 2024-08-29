# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/20 17:23
from fvcore.common.config import CfgNode as CN

_C = CN()
# -------------------------------tokenizer----------------------#
_C.tokenizer = CN()
_C.tokenizer.name = "sentencepiece"
_C.tokenizer.path = ""
_C.tokenizer.vocab_size = 64000
_C.tokenizer.pad_id = 2
# -------------------------------model-------------------------#
_C.model = CN()
_C.model.type = "cdgpt-1b"
_C.model.num_layers = 12
_C.model.num_kv_heads = None
_C.model.num_heads = 24
_C.model.num_hiddens = 2304
_C.model.dropout = 0.
_C.model.bias = False
_C.model.num_classes = 1
_C.model.from_pretrained = False
_C.model.weights = ""
_C.model.eps = 1e-5
_C.model.packed_swiglu = False
_C.model.activation_checkpoint = CN({"enabled": False})
_C.model.use_pretrain = True
_C.model.attn_cls = False
_C.model.return_contacts = False
_C.model.need_attn = False
_C.model.max_len = 1024

def get_config(clone=False):
    if clone:
        return _C.clone()

    return _C
