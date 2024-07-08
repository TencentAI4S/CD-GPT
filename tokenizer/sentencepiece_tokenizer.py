# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/20 16:53
import logging
import os
from typing import Optional, Union

import numpy as np
import sentencepiece as sp
import torch

from config.utils import configurable
from .base_tokenizer import AbstractTokenizer


class SentencePieceTokenizer(AbstractTokenizer):
    """Tokenizer for sentencepiece

    Args:
        model_path: path to sentencepiece model
    """

    @classmethod
    def from_config(cls, cfg):
        return {
            "model_path": cfg.tokenizer.path
        }

    @configurable
    def __init__(self,
                 model_path: str) -> None:
        super().__init__(os.path.basename(model_path).split(".")[0])
        self.model_path = model_path
        self.spm = sp.SentencePieceProcessor(model_file=model_path)
        self.unk_id = self.spm.unk_id()
        self.eos_id = self.spm.eos_id()
        self.bos_id = self.spm.bos_id()
        if self.bos_id < 0:
            neg_bos_id = self.bos_id
            self.bos_id = self.piece_to_id("<s>")
            logging.warning(f"change bos token id from {neg_bos_id} to {self.bos_id}")
            assert self.bos_id > 0

        self.pad_id = self.spm.pad_id()
        if self.pad_id < 0:
            self.pad_id = self.eos_id

        self._vocab = {id: self.spm.id_to_piece(id) for id in range(self.spm.get_piece_size())}

    @property
    def vocab(self):
        return self._vocab

    def id_to_piece(self, token_id):
        return self.spm.IdToPiece(token_id)

    def piece_to_id(self, piece):
        return self.spm.PieceToId(piece)

    @property
    def vocab_size(self) -> int:
        return self.spm.vocab_size()

    @property
    def unk(self):
        return self.unk_id

    @property
    def pad(self):
        return self.pad_id

    @property
    def eod(self):
        # might conflict with <eos>
        return self.eos_id

    @property
    def bos(self):
        # might conflict with <eos>
        return self.bos_id

    @property
    def eos(self):
        # might conflict with <eos>
        return self.eos_id

    def encode(
            self,
            seq: str,
            bos: bool = False,
            eos: bool = False,
            max_length: int = -1,
            pad: bool = False,
            device: Optional[torch.device] = None,
            to_tensor=True
    ) -> torch.Tensor:
        tokens = self.spm.encode(seq)
        if bos:
            tokens = [self.bos_id] + tokens

        if eos:
            tokens = tokens + [self.eos_id]

        if max_length > 0:
            tokens = tokens[:max_length]

        if pad and len(tokens) < max_length:
            tokens += [self.pad_id] * (max_length - len(tokens))

        if to_tensor:
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

        return tokens

    def encode_token(
            self,
            seq: str,
            bos: bool = False,
            eos: bool = False,
            max_length: int = -1,
            pad: bool = False,
            device: Optional[torch.device] = None,
            to_tensor=True
    ) -> torch.Tensor:
        tokens = []
        pieces = [*list(seq)]
        for p in pieces:
            tid = self.piece_to_id(p)
            if isinstance(tid, list):
                tokens.extend(tid)
            else:
                tokens.append(tid)
        if bos:
            tokens = [self.bos_id] + tokens

        if eos:
            tokens = tokens + [self.eos_id]

        if max_length > 0:
            tokens = tokens[:max_length]

        if pad and len(tokens) < max_length:
            tokens += [self.pad_id] * (max_length - len(tokens))

        if to_tensor:
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

        return tokens

    def decode(self, tokens: Union[torch.Tensor, np.ndarray]) -> str:
        if isinstance(tokens, (torch.Tensor, np.ndarray)):
            tokens = tokens.tolist()

        return self.spm.decode(tokens)
