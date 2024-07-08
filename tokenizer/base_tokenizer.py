# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/20 16:53
from abc import ABC
from abc import abstractmethod


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    def __len__(self):
        return self.vocab_size

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        return {v: k for k, v in self.vocab.items()}

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    @abstractmethod
    def encode(self,
               text: str,
               bos: bool = False,
               eos: bool = False,
               max_length: int = -1,
               pad: bool = False):
        pass

    def decode(self, token_ids):
        raise NotImplementedError('detokenizer is not implemented for {} '
                                  'tokenizer'.format(self.name))

    @property
    def cls(self):
        raise NotImplementedError('CLS is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def sep(self):
        raise NotImplementedError('SEP is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def pad(self):
        raise NotImplementedError('PAD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def eod(self):
        raise NotImplementedError('EOD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def mask(self):
        raise NotImplementedError('MASK is not provided for {} '
                                  'tokenizer'.format(self.name))

    def train(self,
              files,
              vocab_size=20000,
              speical_tokens=None,
              model=None):
        pass
