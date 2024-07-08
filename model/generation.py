# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/20 10:58
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import fields
from typing import Any, Tuple
from typing import Optional, List

import torch


def is_tensor(x):
    return isinstance(x, torch.Tensor)


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)
        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for idx, element in enumerate(iterator):
                    if (
                            not isinstance(element, (list, tuple))
                            or not len(element) == 2
                            or not isinstance(element[0], str)
                    ):
                        if idx == 0:
                            # If we do not have an iterator of key/values, set it as attribute
                            self[class_fields[0].name] = first_field
                        else:
                            # If we have a mixed iterator, raise an error
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


def top_k_filtering(logits: torch.Tensor, top_k: int):
    """Set the logits for none top-k values to -inf.

    Args:
        logits: tensor[bs, vocab_size]
        top_k: keep top k probility values
    """
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits.masked_fill_(logits < v[..., [-1]], float('-Inf'))

    return logits


def top_p_filtering(logits: torch.Tensor, top_p: float):
    """Set the logits for none top-p values to -inf.

    Args:
        logits: tensor[bs, vocab_size]
        top_p: probility of filtering
    """
    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Filteration based on the cumulative sum.
    filter = cumulative_probs > top_p
    filter[..., 1:] = filter[..., :-1].clone()
    # Make sure we at least have one token to select from.
    filter[..., 0] = 0
    # Fill in the filtered part
    filter = filter.scatter(1, sorted_indices, filter)
    logits.masked_fill_(filter, float('-Inf'))

    return logits


def sample(logits: torch.Tensor,
           top_k: int = 0,
           top_p: float = 0.0,
           temperature: float = 1.0):
    """ Sample and generate a token.
    Args:
        logits: tensor[bs, v], v is vocab size

    Return:
        output: tensor[bs, ], selected tokens
    """
    assert 0.0 <= top_p <= 1.0, 'p in the range[0, 1]'
    assert 0 <= top_k <= logits.size(1), 'top-k is larger than logit size.'
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)

    # Clone so we do not modify the inputs,
    logits = logits.clone()
    if temperature != 1.0:
        logits.div_(temperature)

    if top_k > 1:
        top_k_filtering(logits, top_k)

    elif top_p > 0.0:
        assert top_p <= 1.0, 'top-p should be in (0, 1].'
        top_p_filtering(logits, top_p)
    # After filtering, we need to recalculate the distribution.
    probs = logits.softmax(dim=-1)

    return torch.multinomial(probs, num_samples=1)


@dataclass
class GenerationOutput(ModelOutput):
    """
    Args:
        sequences: tensor[batch_size, sequence_length], The generated sequences. The second dimension (sequence_length)
            is either equal to `max_length` or shorter if all batches finished early due to the `eos_token_id`.
        scores: `tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed
    """
    sequences: torch.LongTensor = None
    scores: Optional[List[torch.FloatTensor]] = None


class GenerationMixin:

    @torch.no_grad()
    def generate(self,
                 token_ids: torch.Tensor,
                 max_new_tokens: int,
                 *,
                 temperature: float = 1.0,
                 top_k: int = 0,
                 top_p: float = 0.0,
                 output_score=True,
                 stop_ids=None,
                 vocab_size=None):
        """
        Args:
            token_ids: tensor[seq_len, ] or [bs, seq_len]
        """
        only_one = token_ids.dim() == 1
        if only_one:
            token_ids = token_ids[None]

        bs, seq_len = token_ids.shape
        assert seq_len < self.max_len, f"input token is too long"
        max_len = min(self.max_len, max_new_tokens + seq_len)
        output_ids = list(range(bs))
        outputs = [GenerationOutput(scores=[] if output_score else None) for _ in output_ids]

        for cur_pos in range(seq_len, max_len):
            logits = self(token_ids)[:, -1]  # [bs, vocab_size]
            if vocab_size is not None:
                logits = logits[..., :vocab_size]

            next_ids = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)  # [bs, 1]
            # append sampled index to the running sequence and continue
            token_ids = torch.cat((token_ids, next_ids), dim=1)  # [bs, seq_len]

            if output_score:
                probs = logits.softmax(dim=-1)
                for i, p in enumerate(probs):
                    p = p[next_ids[i].item()]
                    outputs[output_ids[i]].scores.append(p)

            remained_batch_ids = list(range(token_ids.size(0)))
            if stop_ids is not None:
                for i, tidx in enumerate(next_ids.view(-1).tolist()):
                    if tidx in stop_ids:
                        outputs[output_ids[i]].sequences = token_ids[i]
                        remained_batch_ids.remove(i)

                if len(remained_batch_ids) == 0:
                    break
                token_ids = token_ids[remained_batch_ids]

            output_ids = [output_ids[idx] for idx in remained_batch_ids]
            if cur_pos == max_len - 1 and len(output_ids) > 0:
                for i, tensor in zip(output_ids, token_ids):
                    outputs[i].sequences = tensor

        return outputs[0] if only_one else outputs
