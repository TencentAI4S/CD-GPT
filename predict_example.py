# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/8/28 14:15
import argparse
import os.path

import torch

from config import get_config
from model import CDGPTResiduePairPrediction, CDGPTTokenPrediction, CDGPTSequencePrediction
from tokenizer import SentencePieceTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='CD-GPT model prediction')
    parser.add_argument('--model', '-m', default="checkpoints/CD-GPT-1b.pth", help='model checkpoint path')
    parser.add_argument('--tokenizer', '-t', default="checkpoints/tokenizer.model", help='tokenizer path')
    parser.add_argument('--head', '-h', default="sequence", help='output head type, must be sequence, token or residuepair')
    parser.add_argument('--num_classes', '-n', default=2, help="number of prediction categories")
    args = parser.parse_args()
    return args

def setup():
    torch.set_grad_enabled(False)

def main(args):
    setup()
    cfg = get_config()
    cfg.tokenizer.path = args.tokenizer
    cfg.model.num_classes = args.num_classes
    tokenizer = SentencePieceTokenizer(args.tokenizer)
    cfg.tokenizer.pad_id = tokenizer.pad
    model_path = args.model
    assert os.path.exists(model_path)
    state = torch.load(model_path, map_location="cpu")

    output_head = args.head
    assert output_head in ('sequence', 'token', 'residuepair')
    if output_head == "sequence":
        model = CDGPTSequencePrediction(cfg)
    elif output_head == "token":
        model = CDGPTTokenPrediction(cfg)
    else:
        model = CDGPTResiduePairPrediction(cfg)
    model.load_state_dict(state["model"])
    print(f"load checkpoint form: {model_path}")
    model.half().cuda().eval()

    input_sequence = "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"
    
    x = tokenizer.encode(input_sequence, eos=False, device=model.device) if output_head == 'sequence' else tokenizer.encode_token(input_sequence, eos=False, device=model.device)
    output = model(x)["output"]
    print(output)
    


if __name__ == '__main__':
    args = parse_args()
    main(args)
