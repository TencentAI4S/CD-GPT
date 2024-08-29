# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/20 16:38
import argparse
import os.path

import torch

from config import get_config
from model import CDGPT
from tokenizer import SentencePieceTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='CD-GPT model generation')
    parser.add_argument('--model', '-m', default="checkpoints/CD-GPT-1b.pth", help='model checkpoint path')
    parser.add_argument('--tokenizer', '-t', default="checkpoints/tokenizer.model", help='tokenizer path')
    args = parser.parse_args()
    return args

def setup():
    torch.set_grad_enabled(False)

def main(args):
    setup()
    cfg = get_config()
    cfg.tokenizer.path = args.tokenizer
    tokenizer = SentencePieceTokenizer(args.tokenizer)
    model_path = args.model
    assert os.path.exists(model_path)
    state = torch.load(model_path, map_location="cpu")
    model = CDGPT(cfg)
    model.load_state_dict(state["model"])
    print(f"load checkpoint form: {model_path}")
    model.half().cuda().eval()
    prompt = (f"<mRNA>ATGTCCGGACTCCACCGATCCTCTAGTGCTCCCATGAAGAAGGGCCTTCCTC"
              f"CCAAGGAGCTCCTCGACGATCTCTGCAGTCGGTTTGTTTTAAATGTGCCAAAAGAAGATCTCGAGTCATTTGAAAGGA"
              f"TTTTATTTCTTGTGGAGAATGCACATTGGTTCTATGAAGACAACTCAGTGGAAAGAGATCCATCACTGAAGTCATTAA"
              f"CTCTGAAGGAATTCACTTCTCTAATATTCAACAGTTGTGATGTCTTAAAACCTTATGTTCCCCATTTGGATGACATAT"
              f"TTAAGGACTTCACTTCCTACAAGGTCCGAGTTCCTGTAACTGGGGCAATAATTTTG</mRNA><translate><:>")
    x = tokenizer.encode(prompt, eos=False, device=model.device)
    output = model.generate(x,
                            max_new_tokens=1024,
                            temperature=0.8,
                            top_k=128,
                            top_p=0.0,
                            stop_ids=(tokenizer.bos, tokenizer.eos, tokenizer.pad)
                            )
    output = tokenizer.decode(output.sequences)
    print(output)


if __name__ == '__main__':
    args = parse_args()
    main(args)
