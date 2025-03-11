# CD-GPT: Biological Foundation Model at Full-molecular Level

English | [简体中文](./README-zh.md)

This repo contains the code and models for "[CD-GPT: A Biological Foundation Model Bridging the Gap between Molecular Sequences Through Central Dogma](https://www.biorxiv.org/content/10.1101/2024.06.24.600337v1.article-info)".

## Contents
1. [Model](#model)
2. [Results](#results)
3. [Setup](#setup)
4. [Getting Start](#getting-start)
5. [Finetune](#finetune-cd-gpt-on-your-own-datasets)
6. [Citation](#citation)

## Model
**CD-GPT** is a generative biological foundation model aiming to capture the intricate system-wide molecular interactions in biological systems. Through pretraining on full molecular level data including DNA, RNA and protein sequences, we demonstrate that CD-GPT can efficiently handle a series of downstream tasks, including prediction and generation tasks across mono-molecular and multi-molecular analyses.


We have released the following checkpoints:

|          Checkpoint           | Description                                                                                                                                |
| :---------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------ |
|           CD-GPT-1b           | Model pretrained through Stage 1 (_Mono-sequence Pretrain_) and Stage 2 (_Central Dogma Pretrain_).                                         |
|          CD-GPT-1b-s          | Model pretrained through Stage 1 (_Mono-sequence Pretrain_), Stage 2 (_Central Dogma Pretrain_) and Stage 3 (_Protein Structure Pretrain_). |
| CD-GPT-1b-reverse-translation | Model finetuned on translation-related pair sequences. Can be used to generate codon sequence from protein. |

You can download the weights from:
- [Tencent Weiyun Disk](https://share.weiyun.com/LpRbEEH4)
- [Google Drive](https://drive.google.com/drive/folders/1ZqelImiYMpmHhTrBGz7Tm8vFoWF32-pJ?usp=drive_link)

## Results

CD-GPT achieves SOTA performance on a series of downstream tasks.

### DNA Promoter Detection

| Model | CD-GPT-1b | NT     | DNABERT-2 | HyenaDNA | Evo   |
| ----- | --------- | ------ | --------- | -------- | ----- |
| MCC   | 🥇0.905     | 0.8771 | 🥈0.8831    | 0.4738   | 0.835 |

### DNA Splice Site Prediction

| Model | CD-GPT-1b | NT     | DNABERT-2 | HyenaDNA |
| ----- | --------- | ------ | --------- | -------- |
| MCC   | 🥇0.894     | 0.7991 | 🥈0.8593    | 0.7267   |

### Protein Solubility Prediction

| Model | CD-GPT-1b | CD-GPT-1b-s | Transformer | LSTM  | CNN   | ResNet | ProtBERT | ESM   |
| ----- | --------- | ----------- | ----------- | ----- | ----- | ------ | -------- | ----- |
| Acc   | 🥈72.48     | 🥇75.8        | 70.12       | 70.18 | 64.43 | 67.33  | 68.15    | 70.23 |

### Protein Secondary Structure Prediction


| Model | CD-GPT-1b-s | Transformer | LSTM  | CNN   | ResNet | ProtBERT | ESM   |
| ----- | ----------- | ----------- | ----- | ----- | ------ | -------- | ----- |
| Acc   | 🥇90.83       | 59.62       | 68.99 | 66.07 | 69.56  | 82.18    | 🥈82.73 |

### Protein Contact Map Prediction

| Model | CD-GPT-1b-s | Transformer | LSTM  | CNN | ResNet | ProtBERT | ESM   |
| ----- | ----------- | ----------- | ----- | --- | ------ | -------- | ----- |
| P@L/5  | 🥇57.29       | 17.5        | 26.34 | 10  | 20.43  | 39.66    | 🥈45.78 |

### RNA Protein Interaction Prediction

for RPI-369:
| Model | CD-GPT-1b | lncPro | RPISeq | IPMiner | RPITER |
| ----- | --------- | ------ | ------ | ------- | ------ |
| MCC   | 🥇0.5224    | 0.009  | 0.426  | 0.428   | 🥈0.461  |
| Acc   | 🥇76.05     | 50.2   | 71.3   | 70      | 🥈72.8   |
| Pre   | 🥈77.98     | 51.2   | 72.4   | 🥇84      | 70.1   |

for RPI-488:
| Model | CD-GPT-1b | lncPro | RPISeq | IPMiner | RPITER |
| ----- | --------- | ------ | ------ | ------- | ------ |
| MCC   | 🥇0.8204    | 0.725  | 0.771  | 🥈0.793   | 🥈0.793  |
| Acc   | 🥇90.8      | 85.6   | 88.3   | 🥈89.3    | 🥈89.3   |
| Pre   | 🥇95.54     | 94     | 93.5   | 🥈95.1    | 94.3   |

## Setup
```shell
# create a virtual environment
conda create -n cdgpt
conda activate cdgpt
# clone repo and install requirements
git clone https://github.com/TencentAI4S/CD-GPT.git
cd CD-GPT/
pip install -r requirements.txt
```
## Getting Start
### Download Model Checkpoint
```shell
# download checkpoints and tokenizer and put them under this directory
mkdir checkpoints
```
### Generation

For generation purposes like translation or reverse translation, you can refer to `generate_examply.ipynb` for guidance.

### Prediction

Equipped with output heads, CD-GPT can be applied to different types of downstream tasks. Currently released checkpoints **do not** include the weight of output head, so the output would be a random guess.

**Sequence Prediction Task**
```shell
python predict_example.py \
    --model checkpoints/CD-GPT-1b.pth \
    --tokenizer checkpoints/tokenizer.model \
    --head sequence \
    --num_classes 2
```

**Token Prediction Task**
```shell
python predict_example.py \
    -model checkpoints/CD-GPT-1b.pth \
    --tokenizer checkpoints/tokenizer.model \
    --head token \
    --num_classes 2
```

**Residue-Pair Prediction Task**
```shell
python predict_example.py \
    -model checkpoints/CD-GPT-1b.pth \
    --tokenizer checkpoints/tokenizer.model \
    --head residuepair \
    --num_classes 2
```

## Finetune CD-GPT on your own datasets
We will provide tutorial of finetuning CD-GPT on your datasets in the future.

## Citation
If you use CD-GPT in your research, please cite our paper

```BibTeX
@article{zhu2024cd,
  title={CD-GPT: A Biological Foundation Model Bridging the Gap between Molecular Sequences Through Central Dogma},
  author={Zhu, Xiao and Qin, Chenchen and Wang, Fang and Yang, Fan and He, Bing and Zhao, Yu and Yao, Jianhua},
  journal={bioRxiv},
  pages={2024--06},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```