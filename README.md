<!--
 * @Author: zhuxiao 1317794623@qq.com
 * @Date: 2024-07-15 17:26:47
 * @LastEditors: zhuxiao 1317794623@qq.com
 * @LastEditTime: 2024-07-17 23:20:10
 * @FilePath: \CD-GPT\README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# CD-GPT: Biological Foundation Model at Full-molecular Level

----------------

English | [简体中文](./README-zh.md)

[TOC]

## Model
**CD-GPT** is a generative biological foundation model, which models biological data at full-molecular level. We demonstrate that CD-GPT can efficiently handle a series of downstream tasks, including prediction and generation tasks across mono-molecular and multi-molecular analyses.

More details are included in the paper ["CD-GPT: A Biological Foundation Model Bridging the Gap between Molecular Sequences Through Central Dogma"](https://www.biorxiv.org/content/10.1101/2024.06.24.600337v1.article-info).

We have released the following checkpoints:

| Checkpoint | Description                                                                                                                       |
| :---------: | --------------------------------------------------------------------------------------------------------------------------------- |
|  CD-GPT-1b  | Model pretrained through Stage 1 (_Mono-sequence Pretrain_) and Stage 2 (_Central Dogma Pretrain_)                                      |
| CD-GPT-1b-s | Model pretrained through Stage 1 (_Mono-sequence Pretrain_), Stage 2 (_Central Dogma Pretrain_) and Stage 3 (_Protein Structure Pretrain_) |

You can download the weights from:
- [Tencent Weiyun Disk](https://share.weiyun.com/LpRbEEH4)
- [Google Drive](https://drive.google.com/drive/folders/1ZqelImiYMpmHhTrBGz7Tm8vFoWF32-pJ?usp=drive_link)

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
```shell
python example.py -m checkpoints/CD-GPT-1b.pth -t checkpoints/tokenizer.model
```

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