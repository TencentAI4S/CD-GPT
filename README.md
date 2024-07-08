# CD-GPT

--------------------------------------------------------------------------------
English | [简体中文](./README-zh.md)

A Biological Foundation Model Bridging the Gap between Molecular Sequences Through Central Dogma

## Model

1. CD-GPT-1b
2. CD-GPT-1b-s
* [Tencent Weiyun Disk](https://share.weiyun.com/LpRbEEH4)
* [Google Drive](https://drive.google.com/drive/folders/1ZqelImiYMpmHhTrBGz7Tm8vFoWF32-pJ?usp=drive_link)

## Install
```shell
pip install -r requirements.txt
```
## Getting Start
### Download Model Checkpoint
```shell
mkdir checkpoints
# download checkpoints and tokenizer form google drive
```
### Generation
```shell
python example.py -m checkpoints/CD-GPT-1b.pth -t checkpoints/tokenizer.model
```

## Citing CD-GPT
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