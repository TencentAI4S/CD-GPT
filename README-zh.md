# CD-GPT

--------------------------------------------------------------------------------
[English](./README.md) | 简体中文

A Biological Foundation Model Bridging the Gap between Molecular Sequences Through Central Dogma

## 模型

1. CD-GPT-1b
2. CD-GPT-1b-s
* [Tencent Weiyun Disk](https://share.weiyun.com/LpRbEEH4)
* [Google Drive](https://drive.google.com/drive/folders/1ZqelImiYMpmHhTrBGz7Tm8vFoWF32-pJ?usp=drive_link)

## 安装
```shell
pip install -r requirements.txt
```

## 开始
### 下载模型参数
```shell
mkdir checkpoints
# download checkpoints and tokenizer form google drive
```
### 序列生成
```shell
python example.py -m checkpoints/CD-GPT-1b.pth -t checkpoints/tokenizer.model
```

## 引用CD-GPT
如果你在研究中使用了CD-GPT， 请引用我们的文章

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