# CD-GPT: 全分子建模的生物基础模型

--------------------------------------------------------------------------------
[English](./README.md) | 简体中文

[TOC]

## 模型

**CD-GPT**是一个生成式的生物基础模型，能够在全分子层面上对生物序列数据进行建模，并处理涉及单分子序列和多分子序列的预测和生成任务。

更多有关模型的细节，请参阅["CD-GPT: A Biological Foundation Model Bridging the Gap between Molecular Sequences Through Central Dogma"](https://www.biorxiv.org/content/10.1101/2024.06.24.600337v1.article-info).


目前，我们提供以下模型参数以供使用：
| 模型 | 描述                                                                                                                       |
| :---------: | --------------------------------------------------------------------------------------------------------------------------------- |
|  CD-GPT-1b  | 经过阶段1（_单分子序列预训练_）和阶段2（_中心法则预训练_）的模型参数                                      |
| CD-GPT-1b-s | 经过阶段1（_单分子序列预训练_），阶段2（_中心法则预训练_）和阶段3（蛋白质结构预训练）的模型参数 |

您可以从以下链接下载权重文件：
* [Tencent Weiyun Disk](https://share.weiyun.com/LpRbEEH4)
* [Google Drive](https://drive.google.com/drive/folders/1ZqelImiYMpmHhTrBGz7Tm8vFoWF32-pJ?usp=drive_link)

## 安装
```shell
# 创建虚拟环境
conda create -n cdgpt
conda activate cdgpt
# 克隆仓库并安装依赖
git clone https://github.com/TencentAI4S/CD-GPT.git
cd CD-GPT/
pip install -r requirements.txt
```

## 开始
### 下载模型参数
```shell
# 下载模型文件和分词器文件，并放置在此文件夹中
mkdir checkpoints
```
### 序列生成
```shell
python example.py -m checkpoints/CD-GPT-1b.pth -t checkpoints/tokenizer.model
```

## 引用
如果您在研究中使用了CD-GPT，请引用我们的文章

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