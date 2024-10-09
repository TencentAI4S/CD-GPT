# CD-GPT: 全分子建模的生物基础模型

[English](./README.md) | 简体中文

此仓库包含"[CD-GPT: A Biological Foundation Model Bridging the Gap between Molecular Sequences Through Central Dogma](https://www.biorxiv.org/content/10.1101/2024.06.24.600337v1.article-info)"中的代码与模型。

## 目录
1. [模型](#模型)
2. [下游任务](#下游任务)
3. [安装](#安装)
4. [开始](#开始)
5. [微调](#微调cd-gpt)
6. [引用](#引用)

## 模型

**CD-GPT**是一个生成性的生物基础模型，能够捕捉生物系统中分子级别的反应与相互作用。通过在包括DNA、RNA和蛋白质序列在内的数据上进行预训练，我们表明了CD-GPT能够有效地处理包括单分子和多分子分析在内的一系列预测和生成任务。


目前，我们提供以下模型参数以供使用：
| 模型 | 描述                                                                                                                       |
| :---------: | --------------------------------------------------------------------------------------------------------------------------------- |
|  CD-GPT-1b  | 经过阶段1（_单分子序列预训练_）和阶段2（_中心法则预训练_）的模型参数                                      |
| CD-GPT-1b-s | 经过阶段1（_单分子序列预训练_），阶段2（_中心法则预训练_）和阶段3（蛋白质结构预训练）的模型参数 |

您可以从以下链接下载权重文件：
* [Tencent Weiyun Disk](https://share.weiyun.com/LpRbEEH4)
* [Google Drive](https://drive.google.com/drive/folders/1ZqelImiYMpmHhTrBGz7Tm8vFoWF32-pJ?usp=drive_link)

## 下游任务

CD-GPT在一系列下游任务上都取得了SOTA表现。

### DNA启动子检测

| Model | CD-GPT-1b | NT     | DNABERT-2 | HyenaDNA | Evo   |
| ----- | --------- | ------ | --------- | -------- | ----- |
| MCC   | 🥇0.905     | 0.8771 | 🥈0.8831    | 0.4738   | 0.835 |

### DNA剪切位点预测

| Model | CD-GPT-1b | NT     | DNABERT-2 | HyenaDNA |
| ----- | --------- | ------ | --------- | -------- |
| MCC   | 🥇0.894     | 0.7991 | 🥈0.8593    | 0.7267   |

### 蛋白质水溶性预测

| Model | CD-GPT-1b | CD-GPT-1b-s | Transformer | LSTM  | CNN   | ResNet | ProtBERT | ESM   |
| ----- | --------- | ----------- | ----------- | ----- | ----- | ------ | -------- | ----- |
| Acc   | 🥈72.48     | 🥇75.8        | 70.12       | 70.18 | 64.43 | 67.33  | 68.15    | 70.23 |

### 蛋白质二级结构预测


| Model | CD-GPT-1b-s | Transformer | LSTM  | CNN   | ResNet | ProtBERT | ESM   |
| ----- | ----------- | ----------- | ----- | ----- | ------ | -------- | ----- |
| Acc   | 🥇90.83       | 59.62       | 68.99 | 66.07 | 69.56  | 82.18    | 🥈82.73 |

### 蛋白质接触图预测

| Model | CD-GPT-1b-s | Transformer | LSTM  | CNN | ResNet | ProtBERT | ESM   |
| ----- | ----------- | ----------- | ----- | --- | ------ | -------- | ----- |
| P@L/5  | 🥇57.29       | 17.5        | 26.34 | 10  | 20.43  | 39.66    | 🥈45.78 |

### RNA蛋白质相互作用预测

RPI-369:
| Model | CD-GPT-1b | lncPro | RPISeq | IPMiner | RPITER |
| ----- | --------- | ------ | ------ | ------- | ------ |
| MCC   | 🥇0.5224    | 0.009  | 0.426  | 0.428   | 🥈0.461  |
| Acc   | 🥇76.05     | 50.2   | 71.3   | 70      | 🥈72.8   |
| Pre   | 🥈77.98     | 51.2   | 72.4   | 🥇84      | 70.1   |

RPI-488:
| Model | CD-GPT-1b | lncPro | RPISeq | IPMiner | RPITER |
| ----- | --------- | ------ | ------ | ------- | ------ |
| MCC   | 🥇0.8204    | 0.725  | 0.771  | 🥈0.793   | 🥈0.793  |
| Acc   | 🥇90.8      | 85.6   | 88.3   | 🥈89.3    | 🥈89.3   |
| Pre   | 🥇95.54     | 94     | 93.5   | 🥈95.1    | 94.3   |

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
### 生成任务
若想使用CD-GPT完成生成任务，如翻译或逆翻译，请参照`generate_examply.ipynb` 。
### 预测任务

搭配输出头，CD-GPT可以应用于不同类型的下游任务。目前我们发布的模型不包括输出头的权重，因此下列实例的输出结果是随机的。

**序列预测任务**
```shell
python predict_example.py \
    -m checkpoints/CD-GPT-1b.pth \
    -t checkpoints/tokenizer.model \
    -h sequence \
    -n 2
```

**残基预测任务**
```shell
python predict_example.py \
    -m checkpoints/CD-GPT-1b.pth \
    -t checkpoints/tokenizer.model \
    -h token \
    -n 2
```

**残基对预测任务**
```shell
python predict_example.py \
    -m checkpoints/CD-GPT-1b.pth \
    -t checkpoints/tokenizer.model \
    -h residuepair \
    -n 2
```

## 微调CD-GPT
我们将会在未来提供微调CD-GPT的教程。

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