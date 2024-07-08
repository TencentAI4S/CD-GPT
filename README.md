# CD-GPT

--------------------------------------------------------------------------------
English | [简体中文](./README-zh.md)

A Biological Foundation Model Bridging the Gap between Molecular Sequences Through Central Dogma

## Model

1. [weiyun disk](https://share.weiyun.com/LpRbEEH4)
2. google drive

## Install
```shell
pip install -r requirements.txt
```
## Getting Start
## Download Model Checkpoint
```shell
mkdir checkpoints
# download checkpoints and tokenizer form google drive
wget checkpoints.pth
```
### Generation
```shell
python example.py -m checkpoints/CD-GPT-1b.pth -t checkpoints/tokenizer.model
```