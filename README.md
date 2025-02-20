# MBID

This repository contains source codes and datasets for the paper:

- Multi-Behavior Intent Disentanglement via Information Bottleneck Principle for Recommendation

## Usage
### Train & Test

- Training MBID on IJCAI15:
```shell
python main.py --dataset=IJCAI_15
```

- Training MBID on Tmall:
```shell
python main.py --dataset=Tmall
```

- Training MBID on Retail:
```shell
python main.py --dataset=retailrocket
```

- Testing MBID using a saved model file:
```shell
ipython evaluation.ipynb
```
