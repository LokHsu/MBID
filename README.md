# MBID

This repository contains source codes and datasets for our **CIKM'25** paper:

- Multi-Behavior Intent Disentanglement for Recommendation via Information Bottleneck Principle

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

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{MBID,
    title = {Multi-Behavior Intent Disentanglement for Recommendation via Information Bottleneck Principle},
    author = {Xu, Tongxin and Bin, Chenzhong and Xiao, Cihan and Li, Yunhui and Gu, Tianlong},
    booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
    year = {2025}
}
```
