This repository contains the official PyTorch code for the Modified Chamfer Distance metric

# Installation with docker 

```
docker run --gpus all \
  --shm-size=10g --ulimit memlock=-1  \
   -it --name gpu1 -v $PWD:/workspace \
   docker1113444/ubuntu:2
   
conda activate py
```

# How to train Point-AE with configuration file

```
model: 
  encoder: graph
  decoder: fold
  loss: cd
  segment: False

argument:
  dropout: 0.5
  feat_dims: 512
  k: 16
  shape: plane

parameter:
  batch_size: 64
  num_workers: 4
  epochs: 2000
  num_points: 2048

dataset: # dataset (modelnet40, shape16)
  dataset: modelnet40

env: # gpu 설정, multi gpu의 경우 옵션은 True로 변경
  gpu: 1
  multi_gpu: False
```

## Baseline models

1. PointNet (point + point)
2. FoldingNet (graph + fold)

# Usage

```python train_ae.py --config {config/graph_enc_fold_dec_cd.yaml}```

After you train the model, the result is saved at lightning_logs dir.
Trained weight also saved at checkpoints dir.
You can visualize the result using tensorboard

## Train with bash file

``sh train.sh``

# Test your trained model

```python test_ae.py```
