# Modified Chamfer Distance (MCD)

This repository contains the official PyTorch code for the Modified Chamfer Distance metric from paper:

[Novel Point Cloud Distance Metrics for Error Sensitivity Control in Deep Point Processing]

## Requirements

### Installation with docker 

```
docker run --gpus all \
  --shm-size=10g --ulimit memlock=-1  \
   -it --name gpu1 -v $PWD:/workspace \
   docker1113444/ubuntu:2
   
conda activate py
```

### Installtion with requirements.txt

```
pip install -r requirements.txt
```

## Usage

### How to train Point-AE with configuration file

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

env: # if you want to use multiple gpus, set the multi_gpu=True
  gpu: 1
  multi_gpu: False
```

### Baseline models

1. PointNet (point + point)
2. FoldingNet (graph + fold)
3. Point-AE (graph + BID)

### Usage

1. You can run our code with prepared config file as follow.
   
```
python train_ae.py --config {your_config_file}
```

2. if your file name is 'grah_enc_fold_dec_cd.yaml', then you can use like
   
```
python train_ae.py --config config/graph_enc_fold_dec_cd.yaml
```

After you train the model, the result is saved at lightning_logs dir.

Trained weight also saved at checkpoints dir.

You can visualize the result using tensorboard

3. You can optimize your models with diverse metric with .sh file

```
sh train.sh
```

4. After you trained your models, then you can test your result like

```
python test_ae.py
```
