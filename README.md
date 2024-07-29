# docker 실행 명령어

```
docker run --gpus all \
  --shm-size=10g --ulimit memlock=-1  \
   -it --name gpu1 -v $PWD:/workspace \
   docker1113444/ubuntu:2
   
conda activate py
```

# configuration 파일 (config 폴더)

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

## 비교하는 학습 모델

1. PointNet (point + point)
2. FoldingNet (graph + fold)

# 학습 코드 실행 방법

```python train_ae.py --config config/graph_enc_fold_dec_cd.yaml```

--config 옵션에 config 파일 경로 입력

학습 후 lightning_logs 폴더에 version_{n} 으로 학습 결과가 저장됨   
network weights 는 checkpoints 폴더에 저장됨  
학습 로그 그래프는 lightning_logs를 tensorboard로 visualization함

## 테스트 케이스 모두 학습 방법

``sh train.sh``

# 테스트 코드 실행

```python test_ae.py```

116 라인의 eval_model 함수에 인자로 ckpt 파일 경로를 입력함  
테스트 데이터에 대해서 평가한 CD, EMD 값을 출력함 