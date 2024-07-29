#! /bin/bash

LOSS_LIST=("cd") # you can add other losses like "emd", "mcd5", "mcd10", "mcd20"
YAML_LIST=("./config/pointAE_cd.yaml") # you can add other yaml files like "./config/pointAE_emd.yaml"

# train
for config in ${YAML_LIST[@]}; do
    echo "Training $config"
    python train_ae.py --config $config
done
