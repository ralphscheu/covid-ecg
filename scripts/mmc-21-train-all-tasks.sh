#!/usr/bin/env bash

echo $0 $@

model=CNN3DSeqReducedMeanStdPool

expDir=data/processed/mmc_tasks_lfcc
python3 covidecg/train.py --model $model --feats raw 


# for exp in data/processed/mmc_tasks_img/*; do
#     python3 covidecg/train.py --model $model --feats img $exp
# done


