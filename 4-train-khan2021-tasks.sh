#!/usr/bin/env bash

for model in CNN3DSeqReducedMeanStdPool CNN3DSeqReducedAttnPool CNN3DSeqMeanStdPool CNN3DSeqAttnPool CNN3DSeqReducedLSTM CNN3DSeqLSTM; do
    for exp in data/processed/khan2021_*; do
        python3 covidecg/train.py --model $model $exp
    done
done
