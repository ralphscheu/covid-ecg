#!/usr/bin/env bash

for model in CNN3DSeqReducedMeanStdPool CNN3DSeqReducedAttnPool CNN3DSeqMeanStdPool CNN3DSeqAttnPool CNN3DSeqReducedLSTM CNN3DSeqLSTM CNN2DSeqReducedMeanStdPool CNN2DSeqReducedAttnPool CNN2DSeqMeanStdPool CNN2DSeqAttnPool CNN2DSeqReducedLSTM CNN2DSeqLSTM; do
    for exp in data/processed/khan_tasks_*; do
        python3 covidecg/train.py --model $model $exp
    done
done
