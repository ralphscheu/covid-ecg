covid-ecg
==============================

## Data structure and naming scheme
- `data/raw`: Dataset extracted from MMC ECG software system
- `data/external`: [ECG Dataset](https://data.mendeley.com/datasets/gwbz3fsgp8/1) published in [Khan2021](https://pubmed.ncbi.nlm.nih.gov/33521183/) 
- `data/interim`: ECG samples in `csv` or `png` form
- `data/processed`: Samples split into train and test sets for different classification tasks


## Set up environment variables
Some environment variables are being read from a `.env` file.
Create `./.env` in the repository root and adjust the following variables according to your setup:
```
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
SAMPLING_RATE=500
PROJECT_ROOT=~/covid-ecg
CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_VISIBLE_DEVICES=0
```

## Run an experiment
Example command:
```
CUDA_VISIBLE_DEVICES=0 python3 ./train_evaluate.py --exp-config ./exp_configs/exp-postcovid-ctrl-img.yaml --model-config ./exp_configs/model-vgg16.yaml
```