import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import covidecg.models.train_eval_utils as utils
import covidecg.data.utils as data_utils
from covidecg.data.dataset import InvertGrayscale, SliceEcgGrid, SliceTimesteps
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
import mlflow
import sklearn.metrics
import torch
from io import StringIO
import random
import time
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')
import torchvision
from skorch.helper import SliceDataset
import warnings
warnings.filterwarnings('ignore')

# Ensure reproducibility
RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.use_deterministic_algorithms(True, warn_only=True)


def update_mlflow_logfile():
    mlflow.log_text(LOG_STREAM.getvalue(), 'train.log')


IMAGE_TRANSFORMS = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.ToTensor(),
    InvertGrayscale(),
    SliceEcgGrid(),
    SliceTimesteps()
    ])

RAW_TRANSFORMS = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    SliceTimesteps()
    ])

LFCC_TRANSFORMS = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    SliceTimesteps()
    ])


@click.command()
@click.option('--model', required=True, type=str)
@click.option('--feats', required=True, type=click.Choice(['raw', 'img', 'imgblur', 'imgsharpen', 'imgsuperres', 'lfcc']))
@click.option('--run-name', default='', type=str)
@click.argument('dataset_root', required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
def run_experiment(model, feats, run_name, dataset_root):
    start_time = time.monotonic()
    mlflow.sklearn.autolog()
    
    experiment = mlflow.set_experiment(experiment_name=Path(dataset_root).stem)
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name):
        conf = utils.load_exp_model_conf(os.path.join(os.getenv('PROJECT_ROOT'), 'conf', 'train_conf.yaml'))
        
        #
        # Load dataset
        #
        LOG_STREAM.write(f"\nLoading train and test data from {dataset_root}")
        update_mlflow_logfile()
        
        mlflow.set_tag('feats', feats)
        
        if feats == 'raw':
            # ECG voltage readings as csv files
            train_dataset = torchvision.datasets.DatasetFolder(
                dataset_root / 'train', 
                loader=lambda f: data_utils.load_signal(f, return_cleaned_signal=True), 
                extensions=['csv'], transform=RAW_TRANSFORMS)
            test_dataset = torchvision.datasets.DatasetFolder(
                dataset_root / 'test',
                loader=lambda f: data_utils.load_signal(f, return_cleaned_signal=True), 
                extensions=['csv'], transform=RAW_TRANSFORMS)
        elif feats in ['img', 'imgblur', 'imgsharpen', 'imgsuperres']:
            # ECG Image features as png files
            train_dataset = torchvision.datasets.ImageFolder(dataset_root / 'train', transform=IMAGE_TRANSFORMS)
            test_dataset = torchvision.datasets.ImageFolder(dataset_root / 'test', transform=IMAGE_TRANSFORMS)
        elif feats == 'lfcc':
            # LFCC features as npy files
            train_dataset = torchvision.datasets.DatasetFolder(
                dataset_root / 'train',
                loader=np.load,
                extensions=['npy'], transform=LFCC_TRANSFORMS)
            test_dataset = torchvision.datasets.DatasetFolder(
                dataset_root / 'test', 
                loader=np.load,
                extensions=['npy'], transform=LFCC_TRANSFORMS)
        
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)
        
        # print(train_dataset[0][0].shape)
        # print(len(train_dataset), len(test_dataset))
        # import sys
        # sys.exit(0)
        
        LOG_STREAM.write(f"\nClass labels targets: {train_dataset.class_to_idx}")
        LOG_STREAM.write(f"\nTrain on {len(y_train)} samples | class distribution: {str(np.unique(y_train, return_counts=True))}")
        LOG_STREAM.write(f"\nTest on {len(y_test)} samples | class distribution: {str(np.unique(y_test, return_counts=True))}")
        update_mlflow_logfile()

        clf = utils.build_model(model, conf, train_dataset)
        update_mlflow_logfile()

        gs = GridSearchCV(clf, conf['grid_search'],
            scoring=sklearn.metrics.get_scorer('roc_auc'),
            cv=int(conf['num_cv_folds']), refit=True, error_score='raise', verbose=4)
        
        LOG_STREAM.write(f"\nStart training on {len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))} GPUs ({os.environ['CUDA_VISIBLE_DEVICES']})")
        update_mlflow_logfile()
        gs.fit(SliceDataset(train_dataset), y_train)
        
        
        import pandas as pd
        pd.DataFrame(gs.cv_results_).to_csv('/tmp/covidecg_cv_results_.csv', index=None, sep=';')
        mlflow.log_artifact('/tmp/covidecg_cv_results_.csv')


        LOG_STREAM.write(f"\nGridSearchCV - Best ROC-AUC Score in CV: {gs.best_score_}")
        LOG_STREAM.write(f"\nGridSearchCV - Best Params: {gs.best_params_}")
        update_mlflow_logfile()
        
        LOG_STREAM.write("\nEvaluating chosen model on test dataset...")
        utils.evaluate_experiment(test_dataset, y_test, gs.best_estimator_)
        update_mlflow_logfile()
        
        end_time = time.monotonic()
        LOG_STREAM.write(f"\nDone. Finished in {timedelta(seconds=end_time - start_time)}")
        
        update_mlflow_logfile()


if __name__ == '__main__':
    load_dotenv(find_dotenv())  # load environment variables set in .env file
    LOG_STREAM = StringIO()
    logging.basicConfig(
        level=os.getenv('LOG_LEVEL', default='INFO'), format=os.getenv('LOG_FORMAT'),
        handlers=[
            logging.StreamHandler(),  # log to stdout
            logging.StreamHandler(stream=LOG_STREAM)  # log to StringIO object for storing in MLFlow
        ])
    logger = logging.getLogger(__name__)
    
    run_experiment()
