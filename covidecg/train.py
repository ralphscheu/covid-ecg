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
        

@click.command()
@click.option('--model', required=True, type=str)
@click.option('--run-name', default='', type=str)
@click.argument('dataset_root', required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
def run_experiment(model, run_name, dataset_root):
    start_time = time.monotonic()
    mlflow.sklearn.autolog()
    experiment = mlflow.set_experiment(experiment_name=Path(dataset_root).stem)
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name):
        conf = utils.load_exp_model_conf(os.path.join(os.getenv('PROJECT_ROOT'), 'conf', 'train_conf.yaml'))
        

        #
        # Load dataset
        #
        logging.info(f"Loading train and test data from {dataset_root}")
        update_mlflow_logfile()
        
        train_dataset = torchvision.datasets.ImageFolder(dataset_root / 'train', transform=IMAGE_TRANSFORMS)
        y_train = np.array(train_dataset.targets)
        test_dataset = torchvision.datasets.ImageFolder(dataset_root / 'test', transform=IMAGE_TRANSFORMS)
        y_test = np.array(test_dataset.targets)
        
        logging.info(f"Class labels targets: {train_dataset.class_to_idx}")
        logging.info(f"Train on {len(y_train)} samples | class distribution: {str(np.unique(y_train, return_counts=True))}")
        logging.info(f"Test on {len(y_test)} samples | class distribution: {str(np.unique(y_test, return_counts=True))}")
        update_mlflow_logfile()

        clf = utils.build_model(model, conf, train_dataset)
        update_mlflow_logfile()

        gs = GridSearchCV(clf, conf['grid_search'],
            scoring=sklearn.metrics.get_scorer('roc_auc'),
            cv=int(conf['num_cv_folds']), refit=True, error_score='raise', verbose=4)
        
        logging.info(f"Start training on {len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))} GPUs ({os.environ['CUDA_VISIBLE_DEVICES']})")
        update_mlflow_logfile()
        gs.fit(SliceDataset(train_dataset), y_train)
        
        
        import pandas as pd
        pd.DataFrame(gs.cv_results_).to_csv('/tmp/covidecg_cv_results_.csv', index=None, sep=';')
        mlflow.log_artifact('/tmp/covidecg_cv_results_.csv')


        logging.info(f"GridSearchCV - Best ROC-AUC Score in CV: {gs.best_score_}")
        logging.info(f"GridSearchCV - Best Params: {gs.best_params_}")
        update_mlflow_logfile()
        
        logging.info("Evaluating chosen model on test dataset...")
        utils.evaluate_experiment(test_dataset, y_test, gs.best_estimator_)
        update_mlflow_logfile()
        
        end_time = time.monotonic()
        logging.info(f"Done. Finished in {timedelta(seconds=end_time - start_time)}")
        
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
    run_experiment()
