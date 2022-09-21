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

# Ensure reproducibility
RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.use_deterministic_algorithms(True, warn_only=True)


@click.command()
@click.option('--model', required=True, type=str)
@click.argument('dataset_root', required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
def run_experiment(model, dataset_root):
    start_time = time.monotonic()
    mlflow.sklearn.autolog()
    experiment = mlflow.set_experiment(experiment_name=Path(dataset_root).stem)
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        conf = utils.load_exp_model_conf(os.path.join(os.getenv('PROJECT_ROOT'), 'conf', 'train_conf.yaml'))

        # Load dataset
        train_dataset = torchvision.datasets.ImageFolder(dataset_root / 'train', transform=torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
            InvertGrayscale(),
            SliceEcgGrid(),
            SliceTimesteps()
            ]))
        y_train = np.array(train_dataset.targets)
        test_dataset = torchvision.datasets.ImageFolder(dataset_root / 'test', transform=torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
            InvertGrayscale(),
            SliceEcgGrid(), 
            SliceTimesteps()
            ]))
        y_test = np.array(test_dataset.targets)

        clf = utils.build_model(model, conf, train_dataset)

        gs = GridSearchCV(clf, conf['grid_search'],
            scoring=sklearn.metrics.get_scorer('roc_auc'),
            cv=int(conf['num_cv_folds']), refit=True, error_score='raise', verbose=4)
        
        logging.info(f"Start training on {len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))} GPUs ({os.environ['CUDA_VISIBLE_DEVICES']})")
        gs.fit(SliceDataset(train_dataset), y_train)

        logging.info(f"GridSearchCV - Best ROC-AUC Score in CV: {gs.best_score_}")
        logging.info(f"GridSearchCV - Best Params: {gs.best_params_}")
        logging.info("Evaluating best model as determined by Grid Search...")
        utils.evaluate_experiment(test_dataset, y_test, gs)
        
        end_time = time.monotonic()
        logging.info(f"Done. Finished in {timedelta(seconds=end_time - start_time)}")
        
        mlflow.log_text(LOG_STREAM.getvalue(), 'train.log')


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
