import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import yaml
import covidecg.data.utils as data_utils
import covidecg.features.utils as feature_utils
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV, GroupShuffleSplit, GroupKFold
import sklearn.pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
import skorch
from skorch.callbacks import EpochScoring, EarlyStopping
from covidecg.models.models import MLP, CNN2D, CNN1D
import mlflow
import torch.nn as nn
import sklearn.metrics
import sklearn.svm
import torch.optim
import torch
from io import StringIO
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import random
# import torchinfo
import imblearn.pipeline
import covidecg.models.train_eval_utils as train_eval_utils
# from dask.distributed import Client
# from joblib import parallel_backend

# client = Client('127.0.0.1:8786')

# ensure reproducibility
RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.use_deterministic_algorithms(True)


@click.command()
@click.option('--exp-config', required=True, type=click.Path(exists=True))
@click.option('--model-config', required=True, type=click.Path(exists=True))
def run_experiment(exp_config, model_config):

    # read experiment config
    with open(exp_config) as f:
        conf_str = '### EXPERIMENT CONFIG ###\n'
        conf_str += f.read()
        conf = yaml.safe_load(conf_str)
    with open(model_config) as f:
        conf_str += '\n\n### MODEL CONFIG ###\n'
        conf_str += f.read()
        conf = {**conf, **yaml.safe_load(conf_str)}  # combine exp and model configs into single dict

    logging.info("Loading dataset...")
    X_train, X_test, y_train, y_test, y_encoder = train_eval_utils.load_dataset(samples_list=conf['samples_list'], root_dir=conf['root_dir'], ecg_type=conf['ecg_type'])

    ###

    preprocessing = train_eval_utils.build_preprocessing_pipeline(conf=conf, sampling_rate=int(os.environ['SAMPLING_RATE']))
    logging.info("Preprocessing data...")
    logging.info(f"Preprocessing steps: {preprocessing.named_steps}")
    X_train = preprocessing.fit_transform(X_train.astype(np.float32))
    X_test = preprocessing.fit_transform(X_test.astype(np.float32))
    logging.info(f"Shapes after preprocessing - X_train: {X_train.shape} - X_test: {X_test.shape}")

    ###

    mlflow.sklearn.autolog()
    experiment = mlflow.set_experiment(Path(exp_config).stem[4:])
    with mlflow.start_run(experiment_id=experiment.experiment_id, 
                          tags={
                              'features': conf['features'], 
                              'imbalance_mitigation': conf['imbalance_mitigation'], 
                              'ecg_type': conf['ecg_type'],
                              'ecg_leads': conf['ecg_leads'],
                              'model': conf['model']}):

        mlflow.log_text(conf_str, 'experiment_config.yaml')

        logging.info("Building model...")
        model = train_eval_utils.build_model(conf, X_train, y_train)

        logging.info("Fitting and evaluating model...")
        gs = GridSearchCV(model, conf['grid_search'], 
            scoring=sklearn.metrics.get_scorer('roc_auc'), 
            cv=int(conf['num_cv_folds']), refit=True, error_score='raise', verbose=4)

        # with parallel_backend('dask'):
        gs.fit(X_train, y_train)

        logging.info(f"GridSearchCV - Best ROC-AUC Score in CV: {gs.best_score_}")
        logging.info(f"GridSearchCV - Best Params: {gs.best_params_}")
        logging.info("Evaluating best model as determined by Grid Search...")
        train_eval_utils.evaluate_experiment(X_test, y_test, y_encoder, gs)

        # Store log in MLFlow
        mlflow.log_text(LOG_STREAM.getvalue(), 'train_evaluate.log')

        # Store model summary in MLFlow
        mlflow.log_text(str(gs.best_estimator_.steps[-1][1].module_), 'model_topology.txt')
        # mlflow.log_text(str(torchinfo.summary(gs.best_estimator_.module, input_size=(gs.best_params_['batch_size'], *X_train.shape[1:]))), 'best_model_summary.txt')


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    LOG_STREAM = StringIO()
    logging.basicConfig(
        level=os.getenv('LOG_LEVEL', default='INFO'), 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # log to stdout
            logging.StreamHandler(stream=LOG_STREAM)  # log to StringIO object for storing in MLFlow
        ])
    run_experiment()
