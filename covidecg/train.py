import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import covidecg.models.train_eval_utils as utils
import covidecg.data.utils as data_utils
from covidecg.data.dataset import EcgImageDataset, EcgImageSequenceDataset, ConcatEcgDataset
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
import mlflow
import sklearn.metrics
import torch
from io import StringIO
import random
import time
from datetime import timedelta

# Ensure reproducibility
RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.use_deterministic_algorithms(True, warn_only=True)


@click.command()
@click.option('--exp-config', required=True, type=click.Path(exists=True))
@click.option('--model-config', required=True, type=click.Path(exists=True))
def run_experiment(exp_config, model_config):
    start_time = time.monotonic()
    exp_name = Path(exp_config).stem.replace('exp-', '')
    experiment = mlflow.set_experiment(exp_name)
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        conf = utils.load_exp_model_conf(exp_config, model_config)

        # Load dataset
        ds_postcovid = EcgImageSequenceDataset(
            recs_list='./data/interim/mmc_recs_stress_postcovid.csv',
            recs_dir='./data/processed/mmc_recs_postcovid',
            min_length=5000, max_length=5000)
        ds_ctrl = EcgImageSequenceDataset(
            recs_list='./data/interim/mmc_recs_stress_ctrl.csv',
            recs_dir='./data/processed/mmc_recs_ctrl',
            min_length=5000, max_length=5000)
        dataset = ConcatEcgDataset([ds_postcovid, ds_ctrl])

        # Split into train/valid and test portion
        train_dataset, test_dataset, y_train, y_test = utils.get_dataset_splits(dataset, test_size=0.2, random_state=RANDOM_SEED)
        
        # X_train = np.stack(list(train_dataset), axis=0)
        # logging.info(f"X_train.shape: {X_train.shape}")

        clf = utils.build_model(conf, train_dataset)

        mlflow.sklearn.autolog()
        gs = GridSearchCV(clf, conf['grid_search'],
            scoring=sklearn.metrics.get_scorer('roc_auc'),
            cv=int(conf['num_cv_folds']), refit=True, error_score='raise', verbose=4)
        gs.fit(train_dataset, y=y_train)

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
        level=os.getenv('LOG_LEVEL', default='INFO'), 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # log to stdout
            logging.StreamHandler(stream=LOG_STREAM)  # log to StringIO object for storing in MLFlow
        ])
    run_experiment()
