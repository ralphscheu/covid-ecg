import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import covidecg.models.train_eval_utils as utils
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
import mlflow
import sklearn.metrics
import torch
from io import StringIO
import random

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

    exp_name = Path(exp_config).stem.replace('exp-', '')
    experiment = mlflow.set_experiment(exp_name)
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        
        conf = utils.load_exp_model_conf(exp_config, model_config)

        X_train, X_test, y_train, y_test, y_encoder = utils.load_dataset(samples_list=conf['samples_list'], root_dir=conf['root_dir'])

        preprocessing = utils.build_preprocessing_pipeline(conf=conf)
        X_train, X_test = utils.preprocess_data(preprocessing, X_train, X_test)

        model = utils.build_model(conf, X_train, y_train)

        mlflow.sklearn.autolog()
        gs = GridSearchCV(model, conf['grid_search'], 
            scoring=sklearn.metrics.get_scorer('roc_auc'), 
            cv=int(conf['num_cv_folds']), refit=True, error_score='raise', verbose=4)
        gs.fit(X_train, y_train)

        logging.info(f"GridSearchCV - Best ROC-AUC Score in CV: {gs.best_score_}")
        logging.info(f"GridSearchCV - Best Params: {gs.best_params_}")
        logging.info("Evaluating best model as determined by Grid Search...")
        utils.evaluate_experiment(X_test, y_test, y_encoder, gs)


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
    
    try:
        run_experiment()
        mlflow.log_text(LOG_STREAM.getvalue(), 'train_evaluate.log')
    except BaseException as e:
        logging.critical(e)
        mlflow.log_text(LOG_STREAM.getvalue(), 'train_evaluate.log')
        mlflow.end_run(status='FAILED')
