import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import yaml
import covidecg.data.utils as data_utils
import covidecg.features.utils as feature_utils
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import sklearn.pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
import skorch
from datetime import datetime
from covidecg.models.mlp_models import MLP, CNN2D, CNN1D
import mlflow
import torch.nn as nn
import sklearn.metrics
from skorch.callbacks import EpochScoring, EarlyStopping
import sklearn.svm
import torch.optim
from typing import Tuple
import copy as cp
import torch
from io import StringIO
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import random
import torchinfo

# set seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def build_preprocessing_pipeline(conf, sampling_rate):
    """ Build data pre-processing pipeline for feature extraction """

    preprocessing = sklearn.pipeline.Pipeline([('clean_signal', data_utils.EcgSignalCleaner())])

    if conf['ecg_leads'] != 'all':
        preprocessing.steps.append(('select_ecg_lead', data_utils.EcgLeadSelector(conf['ecg_leads'])))

    if conf['features'] == 'plain_signal':
        pass
    elif conf['features'] == 'peaks':
        preprocessing.steps.append(('peaks_feats', feature_utils.EcgPeaksFeatsExtractor(sampling_rate=sampling_rate)))
    elif conf['features'] == 'intervals':
        preprocessing.steps.append(('intervals_feats', feature_utils.EcgIntervalsFeatsExtractor(sampling_rate=sampling_rate)))
    elif conf['features'] == 'peaks_intervals':
        preprocessing.steps.append(('peaks_intervals_feats',
                                    sklearn.pipeline.make_union(
                                        feature_utils.EcgPeaksFeatsExtractor(sampling_rate=sampling_rate),
                                        feature_utils.EcgIntervalsFeatsExtractor(sampling_rate=sampling_rate))))

    if conf['flatten_leads']:
        preprocessing.steps.append(('flatten_leads', FunctionTransformer(data_utils.flatten_leads)))

    return preprocessing


def load_dataset(samples_list, root_dir, ecg_type):
    """ Load dataset and encode targets to numerical """
    if ecg_type == 'stress':
        X, y = data_utils.load_stress_ecg_runs(samples_list, root_dir)
    elif ecg_type == 'rest':
        X, y = data_utils.load_rest_ecg_runs(samples_list, root_dir)
    else:
        raise Exception(f"Invalid ecg_type {ecg_type} in experiment configuration! (stress|rest)")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y).astype(np.int64)
    
    # shuffle X and y
    # assert len(X) == len(y)
    # p = np.random.permutation(len(X))
    # X, y = X[p], y[p]
    
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
    logging.info(f"")
    
    return X_train, y_train, X_test, y_test, label_encoder


def build_model(conf:dict, X:np.ndarray, y:np.ndarray, class_weight=None):
    """ Define model and optimizer according to experiment configuration """
    
    if conf['model'] == 'svm':
        if conf['svm_class_weight_balanced']:
            clf = sklearn.svm.SVC(kernel=conf['svm_kernel'], class_weight='balanced')
        else:
            clf = sklearn.svm.SVC(kernel=conf['svm_kernel'])

    elif conf['model'] == 'mlp':
        clf = skorch.NeuralNetClassifier(
            # model config
            module=MLP,
            module__input_size=X[0].size,
            # loss config
            criterion=nn.CrossEntropyLoss,
            criterion__weight=class_weight,
            optimizer=torch.optim.Adam,
            callbacks=[
                EpochScoring(scoring='roc_auc', lower_is_better=False),  # additional scores to observe
                EarlyStopping(patience=conf['early_stopping_patience'])  # Early Stopping based on validation loss
                ],
            max_epochs=conf['early_stopping_max_epochs'], device='cuda', iterator_train__shuffle=True,  # Shuffle training data on each epoch
        )

    elif conf['model'] == 'cnn2d':
        clf = skorch.NeuralNetClassifier(
            # model config
            module=CNN2D,
            # loss config
            criterion=nn.CrossEntropyLoss,
            criterion__weight=class_weight,
            optimizer=torch.optim.Adam,
            # hyperparams
            callbacks=[
                EpochScoring(scoring='roc_auc', lower_is_better=False),  # additional scores to observe
                EarlyStopping(patience=conf['early_stopping_patience'])  # Early Stopping based on validation loss
                ],
            max_epochs=conf['early_stopping_max_epochs'], device='cuda', iterator_train__shuffle=True  # Shuffle training data on each epoch
        )
        
    elif conf['model'] == 'cnn1d':
        clf = skorch.NeuralNetClassifier(
            # model config
            module=CNN1D,
            # loss config
            criterion=nn.CrossEntropyLoss,
            criterion__weight=class_weight,
            optimizer=torch.optim.Adam,
            # hyperparams
            callbacks=[
                EpochScoring(scoring='roc_auc', lower_is_better=False),  # additional scores to observe
                EarlyStopping(patience=conf['early_stopping_patience'])  # Early Stopping based on validation loss
                ],
            max_epochs=conf['early_stopping_max_epochs'], device='cuda', iterator_train__shuffle=True  # Shuffle training data on each epoch
        )

    return clf


def evaluate_experiment(y_true, y_pred, y_encoder, best_estimator, X_test):
    """ Compute scores, create figures and log all metrics to MLFlow """
    accuracy_score = sklearn.metrics.accuracy_score(y_true, y_pred)
    roc_auc_score = sklearn.metrics.roc_auc_score(y_true, y_pred)
    f1_score = sklearn.metrics.f1_score(y_true, y_pred)
    precision_score = sklearn.metrics.precision_score(y_true, y_pred)
    recall_score = sklearn.metrics.recall_score(y_true, y_pred)
    
    conf_matrix_figure = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=y_encoder.classes_, cmap='Blues', normalize='true').figure_
    
    roc_auc_curve_figure = sklearn.metrics.RocCurveDisplay.from_estimator(best_estimator, X_test, y_true).figure_
    
    logging.info("Logging metrics to MLFlow...")
    # # mlflow.log_metrics({'accuracy': accuracy_score, 'roc_auc': roc_auc_score, 'f1': f1_score, 'precision': precision_score, 'recall': recall_score})
    mlflow.log_figure(conf_matrix_figure, 'confusion_matrix.png')
    mlflow.log_figure(roc_auc_curve_figure, 'roc_auc_curve.png')
    
    loss_fig = plt.figure()
    plt.plot(best_estimator.history[:, 'train_loss'], label='train_loss')
    plt.plot(best_estimator.history[:, 'valid_loss'], label='valid_loss')
    plt.legend()
    mlflow.log_figure(loss_fig, 'loss.pdf')


###############################################

@click.command()
@click.option('--config-file', required=True, type=click.Path(exists=True))
def run_experiment(config_file):

    # read experiment config    
    with open(config_file) as _config_file_handle:
        conf = yaml.safe_load(_config_file_handle)

    logging.info("Loading dataset...")
    X_train, X_test, y_train, y_test, y_encoder = load_dataset(samples_list=conf['samples_list'], root_dir=conf['root_dir'], ecg_type=conf['ecg_type'])


    logging.info("Building pre-processing pipeline...")
    preprocessing = build_preprocessing_pipeline(
        conf=conf, 
        sampling_rate=int(os.environ['SAMPLING_RATE']))
    logging.info(f"Preprocessing steps: {preprocessing.named_steps}")


    logging.info("Preprocessing data...")
    X_train = preprocessing.fit_transform(X_train).astype(np.float32)
    X_test = preprocessing.fit_transform(X_test).astype(np.float32)
    logging.info(f"Shapes after preprocessing - X_train: {X_train.shape} - X_test: {X_test.shape}")


    logging.info(f"Shape before class imbalance mitigation: {X_train.shape}")
    logging.info(f"Class distribution before class imbalance mitigation: train {Counter(y_train)} | test {Counter(y_test)}")
    logging.info(f"Applying {conf['imbalance_mitigation']} for mitigating class imbalance...")
    
    class_weight = None
    if conf['imbalance_mitigation'] == 'criterion_weights':
        class_weight = torch.Tensor(sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y))
        
    elif conf['imbalance_mitigation'] == 'random_undersampling':
        shape_original = X_train.shape
        X_train, y_train = RandomUnderSampler().fit_resample(X_train.reshape(shape_original[0], -1), y_train)
        X_train = X_train.reshape(-1, *shape_original[1:])
        
    elif conf['imbalance_mitigation'] == 'smote':
        shape_original = X_train.shape
        X_train, y_train = SMOTE().fit_resample(X_train.reshape(shape_original[0], -1), y_train)
        X_train = X_train.reshape(-1, *shape_original[1:])
        
    else:
        raise f"Invalid option '{conf['imbalance_mitigation']}' for class imbalance mitigation"
    
    logging.info(f"Shape after mitigation: {X_train.shape}")
    logging.info(f"Class distribution after mitigation: train {Counter(y_train)} | test {Counter(y_test)}")


    mlflow.sklearn.autolog()
    experiment = mlflow.set_experiment(Path(config_file).stem)
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        
        with open(config_file) as _config_file_handle:
            mlflow.log_text(_config_file_handle.read(), 'experiment_config.yaml')


        logging.info("Building model...")
        clf = build_model(conf, X_train, y_train, class_weight=class_weight)


        logging.info("Fitting and evaluating model...")
        gs = sklearn.model_selection.GridSearchCV(
            clf, 
            conf['grid_search'], 
            scoring=sklearn.metrics.get_scorer('roc_auc'), 
            cv=int(conf['num_cv_folds']), 
            refit=True, 
            error_score='raise', 
            verbose=4)
        gs.fit(X_train, y_train)
        
        logging.info(f"GridSearchCV - Best ROC-AUC Score in CV: {gs.best_score_}")
        logging.info(f"GridSearchCV - Best Params: {gs.best_params_}")
        
        logging.info("Predicting best model as determined by Grid Search")
        # predict on X_test using model trained on all of X_train using best hyperparameters as determined by GridSearchCV
        y_pred = gs.predict(X_test)

        logging.info("Computing scores and creating figures...")
        evaluate_experiment(y_test, y_pred, y_encoder, gs.best_estimator_, X_test)
        
        
        # Store log in MLFlow
        mlflow.log_text(LOG_STREAM.getvalue(), 'train_evaluate.log')
        
        # Store model summary in MLFlow
        mlflow.log_text(str(torchinfo.summary(gs.best_estimator_.module, input_size=(gs.best_params_['batch_size'], *X_train.shape[1:]))), 'best_model_summary.txt')
        
        


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    
    LOG_STREAM = StringIO()
    logging.basicConfig(
        level='INFO', 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # log to stdout
            logging.StreamHandler(stream=LOG_STREAM)  # log to StringIO for storing in MLFlow
        ])

    run_experiment()
