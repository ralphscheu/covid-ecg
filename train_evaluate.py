# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
import xml.etree.cElementTree as ET
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import yaml
import covidecg.data.utils as data_utils
import covidecg.features.utils as feature_utils
import sklearn.preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
import sklearn.pipeline
from sklearn.preprocessing import FunctionTransformer
import skorch
from datetime import datetime
from covidecg.models.mlp_models import MLP, CNN2D
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

np.random.seed(0)


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
    label_encoder = sklearn.preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y).astype(np.int64)
    return X, y, label_encoder


def build_model(conf:dict, X:np.ndarray, y:np.ndarray):
    """ Define model and optimizer according to experiment configuration """
    
    if conf['model'] == 'svm':
        if conf['svm_class_weight_balanced']:
            clf = sklearn.svm.SVC(kernel=conf['svm_kernel'], class_weight='balanced')
        else:
            clf = sklearn.svm.SVC(kernel=conf['svm_kernel'])

    elif conf['model'] == 'mlp':
        class_weight = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        logging.debug(f"class_weight: {class_weight}")
        class_weight = torch.Tensor(class_weight)
        clf = skorch.NeuralNetClassifier(
            # model config
            module=MLP,
            module__input_size=X[0].size,
            module__hidden_size=int(conf['hidden_size']),
            # loss config
            criterion=nn.CrossEntropyLoss,
            criterion__weight=class_weight,
            optimizer=torch.optim.Adam,
            # hyperparams
            batch_size=int(conf['batch_size']),
            lr=float(conf['lr']),
            callbacks=[
                EpochScoring(scoring='roc_auc', lower_is_better=False),  # additional scores to observe
                EarlyStopping(patience=conf['early_stopping_patience'])  # Early Stopping based on validation loss
                ],
            max_epochs=conf['early_stopping_max_epochs'], device='cuda', iterator_train__shuffle=True,  # Shuffle training data on each epoch
        )

    elif conf['model'] == 'cnn2d':
        class_weight = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        logging.debug(f"class_weight: {class_weight}")
        class_weight = torch.Tensor(class_weight)
        clf = skorch.NeuralNetClassifier(
            # model config
            module=CNN2D,
            module__dense_hidden_size=int(conf['dense_hidden_size']),
            # loss config
            criterion=nn.CrossEntropyLoss,
            criterion__weight=class_weight,
            optimizer=torch.optim.Adam,
            # hyperparams
            batch_size=int(conf['batch_size']),
            lr=float(conf['lr']),
            callbacks=[
                EpochScoring(scoring='roc_auc', lower_is_better=False),  # additional scores to observe
                EarlyStopping(patience=conf['early_stopping_patience'])  # Early Stopping based on validation loss
                ],
            max_epochs=conf['early_stopping_max_epochs'], device='cuda', iterator_train__shuffle=True,  # Shuffle training data on each epoch
        )
    return clf


def cross_val_predict(model, kfold : KFold, X : np.array, y : np.array) -> Tuple[np.array, np.array, np.array]:
    """ Fit models using KFold cross-validation """
    model_ = cp.deepcopy(model)
    num_classes = len(np.unique(y))
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, num_classes]) 

    for fold_i, (train_ndx, test_ndx) in enumerate(kfold.split(X, y)):

        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]
        actual_classes = np.append(actual_classes, test_y)
        model_.fit(train_X, train_y)
        predicted_classes = np.append(predicted_classes, model_.predict(test_X))

        try:
            predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), num_classes), dtype=float), axis=0)

    return actual_classes, predicted_classes, predicted_proba


def evaluate_experiment(y_true, y_pred, y_encoder):
    """ Compute scores, create figures and log all metrics to MLFlow """
    accuracy_score = sklearn.metrics.accuracy_score(y_true, y_pred)
    roc_auc_score = sklearn.metrics.roc_auc_score(y_true, y_pred)
    f1_score = sklearn.metrics.f1_score(y_true, y_pred)
    precision_score = sklearn.metrics.precision_score(y_true, y_pred)
    recall_score = sklearn.metrics.recall_score(y_true, y_pred)
    conf_matrix_figure = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=y_encoder.classes_).figure_
    roc_auc_curve_figure = sklearn.metrics.RocCurveDisplay.from_predictions(y_true, y_pred).figure_

    logging.info("Logging metrics to MLFlow...")
    mlflow.log_metrics({'accuracy': accuracy_score, 'roc_auc': roc_auc_score, 'f1': f1_score, 'precision': precision_score, 'recall': recall_score})
    mlflow.log_figure(conf_matrix_figure, 'confusion_matrix.pdf')
    mlflow.log_figure(roc_auc_curve_figure, 'roc_auc_curve.pdf')


@click.command()
@click.option('--config-file', required=True, type=click.Path(exists=True))
def main(config_file):
    conf = yaml.safe_load(open(config_file))
    logging.info(f"Experiment configuration: {conf}")
    exp_name = Path(config_file).stem

    experiment = mlflow.set_experiment(exp_name)
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.log_params(conf)

        logging.info("Loading dataset...")
        X, y, y_encoder = load_dataset(samples_list=conf['samples_list'], 
                                       root_dir=conf['root_dir'], 
                                       ecg_type=conf['ecg_type'])

        logging.info("Building pre-processing pipeline...")
        preprocessing = build_preprocessing_pipeline(
            conf=conf, 
            sampling_rate=int(os.environ['SAMPLING_RATE']))

        logging.info("Preprocessing data...")
        X = preprocessing.fit_transform(X).astype(np.float32)
        logging.info(f"Shapes after preprocessing - X: {X.shape} - y: {len(y)}")

        logging.info("Building model...")
        clf = build_model(conf, X, y)

        logging.info("Fitting and evaluating model...")
        kfold_ = StratifiedKFold(n_splits=int(conf['num_cv_folds']), shuffle=True)
        y_true, y_pred, y_pred_proba = cross_val_predict(model=clf, kfold=kfold_, X=X, y=y)

        logging.info("Computing scores and creating figures...")
        evaluate_experiment(y_true, y_pred, y_encoder)
        
        # store logging output in MLFlow
        mlflow.log_text(LOG_STREAM.getvalue(), 'train_evaluate.log')


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

    main()
