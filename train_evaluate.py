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
from covidecg.models.mlp_models import MLP
import mlflow
import torch.nn as nn
import sklearn.metrics
from skorch.callbacks import EpochScoring, EarlyStopping
import sklearn.svm
import torch.optim
from typing import Tuple
import copy as cp

np.random.seed(0)


def cross_val_predict(model, kfold : KFold, X : np.array, y : np.array) -> Tuple[np.array, np.array, np.array]:

    model_ = cp.deepcopy(model)
    
    no_classes = len(np.unique(y))
    
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes]) 

    for fold_i, (train_ndx, test_ndx) in enumerate(kfold.split(X, y)):

        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

        actual_classes = np.append(actual_classes, test_y)

        model_.fit(train_X, train_y)
        predicted_classes = np.append(predicted_classes, model_.predict(test_X))

        try:
            predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)

        # log training metrics if model is a PyTorch (Skorch) model
        try:
            for epoch_i, _epoch_history in enumerate( model_.steps[-1][1].history ):
                for _metric in ['train_loss', 'valid_loss', 'valid_acc', 'roc_auc']:
                    mlflow.log_metric(f"{_metric}_fold{fold_i:2}", _epoch_history[_metric], step=epoch_i)
        except:
            pass

    return actual_classes, predicted_classes, predicted_proba


@click.command()
@click.option('--config-file', required=True, type=click.Path(exists=True))
def main(config_file):
    conf = yaml.safe_load(open(config_file))
    logging.info(f"Experiment configuration: {conf}")
    exp_name = Path(config_file).stem
    
    experiment = mlflow.set_experiment(exp_name)
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        
        logging.info("Loading dataset...")
        mlflow.log_param('ecg_type', conf['data']['ecg_type'])
        if conf['data']['ecg_type'] == 'stress':
            X, y = data_utils.load_stress_ecg_runs(conf['data']['samples_list'], conf['data']['root_dir'])
        elif conf['data']['ecg_type'] == 'rest':
            X, y = data_utils.load_rest_ecg_runs(conf['data']['samples_list'], conf['data']['root_dir'])
        else:
            raise Exception("Invalid ecg_type in experiment configuration! (stress|rest)")
        label_encoder = sklearn.preprocessing.LabelEncoder()
        y = label_encoder.fit_transform(y).astype(np.int64)
        ###
        
        logging.info("Building pre-processing pipeline...")
        mlflow.log_param('ecg_leads', conf['ecg_leads'])
        if conf['features'] == 'plain_signal':
            preprocessing = sklearn.pipeline.make_pipeline(
                                    data_utils.EcgSignalCleaner(),
                                    data_utils.EcgLeadSelector(conf['ecg_leads']),
                                    FunctionTransformer(data_utils.flatten_leads))
        elif conf['features'] == 'peaks':
            preprocessing = sklearn.pipeline.make_pipeline(
                                    data_utils.EcgSignalCleaner(),
                                    data_utils.EcgLeadSelector(conf['ecg_leads']),
                                    feature_utils.EcgPeaksFeatsExtractor(sampling_rate=int(os.environ['SAMPLING_RATE'])))
        elif conf['features'] == 'intervals':
            preprocessing = sklearn.pipeline.make_pipeline(
                                    data_utils.EcgSignalCleaner(),
                                    data_utils.EcgLeadSelector(conf['ecg_leads']),
                                    feature_utils.EcgIntervalsFeatsExtractor(sampling_rate=int(os.environ['SAMPLING_RATE'])))
        elif conf['features'] == 'peaks_intervals':
            preprocessing = sklearn.pipeline.make_pipeline(
                                    data_utils.EcgSignalCleaner(),
                                    data_utils.EcgLeadSelector(conf['ecg_leads']),
                                    sklearn.pipeline.make_union(
                                        feature_utils.EcgPeaksFeatsExtractor(sampling_rate=int(os.environ['SAMPLING_RATE']))),
                                        feature_utils.EcgIntervalsFeatsExtractor(sampling_rate=int(os.environ['SAMPLING_RATE'])))
                                    
        ###
        
        logging.info("Building model...")
        mlflow.log_param('model', conf['model'])
        if conf['model'] == 'svm_linear':
            svm_class_weight = None
            mlflow.log_param('svm_class_weight_balanced', conf['svm_class_weight_balanced'])
            if conf['svm_class_weight_balanced'] == 'yes':
                svm_class_weight = 'balanced'
            clf = sklearn.svm.SVC(kernel='linear', class_weight=svm_class_weight)
        elif conf['model'] == 'svm_poly':
            svm_class_weight = None
            mlflow.log_param('svm_class_weight_balanced', conf['svm_class_weight_balanced'])
            if conf['svm_class_weight_balanced'] == 'yes':
                svm_class_weight = 'balanced'
            clf = sklearn.svm.SVC(kernel='poly', class_weight=svm_class_weight)
        elif conf['model'] == 'mlp':
            mlflow.log_param('momentum', float(conf['momentum']))
            mlflow.log_param('batch_size', int(conf['batch_size']))
            mlflow.log_param('criterion', 'CrossEntropyLoss')
            mlflow.log_param('optimizer', 'Adam')
            mlflow.log_param('learning_rate', float(conf['lr']))
            mlflow.log_param('batch_size', int(conf['batch_size']))
            mlflow.log_param('hidden_size', int(conf['hidden_size']))
        
            clf = skorch.NeuralNetClassifier(
                # model config
                module=MLP,
                module__input_size=preprocessing.fit_transform(X[[0]]).size,  # number of features in flattened sample
                module__hidden_size=int(conf['hidden_size']),
                
                # loss config
                criterion=nn.CrossEntropyLoss,
                # criterion__weight=class_weights,
                optimizer=torch.optim.Adam,
                # optimizer__momentum=float(conf['momentum']),
                
                # hyperparams
                batch_size=int(conf['batch_size']),
                lr=float(conf['lr']),
                
                callbacks=[
                    # additional scores to observe
                    EpochScoring(scoring='roc_auc', lower_is_better=False),
                    # Early Stopping based on validation loss
                    EarlyStopping(patience=conf['early_stopping_patience'])
                    ],
                max_epochs=conf['early_stopping_max_epochs'],
                iterator_train__shuffle=True,  # Shuffle training data on each epoch
                device='cuda' # Train on GPU
            )
        ###


        logging.info("Preprocessing data...")
        X = preprocessing.fit_transform(X)
        X = X.astype(np.float32)
        logging.info(f"Shapes after preprocessing - X: {X.shape} - y: {len(y)}")
        
        logging.info("Fitting and evaluating model...")
        mlflow.log_param('num_cv_folds', int(conf['num_cv_folds']))
        kfold_ = StratifiedKFold(n_splits=int(conf['num_cv_folds']), shuffle=True)
        y_true, y_pred, y_pred_proba = cross_val_predict(clf, kfold_, X, y)
        ###
        
        logging.info("Computing scores and creating figures...")
        accuracy_score = sklearn.metrics.accuracy_score(y_true, y_pred)
        roc_auc_score = sklearn.metrics.roc_auc_score(y_true, y_pred)
        f1_score = sklearn.metrics.f1_score(y_true, y_pred)
        precision_score = sklearn.metrics.precision_score(y_true, y_pred)
        recall_score = sklearn.metrics.recall_score(y_true, y_pred)
        
        conf_matrix_figure = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=label_encoder.classes_).figure_
        roc_auc_curve_figure = sklearn.metrics.RocCurveDisplay.from_predictions(y_true, y_pred, ).figure_
        ###
    
        logging.info("Logging metrics to MLFlow...")
        mlflow.log_metric('accuracy', accuracy_score)
        mlflow.log_metric('roc_auc', roc_auc_score)
        mlflow.log_metric('f1', f1_score)
        mlflow.log_metric('precision', precision_score)
        mlflow.log_metric('recall', recall_score)
        mlflow.log_figure(conf_matrix_figure, 'confusion_matrix.pdf')
        mlflow.log_figure(roc_auc_curve_figure, 'roc_auc_curve.pdf')
        ###


if __name__ == '__main__':
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # timestamp = datetime.utcnow().strftime('%Y%m%dT%H%S.%f')[:-3]
    # output_dir = os.path.join(os.environ['PROJECT_ROOT'], 'exp_results', f"{timestamp}__{exp_name}")

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=os.environ['LOG_LEVEL'], format=log_fmt)  #handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(output_dir, 'train_evaluate.log'))]

    main()
