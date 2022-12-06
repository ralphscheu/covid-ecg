import os
import click
import logging
import numpy as np
import yaml
import covidecg.data.utils as data_utils
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, GroupShuffleSplit, GroupKFold
import sklearn.pipeline
from sklearn.preprocessing import FunctionTransformer
import skorch
from skorch.callbacks import EpochScoring, EarlyStopping, MlflowLogger, LRScheduler, ProgressBar
from covidecg.models.cnn3dseq import *
from covidecg.models.cnn2dseq import *
import mlflow
import torch.nn as nn
import sklearn.metrics
import sklearn.svm
import torch.optim
import torch
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
import imblearn.pipeline
import torchvision.models
import matplotlib.pyplot as plt
from typing import Tuple
# import torchinfo
from skorch.helper import SliceDataset
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings('ignore')


def load_exp_model_conf(model_conf_path):
    with open(model_conf_path) as f:
        conf_str = '\n\n### MODEL CONFIG ###\n'
        conf_str += f.read()
        conf = yaml.safe_load(conf_str)  # combine exp and model configs into single dict
    mlflow.log_text(conf_str, 'conf.yaml')
    return conf


def build_model(model_name:str, conf:dict, dataset):
    """ Configure model and optimizer according to configuration files """
    
    logging.debug("Building model...")

    # Compute class weights for loss function to mitigate smaller class imbalances after subject-aware train/test split
    y_train = dataset.targets
    class_weight = sklearn.utils.class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight = torch.Tensor(class_weight)
    
    
    def pad_batch(batch):
        """ Pad sequence examples in batch to common length """
        im, label = list(zip(*batch))
        padded_batch = pad_sequence(im, batch_first=True, padding_value=0.0)
        label = list(label)
        label = np.array(label, dtype=np.uint8)
        label = torch.from_numpy(label).type(torch.LongTensor)
        return padded_batch, label

    # params passed to skorch.NeuralNetClassifier that are the same for all skorch-based models
    skorch_clf_common_params = {
        'optimizer': torch.optim.Adam,
        'criterion': nn.CrossEntropyLoss,
        'criterion__weight': class_weight,
        'callbacks': [
            EpochScoring(name='train_auc', scoring='roc_auc', on_train=True, lower_is_better=False),  # additional scores to observe
            EpochScoring(name='train_accuracy', scoring='accuracy', on_train=True, lower_is_better=False),  # additional scores to observe
            EarlyStopping(patience=conf['early_stopping_patience'], monitor='train_loss'),  # Early Stopping based on train loss
            ProgressBar(),
            ],
        'max_epochs': conf['early_stopping_max_epochs'],
        'device': 'cuda',
        'iterator_train__collate_fn': pad_batch,
        'iterator_train__shuffle': True,  # Shuffle training data on each epoch
        'iterator_valid__collate_fn': pad_batch,
        'iterator_valid__shuffle': False, # not necessary to shuffle validation data each epoch
        'train_split': None  # disable skorch-internal train/validation split since GridSearchCV already takes care of that
    }
    
    try:
        clf = skorch.NeuralNetClassifier(module=eval(model_name), **skorch_clf_common_params)
    except:
        raise Exception(f'Invalid model name "{model_name}!')
    
    print(f"Model: {model_name}")
    mlflow.log_param('model', model_name)
    
    return clf

from skorch.helper import SliceDataset

def evaluate_experiment(test_dataset, y_test, best_model, log_to_mlflow=True) -> None:
    """ Compute scores, create figures and log all metrics to MLFlow """
    
    y_pred = best_model.predict(SliceDataset(test_dataset))
    y_pred_proba = best_model.predict_proba(SliceDataset(test_dataset))
    
    roc_auc = sklearn.metrics.roc_auc_score(y_test, y_pred_proba[:, 1])
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    bal_accuracy = sklearn.metrics.balanced_accuracy_score(y_test, y_pred)
    f1 = sklearn.metrics.f1_score(y_test, y_pred)
    precision = sklearn.metrics.precision_score(y_test, y_pred)
    recall = sklearn.metrics.recall_score(y_test, y_pred)
    
    metrics_dict = {
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'bal_accuracy': bal_accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    # Generate Confusion Matrix
    conf_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
    conf_matrix_fig = sklearn.metrics.ConfusionMatrixDisplay.from_estimator(best_model, SliceDataset(test_dataset), y_test, display_labels=test_dataset.classes, cmap='Blues', normalize='true').figure_
    
    # ROC Curve
    roc_curve_fig = sklearn.metrics.RocCurveDisplay.from_estimator(best_model, SliceDataset(test_dataset), y_test).figure_
    
    # Precision-Recall Curve
    pr_curve_fig = sklearn.metrics.PrecisionRecallDisplay.from_estimator(best_model, SliceDataset(test_dataset), y_test).figure_
    
    # Generate train&valid Loss curve
    loss_fig = plt.figure()
    plt.plot(best_model.history[:, 'train_loss'], label='train loss')
    plt.legend()
    
    figures_dict = { 'conf_matrix': conf_matrix_fig, 'roc_curve': roc_curve_fig, 'pr_curve': pr_curve_fig, 'loss': loss_fig }
    
    if log_to_mlflow:
        # Log metrics and save figures into MLFlow
        
        # Save model history
        best_model.history.to_file('/tmp/covidecg_best_model_history.json')
        
        mlflow.log_metrics(metrics_dict)
        
        mlflow.log_text(str(conf_matrix), 'confusion_matrix.txt')
        mlflow.log_figure(conf_matrix_fig, 'confusion_matrix.png')
        mlflow.log_figure(roc_curve_fig, 'roc_curve.png')
        mlflow.log_figure(pr_curve_fig, 'precision_recall_curve.png')
        mlflow.log_figure(loss_fig, 'train_loss.png')
        
        mlflow.log_artifact('/tmp/covidecg_best_model_history.json')
        
        mlflow.sklearn.log_model(best_model, 'best_model')
        mlflow.log_text(str(best_model), 'model_topology.txt')
        
    plt.clf()
        
    return metrics_dict, conf_matrix, figures_dict, best_model.history
