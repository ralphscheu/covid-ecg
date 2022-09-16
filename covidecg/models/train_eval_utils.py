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
from covidecg.models.cnn3dseq import CNN3DSeqMeanStdPool, CNN3DSeqLSTM
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


def load_exp_model_conf(model_conf_path):
    with open(model_conf_path) as f:
        conf_str = '\n\n### MODEL CONFIG ###\n'
        conf_str += f.read()
        conf = yaml.safe_load(conf_str)  # combine exp and model configs into single dict
    mlflow.log_text(conf_str, 'conf.yaml')
    return conf


def get_dataset_splits(dataset, test_size=0.2, random_state=0):
    train_idx, test_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=test_size,
        shuffle=True,
        stratify=dataset.get_targets(),
        random_state=random_state)
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    y_train = dataset.get_targets()[train_idx]
    y_test = dataset.get_targets()[test_idx]
    
    cnt_train = Counter(y_train)
    cnt_test = Counter(y_test)
    logging.info(f"Training set:\t{cnt_train[0]} ctrl - {cnt_train[1]} postcovid")
    logging.info(f"Test set:\t{cnt_test[0]} ctrl - {cnt_test[1]} postcovid")
    
    ###### ONLY FOR DEVELOPMENT
    # train_dataset = torch.utils.data.Subset(train_dataset, range(5))
    # test_dataset = torch.utils.data.Subset(test_dataset, range(3))
    # y_train = y_train[range(5)]
    # y_test = y_test[range(3)]
    ######
    
    train_dataset = SliceDataset(train_dataset, idx=0)
    test_dataset = SliceDataset(test_dataset, idx=0)
    
    logging.info(f"Train on {len(train_dataset)}, test on {len(test_dataset)} samples")
    return train_dataset, test_dataset, y_train, y_test



def build_model(model_name:str, conf:dict, dataset) -> imblearn.pipeline.Pipeline:
    """ Configure model and optimizer according to configuration files """
    
    logging.debug("Building model...")

    # Compute class weights for loss function if desired
    # if conf['imbalance_mitigation'] == 'criterion_weights':
    #     y_train = dataset.get_targets()
    #     class_weight = sklearn.utils.class_weight.compute_class_weight(
    #         class_weight='balanced', classes=np.unique(y_train), y=y_train)
    #     class_weight = torch.Tensor(class_weight)
    # else:
    class_weight = None
    
    
    def pad_batch(batch):
        im, label = list(zip(*batch))
        from torch.nn.utils.rnn import pad_sequence
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
            EarlyStopping(patience=conf['early_stopping_patience'], monitor='train_loss'),  # Early Stopping based on train loss
            ProgressBar(),
            ],
        'max_epochs': conf['early_stopping_max_epochs'],
        'device': 'cuda',
        'iterator_train__collate_fn': pad_batch,
        'iterator_train__shuffle': True,  # Shuffle training data on each epoch
        'iterator_valid__collate_fn': pad_batch,
        'iterator_valid__shuffle': False,
        'train_split': None  # disable skorch-internal train/validation split since GridSearchCV already takes care of that
    }
    
    logging.info(f"Model: {model_name}")
    mlflow.set_tag('model', model_name)
    
    if model_name == 'CNN3DSeqPool':
        clf = skorch.NeuralNetClassifier(module=CNN3DSeqMeanStdPool, **skorch_clf_common_params)
    if model_name == 'CNN3DSeqLSTM':
        clf = skorch.NeuralNetClassifier(module=CNN3DSeqLSTM, **skorch_clf_common_params)
    elif model_name == 'cnn2d':
        clf = skorch.NeuralNetClassifier(module=CNN2D, **skorch_clf_common_params)
    elif model_name == 'cnn2dimage':
        clf = skorch.NeuralNetClassifier(module=CNN2DImage, **skorch_clf_common_params)
    elif model_name == 'cnn1d':
        clf = skorch.NeuralNetClassifier(module=CNN1D, **skorch_clf_common_params)
    elif model_name == 'vgg16':
       clf = skorch.NeuralNetClassifier(module=PretrainedVGG16Classifier, **skorch_clf_common_params)
    elif model_name == 'resnet18':
        clf = skorch.NeuralNetClassifier(module=PretrainedResNet18Classifier, **skorch_clf_common_params)
    
    # logging.info(f"Model pipeline:\n{pipe.named_steps}")

    return clf

from skorch.helper import SliceDataset

def evaluate_experiment(test_dataset, y_test, gs:imblearn.pipeline.Pipeline) -> None:
    """ Compute scores, create figures and log all metrics to MLFlow """
    
    y_pred = gs.predict(SliceDataset(test_dataset))
    roc_auc = sklearn.metrics.roc_auc_score(y_test, y_pred)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    bal_accuracy = sklearn.metrics.balanced_accuracy_score(y_test, y_pred)
    f1 = sklearn.metrics.f1_score(y_test, y_pred)
    precision = sklearn.metrics.precision_score(y_test, y_pred)
    recall = sklearn.metrics.recall_score(y_test, y_pred)
    mlflow.log_metrics({
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'bal_accuracy': bal_accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    })
    
    # Generate Confusion Matrix
    conf_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
    mlflow.log_text(str(conf_matrix), 'confusion_matrix.txt')
    conf_matrix_fig = sklearn.metrics.ConfusionMatrixDisplay.from_estimator(gs, SliceDataset(test_dataset), y_test, display_labels=test_dataset.classes, cmap='Blues', normalize='true').figure_
    mlflow.log_figure(conf_matrix_fig, 'confusion_matrix.png')
    
    # Generate ROC curve
    roc_curve_fig = sklearn.metrics.RocCurveDisplay.from_estimator(gs, SliceDataset(test_dataset), y_test).figure_
    mlflow.log_figure(roc_curve_fig, 'roc_curve.png')
    
    # Generate train&valid Loss curve
    loss_fig = plt.figure()
    plt.plot(gs.best_estimator_.history[:, 'train_loss'], label='train loss')
    plt.legend()
    mlflow.log_figure(loss_fig, 'train_loss.png')
    
    mlflow.sklearn.log_model(gs.best_estimator_, 'best_model')
    mlflow.log_text(str(gs.best_estimator_), 'model_topology.txt')
    # mlflow.log_text(str(torchinfo.summary(gs.best_estimator_.module, input_size=(gs.best_params_['batch_size'], *X_train.shape[1:]))), 'best_model_summary.txt')
