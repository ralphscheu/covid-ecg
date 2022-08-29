import os
import click
import logging
import numpy as np
import yaml
import covidecg.data.utils as data_utils
import covidecg.features.utils as feature_utils
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, GroupShuffleSplit, GroupKFold
import sklearn.pipeline
from sklearn.preprocessing import FunctionTransformer
import skorch
from skorch.callbacks import EpochScoring, EarlyStopping
from covidecg.models.cnnseqpool import CNNSeqPool
from covidecg.models.cnnseqlstm import CNNSeqLSTM
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


def load_exp_model_conf(exp_conf_path, model_conf_path):
    with open(exp_conf_path) as f:
        conf_str = '### EXPERIMENT CONFIG ###\n'
        conf_str += f.read()
        conf = yaml.safe_load(conf_str)
    with open(model_conf_path) as f:
        conf_str += '\n\n### MODEL CONFIG ###\n'
        conf_str += f.read()
        conf = {**conf, **yaml.safe_load(conf_str)}  # combine exp and model configs into single dict
        
    mlflow.log_text(conf_str, 'exp_model_conf.yaml')
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


def build_preprocessing_pipeline(conf:dict, sampling_rate:int=500) -> sklearn.pipeline.Pipeline:
    """ Build data pre-processing pipeline for feature extraction """
    
    mlflow.set_tags({'features': conf['features']})

    preprocessing = sklearn.pipeline.Pipeline([('clean_signal', data_utils.EcgSignalCleaner())])

    if conf['model'] in ['vgg16', 'resnet18']:
        # apply image transforms specific to pretrained model
        if conf['model'] == 'vgg16':
            preprocessing.steps.append(('vgg16_image_preprocessing', data_utils.PretrainedModelApplyTransforms(torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES.transforms())))
        elif conf['model'] == 'resnet18':
            preprocessing.steps.append(('resnet18_image_preprocessing', FunctionTransformer(torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms())))

    preprocessing.steps.append(('flatten_leads', FunctionTransformer(data_utils.flatten_leads)))

    return preprocessing


def preprocess_data(pipeline, X_train, X_test):    
    logging.info("Preprocessing data...")
    logging.info(f"Preprocessing steps: {pipeline.named_steps}")
    X_train = pipeline.fit_transform(X_train.astype(np.float32))
    X_test = pipeline.fit_transform(X_test.astype(np.float32))
    logging.info(f"Shapes after preprocessing - X_train: {X_train.shape} - X_test: {X_test.shape}")
    return X_train, X_test


def build_model(conf:dict, dataset) -> imblearn.pipeline.Pipeline:
    """ Configure model and optimizer according to configuration files """
    
    logging.info("Building model...")
    # logging.info(f"Applying {conf['imbalance_mitigation']} to mitigate class imbalance in training data...")
    
    # mlflow.set_tags({'imbalance_mitigation': conf['imbalance_mitigation'], 
    #                  'model': conf['model']})

    # Compute class weights for loss function if desired
    if conf['imbalance_mitigation'] == 'criterion_weights':
        y_train = dataset.get_targets()
        class_weight = sklearn.utils.class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weight = torch.Tensor(class_weight)
    else:
        class_weight = None

    # params passed to skorch.NeuralNetClassifier that are the same for all skorch-based models
    skorch_clf_common_params = {
        'optimizer': torch.optim.Adam,
        'criterion': nn.CrossEntropyLoss,
        'criterion__weight': class_weight,
        'callbacks': [
            EpochScoring(name='valid_auc', scoring='roc_auc', on_train=False, lower_is_better=False),  # additional scores to observe
            EarlyStopping(patience=conf['early_stopping_patience'], monitor='train_loss')  # Early Stopping based on validation loss
            ],
        'max_epochs': conf['early_stopping_max_epochs'],
        'device': 'cuda',
        'iterator_train__shuffle': True,  # Shuffle training data on each epoch
        'train_split': None  # disable skorch-internal train/validation split since GridSearchCV already takes care of that
    }
    
    if conf['model'] == 'CNNSeqPool':
        clf = skorch.NeuralNetClassifier(module=CNNSeqPool, **skorch_clf_common_params)
    if conf['model'] == 'CNNSeqLSTM':
        clf = skorch.NeuralNetClassifier(module=CNNSeqLSTM, **skorch_clf_common_params)
    elif conf['model'] == 'cnn2d':
        clf = skorch.NeuralNetClassifier(module=CNN2D, **skorch_clf_common_params)
    elif conf['model'] == 'cnn2dimage':
        clf = skorch.NeuralNetClassifier(module=CNN2DImage, **skorch_clf_common_params)
    elif conf['model'] == 'cnn1d':
        clf = skorch.NeuralNetClassifier(module=CNN1D, **skorch_clf_common_params)
    elif conf['model'] == 'vgg16':
       clf = skorch.NeuralNetClassifier(module=PretrainedVGG16Classifier, **skorch_clf_common_params)
    elif conf['model'] == 'resnet18':
        clf = skorch.NeuralNetClassifier(module=PretrainedResNet18Classifier, **skorch_clf_common_params)
    
    # logging.info(f"Model pipeline:\n{pipe.named_steps}")

    return clf


def evaluate_experiment(test_dataset, y_test, gs:imblearn.pipeline.Pipeline) -> None:
    """ Compute scores, create figures and log all metrics to MLFlow """

    from covidecg.data.dataset import PAT_GROUP_TO_NUMERIC_TARGET
    
    # Generate Confusion Matrix
    conf_matrix_fig = sklearn.metrics.ConfusionMatrixDisplay.from_estimator(gs, test_dataset, y_test, display_labels=PAT_GROUP_TO_NUMERIC_TARGET.values(), cmap='Blues', normalize='true').figure_
    mlflow.log_figure(conf_matrix_fig, 'confusion_matrix.png')
    
    # Generate ROC curve
    roc_curve_fig = sklearn.metrics.RocCurveDisplay.from_estimator(gs, test_dataset, y_test).figure_
    mlflow.log_figure(roc_curve_fig, 'roc_curve.png')
    
    # Generate train&valid Loss curve
    loss_fig = plt.figure()
    plt.plot(gs.best_estimator_.history[:, 'train_loss'], label='train loss')
    plt.legend()
    mlflow.log_figure(loss_fig, 'train_loss.png')
    
    # Store model summary of best model in MLFlow
    mlflow.log_text(str(gs.best_estimator_), 'model_topology.txt')
    # mlflow.log_text(str(torchinfo.summary(gs.best_estimator_.module, input_size=(gs.best_params_['batch_size'], *X_train.shape[1:]))), 'best_model_summary.txt')
