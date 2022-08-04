import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import yaml
import covidecg.data.utils as data_utils
import covidecg.features.utils as feature_utils
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, GroupShuffleSplit, GroupKFold
import sklearn.pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
import skorch
from skorch.callbacks import EpochScoring, EarlyStopping
from covidecg.models.models import MLP, CNN2D, CNN1D, PretrainedVGG16Classifier, PretrainedResNet18Classifier
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


def build_preprocessing_pipeline(conf:dict, sampling_rate:int=500) -> sklearn.pipeline.Pipeline:
    """ Build data pre-processing pipeline for feature extraction """
    
    mlflow.set_tags({'features': conf['features']})

    preprocessing = sklearn.pipeline.Pipeline([('clean_signal', data_utils.EcgSignalCleaner())])

    if conf['features'] == 'plain_signal':
        # no need to do anything - use signal value arrays as-is
        pass
    if conf['features'] == 'signal_image':
        # convert from array of signal values to 2D grayscale image of signal curve
        preprocessing.steps.append(('convert_signal_to_image', feature_utils.EcgSignalToImageConverter(height=conf['signal_image_height'], width=conf['signal_image_width'])))
    elif conf['features'] == 'lfcc':
        preprocessing.steps.append(('extract_lfcc', feature_utils.EcgLfccFeatsExtractor(sampling_rate=sampling_rate)))
    elif conf['features'] == 'peaks':
        preprocessing.steps.append(('peaks_feats', feature_utils.EcgPeaksFeatsExtractor(sampling_rate=sampling_rate)))
    elif conf['features'] == 'intervals':
        preprocessing.steps.append(('intervals_feats', feature_utils.EcgIntervalsFeatsExtractor(sampling_rate=sampling_rate)))
    elif conf['features'] == 'peaks_intervals':
        preprocessing.steps.append(('peaks_intervals_feats',
                                    sklearn.pipeline.make_union(
                                        feature_utils.EcgPeaksFeatsExtractor(sampling_rate=sampling_rate),
                                        feature_utils.EcgIntervalsFeatsExtractor(sampling_rate=sampling_rate))))

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


def load_dataset(samples_list:list, root_dir:Path,) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """ Load dataset and encode targets to numerical """

    logging.info("Loading dataset...")
    logging.info(f"Samples list: {samples_list}")
    X_class0, y_class0, pat_ids_class0 = data_utils.load_runs(samples_list[0], root_dir)
    X_class1, y_class1, pat_ids_class1 = data_utils.load_runs(samples_list[1], root_dir)

    X = np.concatenate((X_class0, X_class1))
    y = np.concatenate((y_class0, y_class1))
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y).astype(np.int64)
    logging.info(f"Classes in dataset: {label_encoder.classes_}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
    
    logging.info(f"Class distribution in training set before imbalance mitigation: {Counter(y_train)}")
    logging.info(f"Class distribution in test set: {Counter(y_test)}")

    return X_train, X_test, y_train, y_test, label_encoder


def build_model(conf:dict, X_train:np.ndarray, y_train:np.ndarray) -> imblearn.pipeline.Pipeline:
    """ Configure model and optimizer according to configuration files """
    
    logging.info("Building model...")
    logging.info(f"Applying {conf['imbalance_mitigation']} to mitigate class imbalance in training data...")
    
    mlflow.set_tags({'imbalance_mitigation': conf['imbalance_mitigation'], 
                     'model': conf['model']})

    # compute class weights for loss function if desired
    if conf['imbalance_mitigation'] == 'criterion_weights':
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
            EarlyStopping(patience=conf['early_stopping_patience'])  # Early Stopping based on validation loss
            ],
        'max_epochs': conf['early_stopping_max_epochs'],
        'device': 'cuda',
        'iterator_train__shuffle': True,  # Shuffle training data on each epoch
    }
    
    if conf['model'] == 'svm':
        clf = sklearn.svm.SVC(kernel=conf['svm_kernel'], class_weight=class_weight)
    elif conf['model'] == 'mlp':
        clf = skorch.NeuralNetClassifier(module=MLP, module__input_size=X_train[0].size, **skorch_clf_common_params)
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

    # add under/oversampling steps to model pipeline if desired
    if conf['imbalance_mitigation'] == 'smote':
        pipe = imblearn.pipeline.Pipeline([('smote', SMOTE()), ('clf', clf)])
    elif conf['imbalance_mitigation'] == 'random_undersampling':
        pipe = imblearn.pipeline.Pipeline([('random_undersampling', RandomUnderSampler()), ('clf', clf)])
    elif conf['imbalance_mitigation'] == 'criterion_weights':
        pipe = imblearn.pipeline.Pipeline([('clf', clf)])
    else:
        raise f"Invalid value {conf['imbalance_mitigation']} for imbalance_mitigation!"
    
    logging.info(f"Model pipeline:\n{pipe}")

    return pipe


def evaluate_experiment(X_test:np.ndarray, y_true:np.ndarray, y_encoder:LabelEncoder, gs:imblearn.pipeline.Pipeline) -> None:
    """ Compute scores, create figures and log all metrics to MLFlow """

    # Generate Confusion Matrix
    conf_matrix_fig = sklearn.metrics.ConfusionMatrixDisplay.from_estimator(gs, X_test, y_true, display_labels=y_encoder.classes_, cmap='Blues', normalize='true').figure_
    mlflow.log_figure(conf_matrix_fig, 'confusion_matrix.png')
    
    # Generate ROC curve
    roc_curve_fig = sklearn.metrics.RocCurveDisplay.from_estimator(gs, X_test, y_true).figure_
    mlflow.log_figure(roc_curve_fig, 'roc_curve.png')
    
    # Generate train&valid Loss curve
    loss_fig = plt.figure()
    plt.plot(gs.best_estimator_.steps[-1][1].history[:, 'train_loss'], label='train loss')
    plt.plot(gs.best_estimator_.steps[-1][1].history[:, 'valid_loss'], label='valid loss')
    plt.legend()
    mlflow.log_figure(loss_fig, 'train_valid_loss.png')
    
    # Store model summary of best model in MLFlow
    mlflow.log_text(str(gs.best_estimator_.steps[-1][1].module_), 'model_topology.txt')
    # mlflow.log_text(str(torchinfo.summary(gs.best_estimator_.module, input_size=(gs.best_params_['batch_size'], *X_train.shape[1:]))), 'best_model_summary.txt')
