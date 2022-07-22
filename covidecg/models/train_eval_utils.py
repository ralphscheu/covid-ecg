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
from covidecg.models.lstm_attn import Attention, Encoder, Classifier
import sklearn.pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
import skorch
from skorch.callbacks import EpochScoring, EarlyStopping
from covidecg.models.models import MLP, CNN2D, CNN1D, VGG16Classifier, ResNet18Classifier, LSTM
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


def build_preprocessing_pipeline(conf:dict, sampling_rate:int) -> sklearn.pipeline.Pipeline:
    """ Build data pre-processing pipeline for feature extraction """

    preprocessing = sklearn.pipeline.Pipeline([('clean_signal', data_utils.EcgSignalCleaner())])

    if conf['ecg_leads'] != 'all':
        preprocessing.steps.append(('select_ecg_lead', data_utils.EcgLeadSelector(conf['ecg_leads'])))
    if conf['features'] == 'plain_signal':
        # no need to do anything - use signal value arrays as-is
        pass
    if conf['features'] == 'signal_image':
        # convert from array of signal values to 2D grayscale image of signal curve
        preprocessing.steps.append(('convert_signal_to_image', feature_utils.EcgSignalToImageConverter(vertical_resolution=conf['signal_image_vertical_resolution'])))
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
        # convert from single channel grayscale to 3-channel RGB image representation
        preprocessing.steps.append(('grayscale_to_rgb', FunctionTransformer(data_utils.grayscale_to_rgb)))
        # convert to torch.Tensor for pretrained model image transforms
        preprocessing.steps.append(('to_tensor', FunctionTransformer(torch.from_numpy)))

        # apply image transforms specific to pretrained model
        if conf['model'] == 'vgg16':
            preprocessing.steps.append(('vgg16_image_preprocessing', FunctionTransformer(torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES.transforms())))
        elif conf['model'] == 'resnet18':
            preprocessing.steps.append(('resnet18_image_preprocessing', FunctionTransformer(torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms())))

        # convert back to numpy array
        preprocessing.steps.append(('to_numpy', FunctionTransformer(lambda x_tensor: x_tensor.detach().cpu().numpy())))

    if conf['flatten_leads']:
        preprocessing.steps.append(('flatten_leads', FunctionTransformer(data_utils.flatten_leads)))

    return preprocessing


def load_dataset(samples_list:list, root_dir:Path, ecg_type:str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """ Load dataset and encode targets to numerical """
    
    logging.info(f"Samples list: {samples_list}")
    if ecg_type == 'stress':
        X_class0, y_class0, pat_ids_class0 = data_utils.load_stress_ecg_runs(samples_list[0], root_dir)
        X_class1, y_class1, pat_ids_class1 = data_utils.load_stress_ecg_runs(samples_list[1], root_dir)
    elif ecg_type == 'rest':
        X_class0, y_class0, pat_ids_class0 = data_utils.load_rest_ecg_runs(samples_list[0], root_dir)
        X_class1, y_class1, pat_ids_class0 = data_utils.load_rest_ecg_runs(samples_list[1], root_dir)
    else:
        raise Exception(f"Found invalid value '{ecg_type}' for ecg_type in experiment configuration! (stress|rest)")

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
    
    logging.info(f"Applying {conf['imbalance_mitigation']} to mitigate class imbalance in training data...")

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
            EpochScoring(scoring='roc_auc', lower_is_better=False),  # additional scores to observe
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
    elif conf['model'] == 'cnn1d':
        clf = skorch.NeuralNetClassifier(module=CNN1D, **skorch_clf_common_params)
    elif conf['model'] == 'lstm':
        clf = skorch.NeuralNetClassifier(module=LSTM, module__input_size=12, **skorch_clf_common_params)
    elif conf['model'] == 'lstmattn':
        bidirectional = True
        hidden_size = 250
        n_layers_rnn = 2
        
        rnn_encoder = Encoder(embedding_dim=12, hidden_dim=hidden_size, nlayers=n_layers_rnn, bidirectional=bidirectional)
        
        attention_dim = hidden_size if not bidirectional else 2 * hidden_size
        attention = Attention(attention_dim, attention_dim, attention_dim)
        
        model = Classifier(encoder=rnn_encoder, attention=attention, hidden_dim=attention_dim, num_classes=2)
        
        clf = skorch.NeuralNetClassifier(module=model, **skorch_clf_common_params)
    elif conf['model'] == 'vgg16':
       clf = skorch.NeuralNetClassifier(module=VGG16Classifier, **skorch_clf_common_params)
    elif conf['model'] == 'resnet18':
        clf = skorch.NeuralNetClassifier(module=ResNet18Classifier, **skorch_clf_common_params)

    # add under/oversampling steps to model pipeline if desired
    if conf['imbalance_mitigation'] == 'smote':
        pipe = imblearn.pipeline.Pipeline([('smote', SMOTE()), ('clf', clf)])
    elif conf['imbalance_mitigation'] == 'random_undersampling':
        pipe = imblearn.pipeline.Pipeline([('random_undersampling', RandomUnderSampler()), ('clf', clf)])
    elif conf['imbalance_mitigation'] == 'criterion_weights':
        pipe = imblearn.pipeline.Pipeline([('clf', clf)])
    else:
        raise f"Invalid value {conf['imbalance_mitigation']} for imbalance_mitigation!"

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
