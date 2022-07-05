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
from covidecg.models.models import MLP, CNN2D, CNN1D, VGG16, LSTM
import mlflow
import torch.nn as nn
import sklearn.metrics
import sklearn.svm
import torch.optim
import torch
from io import StringIO
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
import random
import imblearn.pipeline
import torchvision.models
import matplotlib.pyplot as plt


def build_preprocessing_pipeline(conf:dict, sampling_rate:int) -> sklearn.pipeline.Pipeline:
    """ Build data pre-processing pipeline for feature extraction """

    preprocessing = sklearn.pipeline.Pipeline([('clean_signal', data_utils.EcgSignalCleaner())])

    if conf['ecg_leads'] != 'all':
        preprocessing.steps.append(('select_ecg_lead', data_utils.EcgLeadSelector(conf['ecg_leads'])))

    if conf['features'] == 'plain_signal':
        pass
    if conf['features'] == 'signal_image':
        preprocessing.steps.append(('convert_signal_to_image', feature_utils.EcgSignalToImageConverter()))

        if conf['model'] == 'vgg16':
            
            preprocessing.steps.append(('grayscale_to_rgb', FunctionTransformer(data_utils.grayscale_to_rgb)))
            
            # convert to torch.Tensor for vgg16 image preprocessing
            preprocessing.steps.append(('to_tensor', FunctionTransformer(torch.from_numpy)))
            # apply preprocessing transforms for vgg16 input
            preprocessing.steps.append(('vgg16_image_preprocessing', FunctionTransformer(torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES.transforms())))
            # convert back to numpy array
            preprocessing.steps.append(('to_numpy', FunctionTransformer(lambda x_tensor: x_tensor.detach().cpu().numpy())))

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

    if conf['flatten_leads']:
        preprocessing.steps.append(('flatten_leads', FunctionTransformer(data_utils.flatten_leads)))

    return preprocessing


def load_dataset(samples_list, root_dir, ecg_type):
    """ Load dataset and encode targets to numerical """
    logging.info(f"Samples list: f{samples_list}")
    if ecg_type == 'stress':
        X_class0, y_class0, pat_ids_class0 = data_utils.load_stress_ecg_runs(samples_list[0], root_dir)
        X_class1, y_class1, pat_ids_class1 = data_utils.load_stress_ecg_runs(samples_list[1], root_dir)
    elif ecg_type == 'rest':
        X_class0, y_class0, pat_id_class0 = data_utils.load_rest_ecg_runs(samples_list[0], root_dir)
        X_class1, y_class1, pat_id_class0 = data_utils.load_rest_ecg_runs(samples_list[1], root_dir)
    else:
        raise Exception(f"Found invalid value '{ecg_type}' for ecg_type in experiment configuration! (stress|rest)")

    X = np.concatenate((X_class0, X_class1))
    y = np.concatenate((y_class0, y_class1))
    pat_ids = np.concatenate((pat_ids_class0, pat_ids_class1))
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y).astype(np.int64)
    logging.info(f"Classes in dataset: {label_encoder.classes_}")
    
    train_idx, test_idx = next( GroupShuffleSplit(test_size=0.2).split(X, y, groups=pat_ids) )
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    logging.info(f"Class distribution in training set before imbalance mitigation: {Counter(y_train)}")
    logging.info(f"Class distribution in test set: {Counter(y_test)}")
    
    return X_train, X_test, y_train, y_test, label_encoder


def build_model(conf:dict, X_train:np.ndarray, y_train:np.ndarray) -> imblearn.pipeline.Pipeline:
    """ Define model and optimizer according to experiment configuration """
    
    logging.info(f"Applying {conf['imbalance_mitigation']} to mitigate class imbalance in training data...")

    class_weight = None
    if conf['imbalance_mitigation'] == 'criterion_weights':
        class_weight = sklearn.utils.class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weight = torch.Tensor(class_weight)

    
    if conf['model'] == 'svm':
        if conf['svm_class_weight_balanced']:
            clf = sklearn.svm.SVC(kernel=conf['svm_kernel'], class_weight='balanced')
        else:
            clf = sklearn.svm.SVC(kernel=conf['svm_kernel'])

    elif conf['model'] == 'mlp':
        clf = skorch.NeuralNetClassifier(
            # model config
            module=MLP,
            module__input_size=X_train[0].size,
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

    elif conf['model'] == 'lstm':
        clf = skorch.NeuralNetClassifier(
            # model config
            module=LSTM,
            module__input_size=12,
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
    elif conf['model'] == 'vgg16':
        clf = skorch.NeuralNetClassifier(
            # model config
            module=VGG16,
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


def evaluate_experiment(X_test, y_true, y_encoder, gs):
    """ Compute scores, create figures and log all metrics to MLFlow """
    
    y_pred = gs.predict(X_test)
    
    accuracy_score = sklearn.metrics.accuracy_score(y_true, y_pred)
    roc_auc_score = sklearn.metrics.roc_auc_score(y_true, y_pred)
    f1_score = sklearn.metrics.f1_score(y_true, y_pred)
    precision_score = sklearn.metrics.precision_score(y_true, y_pred)
    recall_score = sklearn.metrics.recall_score(y_true, y_pred)
    
    conf_matrix_figure = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=y_encoder.classes_, cmap='Blues', normalize='true').figure_
    
    roc_auc_curve_figure = sklearn.metrics.RocCurveDisplay.from_estimator(gs, X_test, y_true).figure_
    
    logging.info("Logging metrics to MLFlow...")
    mlflow.log_figure(conf_matrix_figure, 'confusion_matrix.png')
    mlflow.log_figure(roc_auc_curve_figure, 'roc_auc_curve.png')
    
    loss_fig = plt.figure()
    plt.plot(gs.best_estimator_.steps[-1][1].history[:, 'train_loss'], label='train_loss')
    plt.plot(gs.best_estimator_.steps[-1][1].history[:, 'valid_loss'], label='valid_loss')
    plt.legend()
    mlflow.log_figure(loss_fig, 'loss.png')
