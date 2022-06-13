import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import yaml
import covidecg.data.utils as data_utils
import covidecg.features.utils as feature_utils
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import sklearn.pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
import skorch
from skorch.callbacks import EpochScoring, EarlyStopping
from covidecg.models.mlp_models import MLP, CNN2D, CNN1D
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
import matplotlib.pyplot as plt
import random
# import torchinfo
import imblearn.pipeline
# from dask.distributed import Client
# from joblib import parallel_backend

# client = Client('127.0.0.1:8786')

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
    logging.info(f"Samples list: f{samples_list}")
    if ecg_type == 'stress':
        X_a, y_a = data_utils.load_stress_ecg_runs(samples_list[0], root_dir)
        X_b, y_b = data_utils.load_stress_ecg_runs(samples_list[1], root_dir)
    elif ecg_type == 'rest':
        X_a, y_a = data_utils.load_rest_ecg_runs(samples_list[0], root_dir)
        X_b, y_b = data_utils.load_rest_ecg_runs(samples_list[1], root_dir)
    else:
        raise Exception(f"Invalid ecg_type {ecg_type} in experiment configuration! (stress|rest)")
    
    X, y = np.concatenate((X_a, X_b)), np.concatenate((y_a, y_b))
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y).astype(np.int64)
    logging.info(f"Classes in dataset: {label_encoder.classes_}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
    
    return X_train, X_test, y_train, y_test, label_encoder


def build_model(conf:dict, X_train:np.ndarray, y_train:np.ndarray) -> imblearn.pipeline.Pipeline:
    """ Define model and optimizer according to experiment configuration """
    
    logging.info(f"Applying {conf['imbalance_mitigation']} to mitigate class imbalance in training data...")
        
    logging.debug(f"Class distribution in training set: {Counter(y_train)}")

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


###############################################

@click.command()
@click.option('--config-file', required=True, type=click.Path(exists=True))
def run_experiment(config_file):

    # read experiment config    
    with open(config_file) as f:
        conf_str = f.read()
        conf = yaml.safe_load(conf_str)

    logging.info("Loading dataset...")
    X_train, X_test, y_train, y_test, y_encoder = load_dataset(samples_list=conf['samples_list'], root_dir=conf['root_dir'], ecg_type=conf['ecg_type'])

    ###

    preprocessing = build_preprocessing_pipeline(
        conf=conf, 
        sampling_rate=int(os.environ['SAMPLING_RATE']))
    logging.info("Preprocessing data...")
    logging.info(f"Preprocessing steps: {preprocessing.named_steps}")
    X_train = preprocessing.fit_transform(X_train).astype(np.float32)
    X_test = preprocessing.fit_transform(X_test).astype(np.float32)
    logging.info(f"Shapes after preprocessing - X_train: {X_train.shape} - X_test: {X_test.shape}")

    ###

    mlflow.sklearn.autolog()
    experiment = mlflow.set_experiment(Path(config_file).stem)
    with mlflow.start_run(experiment_id=experiment.experiment_id, 
                          tags={
                              'features': conf['features'], 
                              'imbalance_mitigation': conf['imbalance_mitigation'], 
                              'ecg_type': conf['ecg_type'],
                              'ecg_leads': conf['ecg_leads'],
                              'model': conf['model']}):

        mlflow.log_text(conf_str, 'experiment_config.yaml')

        logging.info("Building model...")
        model = build_model(conf, X_train, y_train)

        logging.info("Fitting and evaluating model...")
        gs = sklearn.model_selection.GridSearchCV(model, conf['grid_search'], 
            scoring=sklearn.metrics.get_scorer('roc_auc'), 
            cv=int(conf['num_cv_folds']), refit=True, error_score='raise', verbose=4)

        # with parallel_backend('dask'):
        gs.fit(X_train, y_train)

        logging.info(f"GridSearchCV - Best ROC-AUC Score in CV: {gs.best_score_}")
        logging.info(f"GridSearchCV - Best Params: {gs.best_params_}")
        logging.info("Evaluating best model as determined by Grid Search...")
        evaluate_experiment(X_test, y_test, y_encoder, gs)

        # Store log in MLFlow
        mlflow.log_text(LOG_STREAM.getvalue(), 'train_evaluate.log')

        # Store model summary in MLFlow
        mlflow.log_text(str(gs.best_estimator_.steps[-1][1].module_), 'model_topology.txt')
        # mlflow.log_text(str(torchinfo.summary(gs.best_estimator_.module, input_size=(gs.best_params_['batch_size'], *X_train.shape[1:]))), 'best_model_summary.txt')


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    
    LOG_STREAM = StringIO()
    logging.basicConfig(
        level='INFO', 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # log to stdout
            logging.StreamHandler(stream=LOG_STREAM)  # log to StringIO object for storing in MLFlow
        ])

    run_experiment()
