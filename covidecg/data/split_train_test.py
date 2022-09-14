import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
import torchvision
from skorch.helper import SliceDataset
import re
import logging
import click
import pathlib
import shutil
from dotenv import find_dotenv, load_dotenv
import os


@click.command()
# @click.option('--model', required=True, type=click.Choice(['CNN3DSeqPool', 'CNN3DSeqLSTM']))
@click.argument('in_dir', required=True, type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path))
@click.argument('out_dir', required=True, type=click.Path(exists=False, file_okay=False, path_type=pathlib.Path))
def main(in_dir, out_dir):
    
    dataset = torchvision.datasets.ImageFolder(in_dir)
    out_dir_train = out_dir / 'train'
    out_dir_test = out_dir / 'test'
    os.makedirs(out_dir_train)
    os.makedirs(out_dir_train / dataset.classes[0])
    os.makedirs(out_dir_train / dataset.classes[1])
    os.makedirs(out_dir_test)
    os.makedirs(out_dir_test / dataset.classes[0])
    os.makedirs(out_dir_test / dataset.classes[1])
    
    logging.basicConfig(
        level=os.getenv('LOG_LEVEL', default='INFO'), 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # log to stdout
            logging.FileHandler(out_dir / 'split_train_test.log')
        ])
    
    X = SliceDataset(dataset)
    y = np.array(dataset.targets)
    dataset_filepaths = list(np.array(dataset.samples)[:, 0])
    subjects = np.array([re.search(r'\/[a-z]+(\d+)_', p).group(1) for p in dataset_filepaths])
    test_size = 0.2
    cv = StratifiedGroupKFold(n_splits=int(1/test_size))
    train_idx, test_idx = next(cv.split(X, y, subjects))
    logging.info(f"ORIGINAL POSITIVE RATIO: {y.mean()}")
    logging.info(f"TRAIN RATIO:           : {len(train_idx) / len(dataset)}")
    logging.info(f"TEST  RATIO:           : {len(test_idx) / len(dataset)}")
    logging.info(f"TRAIN POSITIVE RATIO   : {y[train_idx].mean()}")
    logging.info(f"TEST POSITIVE RATIO    : {y[test_idx].mean()}")
    logging.info(f"TRAIN SUBJECTS         : {sorted(set(subjects[train_idx]))}")
    logging.info(f"TEST SUBJECTS          : {sorted(set(subjects[test_idx]))}")
    
    
    for idx in train_idx:
        filepath = pathlib.Path(dataset.samples[idx][0]).resolve()
        target_name = dataset.classes[dataset.targets[idx]]
        shutil.copyfile(filepath, out_dir_train / target_name / filepath.name)
        
    for idx in test_idx:
        filepath = pathlib.Path(dataset.samples[idx][0]).resolve()
        target_name = dataset.classes[dataset.targets[idx]]
        shutil.copyfile(filepath, out_dir_test / target_name / filepath.name)


if __name__ == '__main__':
    load_dotenv(find_dotenv())  # load environment variables set in .env file
    main()
