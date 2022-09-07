import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import click
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
import covidecg.data.utils as data_utils
import os
from pathlib import Path
from PIL import Image


import numpy as np
import os
import cv2
def noisy(image, mode):
    """add noise to image.
    Taken from https://stackoverflow.com/a/30609854/19473738
    
    Args:
        image : ndarray
            Input image data. Will be converted to float.
        mode : str
            One of the following strings, selecting the type of noise to add:

            'gauss'     Gaussian-distributed additive noise.
            'poisson'   Poisson-distributed noise generated from the data.
            's&p'       Replaces random pixels with 0 or 1.
            'speckle'   Multiplicative noise using out = image + n*image,where
                        n is uniform noise with specified mean & variance.

    Returns:
        noisy_image: Input image with added noise
    """
    if mode == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss * 2
        return noisy
    elif mode == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif mode == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif mode =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss * 0.2
        return noisy


@click.command()
@click.argument('in_file', required=True, type=click.Path(exists=True, path_type=Path, dir_okay=False))
@click.argument('out_file', required=True, type=click.Path(exists=False, path_type=Path, dir_okay=False))
@click.option('--aug-method', required=True, type=click.Choice(['noise']))
@click.option('--noise-mode', default='gauss', type=click.Choice(['gauss', 'poisson', 's&p', 'speckle']))
def main(in_file, out_file, aug_method, noise_mode):
    input_img = cv2.imread(str(in_file))  # load base image file
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY )  # convert to grayscale
    input_img = input_img[:, :, np.newaxis]
    if aug_method == 'noise':
        # add background noise to image
        output_img = noisy(image=input_img, mode=noise_mode)
        
    cv2.imwrite(str(out_file), output_img)
    
    
if __name__ == '__main__':
    load_dotenv(find_dotenv())  # load environment variables set in .env file
    main()
