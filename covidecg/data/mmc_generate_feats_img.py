import os
from covidecg.data.dataset import *
from skorch.helper import SliceDataset
from torch.utils.data import DataLoader
import covidecg.data.utils as data_utils
from tqdm import tqdm
import click
from dotenv import find_dotenv, load_dotenv
from PIL import Image
import logging
import pathlib


def signal2image(signal:np.ndarray, img_height, dpi, ecg_value_range=[-1500, 1499], crop_horizontal_padding:int=0):
    # Make a white-on-black line plot and save the buffer to a numpy array
    img_width = signal.shape[0] // 5
    img_render_width = img_width
    fig = plt.figure(figsize=(img_render_width / dpi, img_height / dpi), dpi=dpi)
    plt.gca().set_ylim(ecg_value_range)
    fig.gca().plot(signal, linewidth=1.0, c='black')
    plt.axis('off')
    fig.tight_layout(pad=0)
    fig.canvas.draw()  # trigger drawing
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8) 
    plt.close()  # close image to prevent memory leak

    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, -1]
    assert len(np.unique(img)) > 1, "All pixels have the same value, something went wrong."

    # Crop the img to remove empty space on left and right
    img = ~img
    nonzero_coords = cv2.findNonZero(img) # Find all non-zero points
    x, y, w, h = cv2.boundingRect(nonzero_coords) # Find minimum spanning bounding box
    img = img[:, x - crop_horizontal_padding:x + w + crop_horizontal_padding]
    img = cv2.resize(img, (img_width, img_height))
    img = ~img
    return img


@click.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path))
@click.argument('output_dir', type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path))
@click.option('--img-height', default=200, type=int)
@click.option('--dpi', default=96, type=int)
def main(input_dir, output_dir, img_height, dpi):
    
    files = list(pathlib.Path(input_dir).glob('*.csv'))
    for file in tqdm(files, desc="Processing files"):
        rec_id = file.stem
        rec_signal = data_utils.load_signal(file)
        rec_signal = data_utils.clean_signal(rec_signal)
        imgdata = np.stack([signal2image(lead_signal, img_height, dpi) for lead_signal in rec_signal], axis=0)
        ecggrid = data_utils.generate_ecg_leads_grid(imgdata)
        Image.fromarray(ecggrid).save(output_dir / f"{rec_id}.png")


if __name__ == '__main__':
    load_dotenv(find_dotenv())  # load environment variables set in .env file
    main()