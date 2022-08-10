import os
from covidecg.data.dataset import *
from skorch.helper import SliceDataset
from torch.utils.data import DataLoader
import covidecg.data.utils as data_utils
from tqdm import tqdm
import click
from dotenv import find_dotenv, load_dotenv


def get_lead_signal_img(lead_signal:np.ndarray, img_size, dpi, crop_horizontal_padding:int=0):
    # Make a white-on-black line plot and save the buffer to a numpy array
    img_width = lead_signal.shape[0] // 2
    img_render_width = img_width // 6
    fig = plt.figure(figsize=(img_render_width / dpi, img_size / dpi), dpi=dpi)
    fig.gca().plot(lead_signal, linewidth=1, c='black')
    plt.axis('off')
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    lead_signal_image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8) 
    plt.close()  # close image to prevent memory leak

    lead_signal_image = lead_signal_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, -1]
    assert len(np.unique(lead_signal_image)) > 1, "All pixels have the same value, something went wrong."

    # Crop the image to remove whitespace (well, blackspace)
    lead_signal_image = ~lead_signal_image
    nonzero_coords = cv2.findNonZero(lead_signal_image) # Find all non-zero points
    x, y, w, h = cv2.boundingRect(nonzero_coords) # Find minimum spanning bounding box
    lead_signal_image = lead_signal_image[:, x - crop_horizontal_padding:x + w + crop_horizontal_padding]
    lead_signal_image = cv2.resize(lead_signal_image, (img_width, img_size))
    lead_signal_image = ~lead_signal_image
    
    return lead_signal_image


@click.command()
@click.option('--recordings-file', required=True, type=click.Path(exists=True))
@click.option('--recordings-dir', required=True, type=click.Path(exists=True))
@click.option('--output-dir', required=True, type=click.Path(exists=True))
@click.option('--img-size', default=800, type=int)
@click.option('--dpi', default=96, type=int)
def main(recordings_file, recordings_dir, output_dir, img_size, dpi):
    
    recs_info = pd.read_csv(recordings_file, sep=';')
    imgdict = {}
    for rec_i in tqdm(range(25)):  # range(recs_info.shape[0])
        rec_id = recs_info.iloc[rec_i]['recording']
        signal_path = os.path.join(recordings_dir, rec_id + '.csv')
        rec_signal = data_utils.load_signal(signal_path)
        rec_signal = data_utils.clean_signal(rec_signal)
        imgdata = np.stack([get_lead_signal_img(lead_signal, img_size, dpi) for lead_signal in rec_signal], axis=2)
        imgdict.update({rec_id: imgdata})
    np.savez_compressed(os.path.join(output_dir, 'ecgimgdata.npz'), **imgdict)


if __name__ == '__main__':
    load_dotenv(find_dotenv())  # load environment variables set in .env file
    main()