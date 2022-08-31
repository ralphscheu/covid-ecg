import os
from covidecg.data.dataset import *
from skorch.helper import SliceDataset
from torch.utils.data import DataLoader
import covidecg.data.utils as data_utils
from tqdm import tqdm
import click
from dotenv import find_dotenv, load_dotenv
from PIL import Image


def signal2image(signal:np.ndarray, img_height, dpi, ecg_value_range=[-599, 600], crop_horizontal_padding:int=0):
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

    # Crop the img to remove empty space
    img = ~img
    nonzero_coords = cv2.findNonZero(img) # Find all non-zero points
    x, y, w, h = cv2.boundingRect(nonzero_coords) # Find minimum spanning bounding box
    img = img[:, x - crop_horizontal_padding:x + w + crop_horizontal_padding]
    img = cv2.resize(img, (img_width, img_height))
    img = ~img
    return img


@click.command()
@click.option('--recordings-file', required=True, type=click.Path(exists=True))
@click.option('--recordings-dir', required=True, type=click.Path(exists=True))
@click.option('--output-dir', required=True, type=click.Path(exists=True))
@click.option('--img-height', default=200, type=int)
@click.option('--dpi', default=96, type=int)
def main(recordings_file, recordings_dir, output_dir, img_height, dpi):
    
    recs_info = pd.read_csv(recordings_file, sep=';')
    imgdict = {}
    for rec_i in tqdm(range(recs_info.shape[0])):
        rec_id = recs_info.iloc[rec_i]['recording']
        signal_path = os.path.join(recordings_dir, rec_id + '.csv')
        rec_signal = data_utils.load_signal(signal_path)
        rec_signal = rec_signal[:, 0:5000]  # max length of 10 seconds
        rec_signal = data_utils.clean_signal(rec_signal)
        imgdata = np.stack([signal2image(lead_signal, img_height, dpi) for lead_signal in rec_signal], axis=0)
        imgdata = np.reshape(imgdata, (3, 4, imgdata.shape[1], imgdata.shape[2]))

        # print(f"imgdata: {imgdata.shape}")
        # print(f"imgdata[:, 0]: {imgdata[:, 0].shape}")
        # generate and save ECG printout image
        col0 = np.concatenate(list(imgdata[:, 0]), axis=0)
        col1 = np.concatenate(list(imgdata[:, 1]), axis=0)
        col2 = np.concatenate(list(imgdata[:, 2]), axis=0)
        col3 = np.concatenate(list(imgdata[:, 3]), axis=0)
        # print(f"cols: {col0.shape}, {col1.shape}, {col2.shape}, {col3.shape}")
        ecg_printout = np.concatenate([col0, col1, col2, col3], axis=1)
        # print(f"ecg_printout: {ecg_printout.shape}")
        ecg_printout_savepath = os.path.join(output_dir, rec_id + '.png')
        Image.fromarray(ecg_printout).save(ecg_printout_savepath)
        
        imgdict.update({rec_id: imgdata})
        
    out_filepath = os.path.join(output_dir, 'ecgimgdata.npz')
    print(f"Saving to {out_filepath}")
    np.savez_compressed(out_filepath, **imgdict)


if __name__ == '__main__':
    load_dotenv(find_dotenv())  # load environment variables set in .env file
    main()