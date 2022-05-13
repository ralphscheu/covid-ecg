import os
import numpy as np
import pandas as pd
from  torch.utils.data import Dataset
import neurokit2 as nk
import covidecg.data.utils


class EcgDataset(Dataset):
    """ PyTorch Dataset for retrieving raw ECG signals """

    def __init__(self, runs_info_file, signals_dir, flatten_signal=False, lead=None):
        self.runs_info = pd.read_csv(runs_info_file, sep=';')
        self.labels = self.runs_info.pat_group.to_numpy()
        self.signals_dir = signals_dir
        self.flatten_signal = flatten_signal
        self.lead_index = None

        # convert target labels to numeric
        self.runs_info.pat_group = self.runs_info.pat_group.replace({'ctrl': 0, 'covid': 1, 'postcovid': 2})
        
        if lead is not None:
            # determine indices for leads
            with open(os.path.join(signals_dir, self.runs_info.iloc[0]['run_id'] + '.txt'), "r") as f:
                lead_names = f.readlines()[0:12]
            
            lead_names = [line.strip() for line in lead_names]  # remove newline characters

            try:
                self.lead_index = lead_names.index(lead)
            except ValueError:
                sys.exit(f"Lead {lead} not found in ECG signal files! Available leads: {lead_names}")

    def __len__(self):
        return len(self.runs_info)

    def __getitem__(self, index):
        signal_path = os.path.join(self.signals_dir, self.runs_info.iloc[index]['run_id'] + '.txt')
        signal = covidecg.data.utils.load_run_signal(signal_path)

        if self.lead_index is not None:
            signal = signal[self.lead_index, :]

        if self.flatten_signal:
            signal = signal.flatten()

        label = self.runs_info.iloc[index]['pat_group']
        return signal, label
