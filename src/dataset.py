import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, sosfilt

class GWDataset(Dataset):
    def __init__(self, strain, witnesses, sample_rate, segment_duration=8.0, overlap=7.75, band_start=55.0, band_end=65.0,
                 linear=False):
        """
        strain: array of wave data
        witnesses: matrix of witness channels (N_channels, Time)
        """
        self.sr = sample_rate
        self.seg_len = int(segment_duration * sample_rate) 
        self.step = int((segment_duration - overlap) * sample_rate) 
        
        #sos = butter(8, [band_start, band_end], btype='bandpass', fs=sample_rate, output='sos')
        #strain_filtered = sosfilt(sos, strain)
        strain_filtered = strain

        if linear:
            witnesses_filtered = np.zeros_like(witnesses)
            for i in range(witnesses.shape[0]):
                witnesses_filtered[i] = sosfilt(sos, witnesses[i])
        else:
            witnesses_filtered = witnesses
            
        self.strain_mean = np.mean(strain_filtered)
        self.strain_std = np.std(strain_filtered)
        self.strain = (strain_filtered - self.strain_mean) / self.strain_std
        
        self.wit_mean = np.mean(witnesses_filtered, axis=1, keepdims=True)
        self.wit_std = np.std(witnesses_filtered, axis=1, keepdims=True)
        self.witnesses = (witnesses_filtered - self.wit_mean) / self.wit_std

        self.indices = list(range(0, self.strain.shape[-1] - self.seg_len, self.step))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.seg_len
        
        h_seg = self.strain[start:end]
        w_seg = self.witnesses[:, start:end]
        
        return torch.tensor(w_seg, dtype=torch.float32), torch.tensor(h_seg, dtype=torch.float32).unsqueeze(0)