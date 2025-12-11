import numpy as np
from scipy.signal import butter, sosfilt

def postprocess(prediction_tensor, strain_mean, strain_std, sample_rate, band_min, band_max):
    pred_noise = prediction_tensor.detach().cpu().numpy().squeeze()
    
    noise_physical = (pred_noise * strain_std) + strain_mean
    
    sos = butter(8, [band_min, band_max], btype='bandpass', fs=sample_rate, output='sos')

    if noise_physical.ndim == 1:
        noise_clean = sosfilt(sos, noise_physical)
    else:
        noise_clean = sosfilt(sos, noise_physical, axis=-1)
    
    return noise_clean