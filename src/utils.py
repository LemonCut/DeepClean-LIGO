import numpy as np
from scipy.signal import butter, sosfilt
import matplotlib.pyplot as plt
import random

def postprocess(prediction_tensor, strain_mean, strain_std, sample_rate, band_min, band_max):
    pred_noise = prediction_tensor.detach().cpu().numpy().squeeze()
    
    noise_physical = (pred_noise * strain_std) + strain_mean
    
    sos = butter(8, [band_min, band_max], btype='bandpass', fs=sample_rate, output='sos')

    if noise_physical.ndim == 1:
        noise_clean = sosfilt(sos, noise_physical)
    else:
        noise_clean = sosfilt(sos, noise_physical, axis=-1)
    
    return noise_clean
def visualize_data(data):
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 14))

    # 3. Power Spectral Density (PSD)
    psd = data.psd(fftlength=2, overlap=1)
    axes[0].loglog(psd.frequencies.value, np.sqrt(psd.value), 'g-', linewidth=1)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Amplitude Spectral Density (1/√Hz)')
    axes[0].set_title('Power Spectral Density')
    axes[0].set_xlim(20, 1000)
    #axes.set_ylim(ymin, ymax)
    axes[0].grid(True, alpha=0.3)

    psd = data.psd(fftlength=80, overlap=20)
    axes[1].loglog(psd.frequencies.value, np.sqrt(psd.value), 'g-', linewidth=1)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Amplitude Spectral Density (1/√Hz)')
    axes[1].set_title('Power Spectral Density')
    axes[1].set_xlim(57, 63)
    #axes.set_ylim(ymin, ymax)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
def simulate_mains_hum(f_hum=60, harmonics=4, sample_rate=4096, duration=1024):
    """
    Simulate realistic mains hum with slight frequency jitter and variable harmonic amplitudes.
    """
    t = np.arange(0, duration, 1/sample_rate)
    signal = np.zeros_like(t)
    
    for n in range(1, harmonics + 1):
        # random amplitude for each harmonic (typical real hum: fundamental strongest)
        amp = 1.0 / (n ** 4) * (0.8 + 0.4*np.random.rand())  
        
        # small random frequency fluctuation to simulate real AC power variation
        freq_jitter = f_hum * 0.001 * (np.random.rand() - 0.5)  
        signal += amp * np.sin(2 * np.pi * (f_hum*n + freq_jitter) * t)
    
    # add low-level broadband noise
    signal += 0.01 * np.random.randn(len(t))
    
    return signal

def sideband_nonlinear_coupling(signal, sample_rate=4096):
    """
    Introduce significant sidebands around each harmonic by low-frequency amplitude modulation.
    """
    t = np.arange(len(signal)) / sample_rate
    
    # Modulation signal: multiple low-frequency sinusoids to generate visible sidebands
    mod = 0.001 * np.sin(2*np.pi*(2/3)*t) + 0.001 * np.sin(2*np.pi*(1/3)*t)
    for x in range(20):
        mod += 0.001 * np.sin(2*np.pi*(2/3 + random.uniform(-0.1, 0.1) )*t) + 0.001 * np.sin(2*np.pi*(1/3 + random.uniform(-0.1, 0.1) )*t)
    
    # Amplitude modulation
    modulated_signal = signal * (1 + mod)
    
    return modulated_signal
