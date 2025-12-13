import torch
import torch.nn as nn


class DeepCleanLoss(nn.Module):
    def __init__(self, sample_rate, fft_length=2, overlap=1, alpha=0.5, band_start=55.0, band_end=65.0):
        super(DeepCleanLoss, self).__init__()
        self.alpha = alpha
        self.mse_crit = nn.MSELoss()
        self.fs = sample_rate
        self.n_fft = int(fft_length * sample_rate)
        self.hop = int((fft_length - overlap) * sample_rate)
        self.window = torch.hann_window(self.n_fft)
        
        self.freqs = torch.fft.rfftfreq(self.n_fft, d=1/self.fs)
        self.band_start = band_start
        self.band_end = band_end

    def compute_asd(self, x):
        x_stft = torch.stft(
            x.squeeze(1), 
            n_fft=self.n_fft, 
            hop_length=self.hop, 
            window=self.window.to(x.device), 
            return_complex=True
        )
        psd = torch.abs(x_stft)**2
        psd_avg = torch.mean(psd, dim=2)
        return torch.sqrt(psd_avg + 1e-12)

    def forward(self, noise_pred, target_strain, witness_channels):
        residual = target_strain - noise_pred

        loss_mse = self.mse_crit(residual, torch.zeros_like(residual))

        asd_res = self.compute_asd(residual)
        asd_target = self.compute_asd(target_strain)

        weights = 1.0 / (asd_target.detach() + 1e-8)
        
        mask = (self.freqs >= self.band_start) & (self.freqs <= self.band_end)
        mask = mask.to(weights.device).float()
        
        weights = weights * mask
        
        loss_asd = torch.mean(weights * asd_res)

        return self.alpha * loss_asd + (1 - self.alpha) * loss_mse