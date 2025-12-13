import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class DeepCleanAutoencoder(nn.Module):
    def __init__(self, num_witnesses):
        super(DeepCleanAutoencoder, self).__init__()
        
        kernel_size = 15
        padding = 7
        
        self.in_conv = nn.Sequential(
            nn.Conv1d(num_witnesses, 8, kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(8),
            nn.Tanh()
        )
        
        self.down1 = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size, stride=2, padding=padding),
            nn.BatchNorm1d(8),
            nn.Tanh()
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size, stride=2, padding=padding),
            nn.BatchNorm1d(16),
            nn.Tanh()
        )
        self.down3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size, stride=2, padding=padding),
            nn.BatchNorm1d(32),
            nn.Tanh()
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(32, 32, kernel_size, stride=2, padding=padding, output_padding=1),
            nn.BatchNorm1d(32),
            nn.Tanh()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size, stride=2, padding=padding, output_padding=1),
            nn.BatchNorm1d(16),
            nn.Tanh()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size, stride=2, padding=padding, output_padding=1),
            nn.BatchNorm1d(8),
            nn.Tanh()
        )

        self.out_conv = nn.Conv1d(8, 1, kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        
        noise_pred = self.out_conv(x)
        return noise_pred

