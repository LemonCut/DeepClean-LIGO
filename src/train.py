import torch
from torch.utils.data import DataLoader
from src.model import DeepCleanAutoencoder
from src.loss import DeepCleanLoss
from src.dataset import GWDataset

def train(strain_data, witness_data, sample_rate, band_min, band_max, alpha=0.5, overlap=7.75, batch_size=32, epochs=50):
    """
    Train the model
    
    Args:
        strain_data (array): 1D array of strain data.
        witness_data (array): 2D array of witness channels (N_channels x Time).
        sample_rate (int): Sampling rate in Hz (e.g., 2048 or 1024).
        band_min (float): Lower frequency of the noise band to target (Hz).
        band_max (float): Upper frequency of the noise band to target (Hz).
        alpha (float): Loss weighting factor (0.0 to 1.0). 
                       Use high values for broadband noise, low for spectral lines.
        overlap (float): Duration of overlap between segments in seconds.
                         Paper suggests 7.75s for training, 4.0s for inference.
        batch_size (int): Training batch size.
        epochs (int): Number of training epochs.
    """
    
    dataset = GWDataset(
        strain_data, 
        witness_data, 
        sample_rate,
        segment_duration=8.0,
        overlap=overlap,
        band_start=band_min, 
        band_end=band_max
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    num_witnesses = witness_data.shape[0]
    model = DeepCleanAutoencoder(num_witnesses).cuda()
    
    criterion = DeepCleanLoss(
        sample_rate, 
        alpha=alpha, 
        band_start=band_min, 
        band_end=band_max
    ).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    print(f"Starting training on {sample_rate}Hz data targeting {band_min}-{band_max}Hz")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (w, h) in enumerate(loader):
            w, h = w.cuda(), h.cuda()
            
            optimizer.zero_grad()

            noise_pred = model(w)
            loss = criterion(noise_pred, h, w)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

    return model