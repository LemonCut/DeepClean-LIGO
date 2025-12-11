import torch
from torch.utils.data import DataLoader
from src.model import DeepCleanAutoencoder
from src.loss import DeepCleanLoss
from src.dataset import GWDataset

def train(strain_train, witness_train, strain_val, witness_val, sample_rate, band_min, band_max, alpha=0.5, 
          overlap=7.75, val_overlap=4.0, batch_size=32, epochs=50, save_path='best_model.pth', patience=5,
          linear=True):
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
        save_path (file path): Path to store best performing model
        patience (int): How many epochs to wait if validation loss isn't improving before stopping.
    """

    train_dataset = GWDataset(
        strain_train, witness_train, sample_rate,
        overlap=overlap,
        band_start=band_min, band_end=band_max,
        linear=linear
    )
    
    val_dataset = GWDataset(
        strain_val, witness_val, sample_rate,
        overlap=val_overlap,
        band_start=band_min, band_end=band_max,
        linear=linear
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    num_witnesses = witness_train.shape[0]
    model = DeepCleanAutoencoder(num_witnesses).cuda()
    
    criterion = DeepCleanLoss(
        sample_rate, 
        alpha=alpha, 
        band_start=band_min, 
        band_end=band_max
    ).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    print(f"Starting training: {sample_rate}Hz | Band: {band_min}-{band_max}Hz")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for w, h in train_loader:
            w, h = w.cuda(), h.cuda()
            optimizer.zero_grad()
            
            pred = model(w)
            loss = criterion(pred, h, w)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for w, h in val_loader:
                w, h = w.cuda(), h.cuda()
                pred = model(w)
                loss = criterion(pred, h, w)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print("Stopped training: Validation loss stopped improving.")
                break
            
    print(f"Training complete. Best validation loss: {best_val_loss}")
    
    model.load_state_dict(torch.load(save_path))
    return model