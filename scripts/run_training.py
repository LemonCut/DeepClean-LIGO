import numpy as np
import sys
sys.path.append('..')

#only works if you run this file while in the scripts directory
from src.train import train

# Configuration matching O2/O3 requirements
DATA_PATH = '../data/train_data.npz'
MODEL_SAVE_PATH = '../models/best_model.pth'
SAMPLE_RATE = 1024
BAND_MIN = 55.0      # Target 60Hz mains
BAND_MAX = 65.0
ALPHA = 0.5          # Balanced loss

def main():
    data = np.load(DATA_PATH)
    strain = data['strain']
    witnesses = data['witnesses']
    
    split = int(len(strain) * 0.8)
    
    strain_train, strain_val = strain[:split], strain[split:]
    wit_train, wit_val = witnesses[:, :split], witnesses[:, split:]
    
    print(f"Training on {len(strain_train)/SAMPLE_RATE:.1f}s of data.")
    
    model = train(
        strain_train, wit_train, 
        strain_val, wit_val,
        sample_rate=SAMPLE_RATE,
        band_min=BAND_MIN, 
        band_max=BAND_MAX,
        alpha=ALPHA,
        epochs=30,
        save_path=MODEL_SAVE_PATH
    )
    
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    import os
    os.makedirs('../models', exist_ok=True)
    main()