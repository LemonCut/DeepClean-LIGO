import numpy as np
import sys
sys.path.append('..')

#only works if you run this file while in the scripts directory
from src.train import train

# Configuration matching O2/O3 requirements
# this is config for O2. O3 use 1024, 55, 65, 0.1, False
DATA_PATH = '../data/train_data.npz'
MODEL_SAVE_PATH = '../models/best_model.pth'
SAMPLE_RATE = 4096
BAND_MIN = 60
BAND_MAX = 62
ALPHA = 0
LINEAR = False

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
        epochs=50,
        save_path=MODEL_SAVE_PATH,
        linear=LINEAR
    )
    
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    import os
    os.makedirs('../models', exist_ok=True)
    main()