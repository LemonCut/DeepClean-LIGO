import numpy as np
import random
import sys
sys.path.append('..')
from src.utils import simulate_mains_hum, sideband_nonlinear_coupling


def generatedSimulatedData(sourceDataPath, outputPath):

    data = np.load(sourceDataPath)
    strain_raw = data['strain']
    witnesses_raw = data['witnesses']
    mainsHumWitness = simulate_mains_hum()
    modulatedNoise = sideband_nonlinear_coupling(mainsHumWitness)
    
    simulatedNoisedData = data['strain'] + modulatedNoise * 1e-20
    
    for i, channel in enumerate(data['witnesses']):
        channel = np.array(channel)  
        # normalize to [-1, 1]
        normalizedWitnessChannel = 2 * (channel - channel.min()) / (channel.max() - channel.min()) - 1
        simulatedNoisedData += normalizedWitnessChannel * 3e-22
        
            
    newWitnessData = np.concatenate([data['witnesses'], mainsHumWitness.reshape(1, -1)], axis=0)
    np.savez(
        outputPath,
        strain=simulatedNoisedData,
        witnesses=newWitnessData,
        times=data['times'],
        sample_rate=data['sample_rate'],
        gps_start=data['gps_start'],
        duration=data['duration'],
        detector=data['detector'],
        witness_channels=data['witness_channels'],
    )
if __name__ == "__main__":
    generatedSimulatedData('../data/train_data_raw.npz', '../data/train_data.npz')
    generatedSimulatedData('../data/test_data_raw.npz', '../data/test_data.npz')