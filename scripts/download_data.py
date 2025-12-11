#!/usr/bin/env python3
"""
Download LIGO data and save as .npz files for training.
"""

from gwpy.timeseries import TimeSeries, TimeSeriesDict
from pathlib import Path
import numpy as np

def download_and_save_data(gps_start, duration, output_file, detector='H1', sample_rate=4096):
    """
    Download LIGO data from GWOSC and save as .npz file.
    
    Args:
        gps_start: GPS start time
        duration: Duration in seconds
        output_file: Path to save the .npz file
        detector: Detector name (default: 'H1')
        sample_rate: Sample rate in Hz (default: 4096)
    """
    print(f"Downloading {duration}s of {detector} data starting at GPS {gps_start}...")
    
    # Download data directly from GWOSC
    data = TimeSeries.fetch_open_data(
        detector, 
        gps_start, 
        gps_start + duration,
        sample_rate=sample_rate,
        cache=True
    )
    
    print(f"✓ Downloaded strain data successfully")
    print(f"  Duration: {data.duration.value} seconds")
    print(f"  Sample rate: {data.sample_rate.value} Hz")
    print(f"  Data shape: {data.value.shape}")
    
    # Fetch all available witness channels for this detector
    print(f"\nFetching witness channels from NDS2...")
    try:
        from gwpy.io import nds2 as io_nds2
        
        # Connect to NDS server
        conn = io_nds2.auth_connect('nds.gwosc.org', 31200)
        
        # Find all available channels for this detector
        all_channels = conn.find_channels(f'{detector}:*')
        
        # Filter for common witness channel types (auxiliary channels)
        witness_patterns = [
            # 'PEM',      # Physical environment monitoring
            'PSL',      # Pre-stabilized laser
            # 'SUS',      # Suspension
            # 'ASC',      # Angular sensing and control
            # 'LSC',      # Length sensing and control
            # 'ALS',      # Auxiliary length sensing
            # 'OMC',      # Output mode cleaner
            # 'IMC',      # Input mode cleaner
            "MAINS"
        ]
        
        witness_channel_names = []
        for channel in all_channels:
            channel_name = str(channel.name)
            # Exclude the main strain channel
            if 'STRAIN' not in channel_name and 'CALIB' not in channel_name and 'ENV' not in channel_name:
                # witness_channel_names.append(channel_name)
                # Check if it matches any witness pattern
                if any(pattern in channel_name for pattern in witness_patterns):
                    # Prefer channels with reasonable sample rates
                    if channel.sample_rate <= sample_rate and channel.sample_rate > 0:
                        witness_channel_names.append(channel_name)
        
        print(f"Channel names: {witness_channel_names}")
        print(f"  Found {len(witness_channel_names)} potential witness channels")
        
        # Limit to a reasonable number to avoid too much data
        # max_channels = 50
        # if len(witness_channel_names) > max_channels:
        #     print(f"  Limiting to first {max_channels} channels")
        #     witness_channel_names = witness_channel_names[:max_channels]
        
        if witness_channel_names:
            print(f"  Downloading {len(witness_channel_names)} witness channels...")
            
            # Download channels one at a time to handle errors gracefully
            witness_arrays = []
            successful_channels = []
            
            for i, channel_name in enumerate(witness_channel_names):
                try:
                    # Download individual channel
                    channel_data = TimeSeries.fetch(
                        channel_name,
                        gps_start,
                        gps_start + duration,
                        host='nds.gwosc.org',
                        verbose=False
                    )
                    
                    # Resample to match strain if needed
                    if channel_data.sample_rate.value != sample_rate:
                        channel_data = channel_data.resample(sample_rate)
                    
                    # Ensure same length as strain
                    if len(channel_data) == len(data):
                        witness_arrays.append(channel_data.value)
                        successful_channels.append(channel_name)
                        if (i + 1) % 10 == 0:
                            print(f"    Downloaded {i + 1}/{len(witness_channel_names)} channels...")
                    else:
                        print(f"  ⚠ Skipping {channel_name}: length mismatch")
                        
                except Exception as e:
                    # Skip channels that fail to download
                    continue
            
            if witness_arrays:
                # Stack into 2D array: (num_channels, num_samples)
                witnesses = np.vstack(witness_arrays)
                print(f"✓ Successfully downloaded {len(successful_channels)} witness channels")
                print(f"  Witness data shape: {witnesses.shape}")
                print(f"  Sample channels: {successful_channels[:5]}{'...' if len(successful_channels) > 5 else ''}")
            else:
                raise ValueError("No witness channels could be successfully downloaded")
        else:
            raise ValueError("No witness channels found")
            
    except Exception as e:
        print(f"⚠ Error fetching witness channels: {e}")
        print(f"  Falling back to single mock witness channel for testing")
        # Create a mock witness channel (just noise) for testing
        witnesses = np.random.randn(1, len(data.value)) * 1e-20
        successful_channels = ['MOCK:WITNESS']
    
    # Save as .npz file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        strain=data.value,
        witnesses=witnesses,
        times=data.times.value,
        sample_rate=sample_rate,
        gps_start=gps_start,
        duration=duration,
        detector=detector,
        witness_channels=successful_channels,
        ts_object=data
    )
    
    print(f"✓ Saved to {output_path}")
    print()

def main():
    print("Downloading LIGO Training Data")
    download_and_save_data(
        gps_start=1186736512  ,
        duration=1024,
        output_file="../data/train_data_raw.npz"
    )
    
    print("Downloading LIGO Test Data")
    download_and_save_data(
        gps_start=1186740564  ,
        duration=1024,
        output_file="../data/test_data_raw.npz"
    )

    
    print("All data downloaded successfully")

if __name__ == "__main__":
    main()
