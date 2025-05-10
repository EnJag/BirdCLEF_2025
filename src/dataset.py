# src/simple_dataset.py
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
import joblib

from . import config as cfg

class BirdSoundDataset(Dataset):
    """
    A simplified dataset class for loading and processing bird audio.
    """
    def __init__(self, filepaths, labels, audio_cfg, mel_transform, db_transform, is_train=True, dataset_name="unknown"):
        self.filepaths = filepaths
        self.labels = labels # These are integer labels
        self.audio_cfg = audio_cfg
        self.mel_transform = mel_transform # .to(cfg.DEVICE) # Ensure transform is on device
        self.db_transform = db_transform # .to(cfg.DEVICE)   # Ensure transform is on device
        self.is_train = is_train
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        try:
            waveform, sr = torchaudio.load(filepath)
            # waveform = waveform.to(cfg.DEVICE) # Move waveform to device early

            # 1. Resample if necessary
            if sr != self.audio_cfg['sample_rate']:
                resampler = T.Resample(orig_freq=sr, new_freq=self.audio_cfg['sample_rate']) # .to(cfg.DEVICE)
                waveform = resampler(waveform)

            # 2. Convert to Mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 3. Apply Mel Spectrogram and dB conversion
            mel_spec = self.mel_transform(waveform)
            mel_spec_db = self.db_transform(mel_spec)

            # 4. Fix Spectrogram Length (padding or truncation)
            current_frames = mel_spec_db.shape[2]
            target_frames = self.audio_cfg['fixed_spec_frames']

            if current_frames > target_frames:
                if self.is_train: # Random crop for training augmentation
                    start = torch.randint(0, current_frames - target_frames + 1, (1,)).item()
                    mel_spec_db = mel_spec_db[:, :, start:start + target_frames]
                else: # Center or start crop for validation
                    mel_spec_db = mel_spec_db[:, :, :target_frames] # Simple truncation from start
            elif current_frames < target_frames:
                padding_needed = target_frames - current_frames
                # Pad with a value representing silence in dB scale
                mel_spec_db = torch.nn.functional.pad(mel_spec_db, (0, padding_needed), mode='constant', value=-80.0)
            
            # 5. Normalize (per spectrogram)
            mean = mel_spec_db.mean()
            std = mel_spec_db.std()
            if std > 1e-6: # Avoid division by zero for silent spectrograms
                mel_spec_db = (mel_spec_db - mean) / std
            
            # 6. Add channel dimension for CNN (e.g., ResNet expects 3 channels)
            # Input shape: (1, n_mels, fixed_spec_frames) -> (3, n_mels, fixed_spec_frames)
            mel_spec_db = mel_spec_db.repeat(3, 1, 1) # Repeat the single channel 3 times

            return mel_spec_db, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"!!!!!!!! ERROR IN {self.dataset_name.upper()} DATASET !!!!!!!!")
            print(f"Error processing file: {filepath}")
            print(f"Specific error: {type(e).__name__}: {e}")
            # import traceback # Uncomment for full traceback during deep debugging
            # traceback.print_exc()
            # Return a dummy tensor and an invalid label to be filtered by collate_fn
            dummy_spec = torch.zeros((3, self.audio_cfg['n_mels'], self.audio_cfg['fixed_spec_frames']), device=cfg.DEVICE)
            return dummy_spec, torch.tensor(-1, dtype=torch.long)

def simple_collate_fn(batch):
    """
    Filters out samples that failed to process (where label is -1).
    """
    # Filter out items where the label is -1 (indicating an error during __getitem__)
    batch = [item for item in batch if item[1].item() != -1]
    if not batch: # If all items in batch failed
        # Return empty tensors with correct first dimension for batch size
        # and subsequent dimensions matching expected output.
        # This helps avoid crashes if an entire batch fails.
        # Note: The shape of the empty tensor's data part (e.g., spec shape) needs to be correct.
        # For simplicity, if all fail, we return completely empty tensors.
        # The training loop should handle inputs.numel() == 0.
        return torch.empty(0), torch.empty(0)
    # Proceed with default collation if batch is not empty
    return torch.utils.data.dataloader.default_collate(batch)


def get_data_loaders():
    """
    Reads data from CSV, creates datasets and dataloaders.
    Returns: train_loader, val_loader, label_encoder, num_classes
    """
    print(f"Loading data from CSV: {cfg.TRAIN_CSV_PATH}")
    try:
        df = pd.read_csv(cfg.TRAIN_CSV_PATH)
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at {cfg.TRAIN_CSV_PATH}")
        raise
    
    # Construct full file paths
    all_filepaths = [os.path.join(cfg.TRAIN_AUDIO_DIR, fname) for fname in df['filename']]
    labels_str = df[cfg.TARGET_LABEL_COLUMN].astype(str).tolist() # Ensure labels are strings

    # Check how many files actually exist
    existing_filepaths = []
    existing_labels_str = []
    for fp, label_s in zip(all_filepaths, labels_str):
        if os.path.exists(fp):
            existing_filepaths.append(fp)
            existing_labels_str.append(label_s)
        # else: # Optional: print missing files
            # print(f"Warning: File not found {fp}, skipping.")
    
    if not existing_filepaths:
        raise ValueError("No valid audio file paths found. Check TRAIN_AUDIO_DIR and filenames in CSV.")
    print(f"Found {len(existing_filepaths)} existing audio files out of {len(all_filepaths)} listed in CSV.")

    # Encode labels
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(existing_labels_str)
    num_classes = len(label_encoder.classes_)
    print(f"Number of unique classes found ('{cfg.TARGET_LABEL_COLUMN}'): {num_classes}")
    
    # Save LabelEncoder
    os.makedirs(os.path.dirname(cfg.LABEL_ENCODER_SAVE_PATH), exist_ok=True)
    joblib.dump(label_encoder, cfg.LABEL_ENCODER_SAVE_PATH)
    print(f"LabelEncoder saved to {cfg.LABEL_ENCODER_SAVE_PATH}")

    # Split data
    train_files, val_files, train_labels, val_labels = train_test_split(
        existing_filepaths, integer_labels,
        test_size=cfg.VALIDATION_SPLIT_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=integer_labels if num_classes > 1 else None # Stratify if more than one class
    )

    # Audio processing parameters dictionary
    audio_params = {
        'sample_rate': cfg.SAMPLE_RATE,
        'n_mels': cfg.N_MELS,
        'n_fft': cfg.N_FFT,
        'hop_length': cfg.HOP_LENGTH,
        'fixed_spec_frames': cfg.FIXED_SPEC_FRAMES
    }

    # Define Mel Spectrogram transformations (to be passed to dataset)
    # These are initialized here and moved to device within the dataset or here.
    # For simplicity, let's assume they are stateless enough to be created on CPU first.
    mel_transform = T.MelSpectrogram(
        sample_rate=cfg.SAMPLE_RATE,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS
    ) # .to(cfg.DEVICE) # Can be moved to device here or in Dataset
    db_transform = T.AmplitudeToDB() # .to(cfg.DEVICE)

    # Create Datasets
    train_dataset = BirdSoundDataset(
        train_files, train_labels, audio_params, mel_transform, db_transform, is_train=True, dataset_name="train"
    )
    val_dataset = BirdSoundDataset(
        val_files, val_labels, audio_params, mel_transform, db_transform, is_train=False, dataset_name="val"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=simple_collate_fn # num_workers=0 for simplicity/debugging
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=simple_collate_fn
    )
    print(f"Train DataLoader: {len(train_loader)} batches. Val DataLoader: {len(val_loader)} batches.")
    return train_loader, val_loader, label_encoder, num_classes
