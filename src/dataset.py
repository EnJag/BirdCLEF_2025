# src/data_utils.py
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pandas as pd # Import pandas
import joblib

from . import config # Import from our config file

# --- Audio Transformations (defined once, potentially moved to device) ---
mel_spectrogram_transform = T.MelSpectrogram(
    sample_rate=config.SAMPLE_RATE,
    n_fft=config.N_FFT,
    hop_length=config.HOP_LENGTH,
    n_mels=config.N_MELS
).to(config.DEVICE)

amplitude_to_db_transform = T.AmplitudeToDB().to(config.DEVICE)


class BirdSoundDataset(Dataset): # Keep this class as is
    def __init__(self, filepaths, labels, target_sample_rate, target_length_frames,
                 mel_transform, db_transform, device, is_train=True):
        self.filepaths = filepaths
        self.labels = labels
        self.target_sample_rate = target_sample_rate
        self.target_length_frames = target_length_frames
        self.mel_transform = mel_transform
        self.db_transform = db_transform
        self.device = device
        self.is_train = is_train

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        try:
            waveform, sample_rate = torchaudio.load(filepath)
            waveform = waveform.to(self.device)

            if sample_rate != self.target_sample_rate:
                resampler = T.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate).to(self.device)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            mel_spec = self.mel_transform(waveform)
            mel_spec_db = self.db_transform(mel_spec)

            current_length = mel_spec_db.shape[2]
            if current_length > self.target_length_frames:
                if self.is_train:
                    start = torch.randint(0, current_length - self.target_length_frames + 1, (1,)).item()
                    mel_spec_db = mel_spec_db[:, :, start:start + self.target_length_frames]
                else:
                    mel_spec_db = mel_spec_db[:, :, :self.target_length_frames]
            elif current_length < self.target_length_frames:
                padding_needed = self.target_length_frames - current_length
                pad_value = -80.0
                mel_spec_db = torch.nn.functional.pad(mel_spec_db, (0, padding_needed), mode='constant', value=pad_value)

            mean = mel_spec_db.mean()
            std = mel_spec_db.std()
            if std > 1e-6:
                 mel_spec_db = (mel_spec_db - mean) / std

            mel_spec_db = mel_spec_db.repeat(1, 3, 1, 1).squeeze(0)

            return mel_spec_db, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            # Fallback for problematic files - consider how you want to handle these
            dummy_spec = torch.zeros((3, config.N_MELS, self.target_length_frames), device=self.device)
            dummy_label = torch.tensor(-1, dtype=torch.long) # Invalid label
            return dummy_spec, dummy_label

def collate_fn(batch): # Keep this function as is
    batch = [item for item in batch if item[1].item() != -1]
    if not batch:
        return torch.empty(0), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)

def get_data_loaders(data_dir, csv_path, target_label_col, batch_size,
                     val_split_size, random_state, label_encoder_path):
    """
    Loads data based on a CSV file.
    Args:
        data_dir (str): Base directory where audio subfolders (like '1139490') are located.
        csv_path (str): Path to the train.csv file.
        target_label_col (str): Name of the column in CSV to use as labels.
        batch_size (int): Batch size for DataLoaders.
        val_split_size (float): Proportion of data for validation.
        random_state (int): Random seed for reproducibility.
        label_encoder_path (str): Path to save/load the LabelEncoder.
    Returns:
        train_loader, val_loader, label_encoder
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        raise
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        raise

    all_filepaths = []
    labels_str = []
    valid_files_count = 0
    missing_files_count = 0

    print(f"Reading audio file paths and labels from {csv_path}...")
    for index, row in df.iterrows():
        # The 'filename' column already contains the path relative to 'train_audio'
        # e.g., '1139490/CSA36385.ogg'
        relative_path = row['filename']
        full_path = os.path.join(data_dir, relative_path)

        if os.path.exists(full_path):
            all_filepaths.append(full_path)
            labels_str.append(str(row[target_label_col])) # Ensure label is string for encoder
            valid_files_count += 1
        else:
            # print(f"Warning: File not found: {full_path} (skipped)") # Optional: for debugging
            missing_files_count += 1

    if missing_files_count > 0:
        print(f"Warning: {missing_files_count} out of {len(df)} files listed in CSV were not found on disk.")

    if not all_filepaths:
        raise FileNotFoundError(f"No valid audio files found based on CSV {csv_path} and DATA_DIR {data_dir}. "
                                "Check paths and file existence.")

    print(f"Found {len(all_filepaths)} existing audio files with labels.")

    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(labels_str)

    # Save the fitted LabelEncoder
    try:
        joblib.dump(label_encoder, label_encoder_path)
        print(f"LabelEncoder saved to {label_encoder_path}")
    except Exception as e:
        print(f"Error saving LabelEncoder to {label_encoder_path}: {e}")


    num_unique_labels = len(label_encoder.classes_)
    print(f"Number of unique labels found: {num_unique_labels}")
    if num_unique_labels != config.NUM_CLASSES:
        print(f"WARNING: Number of unique labels ({num_unique_labels}) from CSV column '{target_label_col}' "
              f"does not match NUM_CLASSES in config ({config.NUM_CLASSES}). "
              f"Please verify NUM_CLASSES in config.py or your target_label_col.")
    # You might want to make this an assert or allow NUM_CLASSES to be determined dynamically:
    # config.NUM_CLASSES = num_unique_labels # If you want to set it dynamically

    train_filepaths, val_filepaths, train_labels, val_labels = train_test_split(
        all_filepaths, integer_labels,
        test_size=val_split_size,
        random_state=random_state,
        stratify=integer_labels if num_unique_labels > 1 else None # Stratify only if more than 1 class
    )

    print(f"Total files for training/validation: {len(all_filepaths)}")
    print(f"Training samples: {len(train_filepaths)}, Validation samples: {len(val_filepaths)}")

    train_dataset = BirdSoundDataset(
        train_filepaths, train_labels, config.SAMPLE_RATE, config.FIXED_LENGTH_FRAMES,
        mel_spectrogram_transform, amplitude_to_db_transform, config.DEVICE, is_train=True
    )
    val_dataset = BirdSoundDataset(
        val_filepaths, val_labels, config.SAMPLE_RATE, config.FIXED_LENGTH_FRAMES,
        mel_spectrogram_transform, amplitude_to_db_transform, config.DEVICE, is_train=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn
    )

    return train_loader, val_loader, label_encoder