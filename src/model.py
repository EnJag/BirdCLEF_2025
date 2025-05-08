import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import glob
from tqdm.notebook import tqdm  # Use tqdm.notebook for Jupyter, tqdm otherwise
import time
import copy

# --- 1. Configuration & Setup ---
DATA_DIR = 'train_audio'
NUM_CLASSES = 206
BATCH_SIZE = 32
NUM_EPOCHS = 25  # Adjust as needed
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Audio processing parameters
SAMPLE_RATE = 16000  # Target sample rate
N_MELS = 128        # Number of Mel bands
N_FFT = 1024        # FFT size
HOP_LENGTH = 512    # Hop length for STFT
FIXED_LENGTH_SECONDS = 5 # Process audio in fixed chunks (adjust as needed)
FIXED_LENGTH_FRAMES = int(FIXED_LENGTH_SECONDS * SAMPLE_RATE / HOP_LENGTH) + 1

# --- 2. Data Preparation ---

# Find all audio files and extract labels
all_filepaths = glob.glob(os.path.join(DATA_DIR, '*/*.wav')) # Adjust wildcard if needed (e.g., *.mp3)
if not all_filepaths:
    all_filepaths = glob.glob(os.path.join(DATA_DIR, '*/*.mp3')) # Try mp3
    if not all_filepaths:
        all_filepaths = glob.glob(os.path.join(DATA_DIR, '*/*.ogg')) # Try ogg
        if not all_filepaths:
             raise FileNotFoundError(f"Could not find audio files in {DATA_DIR}. Check paths and extensions.")


labels = [os.path.basename(os.path.dirname(fp)) for fp in all_filepaths]

# Encode labels to integers
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(labels)
print(f"Found {len(all_filepaths)} files belonging to {len(label_encoder.classes_)} classes.")
assert len(label_encoder.classes_) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes but found {len(label_encoder.classes_)}"


# Split data into training and validation sets
train_filepaths, val_filepaths, train_labels, val_labels = train_test_split(
    all_filepaths,
    integer_labels,
    test_size=0.2,        # 20% for validation
    random_state=42,      # for reproducibility
    stratify=integer_labels # ensure class distribution is similar in both sets
)

print(f"Training samples: {len(train_filepaths)}, Validation samples: {len(val_filepaths)}")

# --- 3. Audio Transformations ---

# Define the Mel Spectrogram transformation pipeline
mel_spectrogram_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
).to(DEVICE) # Move transform to device if possible

amplitude_to_db_transform = T.AmplitudeToDB().to(DEVICE)

# --- 4. PyTorch Dataset ---

class BirdSoundDataset(Dataset):
    def __init__(self, filepaths, labels, target_sample_rate, target_length_frames, transforms):
        self.filepaths = filepaths
        self.labels = labels
        self.target_sample_rate = target_sample_rate
        self.target_length_frames = target_length_frames
        self.transforms = transforms
        self.amplitude_to_db = T.AmplitudeToDB() # Keep on CPU for initial load flexibility

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        try:
            # Load audio waveform
            waveform, sample_rate = torchaudio.load(filepath)
            waveform = waveform.to(DEVICE) # Move waveform to device early

            # Resample if necessary
            if sample_rate != self.target_sample_rate:
                resampler = T.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate).to(DEVICE)
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Apply Mel Spectrogram transformation
            mel_spec = self.transforms(waveform) # Already on device
            mel_spec_db = amplitude_to_db_transform(mel_spec) # Convert to dB scale, also on device

            # --- Handle Fixed Length ---
            current_length = mel_spec_db.shape[2] # Shape: (channels, n_mels, time_frames)
            if current_length > self.target_length_frames:
                # Truncate (randomly or from start)
                start = torch.randint(0, current_length - self.target_length_frames + 1, (1,)).item()
                mel_spec_db = mel_spec_db[:, :, start:start + self.target_length_frames]
                # Alternatively: mel_spec_db = mel_spec_db[:, :, :self.target_length_frames] # Truncate from start
            elif current_length < self.target_length_frames:
                # Pad (usually with zeros, dB scale means padding with a low value like -80)
                padding_needed = self.target_length_frames - current_length
                # Pad reflects edges, constant pads with value=min_db, zero padding is not ideal in dB scale
                # Using constant padding with a value representing silence
                pad_value = -80.0 # A common value for silence in dB Mel spectrograms
                mel_spec_db = torch.nn.functional.pad(mel_spec_db, (0, padding_needed), mode='constant', value=pad_value)


            # Normalize (optional but often helpful) - per spectrogram
            mean = mel_spec_db.mean()
            std = mel_spec_db.std()
            if std > 1e-6: # Avoid division by zero
                 mel_spec_db = (mel_spec_db - mean) / std


            # Add channel dimension if needed for CNN (some models expect 3 channels)
            # If using a model like ResNet pre-trained on ImageNet (3 channels)
            mel_spec_db = mel_spec_db.repeat(1, 3, 1, 1) # Repeat across channel dim -> (1, 3, n_mels, time)
            mel_spec_db = mel_spec_db.squeeze(0) # Remove batch dim -> (3, n_mels, time)

            # --- End Handle Fixed Length ---

            return mel_spec_db, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            # Return a dummy tensor and label - or implement better error handling
            # Be careful with dummy data as it might skew training
            dummy_spec = torch.zeros((3, N_MELS, self.target_length_frames), device=DEVICE)
            dummy_label = torch.tensor(-1, dtype=torch.long) # Use an invalid label
            return dummy_spec, dummy_label

# Custom collate function to filter out bad samples
def collate_fn(batch):
    # Filter out items where the label is -1 (indicating an error)
    batch = [item for item in batch if item[1].item() != -1]
    if not batch: # If all items in batch failed
        return torch.empty(0), torch.empty(0) # Return empty tensors
    # Proceed with default collation if batch is not empty
    return torch.utils.data.dataloader.default_collate(batch)


# Create Datasets
train_dataset = BirdSoundDataset(train_filepaths, train_labels, SAMPLE_RATE, FIXED_LENGTH_FRAMES, mel_spectrogram_transform)
val_dataset = BirdSoundDataset(val_filepaths, val_labels, SAMPLE_RATE, FIXED_LENGTH_FRAMES, mel_spectrogram_transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)


# --- 5. Model Definition ---

# Example using a pre-trained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) # Load pre-trained weights

# Modify the first convolutional layer if input is not 3 channels (already handled above by repeating channel)
# If your spectrogram processing results in 1 channel, you might do:
# original_conv1 = model.conv1
# model.conv1 = nn.Conv2d(1, original_conv1.out_channels,
#                         kernel_size=original_conv1.kernel_size,
#                         stride=original_conv1.stride,
#                         padding=original_conv1.padding,
#                         bias=False)
# # Optional: Initialize weights of the new conv layer

# Modify the final fully connected layer for NUM_CLASSES outputs
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(DEVICE) # Move model to GPU/CPU

# --- 6. Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Optional: Learning rate scheduler
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# --- 7. Training Loop ---

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # Iterate over data.
            progress_bar = tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {epoch}")
            for inputs, labels in progress_bar:
                # Skip empty batches resulting from collation errors
                if inputs.numel() == 0:
                    continue

                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                batch_loss = loss.item() * inputs.size(0)
                batch_corrects = torch.sum(preds == labels.data).item()
                running_loss += batch_loss
                running_corrects += batch_corrects
                total_samples += labels.size(0) # Use labels.size(0) which accounts for filtered samples

                progress_bar.set_postfix(loss=batch_loss/inputs.size(0), acc=batch_corrects/inputs.size(0))


            # Ensure total_samples is not zero before division
            if total_samples == 0:
                 print(f"Warning: No valid samples processed in {phase} phase for epoch {epoch}. Skipping metrics calculation.")
                 epoch_loss = 0.0
                 epoch_acc = 0.0
            else:
                epoch_loss = running_loss / total_samples
                epoch_acc = running_corrects / total_samples


            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
                # if scheduler: scheduler.step() # Step the scheduler if using one
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                # Deep copy the model if it's the best validation accuracy so far
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f"*** New best validation accuracy: {best_acc:.4f} ***")
                    # Save the best model
                    torch.save(model.state_dict(), 'best_bird_classifier_model.pth')
                    print("Best model saved to best_bird_classifier_model.pth")


        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# Start Training
model_ft, history = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=NUM_EPOCHS)


# --- 8. (Optional) Plot Training History ---
import matplotlib.pyplot as plt

def plot_history(history):
    epochs = range(len(history['train_loss']))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)

# --- 9. (Optional) Inference Example ---

def predict_species(model, filepath, label_encoder, device, target_sample_rate, target_length_frames, transforms):
    model.eval() # Set model to evaluation mode

    try:
        waveform, sample_rate = torchaudio.load(filepath)
        waveform = waveform.to(device)

        # Apply the same preprocessing as during training
        if sample_rate != target_sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate).to(device)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
             waveform = torch.mean(waveform, dim=0, keepdim=True)

        mel_spec = transforms(waveform)
        mel_spec_db = T.AmplitudeToDB()(mel_spec)

        # Handle fixed length
        current_length = mel_spec_db.shape[2]
        if current_length > target_length_frames:
             mel_spec_db = mel_spec_db[:, :, :target_length_frames] # Truncate from start for consistency
        elif current_length < target_length_frames:
             padding_needed = target_length_frames - current_length
             pad_value = -80.0
             mel_spec_db = torch.nn.functional.pad(mel_spec_db, (0, padding_needed), mode='constant', value=pad_value)

        # Normalize (using stats from training if available, or per-spectrogram)
        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        if std > 1e-6:
            mel_spec_db = (mel_spec_db - mean) / std

        # Add batch and channel dimensions (repeat for 3 channels)
        mel_spec_db = mel_spec_db.repeat(1, 3, 1, 1) # -> (1, 3, n_mels, time)
        # mel_spec_db = mel_spec_db.unsqueeze(0) # Add batch dim -> (1, 1, n_mels, time) if using 1 channel


        with torch.no_grad():
            output = model(mel_spec_db)
            probabilities = torch.softmax(output, dim=1)
            predicted_index = torch.argmax(probabilities, dim=1).item()

        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        confidence = probabilities[0, predicted_index].item()

        return predicted_label, confidence

    except Exception as e:
        print(f"Error processing file {filepath} for prediction: {e}")
        return None, 0.0

# Example usage:
# test_audio_file = 'path/to/some/test_bird_sound.wav' # Replace with an actual path
# predicted_species, confidence = predict_species(
#     model_ft, # Use the trained model
#     test_audio_file,
#     label_encoder,
#     DEVICE,
#     SAMPLE_RATE,
#     FIXED_LENGTH_FRAMES,
#     mel_spectrogram_transform
# )

# if predicted_species:
#     print(f"Predicted Species: {predicted_species} with confidence {confidence:.4f}")