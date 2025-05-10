# src/simple_config.py
import torch
import os

# --- Project Root (assuming this file is in project_root/src/) ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Data Configuration ---
# Adjust DATA_BASE_DIR to where your 'data' folder (containing 'train_audio' and 'train.csv') is located
# If 'data' is in the project root, this is correct.
DATA_BASE_DIR = os.path.join(ROOT_DIR, 'data')
TRAIN_AUDIO_DIR = os.path.join(DATA_BASE_DIR, 'train_audio')
TRAIN_CSV_PATH = os.path.join(DATA_BASE_DIR, 'metadata', 'train.csv')
LABEL_ENCODER_SAVE_PATH = os.path.join(ROOT_DIR, 'models', 'simple_label_encoder.joblib')
MODEL_SAVE_DIR = os.path.join(ROOT_DIR, 'models') # Directory to save models

# Ensure model save directory exists
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)
    print(f"Created directory: {MODEL_SAVE_DIR}")

BEST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'simple_best_model.pth')


# --- Labeling ---
# Choose the column from your train.csv to use as the target label
# Options: 'primary_label', 'common_name', 'scientific_name'
TARGET_LABEL_COLUMN = 'common_name'
# NUM_CLASSES will be determined dynamically from the data by the LabelEncoder

# --- Training Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 # Smaller batch size for easier debugging initially
NUM_EPOCHS = 20  # Fewer epochs for quick testing
LEARNING_RATE = 0.001
VALIDATION_SPLIT_SIZE = 0.2
RANDOM_STATE = 42 # For reproducibility

# --- Audio Processing Parameters ---
SAMPLE_RATE = 16000  # Target sample rate for audio
N_MELS = 128         # Number of Mel frequency bands in the spectrogram
N_FFT = 1024         # Size of the Fast Fourier Transform window
HOP_LENGTH = 512     # Number of samples between successive STFT columns
FIXED_DURATION_SECONDS = 5 # Duration to pad/truncate audio to (in seconds)
# Calculated length of spectrogram frames based on duration
FIXED_SPEC_FRAMES = int(FIXED_DURATION_SECONDS * SAMPLE_RATE / HOP_LENGTH) + 1

print(f"Configuration Loaded: Device={DEVICE}, Audio Dir={TRAIN_AUDIO_DIR}, CSV={TRAIN_CSV_PATH}")
