# src/config.py
import os
import torch

# --- Project Root ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Overall Configuration ---
# DATA_DIR should point to the folder containing the subfolders named by 'primary_label' IDs
# e.g., if your CSV has 'filename' as '1139490/CSA36385.ogg',
# and your files are in 'your_project_root_name/data/train_audio/1139490/CSA36385.ogg',
# then DATA_DIR should be 'your_project_root_name/data/train_audio'
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'train_audio')
TRAIN_CSV_PATH = os.path.join(ROOT_DIR, 'data', 'metadata', 'train.csv') # Path to your CSV file

MODEL_SAVE_DIR = os.path.join(ROOT_DIR, 'models')
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)
    print(f"Created directory: {MODEL_SAVE_DIR}")

MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'best_bird_classifier_model.pth')
LABEL_ENCODER_PATH = os.path.join(MODEL_SAVE_DIR, 'label_encoder.joblib')

# IMPORTANT: NUM_CLASSES should be the number of unique primary_labels OR common_names
# you intend to classify. You might need to determine this from your CSV.
# For now, let's assume it's still 206, but verify this.
# If you use 'common_name' as your label, count unique common_names in train.csv
NUM_CLASSES = 206 # Placeholder - VERIFY THIS from your CSV and chosen label column
TARGET_LABEL_COLUMN = 'common_name' # Choose: 'primary_label', 'common_name', or 'scientific_name'


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Training Hyperparameters ---
BATCH_SIZE = 32 # Make sure this is the variable you used in the notebook
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
VALIDATION_SPLIT_SIZE = 0.2 # Make sure this is the variable you used in the notebook
RANDOM_STATE = 42 # Make sure this is the variable you used in the notebook

# --- Audio Processing Parameters ---
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
FIXED_LENGTH_SECONDS = 5
FIXED_LENGTH_FRAMES = int(FIXED_LENGTH_SECONDS * SAMPLE_RATE / HOP_LENGTH) + 1