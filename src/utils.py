# src/simple_train_utils.py
import torch
import torch.optim as optim
import torch.nn as nn
import time
import copy
from tqdm import tqdm

from . import config as cfg
from .resnet18 import get_bird_model

def train_one_epoch(model, dataloader, criterion, optimizer, device, phase_name="Train"):
    """
    Trains the model for one epoch.
    Returns: epoch_loss, epoch_accuracy
    """
    if phase_name == "Train":
        model.train()  # Set model to training mode
    else:
        model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    total_samples_processed = 0
    
    # Use tqdm for progress if not in a notebook, or a simpler loop
    # For script execution, tqdm is fine. For notebook, you might control iteration manually.
    # For simplicity here, we'll include tqdm.
    progress_bar = tqdm(dataloader, desc=f"{phase_name} Phase", unit="batch", leave=False)

    for inputs, labels in progress_bar:
        if inputs.numel() == 0: # Skip if batch is empty due to collate_fn filtering
            continue

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase_name == "Train"): # Enable grads only in training
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            if phase_name == "Train":
                loss.backward()
                optimizer.step()
        
        current_batch_size = inputs.size(0)
        running_loss += loss.item() * current_batch_size
        running_corrects += torch.sum(preds == labels.data).item()
        total_samples_processed += current_batch_size
        
        if total_samples_processed > 0:
             progress_bar.set_postfix(
                 loss=running_loss/total_samples_processed,
                 acc=running_corrects/total_samples_processed
             )

    if total_samples_processed == 0:
        print(f"Warning: No valid samples processed in {phase_name} phase this epoch.")
        return 0.0, 0.0

    epoch_loss = running_loss / total_samples_processed
    epoch_acc = running_corrects / total_samples_processed
    
    return epoch_loss, epoch_acc

def run_training_pipeline(num_classes_for_model):
    """
    Main function to set up and run the training pipeline.
    This can be called from a Jupyter notebook.
    Args:
        num_classes_for_model (int): Number of classes determined by LabelEncoder.
    Returns:
        trained_model, history_log
    """
    print(f"--- Starting Training Pipeline ---")
    print(f"Device: {cfg.DEVICE}")
    print(f"Number of epochs: {cfg.NUM_EPOCHS}")
    print(f"Learning rate: {cfg.LEARNING_RATE}")
    print(f"Batch size: {cfg.BATCH_SIZE}")

    # 1. Get DataLoaders (label_encoder is also returned but we mainly need num_classes here)
    # The actual label_encoder object might be useful for prediction later.
    # get_simple_data_loaders is defined in simple_dataset.py
    # For this script to call it, it would need to be imported.
    # However, it's better if the notebook calls get_simple_data_loaders and passes them.
    # For simplicity, let's assume data_loaders are passed or this function is part of a larger flow.
    # For now, this function will focus on model init and training loop.
    # The notebook will handle data loading and then call this.

    # This function will expect train_loader and val_loader as arguments
    # OR it will call get_simple_data_loaders itself.
    # To keep it callable from notebook which already loads data:
    # We'll assume train_loader, val_loader are obtained *before* calling this.
    # This function needs num_classes.

    # 2. Initialize Model
    print(f"Initializing model for {num_classes_for_model} classes...")
    model = get_bird_model(num_classes=num_classes_for_model, pretrained=True)
    model = model.to(cfg.DEVICE)

    # 3. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    # --- Training Loop ---
    history_log = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_model_state = None

    start_time = time.time()

    # The notebook will call get_simple_data_loaders
    # For this function to be self-contained for script execution (if desired), it would need to load data.
    # Let's refine this: the notebook will get data_loaders and num_classes, then pass them.

    print(f"\n--- This function expects train_loader and val_loader to be provided ---")
    print(f"--- Please ensure they are loaded in your calling script/notebook ---")
    # This function will be modified to accept train_loader and val_loader

    # The actual training loop will be called by the notebook, passing the loaders.
    # So, this function is more of a conceptual placeholder for the pipeline logic
    # that the notebook will orchestrate.

    # Let's restructure: Notebook calls data loading, then calls a training function.

    # This function will be simplified to just orchestrate the training if loaders are provided.
    # The main call from the notebook will look like:
    # train_loader, val_loader, _, num_classes = get_simple_data_loaders()
    # model, history = run_training_process(train_loader, val_loader, num_classes)

    # So, let's rename and repurpose this.
    # The notebook will be the primary driver.
    # This file will contain the epoch training logic.
    # The `main` part of this script (if run directly) would be for testing the script itself.

    print("--- `run_training_pipeline` is a conceptual guide ---")
    print("--- The actual training loop will be orchestrated by the notebook ---")
    print("--- by calling `train_one_epoch` repeatedly. ---")
    
    # This function is not directly callable as a full pipeline without data loaders.
    # The notebook will handle the overall flow.
    # Return None or raise an error if called directly without proper setup.
    return None, None # Placeholder

# Example of how the notebook would use train_one_epoch:
# (This is conceptual, actual execution is in the notebook)
# if __name__ == '__main__':
#     print("This script contains utility functions for training.")
#     print("Please run the training process from the experiment_notebook.ipynb.")
    
    # To make this script runnable for testing its own components:
    # from .simple_dataset import get_simple_data_loaders # For script-based test
    # print("Attempting to run a test sequence (if script is run directly):")
    # try:
    #     train_loader, val_loader, _, num_classes_test = get_simple_data_loaders()
    #     print(f"Test: Loaded data, num_classes = {num_classes_test}")
    #     test_model = get_simple_bird_model(num_classes=num_classes_test, pretrained=False).to(cfg.DEVICE)
    #     test_criterion = nn.CrossEntropyLoss()
    #     test_optimizer = optim.Adam(test_model.parameters(), lr=cfg.LEARNING_RATE)
        
    #     print("Test: Training one epoch...")
    #     loss, acc = train_one_epoch(test_model, train_loader, test_criterion, test_optimizer, cfg.DEVICE, "Test Train")
    #     print(f"Test Train Epoch 1: Loss={loss:.4f}, Acc={acc:.4f}")
        
    #     print("Test: Validating one epoch...")
    #     loss, acc = train_one_epoch(test_model, val_loader, test_criterion, test_optimizer, cfg.DEVICE, "Test Val")
    #     print(f"Test Val Epoch 1: Loss={loss:.4f}, Acc={acc:.4f}")
        
    # except Exception as e:
    #     print(f"Error during script self-test: {e}")
    #     import traceback
    #     traceback.print_exc()

