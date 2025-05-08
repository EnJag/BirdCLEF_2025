# train.py
import torch
import torch.optim as optim
import torch.nn as nn
import time
import copy
from tqdm import tqdm # Use tqdm for command line

# Import from our modules
from config import *
from dataset import get_data_loaders
from resnet18 import get_bird_classifier_model
from utils import plot_training_history

def train_model(model, criterion, optimizer, train_loader, val_loader, device,
                num_epochs=25, model_save_path='best_model.pth'):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"Starting training on {device}...")

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            progress_bar = tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {epoch}", unit="batch")
            for inputs, labels in progress_bar:
                if inputs.numel() == 0: # Skip empty batches from collate_fn
                    continue

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                current_batch_size = inputs.size(0)
                batch_loss = loss.item() * current_batch_size
                batch_corrects = torch.sum(preds == labels.data).item()
                running_loss += batch_loss
                running_corrects += batch_corrects
                total_samples += current_batch_size

                progress_bar.set_postfix(loss=batch_loss/current_batch_size, acc=batch_corrects/current_batch_size)
            
            if total_samples == 0:
                print(f"Warning: No valid samples processed in {phase} phase for epoch {epoch}.")
                epoch_loss = 0.0
                epoch_acc = 0.0
            else:
                epoch_loss = running_loss / total_samples
                epoch_acc = running_corrects / total_samples

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), model_save_path)
                    print(f"*** New best val acc: {best_acc:.4f}. Model saved to {model_save_path} ***")

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history

def main():
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    print("Loading data...")
    train_loader, val_loader, label_encoder_obj = get_data_loaders(
        data_dir=DATA_DIR,
        csv_path=TRAIN_CSV_PATH,
        target_label_col=TARGET_LABEL_COLUMN, # Pass the target label column
        batch_size=BATCH_SIZE,
        val_split_size=VALIDATION_SPLIT_SIZE,
        random_state=RANDOM_STATE,
        label_encoder_path=LABEL_ENCODER_PATH
    )
    print("Data loaded.")

    # 2. Initialize Model
    print("Initializing model...")
    model = get_bird_classifier_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)
    print("Model initialized.")

    # 3. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # Optional

    # 4. Train Model
    trained_model, history = train_model(
        model, criterion, optimizer, train_loader, val_loader,
        DEVICE, num_epochs=NUM_EPOCHS, model_save_path=MODEL_SAVE_PATH
    )

    # 5. Plot history
    if history['train_loss']: # Check if history is not empty
        print("Plotting training history...")
        plot_training_history(history)
    else:
        print("No training history to plot (perhaps training was interrupted or had no valid data).")

    print("Training finished.")

if __name__ == '__main__':
    main()