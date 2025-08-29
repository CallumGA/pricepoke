

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from network import PricePredictor, get_dataloaders
import config


def train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    num_epochs,
    patience,
    model_save_path,
):
    """
    Training loop with early stopping.
    """
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_no_improve = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Starting training on {device}...")

    for epoch in range(num_epochs):
        # training
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass + back prop + optimize weights/bias
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item() * features.size(0)

            # calculate accuracy
            predicted = torch.round(torch.sigmoid(outputs))
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        epoch_train_loss = train_running_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct / train_total

        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * features.size(0)

                # calculate accuracy
                predicted = torch.round(torch.sigmoid(outputs))
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}"
        )

        # early stopping check - we stop the training if the model has not improved by 10 epochs
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation loss decreased. Saving model with Val Acc: {epoch_val_acc:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    print("Finished Training.")
    print(f"Best Model Performance -> Validation Loss: {best_val_loss:.4f} | Validation Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    # high number, early stopping will find the best one
    NUM_EPOCHS = 200
    # stop if val loss doesn't improve for 10 epochs in a row
    PATIENCE = 10

    # data, model, loss, optimizer
    train_loader, val_loader, input_size = get_dataloaders(batch_size=BATCH_SIZE)
    model = PricePredictor(input_size=input_size)
    
    # --- Handle Class Imbalance with Weighted Loss ---
    # Calculate weights to penalize errors on the minority class more heavily.
    # This is crucial when one class (e.g., 'False') dominates the dataset.
    full_data = pd.read_csv(config.INPUT_CSV_PATH)
    class_counts = full_data[config.TARGET_COL].value_counts()
    count_false = class_counts.get(0, 0)
    count_true = class_counts.get(1, 0)

    if count_true > 0:
        # The raw ratio can be too aggressive, leading the model to always predict the
        # minority class. Taking the square root is a common technique to dampen
        # the weight while still penalizing errors on the minority class.
        raw_ratio = count_false / count_true
        pos_weight = np.sqrt(raw_ratio)
        print(f"Class Imbalance Detected: {count_false} 'False' vs {count_true} 'True' (Ratio: {raw_ratio:.2f}).")
        print(f"Applying a dampened positive class weight of {pos_weight:.2f} to compensate.")
    else:
        pos_weight = 1.0  # Default if there are no positive samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight_tensor = torch.tensor([pos_weight], device=device)

    # pass in a penalty weight to the loss function for getting a "true" wrong
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # begin training
    train_model(model, criterion, optimizer, train_loader, val_loader, NUM_EPOCHS, PATIENCE, config.MODEL_SAVE_PATH)