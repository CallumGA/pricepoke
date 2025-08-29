

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from network import PricePredictor, get_dataloaders


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
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    print("Finished Training.")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    # high number, early stopping will find the best one
    NUM_EPOCHS = 200
    # stop if val loss doesn't improve for 10 epochs in a row
    PATIENCE = 10
    MODEL_SAVE_PATH = "/Users/callumanderson/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/models/price_predictor.pth"

    # data, model, loss, optimizer
    train_loader, val_loader, input_size = get_dataloaders(batch_size=BATCH_SIZE)
    model = PricePredictor(input_size=input_size)
    
    # BCEWithLogitsLoss for binary classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # begin training
    train_model(model, criterion, optimizer, train_loader, val_loader, NUM_EPOCHS, PATIENCE, MODEL_SAVE_PATH)