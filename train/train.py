import torch
import wandb

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, start_epoch=0):
    model = model.to(device)
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # Validation loop omitted for brevity
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            # "val_loss": val_loss / len(val_loader), # Add validation metrics
        })