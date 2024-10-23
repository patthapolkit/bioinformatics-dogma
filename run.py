import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import esm
import numpy as np
import wandb
import os

# Define constants
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
ESM_EMBEDDING_DIM = 1280  # ESM2 output dimension
HIDDEN_DIM = 512

def parse_fasta_file(file_path):
    """
    Parse a FASTA file and return sequences and labels.
    """
    sequences = []
    labels = []
    
    current_sequence = ""
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_sequence:
                    sequences.append(current_sequence)
                    current_sequence = ""
                
                label = line.split()[-1]
                binary_label = 1 if label == "A-0" else 0
                labels.append(binary_label)
            else:
                current_sequence += line
        
        if current_sequence:
            sequences.append(current_sequence)
    
    return sequences, labels

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, model, alphabet, device='cpu'):
        self.sequences = sequences
        self.labels = labels
        self.model = model
        self.alphabet = alphabet
        self.device = device
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Convert sequence to tokens
        batch_tokens = self.alphabet.encode(sequence)
        
        # Convert to tensor and add batch dimension
        batch_tokens_tensor = torch.tensor(batch_tokens, dtype=torch.long)
        
        # Generate ESM2 embeddings
        with torch.no_grad():
            results = self.model(
                batch_tokens_tensor.unsqueeze(0).to(self.device), 
                repr_layers=[33],
                return_contacts=False
            )
        
        # Get per-token representations from the last layer
        token_embeddings = results["representations"][33].squeeze(0)
        
        # Use mean pooling to get sequence representation
        sequence_embedding = token_embeddings.mean(0)
        
        return (
            sequence_embedding.cpu(),  # Move back to CPU for DataLoader
            torch.tensor([label], dtype=torch.float)
        )

class ProteinSolubilityPredictor(nn.Module):
    def __init__(self, input_dim=ESM_EMBEDDING_DIM):
        super(ProteinSolubilityPredictor, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, 128)
        self.fc3 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

def collate_batch(batch):
    """
    Custom collate function for DataLoader
    """
    embeddings, labels = zip(*batch)
    embeddings = torch.stack(embeddings)
    labels = torch.stack(labels)
    return embeddings, labels

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, start_epoch=0):
    model = model.to(device)
    
    for epoch in range(start_epoch, num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0
        for batch_idx, (embeddings, labels) in enumerate(train_loader):
            print(f"Processing batch {batch_idx+1}/{len(train_loader)}")
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print(f"Finished epoch {epoch+1}/{num_epochs}")
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {train_loss_avg:.4f}')
        print(f'Validation Loss: {val_loss_avg:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.2f}%\n')
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss_avg,
            "val_loss": val_loss_avg,
            "val_accuracy": val_accuracy
        })
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model and optimizer state from checkpoint.
    """
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    else:
        print(f"No checkpoint found at '{checkpoint_path}', starting from scratch.")
        start_epoch = 0
    return model, optimizer, start_epoch

def main():
    print("Starting main function")
    
    # Initialize wandb
    wandb.init(project="protein-solubility-prediction")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load ESM-2 model
    print("Loading ESM-2 model...")
    model_name = "esm2_t33_650M_UR50D"
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    esm_model = esm_model.to(device)
    esm_model.eval()
    
    # Load and parse FASTA file
    print("Loading FASTA data...")
    fasta_file = "your_fasta_file.fasta"  # Replace with your file path
    sequences, labels = parse_fasta_file(fasta_file)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = ProteinDataset(X_train, y_train, esm_model, alphabet, device)
    val_dataset = ProteinDataset(X_val, y_val, esm_model, alphabet, device)

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_batch
    )
    
    # Initialize model, criterion, and optimizer
    print("Initializing model...")
    model = ProteinSolubilityPredictor(input_dim=ESM_EMBEDDING_DIM)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Log hyperparameters
    wandb.config.update({
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "hidden_dim": HIDDEN_DIM
    })
    
    # Check for checkpoint
    checkpoint_path = 'checkpoint.pth'
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    
    # Train the model
    print("Starting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device, start_epoch)

    # Save the model
    print("Saving model...")
    torch.save(model.state_dict(), 'protein_solubility_model.pt')
    wandb.save('protein_solubility_model.pt')

if __name__ == "__main__":
    main()