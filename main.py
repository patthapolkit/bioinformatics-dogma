import wandb
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from models.biLSTM_TextCNN import biLSTM_TextCNN
from models.predictor import ProteinSolubilityPredictor
from utils.fasta_parser import parse_fasta_file
from utils.collate import collate_batch
from utils.checkpoint import load_checkpoint
from datasets import EmbeddingsDataset
from train.train import train_model
from train.test import test_model
from transformers import AutoTokenizer, EsmForProteinFolding

def main():
    # Load configuration
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    wandb.init(project="protein-solubility-prediction")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_name = config['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm_model = EsmForProteinFolding.from_pretrained(model_name)
    esm_model = esm_model.to(device)
    esm_model.esm = esm_model.esm.half()
    torch.backends.cuda.matmul.allow_tf32 = True
    esm_model.eval()
    
    fasta_file = config['data']['val_path']
    sequences, labels = parse_fasta_file(fasta_file)

    sequences = sequences[:100]
    labels = labels[:100]

    X_train, X_temp, y_train, y_temp = train_test_split(sequences, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_dataset = EmbeddingsDataset(X_train, y_train, esm_model, tokenizer, device)
    val_dataset = EmbeddingsDataset(X_val, y_val, esm_model, tokenizer, device)
    # test_dataset = EmbeddingsDataset(X_test, y_test, esm_model, tokenizer, device)

    print(train_dataset[0][0].shape)
    # print(train_dataset[0][0])

    # if len(train_dataset[0][0].shape) == 2:  # if we have per residue embeddings they have an additional length dim
    #     collate_function = collate_batch
    # else:  # if we have reduced sequence wise embeddings use the default collate function by passing None
    #     collate_function = None

    # train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_function,drop_last=True)
    # val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],shuffle=True, collate_fn=collate_function,drop_last=True)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], collate_fn=collate_batch)
    # test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], collate_fn=collate_batch)
    
    # model = biLSTM_TextCNN(embeddings_dim=config['model']['embedding_dim'])
    model = ProteinSolubilityPredictor(input_dim=config['model']['embedding_dim'])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, config['training']['checkpoint_path'])
    
    train_model(model, train_loader, val_loader, criterion, optimizer, config['training']['epochs'], device, start_epoch)
    
    # Save the model state
    torch.save(model.state_dict(), config['output']['save_path'])
    wandb.save(config['output']['save_path'])
    
    # Test the model
    # test_model(model, test_loader, criterion, device)

if __name__ == "__main__":
    main()