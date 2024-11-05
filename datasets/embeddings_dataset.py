import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, EsmModel

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, model, tokenizer, device='cpu'):
        self.sequences = sequences
        self.labels = labels
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(sequence, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            results = self.model(**tokens.to(self.device))
        token_embeddings = results.last_hidden_state.squeeze(0)
        sequence_embedding = token_embeddings.mean(0)
        return sequence_embedding.cpu(), torch.tensor([label], dtype=torch.float)