import torch
from torch.utils.data import Dataset

class EmbeddingsDataset(Dataset):
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
        print(len(sequence))
        label = self.labels[idx]
        tokens = self.tokenizer(sequence, return_tensors='pt', add_special_tokens=False)
        with torch.no_grad():
            results = self.model(**tokens.to(self.device))
        sequence_embedding = results['states'][-1].mean(dim=1).type(torch.float16)
        return sequence_embedding, torch.tensor([label], dtype=torch.int8)