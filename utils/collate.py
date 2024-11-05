import torch

def collate_batch(batch):
    """
    Custom collate function for DataLoader
    """
    embeddings, labels = zip(*batch)
    embeddings = torch.stack(embeddings)
    labels = torch.stack(labels)
    return embeddings, labels