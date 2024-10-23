import torch
from Bio import SeqIO
import esm
import numpy as np
import h5py
import argparse

def load_esm_model():
    """
    Load the ESM2 model and tokenizer
    """
    # Load ESM-2 model (8M parameter version)
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()  # Set to evaluation mode
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, alphabet, device

def read_fasta(fasta_file):
    """
    Read sequences from a FASTA file
    """
    sequences = []
    ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
        ids.append(record.id)
    return ids, sequences

def get_embeddings(model, alphabet, sequences, device, batch_tokens=1024):
    """
    Generate embeddings for sequences using ESM2 model
    """
    embeddings = []
    
    # Process each sequence
    for seq in sequences:
        # Tokenize sequence
        batch_converter = alphabet.get_batch_converter()
        _, _, tokens = batch_converter([("", seq)])
        tokens = tokens.to(device)
        
        # Generate embeddings
        with torch.no_grad():
            results = model(tokens, repr_layers=[6])
            embedding = results["representations"][6].cpu().numpy()
        
        # Take mean over sequence length (excluding start/end tokens)
        per_protein = embedding.mean(axis=1)
        embeddings.append(per_protein[0])
    
    return np.array(embeddings)

def save_embeddings(output_file, ids, embeddings):
    """
    Save embeddings to HDF5 file
    """
    with h5py.File(output_file, 'w') as f:
        # Create main dataset for embeddings
        embed_dset = f.create_dataset('embeddings', data=embeddings)
        
        # Create dataset for sequence IDs
        dt = h5py.special_dtype(vlen=str)
        id_dset = f.create_dataset('sequence_ids', (len(ids),), dtype=dt)
        id_dset[:] = ids

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate ESM2 embeddings from FASTA file')
    parser.add_argument('input_fasta', help='Input FASTA file')
    parser.add_argument('output_file', help='Output HDF5 file')
    args = parser.parse_args()
    
    # Load model
    print("Loading ESM2 model...")
    model, alphabet, device = load_esm_model()
    
    # Read sequences
    print("Reading sequences from FASTA file...")
    ids, sequences = read_fasta(args.input_fasta)
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = get_embeddings(model, alphabet, sequences, device)
    
    # Save results
    print("Saving embeddings to file...")
    save_embeddings(args.output_file, ids, embeddings)
    
    print(f"Done! Processed {len(sequences)} sequences.")

if __name__ == "__main__":
    main()