import h5py

# Read the embeddings
with h5py.File('embed.h5', 'r') as f:
    # Get embeddings
    embeddings = f['embeddings'][:]
    # Get sequence IDs
    sequence_ids = f['sequence_ids'][:]
    
# Print results
for i, (seq_id, embedding) in enumerate(zip(sequence_ids, embeddings)):
    print(f"Sequence {seq_id}: Embedding shape {embedding.shape}")