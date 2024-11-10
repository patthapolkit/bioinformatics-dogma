import torch
import yaml
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding
from utils.fasta_parser import parse_fasta_file


def generate_embeddings():
    torch.cuda.empty_cache() # Clear GPU memory
    
    # Load configuration
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    print("Loading ESMFold model...")
    model_name = config['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm_model = EsmForProteinFolding.from_pretrained(model_name)
    esm_model = esm_model.to(device)
    esm_model.eval()

    # Load dataset
    # fasta_file = config['data']['test_path']
    # sequences, labels = parse_fasta_file(fasta_file)

    df = pd.read_csv('data/train_sample.csv')
    sequences = df['sequence'].tolist()
    labels = df['label'].tolist()

    # print(f"Loaded {len(sequences)} sequences")
    # for i in range(10):
    #     print(sequences[i])
    #     print(labels[i])

    # sequences = sequences[:20]
    # labels = labels[:20]

    # sequences = ['MARAGNLAGRRMRKARAAGALTLVGALALAACSGGGGDTNADGEAAELECSSEAVADQPWKAAEPREFSLLWTDWADYPITDTWEFFDEIEKRTNVKLKLTNIPFSDATEKRSLLISAGDAPQIIPLVYTGEERQFAASGAVVPLSDYIDYMPNFKKYTEEWDLVDMVDDLRQEDGKYYMTPGLQEVSVPVFTLIIRKDVFDEVGAPEPDTWEDLQEGLALIKEKYPDSYPLADGFEAWSMINYAAHAFGTVGGWGFGDGAWWDEEKGEFVYAATTDGYKDMVTYFRGLHDAGLLDAESFTASNDGGGTVVEKVAAEKVFAFSGGSWTVQEFGTALEAAGVTDYELVQIAPPAGPAGNNVEPRNFWNGFMLTADAAKDENFCDLLHFTDWLYYNPEARELIQWGVEGKHFTKEGGKYTLNPEFSLKNLNMNPDAPVDLKKDLGYANDVFAGSTESRELKESYNVPAFVQYIDDVQTKREPREPFPPHPLDEAELEQSSLLGTPLKDTVDTATLEFILGQRPLSDWDAYVAQLEGQGLQSYMDLINGAYKRAAEGQD']
    # labels = [1]

    output_file = config['data']['train_embedding_path']

    with h5py.File(output_file, 'w') as f:
        labels_list = []
        
        with torch.no_grad():
            for i, (sequence, label) in enumerate(tqdm(zip(sequences, labels), desc="Generating embeddings", total=len(sequences))):
                
                print(len(sequence))

                if len(sequence) > 556:
                    print(f"Skipping sequence {i} due to length > 556")
                    continue

                # Tokenize and encode each sequence individually
                inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False).to(device)
                outputs = esm_model(inputs.input_ids)
                
                # Store the full embedding for each sequence
                seq_id = f"val_{i}"
                f.create_dataset(seq_id, data=outputs['states'][-1].cpu().numpy())  # Shape: (seq_len, embedding_dim)

                labels_list.append(label)

        # Save labels in a separate dataset
        f.create_dataset("labels", data=np.array(labels_list))

    print(f"Saved embeddings to {output_file}")

if __name__ == "__main__":
    generate_embeddings()
