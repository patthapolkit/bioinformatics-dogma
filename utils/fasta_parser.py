def parse_fasta_file(file_path):
    """
    Parse a FASTA file and return sequences and labels.
    """
    sequences, labels = [], []
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