#calling main to train the data/h5
def main():
    # Load configuration
    # with open('configs/config.yaml', 'r') as file:
    #     config = yaml.safe_load(file)

    # wandb.init(project="protein-solubility-prediction")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    # Load precomputed embeddings
    dataset = ProteinEmbeddingsDataset('/content/drive/MyDrive/CU_Bioinformatics/project/sample_embeddings.h5')

    # Split data into train, validation, and test sets
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'])

    for i, (inputs, labels) in enumerate(train_loader):
        print(inputs.shape)
        print(labels.shape)
        break

    # Initialize model
    # model = ProteinSolubilityPredictor(input_dim=config['model']['embedding_dim'])
    model = biLSTM_TextCNN(embeddings_dim=config['model']['embedding_dim'])
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])

    # Load checkpoint if available
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, config['training']['checkpoint_path'])

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, config['training']['epochs'], device, start_epoch)

    # Save the model state
    # torch.save(model.state_dict(), config['output']['save_path'])
    # wandb.save(config['output']['save_path'])

    # Test the model
    test_model(model, test_loader, criterion, device)

if __name__ == "__main__":
    main()
