import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console
from rich.progress import track
from model import ModelHandler  # Import your ModelHandler class
from datasets import RadiologyDataset, collate_fn, transform
from vocab import build, decode_caption
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence

console = Console()

def train_model(model_handler, train_loader, val_loader, epochs, save_path, vocab_required=True, tokenizer=None, lr=1e-4, vocab=None):
    """
    Trains the model and runs validation at the end of each epoch.

    Args:
        model_handler (ModelHandler): The model handler instance.
        train_loader (DataLoader): Training dataset loader.
        val_loader (DataLoader): Validation dataset loader.
        epochs (int): Number of training epochs.
        save_path (str): Path to save the trained model weights.
        vocab_required (bool): Whether the model requires a vocabulary.
        tokenizer (AutoTokenizer, optional): Pretrained tokenizer if using CLIP or BLIP.
        lr (float): Learning rate for the optimizer.
    """
    device = model_handler.device
    model = model_handler.model

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()  # You can modify this if needed

    for epoch in range(epochs):
        console.print(f"[bold green]Epoch {epoch+1}/{epochs}[/bold green]")

        # Training Phase
        model.train()
        train_loss = 0.0
        for images, captions, lengths in track(train_loader, description="Training...", console=console):
            images, captions = images.to(device), captions.to(device)
            
            optimizer.zero_grad()
            outputs = model_handler.forward_train(images, captions, lengths, vocab_required, tokenizer)
            # Pack captions to align them with LSTM outputs
            packed_captions = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False)
            # Flatten outputs and packed captions for loss calculation
            loss = criterion(outputs, packed_captions.data)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        console.print(f"Train Loss: {avg_train_loss:.4f}")

        # Validation Phase
        model.eval()
        val_loss, bleu_references, candidates = 0.0, [], []
        with torch.no_grad():
            for images, captions, lengths in track(val_loader, description="Validating...", console=console):
                images, captions = images.to(device), captions.to(device)
                
                outputs = model_handler.forward_train(images, captions, lengths, vocab_required, tokenizer)
                packed_captions = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False)
            
                loss = criterion(outputs, packed_captions.data)
                val_loss += loss.item()

                decoded_outputs = model_handler.generate_caption(images, vocab=vocab, lengths=lengths)
                # decoded_captions = [decode_caption(caption.tolist(), vocab) for caption in captions]
                decoded_captions = [caption[:length].tolist() for caption, length in zip(captions, lengths)]

                bleu_references.extend([[ref] for ref in decoded_captions])
                candidates.extend(decoded_outputs)

        avg_val_loss = val_loss / len(val_loader)
        console.print(f"Validation Loss: {avg_val_loss:.4f}")
        print(bleu_references[0], candidates[0])
        # Compute BLEU score for validation
        bleu_score = corpus_bleu(bleu_references, candidates)
        console.print(f"Validation BLEU Score: {bleu_score:.4f}")

        # Save model weights after each epoch
        save_file = f"{save_path}_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), save_file)
        console.print(f"[bold yellow]Model saved at {save_file}[/bold yellow]")

def load_datasets(dataset_path, vocab_required, tokenizer, batch_size=4):
    """
    Loads the training and validation datasets.

    Args:
        dataset_path (str): Path to dataset.
        vocab_required (bool): Whether vocabulary is required.
        tokenizer (AutoTokenizer, optional): Tokenizer if using pretrained models like CLIP or BLIP.
        batch_size (int): Batch size for data loading.

    Returns:
        Tuple[DataLoader, DataLoader, vocab]: Train and validation data loaders, and the vocabulary object.
    """
    console.print("[bold blue]Loading datasets...[/bold blue]")

    if vocab_required:
        vocab = build()
    else:
        vocab = None  # If not using vocab, pass tokenizer
    
    # Assuming dataset structure like "train/radiology" and "val/radiology"

    train_dataset = RadiologyDataset(
        f"{dataset_path}/train/radiology/captions.txt", 
        f"{dataset_path}/train/radiology/images", 
        vocab, 
        transform
    )

    val_dataset = RadiologyDataset(
        f"{dataset_path}/validation/radiology/captions.txt", 
        f"{dataset_path}/validation/radiology/images", 
        vocab, 
        transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    return train_loader, val_loader, vocab

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Fine-tune Image Captioning Models")
    parser.add_argument('--model_type', type=str, required=True, help="Type of model (e.g., baseline, CLIP, BLIP)")
    parser.add_argument('--model_weights', type=str, help="Path to load pretrained model weights")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to dataset directory")
    parser.add_argument('--vocab_required', type=bool, default=True, help="Whether the model requires a vocabulary (True/False)")
    parser.add_argument('--tokenizer', type=str, help="Pretrained tokenizer (optional, for models like CLIP or BLIP)")
    parser.add_argument('--save_path', type=str, required=True, help="Directory path to save model weights")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for optimizer")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets and vocabulary
    train_loader, val_loader, vocab = load_datasets(
        dataset_path=args.dataset_path,
        vocab_required=args.vocab_required,
        tokenizer=args.tokenizer,
        batch_size=args.batch_size
    )

    # Initialize ModelHandler with vocab size
    vocab_size = len(vocab) if args.vocab_required else None

    model_handler = ModelHandler(
        model_type=args.model_type,
        model_weights=args.model_weights,
        device=device,
        vocab_size=vocab_size,  # Pass vocab_size if required
        tokenizer=args.tokenizer
    )

    # Train the model
    train_model(
        model_handler=model_handler,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_path=args.save_path,
        vocab_required=args.vocab_required,
        vocab=vocab,
        tokenizer=model_handler.processor if args.tokenizer else None,
        lr=args.learning_rate
    )
