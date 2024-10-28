import torch
import argparse
from vocab import build, decode_caption
from datasets import RadiologyDataset, collate_fn, transform
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from rich.console import Console
from rich.table import Table
from rich.progress import track
from model import ModelHandler  

console = Console()

def evaluate_model(model_handler, dataset_path, vocab_required=True, tokenizer=None):
    console.print("[bold green]Starting evaluation...[/bold green]")

    # Build vocabulary if required
    if vocab_required:
        console.print("[bold blue]Building vocabulary ...[/bold blue]")
        vocab = build()
    else:
        vocab = None

    # Load dataset
    console.print("[bold blue]Loading dataset ...[/bold blue]")
    test_dataset = RadiologyDataset(
        f"{dataset_path}/captions.txt",
        f"{dataset_path}/images",
        vocab if vocab_required else tokenizer,  # Use vocab or tokenizer
        transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    bleu_references = []
    references = []
    candidates = []

    # Iterate over dataset using Rich's progress tracking
    for images, captions, lengths in track(test_loader, description="Evaluating...", console=console):
        with torch.no_grad():
            # Generate captions using ModelHandler
            decoded_outputs = model_handler.generate_caption(images, vocab=vocab, lengths=lengths)

            # Decode captions according to vocab usage
            if vocab_required:
                decoded_captions = [decode_caption(caption.tolist(), vocab) for caption in captions]
            else:
                decoded_captions = captions

            # Update references and candidates for evaluation metrics
            bleu_references.extend([[ref] for ref in decoded_captions])
            references.extend(decoded_captions)
            candidates.extend(decoded_outputs)

    # Calculate BLEU score
    bleu_score = corpus_bleu(bleu_references, candidates)

    # Calculate ROUGE score
    rouge = Rouge()
    rouge_score, _ = rouge.compute_score(
        {i: [" ".join(ref)] for i, ref in enumerate(references)},
        {i: [" ".join(candidate)] for i, candidate in enumerate(candidates)}
    )

    # Calculate CIDEr score
    cider = Cider()
    cider_score, _ = cider.compute_score(
        {i: [" ".join(ref)] for i, ref in enumerate(references)},
        {i: [" ".join(candidate)] for i, candidate in enumerate(candidates)}
    )

    # Display results using Rich's table
    table = Table(title="Evaluation Metrics")
    table.add_column("Metric", justify="center", style="cyan", no_wrap=True)
    table.add_column("Score", justify="center", style="magenta")

    table.add_row("BLEU", f"{bleu_score:.4f}")
    table.add_row("ROUGE", f"{rouge_score:.4f}")
    table.add_row("CIDEr", f"{cider_score:.4f}")

    console.print(table)
    console.print("[bold green]Evaluation completed![/bold green]")

# Main function to handle command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Image Captioning Models")
    parser.add_argument('--model_type', type=str, required=True, help="Type of model (e.g., baseline, CLIP, BLIP)")
    parser.add_argument('--model_weights', type=str, required=True, help="Path to model weights (e.g., 'model.pth')")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--vocab_required', type=bool, default=True, help="Whether the model requires a vocabulary (True/False)")
    parser.add_argument('--tokenizer', type=str, default=None, help="Pretrained tokenizer (optional, only for tokenizer-based models)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the ModelHandler with necessary parameters
    model_handler = ModelHandler(
        model_type=args.model_type,
        model_weights=args.model_weights,
        device=device,
        vocab_required=args.vocab_required,
        tokenizer=args.tokenizer
    )

    # Evaluate the model using ModelHandler
    evaluate_model(
        model_handler=model_handler, 
        dataset_path=args.dataset_path, 
        vocab_required=args.vocab_required, 
        tokenizer=model_handler.processor if args.tokenizer else None
    )
