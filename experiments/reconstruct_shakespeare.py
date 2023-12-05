# General imports
import argparse
import torch

# Imports for the tokenizer, the dataset, and the model
from transformers import GPT2Tokenizer
from utils.datasets import TextDataLoader
from models.transformer_model import TransformerModel
from utils.training import train_shakespeare_transformer

# Set random seed for reproducibility
torch.manual_seed(0)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="VanillaTransformer", help="Name of model. Options are \
                        VanillaTransformer, StatefulTransformer, NanoGPT, TransformerXL")
    parser.add_argument("--data_path", type=str, default="data/shakespeare/tinyshakespeare.txt", help="Path to the data file")
    parser.add_argument("--chunk_size", type=int, default=2048, help="Size of the vocabulary")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--window_step_size", type=int, default=64, help="Size of the vocabulary")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="Dimension of the feedforward network")
    args = parser.parse_args()

    # Define the device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create the tokenizer
    tokenizer = 'gpt2'
    vocab_size = 50257  # NOTE: HARD CODED FOR GPT2

    # Create the data loader
    data_loader = TextDataLoader(file_path=args.data_path, seq_length=args.chunk_size, bpe_tokenizer=tokenizer, 
                                 batch_size=args.batch_size, vocab_size=vocab_size, split_ratio=0.8)
    train_loader, test_loader = data_loader.create_loaders()

    # Confirm model name is valid
    assert args.model_name in ['VanillaTransformer', 'StatefulTransformer', 'NanoGPT', 'TransformerXL'], \
        f"Invalid model name {args.model_name}"
    
    # Create the model
    if args.model_name == 'VanillaTransformer':
        model = TransformerModel(vocab_size=vocab_size, max_seq_length=args.max_seq_length, d_model=512, nhead=args.nhead, 
                                 num_encoder_layers=args.num_encoder_layers, num_decoder_layers=args.num_decoder_layers, 
                                 dim_feedforward=args.dim_feedforward).to(device)
    elif args.model_name == 'StatefulTransformer':
        raise NotImplementedError
    elif args.model_name == 'NanoGPT':
        raise NotImplementedError
    elif args.model_name == 'TransformerXL':
        raise NotImplementedError
    
    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    print(f"Training {args.model_name} on {args.data_path} with chunk size {args.chunk_size} and context window size {args.max_seq_length}")
    train_shakespeare_transformer(model=model, context_window=args.chunk_size, step_size=args.window_step_size, 
                                  data_loader=train_loader, optimizer=optimizer, num_epochs=args.num_epochs, 
                                  device=device, mask=False)