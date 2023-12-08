"""
Module containing main training loop for experimentation. Future work should expand upon the training loop to track
and store metrics, and to save the model at regular intervals.

Lightweight Example usage:
python -m experiments.reconstruct_shakespeare --data_path='data/shakespeare/tinyshakespeare_100_lines.txt' --num_epochs=5 --chunk_size=512 --max_seq_length=256 --num_decoder_layers=2 --nhead=1

NanoGPT Example usage:
python -m experiments.reconstruct_shakespeare --data_path='data/shakespeare/tinyshakespeare_100_lines.txt' --model_name=NanoGPT --num_epochs=5 --chunk_size=256 --block_size=256 --nembd=384 --nhead=6 --num_layer=6 --max_iters=20 --batch_size=12
"""

# General imports
import argparse
import torch
import os

import torch.multiprocessing as mp

# Imports for the tokenizer, the dataset, and models
from transformers import GPT2Tokenizer
from utils.datasets import TextDataLoader
from models.vanilla_transformer_model import VanillaTransformerModel
from models.recurrent_neuron_transformer import RecurrentNeuronTransformer, ModelConfig
from models.nanogpt_model import NanoGPT, GPTConfig
from models.transformer_xl import TransformerXL
from utils.training import train_shakespeare, fsdp_main

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
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="Dimension of the feedforward network")
    parser.add_argument("--dmodel", type=int, default=512, help="Dimension of the model")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Ratio of train to test data")
    parser.add_argument("--window_step_size", type=int, default=1, help="Size of the vocabulary")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--nembd", type=int, default=768, help="Dimension of the embedding layer for NanoGPT")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers for NanoGPT")
    parser.add_argument("--block_size", type=int, default=1024, help="Block size for NanoGPT")
    parser.add_argument("--max_iters", type=float, default=0.0, help="Max iternations for NanoGPT")
    parser.add_argument("--recurrent_layers", type=str, default="all", help="Which layers to make recurrent in Recurrent Neuron Transformer. Possible values: all, qkv, proj, none")
    parser.add_argument("--distributed", type=bool, default=False, help="Whether to run training in distributed mode or on a single GPU")
    args = parser.parse_args()

    print(os.getcwd())
    if not os.path.isdir('experiment_results'):
        os.mkdir('experiment_results')

    # Define the device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # If using a CUDA GPU, clear the memory
    if device == torch.device('cuda'):
        torch.cuda.empty_cache()

    # Create the tokenizer
    tokenizer = 'gpt2'
    vocab_size = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency

    # Create the data loader
    data_loader = TextDataLoader(file_path=args.data_path, seq_length=args.chunk_size, bpe_tokenizer=tokenizer, 
                                 batch_size=args.batch_size, vocab_size=vocab_size, split_ratio=args.split_ratio, 
                                 device=device)
    train_loader, test_loader = data_loader.create_loaders()

    # Confirm model name is valid
    assert args.model_name in ['VanillaTransformer', 'StatefulTransformer', 'NanoGPT', 'TransformerXL'], \
        f"Invalid model name {args.model_name}"
    
    # Create the model
    if args.model_name == 'VanillaTransformer' and not args.distributed:
        model = VanillaTransformerModel(vocab_size=vocab_size, max_seq_length=args.max_seq_length, d_model=args.dmodel, 
                                        nhead=args.nhead, num_decoder_layers=args.num_layers, 
                                        dim_feedforward=args.dim_feedforward)
        
    elif args.model_name == 'StatefulTransformer' and not args.distributed:
        model_config = ModelConfig(max_length=args.max_seq_length, vocab_size=vocab_size, 
                                   n_layer=args.num_layers, num_heads=args.nhead, hidden_dim=args.dmodel,
                                   dropout=args.dropout, device=device, recurrent_layers=args.recurrent_layers)
        model = RecurrentNeuronTransformer(config=model_config)

    elif args.model_name == 'NanoGPT' and not args.distributed:
        model_args = dict(n_layer=args.num_layers, n_head=args.nhead, n_embd=args.nembd, block_size=args.max_seq_length,
                          bias=False, vocab_size=vocab_size, dropout=args.dropout) # start with model_args from command line
        gptconf = GPTConfig(**model_args)
        model = NanoGPT(gptconf)
    
    elif args.model_name == 'TransformerXL' and not args.distributed:
        model = TransformerXL(vocab_size=vocab_size, chunk_size=args.chunk_size, max_seq_length=args.max_seq_length, 
                              d_model=args.dmodel, nhead=args.nhead, num_layers=args.num_layers, 
                              dropout=args.dropout)

    # Define the name of the best model and the loss curves
    save_model_name = f"{args.model_name}_best_model.pth"
    save_loss_curves_name = f"{args.model_name}_loss_curves.png"
    save_losses_csv_name = f"{args.model_name}_losses.csv"

    # Train the model
    print(f"Training {args.model_name} on {args.data_path} with chunk size {args.chunk_size}, context window size {args.max_seq_length}, and step size {args.window_step_size}")
    
    if not args.distributed:
        # Create the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_shakespeare(model=model, train_loader=train_loader, eval_loader=test_loader,
                                            context_window=args.max_seq_length, step_size=args.window_step_size,
                                            optimizer=optimizer, num_epochs=args.num_epochs, args=vars(args), device=device, 
                                            mask=False, save_model_name=save_model_name, 
                                            save_loss_curves_name=save_loss_curves_name, 
                                            save_losses_csv_name=save_losses_csv_name)
    else:
        print(f"Starting distributed run for {args.model_name}")
        WORLD_SIZE = torch.cuda.device_count()

        args = vars(args)
        args["save_model_name"] = save_model_name
        args["save_loss_curves_name"] = save_loss_curves_name
        args["save_losses_csv_name"] = save_losses_csv_name
        args["device"] = device
        mp.spawn(fsdp_main,
            args=(WORLD_SIZE, args),
            nprocs=WORLD_SIZE,
            join=True)