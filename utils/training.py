# General imports
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from contextlib import nullcontext
import time
import math
import os
import functools

# Torch imports
import torch
import torch.nn as nn
import wandb

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from utils.datasets import TextDataLoader
from models.recurrent_neuron_transformer import RecurrentNeuronTransformer, ModelConfig


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12352'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# TODO: Using a mask breaks the training process due to a shape error. Needs to be fixed
def create_look_ahead_mask(size):
    """
    Creates a mask for the target sequence to prevent the model from cheating by looking ahead in the sequence.

    Args:
        size (int): The size of the target sequence
    
    Returns:
        torch.Tensor: The mask
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))

def fsdp_main(rank, world_size, args):
    if args["model_name"] == "StatefulTransformer":
        # Create the tokenizer
        tokenizer = 'gpt2'
        vocab_size = 50257  # NOTE: HARD CODED FOR GPT2

        # Create the data loader
        data_loader = TextDataLoader(file_path=args["data_path"], seq_length=args["chunk_size"], bpe_tokenizer=tokenizer, 
                                    batch_size=args["batch_size"], vocab_size=vocab_size, split_ratio=args["split_ratio"], 
                                    device=args["device"])
        train_loader, test_loader = data_loader.create_loaders()
        config_args = dict()

        sampler1 = DistributedSampler(train_loader, rank=rank, num_replicas=world_size, shuffle=True)
        sampler2 = DistributedSampler(test_loader, rank=rank, num_replicas=world_size, shuffle=False)

        for k, v in args.items():
            if k not in set(["model", "train_loader", "eval_loader", "optimizer", "devices", "save_model_name", "save_loss_curves_name", "save_losses_csv_name"]):
                config_args[k] = v

        model_config = ModelConfig(max_length=args["max_seq_length"], vocab_size=vocab_size, 
                                   n_layer=args["num_layers"], num_heads=args["nhead"], hidden_dim=args["dmodel"],
                                   dropout=args["dropout"], device=args["device"], recurrent_layers=args["recurrent_layers"])
        
        model = RecurrentNeuronTransformer(config=model_config)

        train_recurrent_shakespeare_transformer(model=model, train_loader=train_loader, eval_loader=test_loader,
                                                context_window=args["max_seq_length"], step_size=args["window_step_size"],
                                                optimizer=None, num_epochs=args["num_epochs"], args=config_args, device=args["device"], 
                                                mask=False, save_model_name=args["save_model_name"], 
                                                save_loss_curves_name=args["save_loss_curves_name"], 
                                                save_losses_csv_name=args["save_losses_csv_name"], distributed=True, rank=rank, world_size=world_size, sampler=sampler1)
        
    elif args.model_name == 'NanoGPT':
        raise NotImplementedError

    elif args.model_name == "TransformerXL":
        raise NotImplementedError
    
    else:
        raise NotImplementedError
        
    cleanup()


def train_shakespeare_transformer(model, context_window, step_size, train_loader, eval_loader, optimizer, num_epochs, 
                                  device, args, mask=False, save_model_name='best_model.pth', 
                                  save_loss_curves_name='loss_curves.png', save_losses_csv_name='losses.csv'):
    """
    Trains and validates a transformer model. Saves the best model based on validation loss and plots training curves.

    Args:
        model (nn.Module): The model to train and validate.
        train_loader (DataLoader): The data loader to use for training.
        eval_loader (DataLoader): The data loader to use for validation.
        optimizer (nn.optim): The optimizer to use for training.
        num_epochs (int): The number of epochs to train for.
        device (torch.device): The device to use for training.
        mask (bool): Whether to use a mask for the target sequence.
        save_model_name (str): The name of the file to save the best model to.
        save_loss_curves_name (str): The name of the file to save the loss curves to.
        save_losses_csv_name (str): The name of the file to save the loss records to.
    
    Returns:
        None
    """
    args["architecture"] = "vanilla_transformer"
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="transformer-testing",
        
        # track hyperparameters and run metadata
        config=args
    )
    wandb.define_metric("epoch")
    # set all other metrics to use this step
    wandb.define_metric("*", step_metric="epoch")

    model.to(device)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    loss_records = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        train_progress_bar = tqdm(train_loader, desc=f'Training: Epoch {epoch+1}/{num_epochs}', leave=False)

        for batch_idx, (input_chunk, target_chunk) in enumerate(train_progress_bar):
            batch_loss = 0

            for i in range(0, input_chunk.size(1) - context_window, step_size):

                # Create the input and target sequences
                input_seq = input_chunk[:, i:i+context_window].to(device)
                target_seq = target_chunk[:, i+1:i+context_window+1].to(device)

                # Optionally mask the target sequence
                if mask:
                    target_seq_mask = create_look_ahead_mask(target_seq.size(1)).to(device)
                    target_seq_mask = target_seq_mask.unsqueeze(0)
                else:
                    target_seq_mask = None
                
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(input_seq, tgt_mask=target_seq_mask)
                outputs = outputs.reshape(-1, outputs.size(-1))
                target_seq = target_seq.reshape(-1)

                # Calculate loss and backpropagate
                loss = nn.CrossEntropyLoss()(outputs, target_seq)
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()

            epoch_train_loss += batch_loss
            train_progress_bar.set_postfix(loss=batch_loss)

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Phase
        model.eval()
        epoch_val_loss = 0
        eval_progress_bar = tqdm(eval_loader, desc=f'Evaluating: Epoch {epoch+1}/{num_epochs}', leave=False)
        with torch.no_grad():
            for batch_idx, (input_chunk, target_chunk) in enumerate(eval_progress_bar):
                batch_loss = 0

                for i in range(0, input_chunk.size(1) - context_window, step_size):
                    # Create the input and target sequences
                    input_seq = input_chunk[:, i:i+context_window].to(device)
                    target_seq = target_chunk[:, i+1:i+context_window+1].to(device)

                    # Optionally mask the target sequence
                    if mask:
                        target_seq_mask = create_look_ahead_mask(target_seq.size(1)).to(device)
                        target_seq_mask = target_seq_mask.unsqueeze(0)
                    else:
                        target_seq_mask = None

                    # Forward pass
                    outputs = model(input_seq, tgt_mask=target_seq_mask)
                    outputs = outputs.reshape(-1, outputs.size(-1))
                    target_seq = target_seq.reshape(-1)

                    # Calculate loss
                    loss = nn.CrossEntropyLoss()(outputs, target_seq)

                    batch_loss += loss.item()

                epoch_val_loss += batch_loss
                eval_progress_bar.set_postfix(loss=batch_loss)

        avg_val_loss = epoch_val_loss / len(eval_loader)
        val_losses.append(avg_val_loss)

        # Create dictionary of loss records, append to list of results
        loss_records.append({
            'epoch': epoch+1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        })

        wandb.log({'epoch': epoch+1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss}, step=epoch+1)

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"experiment_results/{save_model_name}")

        print(f"Epoch {epoch+1}/{num_epochs} completed. Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")
    
    # Save the loss records to a CSV
    df = pd.DataFrame(loss_records)
    df.to_csv(f"experiment_results/{save_losses_csv_name}", index=False)

    # Plot the training and validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot([1 + int(x) for x in range(num_epochs)], train_losses, label='Training Loss')
    plt.plot([1 + int(x) for x in range(num_epochs)], val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"experiment_results/{save_loss_curves_name}")
    plt.show()


def train_recurrent_shakespeare_transformer(model, context_window, step_size, train_loader, eval_loader, optimizer, num_epochs, args, device='cuda', mask=False,
                                            save_model_name='best_model.pth', save_loss_curves_name='loss_curves.png',
                                            save_losses_csv_name='losses.csv', distributed = False, rank = -1, world_size = -1, sampler = None):
    """
    Trains and validates a recurrent transformer model. Saves the best model based on validation loss and plots training curves.

    Args:
        model (nn.Module): The model to train and validate.
        context_window (int): The size of the context window.
        step_size (int): The number of words to slide the context window by.
        train_loader (DataLoader): The data loader to use for training.
        eval_loader (DataLoader): The data loader to use for validation.
        optimizer (nn.optim): The optimizer to use for training.
        num_epochs (int): The number of epochs to train for.
        device (torch.device): The device to use for training.
        mask (bool): Whether to use a mask for the target sequence.
        save_model_name (str): The name of the file to save the best model to.
        save_loss_curves_name (str): The name of the file to save the loss curves to.
        save_losses_csv_name (str): The name of the file to save the loss records to.
    
    Returns:
        None
    """
    if distributed:
        assert rank >= 0 and world_size > 0
        setup(rank, world_size)
        device = rank
        torch.cuda.set_device(device)
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=100
        )
        model.to(device)
        model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy)
        optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
    else:
        model.to(device)
    
    if not distributed or rank == 0:
        args["architecture"] = "recurrent_neuron_transformer"
        
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="transformer-testing",
            
            # track hyperparameters and run metadata
            config=args
        )
        wandb.define_metric("epoch")
        # set all other metrics to use this step
        wandb.define_metric("*", step_metric="epoch")

    
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        loss_records = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0 if not distributed else torch.zeros(1).to(device)
        train_progress_bar = tqdm(train_loader, desc=f'Training: Epoch {epoch+1}/{num_epochs}', leave=False)

        if sampler:
            sampler.set_epoch(epoch)

        for batch_idx, (input_chunk, target_chunk) in enumerate(train_progress_bar):
            print(f"Batch {batch_idx}, rank/device {device}")
            batch_loss = 0
            hidden_layers = dict()

            for i in range(0, input_chunk.size(1) - context_window, step_size):
                # Create the input and target sequences
                input_seq = input_chunk[:, i:i+context_window].to(device)
                target_seq = target_chunk[:, i+1:i+context_window+1].to(device)
                
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs, hidden_layers = model(inputs=input_seq, hidden_layers=hidden_layers)
                outputs = outputs.reshape(-1, outputs.size(-1))
                target_seq = target_seq.reshape(-1)

                # Calculate loss and backpropagate
                loss = nn.CrossEntropyLoss()(outputs, target_seq)
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()

            epoch_train_loss += batch_loss
            train_progress_bar.set_postfix(loss=batch_loss)

        if distributed:
            dist.all_reduce(epoch_train_loss, op=dist.ReduceOp.SUM)
        if distributed and rank == 0:
            epoch_train_loss = epoch_train_loss.item()
            print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, epoch_train_loss / (len(train_progress_bar) * world_size)))
            avg_train_loss = epoch_train_loss / (len(train_progress_bar) * world_size)
            train_losses.append(avg_train_loss)
        elif not distributed:
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

        # Validation Phase
        model.eval()
        epoch_val_loss = 0 if not distributed else torch.zeros(1).to(device)
        eval_progress_bar = tqdm(eval_loader, desc=f'Evaluating: Epoch {epoch+1}/{num_epochs}', leave=False)
        with torch.no_grad():
            for batch_idx, (input_chunk, target_chunk) in enumerate(eval_progress_bar):
                batch_loss = 0
                hidden_layers = dict()

                for i in range(0, input_chunk.size(1) - context_window, step_size):
                    # Create the input and target sequences
                    input_seq = input_chunk[:, i:i+context_window].to(device)
                    target_seq = target_chunk[:, i+1:i+context_window+1].to(device)

                    # Forward pass
                    outputs, hidden_layers = model(inputs=input_seq, hidden_layers=hidden_layers)
                    outputs = outputs.reshape(-1, outputs.size(-1))
                    target_seq = target_seq.reshape(-1)

                    # Calculate loss
                    loss = nn.CrossEntropyLoss()(outputs, target_seq)

                    batch_loss += loss.item()

                epoch_val_loss += batch_loss
                eval_progress_bar.set_postfix(loss=batch_loss)

        if distributed:
            dist.all_reduce(epoch_val_loss, op=dist.ReduceOp.SUM)
        if distributed and rank == 0:
            epoch_val_loss = epoch_val_loss.item()
            print('Eval Epoch: {} \tLoss: {:.6f}'.format(epoch, epoch_val_loss / (len(eval_progress_bar) * world_size)))
            avg_val_loss = epoch_val_loss / (len(eval_loader) * world_size)
            val_losses.append(avg_val_loss)
        elif not distributed:
            avg_val_loss = epoch_val_loss / len(eval_loader)
            val_losses.append(avg_val_loss)
        
        if rank == 0 or not distributed:
            # Create dictionary of loss records, append to list of results
            loss_records.append({'epoch': epoch+1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})
            wandb.log({'epoch': epoch+1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss}, step=epoch+1)

            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if not distributed:
                    torch.save(model.state_dict(), f"experiment_results/{save_model_name}")
        if distributed:
            dist.barrier()
            states = model.state_dict()
            if rank == 0:
                torch.save(states, f"experiment_results/{save_model_name}")

                print(f"Epoch {epoch+1}/{num_epochs} completed. Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

    if rank == 0 or not distributed:
        # Save the loss records to a CSV
        df = pd.DataFrame(loss_records)
        df.to_csv(f"experiment_results/{save_losses_csv_name}", index=False)

        # Plot the training and validation loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"experiment_results/{save_loss_curves_name}")
        plt.show()        


def train_nanogpt(model, train_data_loader, val_data_loader, num_epochs, args, device='cuda'):
    args["architecture"] = "nano_gpt"
    # Initialize wandb
    wandb.init(project="transformer-testing", config=args)
    wandb.define_metric("epoch")
    # set all other metrics to use this step
    wandb.define_metric("*", step_metric="epoch")
    
    model.to(device)
    # Configuration for training
    weight_decay = 1e-1
    learning_rate = 6e-4
    beta1 = 0.9
    beta2 = 0.95
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    loss_records = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        train_progress_bar = tqdm(train_data_loader, desc=f'Training: Epoch {epoch+1}/{num_epochs}', leave=False)

        for batch_idx, (input_chunk, target_chunk) in enumerate(train_progress_bar):
            input_seq, target_seq = input_chunk.to(device), target_chunk.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs, loss = model(input_seq, target_seq)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            train_progress_bar.set_postfix(loss=epoch_train_loss / (batch_idx + 1))

        avg_train_loss = epoch_train_loss / len(train_data_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        val_progress_bar = tqdm(val_data_loader, desc=f'Evaluating: Epoch {epoch+1}/{num_epochs}', leave=False)

        with torch.no_grad():
            for batch_idx, (input_chunk, target_chunk) in enumerate(val_progress_bar):
                input_seq, target_seq = input_chunk.to(device), target_chunk.to(device)
                
                # Forward pass
                outputs, loss = model(input_seq, target_seq)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_data_loader)
        val_losses.append(avg_val_loss)

        # Record losses and log them
        loss_records.append({'epoch': epoch+1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})
        wandb.log({'epoch': epoch+1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss}, step=epoch+1)

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'best_model_epoch_{epoch+1}.pth')

        print(f"Epoch {epoch+1}/{num_epochs} completed. Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

def train_shakespeare_transformer_xl(model, context_window, step_size, train_loader, eval_loader, optimizer, num_epochs,
                                     device, args, save_model_name='best_model.pth',
                                     save_loss_curves_name='loss_curves.png', save_losses_csv_name='losses.csv'):
    """
    Trains and validates a Transformer XL model on long sequences.

    Args:
        model (nn.Module): The Transformer XL model to train.
        context_window (int): The context window size.
        step_size (int): The step size for sliding the context window.
        train_loader (DataLoader): The data loader for training data.
        eval_loader (DataLoader): The data loader for validation data.
        optimizer (nn.optim): The optimizer to use for training.
        num_epochs (int): The number of epochs to train for.
        device (torch.device): The device to use for training.
        mask (bool): Whether to use a mask for the target sequence.
        save_model_name (str): The name of the file to save the best model to.
        save_loss_curves_name (str): The name of the file to save the loss curves to.
        save_losses_csv_name (str): The name of the file to save the loss records to.

    Returns:
        None
    """
    args["architecture"] = "transformer_xl"
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="transformer-testing",
        
        # track hyperparameters and run metadata
        config=args
    )
    wandb.define_metric("epoch")
    # set all other metrics to use this step
    wandb.define_metric("*", step_metric="epoch")

    model.to(device)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    loss_records = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        train_progress_bar = tqdm(train_loader, desc=f'Training: Epoch {epoch+1}/{num_epochs}', leave=False)

        for batch_idx, (input_chunk, target_chunk) in enumerate(train_progress_bar):
            batch_loss = 0

            # Initialize empty memory of zeros
            mems = None

            # Slide the context window along the long sequence
            for i in range(0, input_chunk.size(1) - context_window, step_size):

                # Create the input and target sequences
                input_seq = input_chunk[:, i:i + context_window].to(device)
                target_seq = target_chunk[:, i + 1:i + context_window + 1].to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs, mems = model(input_seq, mems)  # Update mems

                # Reshape outputs and target for CrossEntropyLoss
                outputs = outputs.reshape(-1, outputs.size(-1))
                target_seq = target_seq.reshape(-1)

                # Calculate loss and backpropagate
                loss = nn.CrossEntropyLoss()(outputs, target_seq)
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()

            epoch_train_loss += batch_loss
            train_progress_bar.set_postfix(loss=batch_loss)

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        val_progress_bar = tqdm(eval_loader, desc=f'Evaluating: Epoch {epoch+1}/{num_epochs}', leave=False)

        with torch.no_grad():
            for batch_idx, (input_chunk, target_chunk) in enumerate(val_progress_bar):
                batch_loss = 0

                # Initialize empty memory of zeros
                mems = None

                # Slide the context window along the long sequence
                for i in range(0, input_chunk.size(1) - context_window, step_size):

                    # Create the input and target sequences
                    input_seq = input_chunk[:, i:i + context_window].to(device)
                    target_seq = target_chunk[:, i + 1:i + context_window + 1].to(device)
    
                    # Forward pass
                    outputs, mems = model(input_seq, mems)  # Update mems
    
                    # Reshape outputs and target for CrossEntropyLoss
                    outputs = outputs.reshape(-1, outputs.size(-1))
                    target_seq = target_seq.reshape(-1)
    
                    # Calculate loss and backpropagate
                    loss = nn.CrossEntropyLoss()(outputs, target_seq)
    
                    batch_loss += loss.item()
    
                epoch_val_loss += batch_loss
                val_progress_bar.set_postfix(loss=batch_loss)
    
            avg_val_loss = epoch_val_loss / len(eval_loader)
            val_losses.append(avg_val_loss)

            # Create dictionary of loss records, append to list of results
            loss_records.append({'epoch': epoch+1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})

            wandb.log({'epoch': epoch+1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss}, step=epoch+1)
    
            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"experiment_results/{save_model_name}")
    
            print(f"Epoch {epoch+1}/{num_epochs} completed. Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")
    
    # Save the loss records to a CSV
    df = pd.DataFrame(loss_records)
    df.to_csv(f"experiment_results/{save_losses_csv_name}", index=False)

    # Plot the training and validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"experiment_results/{save_loss_curves_name}")
    plt.show()