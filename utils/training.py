# General imports
import os
import numpy as np
from tqdm import tqdm
import copy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer

# Torch imports
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

# Imports for the tokenizer, the dataset, and the model
from transformers import GPT2Tokenizer
from utils.datasets import TextDataLoader
from models.transformer_model import TransformerModel



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


def train_shakespeare_transformer(model, data_loader, optimizer, num_epochs, device, mask=False):
    """
    Trains a transformer model to reproduce large chunks of Shakespeare's plays by sliding a context window along a 
    larger piece of text. The model is trained to predict the next word in the sequence given the context window.

    Args:
        model (nn.Module): The model to train
        context_window (int): The size of the context window
        step_size (int): The number of words to slide the context window by
        data_loader (DataLoader): The data loader to use for training
        optimizer (nn.optim): The optimizer to use for training
        num_epochs (int): The number of epochs to train for
        device (torch.device): The device to use for training
        mask (bool): Whether to use a mask for the target sequence or not
    
    Returns:
        None
    """

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, labels in tqdm(data_loader, total=len(data_loader), 
                                   desc=f'Training: Epoch {epoch+1}/{num_epochs}', unit='batch', leave=False):
            # Move the data to the device
            input_seq = inputs.to(device)
            target_seq = labels.to(device)
            
            # Optionally create a mask for the target sequence
            if mask == True:
                target_seq_mask = create_look_ahead_mask(target_seq.size(1)).to(device)
                target_seq_mask = target_seq_mask.unsqueeze(0)
            else:
                target_seq_mask = None

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_seq, target_seq, tgt_mask=target_seq_mask)
            outputs = outputs.view(-1, outputs.size(-1))
            target_seq = target_seq.view(-1)

            # Calculate loss and backpropagate
            loss = nn.CrossEntropyLoss()(outputs, target_seq)
            loss.backward()
            optimizer.step()

            # Logging the loss and update progress bar
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} completed. Loss: {epoch_loss/len(data_loader)}")


def train_recurrent_shakespeare_transformer(model, context_window, step_size, data_loader, optimizer, num_epochs, 
                                            device='cuda', mask=False):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)

        for batch_idx, (input_chunk, target_chunk) in enumerate(progress_bar):
            # Initialize batch loss
            batch_loss = 0

            # Reset hidden layers at the start of each batch
            hidden_layers = dict()

            for i in range(0, input_chunk.size(1) - context_window, step_size):
                print(f"Chunk starting at position {i} in batch {batch_idx}")

                # Create input and target sequences
                input_seq = input_chunk[:, i:i+context_window].to(device)
                target_seq = target_chunk[:, i+1:i+context_window+1].to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs, hidden_layers = model(inputs=input_seq, hidden_layers=hidden_layers)
                outputs = outputs.view(-1, outputs.size(-1))
                target_seq = target_seq.view(-1)

                # Calculate loss
                loss = nn.CrossEntropyLoss()(outputs, target_seq)
                loss.backward()  # Backpropagate on each loss
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_loss += loss.item()  # Accumulate the scalar loss

            # Update running loss for the epoch
            epoch_loss += batch_loss

            # Update progress bar
            progress_bar.set_postfix(loss=epoch_loss / (batch_idx + 1))

        print(f"Epoch {epoch+1}/{num_epochs} completed. Average batch loss: {epoch_loss / len(data_loader)}")