# General imports
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt

# Torch imports
import torch
import torch.nn as nn



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


def train_shakespeare_transformer(model, train_loader, eval_loader, optimizer, num_epochs, device, mask=False,
                                  save_model_name='best_model.pth', save_loss_curves_name='loss_curves.png',
                                  save_losses_csv_name='losses.csv'):
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

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    loss_records = []

    for epoch in range(num_epochs):

        # Training Phase
        model.train()
        epoch_train_loss = 0
        for inputs, labels in tqdm(train_loader, total=len(train_loader), desc=f'Training: Epoch {epoch+1}/{num_epochs}', unit='batch', leave=False):
            input_seq = inputs.to(device)
            target_seq = labels.to(device)

            if mask:
                target_seq_mask = create_look_ahead_mask(target_seq.size(1)).to(device)
                target_seq_mask = target_seq_mask.unsqueeze(0)
            else:
                target_seq_mask = None

            optimizer.zero_grad()
            outputs = model(input_seq, tgt_mask=target_seq_mask)
            outputs = outputs.view(-1, outputs.size(-1))
            target_seq = target_seq.view(-1)
            loss = nn.CrossEntropyLoss()(outputs, target_seq)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for inputs, labels in tqdm(eval_loader, total=len(eval_loader), desc=f'Validating: Epoch {epoch+1}/{num_epochs}', unit='batch', leave=False):
                input_seq = inputs.to(device)
                target_seq = labels.to(device)

                if mask:
                    target_seq_mask = create_look_ahead_mask(target_seq.size(1)).to(device)
                    target_seq_mask = target_seq_mask.unsqueeze(0)
                else:
                    target_seq_mask = None

                outputs = model(input_seq, tgt_mask=target_seq_mask)
                outputs = outputs.view(-1, outputs.size(-1))
                target_seq = target_seq.view(-1)
                loss = nn.CrossEntropyLoss()(outputs, target_seq)

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(eval_loader)
        val_losses.append(avg_val_loss)

        # Create dictionary of loss records, append to list of results
        loss_records.append({
            'epoch': epoch+1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        })

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


def train_recurrent_shakespeare_transformer(model, context_window, step_size, train_loader, eval_loader, optimizer, num_epochs, device='cuda', mask=False,
                                            save_model_name='best_model.pth', save_loss_curves_name='loss_curves.png',
                                            save_losses_csv_name='losses.csv'):
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
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    loss_records = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Training: Epoch {epoch+1}/{num_epochs}', leave=False)

        for batch_idx, (input_chunk, target_chunk) in enumerate(progress_bar):
            batch_loss = 0
            hidden_layers = None  # Reset hidden layers for each batch

            for i in range(0, input_chunk.size(1) - context_window, step_size):
                input_seq = input_chunk[:, i:i+context_window].to(device)
                target_seq = target_chunk[:, i+1:i+context_window+1].to(device)

                optimizer.zero_grad()
                outputs, hidden_layers = model(inputs=input_seq, hidden_layers=hidden_layers)
                outputs = outputs.view(-1, outputs.size(-1))
                target_seq = target_seq.view(-1)

                loss = nn.CrossEntropyLoss()(outputs, target_seq)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_loss += loss.item()

            epoch_train_loss += batch_loss / (len(input_chunk) // step_size)
            progress_bar.set_postfix(loss=epoch_train_loss / (batch_idx + 1))

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for input_chunk, target_chunk in tqdm(eval_loader, total=len(eval_loader), desc=f'Validating: Epoch {epoch+1}/{num_epochs}', unit='batch', leave=False):
                batch_loss = 0
                hidden_layers = None  # Reset hidden layers for each batch

                for i in range(0, input_chunk.size(1) - context_window, step_size):
                    input_seq = input_chunk[:, i:i+context_window].to(device)
                    target_seq = target_chunk[:, i+1:i+context_window+1].to(device)

                    outputs, hidden_layers = model(inputs=input_seq, hidden_layers=hidden_layers)
                    outputs = outputs.view(-1, outputs.size(-1))
                    target_seq = target_seq.view(-1)

                    loss = nn.CrossEntropyLoss()(outputs, target_seq)
                    batch_loss += loss.item()

                epoch_val_loss += batch_loss / (len(input_chunk) // step_size)

        avg_val_loss = epoch_val_loss / len(eval_loader)
        val_losses.append(avg_val_loss)

        # Create dictionary of loss records, append to list of results
        loss_records.append({'epoch': epoch+1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})

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