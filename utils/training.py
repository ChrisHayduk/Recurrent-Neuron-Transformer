# General imports
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from contextlib import nullcontext
import time
import math

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


def train_nanogpt(model, device, train_data_loader, val_data_loader, max_iters, batch_size):
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    weight_decay=1e-1
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    learning_rate = 6e-4 # max learning rate
    beta1=0.9
    beta2=0.95
    grad_clip=1.0
    eval_iters = 200
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    warmup_iters = max(1, int(max_iters / 30.0)) # how many steps to warm up for
    lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
    min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    decay_lr = True # whether to decay the learning rate
    master_process = True
    eval_only = False 
    gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
    log_interval = 1
    iter_num = 0

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

    def get_batch(split):
        loader = train_data_loader if split == 'train' else val_data_loader

        # randomly select a batch of sequences
        for batch_idx, (input_chunk, target_chunk) in enumerate(loader):
            x, y = input_chunk.to(device_type), target_chunk.to(device_type)
            return x, y

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)


    # training loop
    X, Y = get_batch('train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model # unwrap DDP container if needed
    running_mfu = -1.0
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break


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