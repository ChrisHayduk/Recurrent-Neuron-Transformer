import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tiktoken
import os


class TextDataset(Dataset):
    def __init__(self, tokens, seq_length, bpe_tokenizer, vocab_size, device):
        self.tokens = tokens
        self.tokenizer = tiktoken.get_encoding(bpe_tokenizer)
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.device = device

    def __len__(self):
        return len(self.tokens) - self.seq_length - 1

    def __getitem__(self, idx):
        input_seq = torch.tensor(self.tokens[idx : idx+self.seq_length], device=self.device)
        target_seq = torch.tensor(self.tokens[idx+1 : idx+1+self.seq_length], device=self.device)
        return input_seq, target_seq


class TextDataLoader:
    def __init__(self, file_path, seq_length, bpe_tokenizer, batch_size, vocab_size, device, split_ratio=0.8):
        self.file_path = file_path
        self.seq_length = seq_length
        self.bpe_tokenizer = bpe_tokenizer
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.device = device
        self.split_ratio = split_ratio
    
    def load_and_tokenize(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return text
        except IOError:
            print(f"Error opening/reading {self.file_path}")
            return None
    
    def _create_datasets(self):
        text = self.load_and_tokenize()
        tokenizer = tiktoken.get_encoding(self.bpe_tokenizer)
        tokens = tokenizer.encode_ordinary(text)
        split_index = int(len(tokens) * self.split_ratio)
        train_tokens = tokens[:split_index]
        test_tokens = tokens[split_index:]
        train_dataset = TextDataset(train_tokens, self.seq_length, self.bpe_tokenizer, self.vocab_size, self.device)
        test_dataset = TextDataset(test_tokens, self.seq_length, self.bpe_tokenizer, self.vocab_size, self.device)
        return train_dataset, test_dataset

    def create_loaders(self):
        train_dataset, test_dataset = self._create_datasets()
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        return train_loader, test_loader


def print_sequences(loader, tokenizer, mode, max_batches):
    current_batch = 1
    for input_seq, target_seq in loader:
        if current_batch > max_batches:
            break
        print(f'----- {mode} Batch {current_batch} -----')
        print(f'input seq: {input_seq}')
        print(f'decoded input texts: {[tokenizer.decode(seq.tolist()) for seq in input_seq]}\n')
        print(f'target_seq: {target_seq}')
        print(f'decoded target text: {[tokenizer.decode(seq.tolist()) for seq in target_seq]}\n')
        current_batch += 1


if __name__ == '__main__':
    seq_length = 5
    batch_size = 2
    batch_count = 4
    file_path = os.path.join(os.getcwd(), 'data/shakespeare', 'tinyshakespeare.txt')
    bpe_tokenizer = 'gpt2'
    vocab_size = 50257
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = TextDataLoader(file_path, seq_length, bpe_tokenizer, batch_size, vocab_size, device)
    train_loader, test_loader = data_loader.create_loaders()
    tokenizer = tiktoken.get_encoding(bpe_tokenizer)
    print_sequences(train_loader, tokenizer, 'Training', batch_count)
    print_sequences(test_loader, tokenizer, 'Test', batch_count)