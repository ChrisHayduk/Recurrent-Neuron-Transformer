import pandas as pd
import torch
from torch.utils.data import Dataset

class ShakespeareDataset(Dataset):
    def __init__(self, csv_file, chunk_length, tokenizer):
        # Read and concatenate text as before
        self.df = pd.read_csv(csv_file)
        self.all_text = ' '.join(self.df['PlayerLine'].tolist())

        # Tokenize the text
        self.tokenizer = tokenizer
        self.tokens = self.tokenizer.encode(self.all_text)

        # Set chunk length (2k)
        self.chunk_length = chunk_length

    def __len__(self):
        return (len(self.tokens) - self.chunk_length) // self.chunk_length + 1

    def __getitem__(self, index):
        start = index * self.chunk_length
        end = start + self.chunk_length
        return torch.tensor(self.tokens[start:end])

