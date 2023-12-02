import pandas as pd
import torch
from torch.utils.data import Dataset

class ShakespeareDataset(Dataset):
    def __init__(self, csv_file, chunk_length, tokenizer):
        self.df = pd.read_csv(csv_file)
        all_text = ' '.join(self.df['PlayerLine'].tolist())

        # Tokenize the text in smaller chunks
        max_length = tokenizer.model_max_length  # 1024 for models like GPT-2
        self.tokens = []

        for i in range(0, len(all_text), max_length):
            chunk = all_text[i:i+max_length]
            self.tokens.extend(tokenizer.encode(chunk))

        self.chunk_length = chunk_length

    def __len__(self):
        return (len(self.tokens) - self.chunk_length) // self.chunk_length + 1

    def __getitem__(self, index):
        start = index * self.chunk_length
        end = min(start + self.chunk_length, len(self.tokens))
        return torch.tensor(self.tokens[start:end])

