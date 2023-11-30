import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

class ShakespeareDataset(Dataset):
    def __init__(self, csv_file, seq_length):
        # Read the CSV file
        self.df = pd.read_csv(csv_file)

        # Concatenate all the lines into one large string
        self.all_text = ' '.join(self.df['PlayerLine'].tolist())

        # Initialize the tokenizer (GPT-2 tokenizer as an example)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Tokenize the text
        self.tokens = self.tokenizer.encode(self.all_text)

        # Sequence length
        self.seq_length = seq_length

    def __len__(self):
        return len(self.tokens) - self.seq_length

    def __getitem__(self, index):
        # Fetch a sequence of tokens and its next token (for prediction)
        return (torch.tensor(self.tokens[index:index+self.seq_length]),
                torch.tensor(self.tokens[index+1:index+self.seq_length+1]))
