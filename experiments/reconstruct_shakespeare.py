# General imports
import os
import numpy as np
from tqdm.notebook import tqdm
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

# Set random seed for reproducibility
torch.manual_seed(0)