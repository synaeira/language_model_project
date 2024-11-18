from torch.utils.data import Dataset
import torch

class CharDataset(Dataset):
    """
    Emits batches of characters.

    Adapted from "https://github.com/karpathy/minGPT".
    """

    def __init__(self, config, data):

        with open(f'{data}', "r") as file:
            self.data = file.read()
        
        # get characters from the input data
        chars = sorted(set(self.data))
        self.stoi = { ch:i for i,ch in enumerate(chars) } # map characters to integer indices

    def get_vocab_size(self):
        return len(self.stoi)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        # encode every character to an integer
        # return the chunk and the shifted version as tensors
        
        block_size = 50
        chunk = [self.stoi[char] for char in self.data[idx : idx+block_size+1]]
        return torch.tensor(chunk[:-1], dtype=torch.int64), torch.tensor(chunk[1:], dtype=torch.int64)