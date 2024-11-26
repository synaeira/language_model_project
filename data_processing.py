from torch.utils.data import Dataset
import torch
import random

class CharDataset(Dataset):

    def __init__(self, block_size, data):

        with open(f'{data}', "r") as file:
            self.data = file.read()
        
        # get characters from the input data
        chars = sorted(set(self.data))
        self.stoi = { ch:i for i,ch in enumerate(chars) } # map characters to integer indices

        self.block_size = block_size

    def get_vocab_size(self):
        return len(self.stoi)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        # encode every character to an integer
        # return the chunk and the shifted version as tensors

        end = self.__len__() - self.block_size - 1
        idx = random.randint(0, end)
        
        chunk = [self.stoi[char] for char in self.data[idx : idx+self.block_size+1]]
        x = torch.tensor(chunk[:-1], dtype=torch.int64)
        y =  torch.tensor(chunk[1:], dtype=torch.int64)
        
        return x,y