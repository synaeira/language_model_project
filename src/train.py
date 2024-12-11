from data_processing import CharDataset
from model.transformer import Transformer
import random
from torch import nn
import torch.optim as optim
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb

class Trainer() :

    def __init__(self, datafile=None, block_size=None, batch_size=None, dim_emb=None, hidden_layer=None, 
                 num_head=None, num_transformer=None, learning_rate=None, iteration=None, load_path=None):
        
        if load_path:
            self.load_model(load_path)
        else:
            if not all([datafile, block_size, batch_size, dim_emb, hidden_layer, num_head, num_transformer, learning_rate, iteration]):
                raise ValueError("Missing parameters.")
        
        self.datafile = datafile

        self.dataset = CharDataset(block_size, datafile)
        self.dl_train = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
        
        self.model = Transformer(self.dataset.stoi, dim_emb, num_head, hidden_layer, num_transformer, block_size)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.window_size = block_size
        self.it_max = iteration
        self.running_loss = []


    def run(self) :

        wandb.init(project=f"loss")

        self.model.train()

        with tqdm(total=self.it_max, desc="Training Progress", unit="batch") as pbar:

            for i, (x_train, y_train) in enumerate(self.dl_train):

                self.optimizer.zero_grad()
                output = self.model(x_train)


                loss = self.criterion(output.transpose(-1, -2), y_train)
                loss.backward()

                wandb.log({f"loss" : loss})


                self.optimizer.step()

                self.running_loss.append(loss.item())

                pbar.update(1)
                if i > self.it_max :
                    break
        wandb.finish()


    def save_model(self, path="model.pth"):
        torch.save({'model_state_dict': self.model.state_dict()}, path)

    def load_model(self, path="model.pth"):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])