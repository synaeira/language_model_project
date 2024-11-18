from data_processing import CharDataset
from transformer import Transformer
import random
from torch import nn
import torch.optim as optim
import torch
from tqdm import tqdm

class Trainer() :

    def __init__(self, datafile, block_size, dim_emb, hidden_layer, num_head, num_transformer, learning_rate, iteration):
        
        self.dataloader = CharDataset(block_size, datafile)

        self.model = Transformer(self.dataloader.stoi, dim_emb, num_head, hidden_layer, num_transformer)

        # à vérifier
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.window_size = block_size
        self.it = iteration
        self.running_loss = []

    def run(self) :
        
        self.model.train()

        for _ in tqdm(range(self.it)) :

            end = self.dataloader.__len__() - self.window_size - 1
            idx = random.randint(0, end)
            x_train, y_train = self.dataloader.__getitem__(idx)

            self.optimizer.zero_grad()
            output = self.model(x_train)

            y_train_ont_hot = nn.functional.one_hot(y_train, self.dataloader.get_vocab_size()).to(torch.float)

            loss = self.criterion(output, y_train_ont_hot)

            loss.backward()

            self.optimizer.step()

            self.running_loss.append(loss.item())


    def save_model(self, path="model.pth"):
        torch.save({'model_state_dict': self.model.state_dict()}, path)

    def load_model(self, path="model.pth"):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

