from data_processing import CharDataset
from transformer import Transformer
import random
from torch import nn
import torch.optim as optim
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

class Trainer() :

    def __init__(self, datafile, block_size, batch_size, dim_emb, hidden_layer, num_head, num_transformer, learning_rate, iteration, batch_it_max, strong_residual = False):
        
        self.dataset = CharDataset(block_size, datafile)
        self.dataloader = DataLoader(self.dataset, batch_size, shuffle=True)

        self.model = Transformer(self.dataset.stoi, dim_emb, num_head, hidden_layer, num_transformer, block_size, strong_residual)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.window_size = block_size
        self.it = iteration
        self.batch_it_max = batch_it_max
        self.running_loss = []


    def run(self) :
        
        self.model.train()

        for _ in tqdm(range(self.it)):

            batch_it = 0
            for x_train, y_train in self.dataloader :
                
                batch_it += 1
                if batch_it > self.batch_it_max :
                    break

                self.optimizer.zero_grad()
                output = self.model(x_train)

                y_train_ont_hot = nn.functional.one_hot(y_train, self.dataset.get_vocab_size()).to(torch.float)
                loss = self.criterion(output, y_train_ont_hot)

                loss.backward()

                self.optimizer.step()

                self.running_loss.append(loss.item())


    def save_model(self, path="model.pth"):
        torch.save({'model_state_dict': self.model.state_dict()}, path)

    def load_model(self, path="model.pth"):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])



          # x_train = []
            # y_train = []
            # for _ in range(128):
            #     x, y = self.dataset.__getitem__(None)
            #     x_train.append(x)
            #     y_train.append(y)

            # x_train = torch.stack(x_train)
            # y_train = torch.stack(y_train)