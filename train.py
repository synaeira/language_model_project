from data_processing import CharDataset
from transformer import Transformer
import random
from torch import nn
import torch.optim as optim


class Trainer() :

    def __init__(self, datafile):
        
        self.dataloader = CharDataset(config=None, data=datafile)
        self.model = Transformer(self.dataloader.stoi, hidden_layer=50)

        # à vérifier
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam()

    def run(self) :
        
        epoch = 100
        window_size = 5
        
        self.model.train()
        running_loss = 0

        for e in range(epoch) :

            end = self.dataloader.__len__() - window_size - 1
            idx = random.randint(0, end)
            x_train, y_train = self.dataloader.__getitem__(idx)

            self.optimizer.zero_grad()
            output = self.model(x_train)

            loss = self.criterion(output, y_train)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            