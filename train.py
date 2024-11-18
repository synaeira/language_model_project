from data_processing import CharDataset
from transformer import Transformer
import random
from torch import nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt


class Trainer() :

    def __init__(self, datafile):
        
        self.dataloader = CharDataset(config=None, data=datafile)
        self.model = Transformer(self.dataloader.stoi, hidden_layer=50)

        # à vérifier
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.running_loss = []

    def run(self) :
        
        epoch = 1000
        window_size = 5
        
        self.model.train()

        for e in range(epoch) :

            end = self.dataloader.__len__() - window_size - 1
            idx = random.randint(0, end)
            x_train, y_train = self.dataloader.__getitem__(idx)

            self.optimizer.zero_grad()
            output = self.model(x_train)

            y_train_ont_hot = nn.functional.one_hot(y_train, self.dataloader.get_vocab_size()).to(torch.float)

            loss = self.criterion(output, y_train_ont_hot)

            loss.backward()

            self.optimizer.step()

            self.running_loss.append(loss.item())

train = Trainer("shakespeare-data.txt")

train.run()

plt.plot(train.running_loss)
plt.show()