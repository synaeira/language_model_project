import torch
from torch import nn




class Embedding_(nn.Module):
     def __init__(self,dico):
        super().__init__()

        # j'ai mis 64 mais reste à determiner combien on veut vraiment
        self.embedLettre = nn.Embedding(len(dico),64)

        # j'ai mis 64 mais reste à determiner combien on veut vraiment
        self.embedPosition = nn.Embedding(len(dico),64)


     def forward(self,x):
         y_1 = self.embedLettre(x)

         position = torch.tensor([i for i in range(len(x))])

         y_2 = self.embedPosition(position)

         return y_1 + y_2