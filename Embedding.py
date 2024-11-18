import torch
from torch import nn


class Embedding(nn.Module):
     def __init__(self, dico, dim_emb):
        super().__init__()

        self.embedLettre = nn.Embedding(len(dico), dim_emb)
        self.embedPosition = nn.Embedding(len(dico), dim_emb)


     def forward(self,x):
         
         y1 = self.embedLettre(x)

         position = torch.arange(len(x))

         y2 = self.embedPosition(position)

         return y1 + y2