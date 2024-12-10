import torch
from torch import nn


class Embedding(nn.Module):
     def __init__(self, dico, dim_emb, block_size):
        super().__init__()

        self.embedLettre = nn.Embedding(len(dico), dim_emb)
        self.embedPosition = nn.Embedding(block_size, dim_emb)
        self.dropout = nn.Dropout(p=0.1)


     def forward(self,x):
         
         y1 = self.embedLettre(x)

         position = torch.arange(x.size(-1))
         y2 = self.embedPosition(position)

         return y1 + y2
