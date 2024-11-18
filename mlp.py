from torch import nn
import torch


class MLP(nn.Module):
     

     def __init__(self, hidden_layer, dim_emb):
        super().__init__()

        self.firstLinearLayer = nn.Linear(dim_emb, hidden_layer)
        self.SecondeLinearLayer = nn.Linear(hidden_layer, dim_emb)


     def forward(self,x):
         
         y1 = self.firstLinearLayer(x)
         y1 = torch.relu(y1)
         y2 = self.SecondeLinearLayer(y1)

         return y2
