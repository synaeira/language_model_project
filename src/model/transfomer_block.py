from torch import nn
from model.mha import MHA
from model.mlp import MLP

class TBlock(nn.Module):
     

     def __init__(self, dim_emb, num_head, hidden_layer):

        super().__init__()

        self.ln1 = nn.LayerNorm(normalized_shape=dim_emb)
        self.mha = MHA(dim_emb, num_head)
        self.ln2 = nn.LayerNorm(normalized_shape=dim_emb)
        self.mlp = MLP(hidden_layer, dim_emb)

     def forward(self, x):
         
         x_n = self.ln1(x)
         x1, _ = self.mha(x_n)

         x1 = x1 + x

         x1_n = self.ln2(x1)
         x2 = self.mlp(x1_n)

         y = x2 + x1

         return y