from torch import nn
from mha import MHA
from mlp import MLP

class TBlock(nn.Module):
     

     def __init__(self, dim_emb, num_head, hidden_layer, strong_residual):

        super().__init__()

        self.strong_residual = strong_residual

        self.ln1 = nn.LayerNorm(normalized_shape=dim_emb)
        self.mha = MHA(dim_emb, num_head)
        self.ln2 = nn.LayerNorm(normalized_shape=dim_emb)
        self.mlp = MLP(hidden_layer, dim_emb)

     def forward(self, x):
         
         x_n = self.ln1(x)
         x1, _ = self.mha(x_n)

         if self.strong_residual :

            x1 = x1 + x

         x1_n = self.ln2(x1)
         x2 = self.mlp(x1_n)

         y = x2 + x


         return y