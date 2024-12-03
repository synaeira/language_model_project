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

         x1_res = x1 + x

         x1_n = self.ln2(x1_res)
         x2 = self.mlp(x1_n)

         if self.strong_residual :
            y = x2 + x
         else :
            y = x2 + x1_res

         return y