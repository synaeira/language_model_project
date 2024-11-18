from torch import nn
from mha import MHA_
from MLP import MLP_

class TBlock(nn.Module):
     

     def __init__(self):

        super().__init__()

        self.ln1 = nn.LayerNorm(normalized_shape=64)
        self.mha = MHA_(8)
        self.ln2 = nn.LayerNorm(normalized_shape=64)
        self.mlp = MLP_(50)

     def forward(self, x):
         
        x_n = self.ln1(x)
        x1, _ = self.mha(x_n)
        x1_n = self.ln2(x1)
        x2 = self.mlp(x1_n)

        y = x2+x

        return y