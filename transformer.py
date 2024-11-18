from torch import nn
from transfomer_block import TBlock
from Embedding import Embedding_

class Transformer(nn.Module):
     

    def __init__(self, dico, hidden_layer):

        super().__init__()

        self.emb = Embedding_(dico)
        self.tblock = nn.Sequential(
            *(TBlock()
              for _ in range(4))
        ) 
        self.l1 = nn.Linear(64, len(dico))



    def forward(self, x):
         
        x_emb = self.emb(x)
        x_t = self.tblock(x_emb)
        # H = self.l1(x_t)
        # y = x_t.T @ H

        y = self.l1(x_t)

        return y