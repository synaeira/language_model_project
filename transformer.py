from torch import nn
from transfomer_block import TBlock
from embedding import Embedding

class Transformer(nn.Module):
     

    def __init__(self, dico, dim_emb, num_head, hidden_layer, num_transformer):

        super().__init__()

        self.emb = Embedding(dico, dim_emb)
        self.tblock = nn.Sequential(
            *(TBlock(dim_emb, num_head, hidden_layer)
              for _ in range(num_transformer))
        ) 
        self.l1 = nn.Linear(dim_emb, len(dico))



    def forward(self, x):
         
        x_emb = self.emb(x)
        x_t = self.tblock(x_emb)
        y = self.l1(x_t)

        return y