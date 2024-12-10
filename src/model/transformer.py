from torch import nn
from model.transfomer_block import TBlock
from model.embedding import Embedding

class Transformer(nn.Module):
     

    def __init__(self, dico, dim_emb, num_head, hidden_layer, num_transformer, block_size):

        super().__init__()

        self.emb = Embedding(dico, dim_emb, block_size)
        self.tblock = nn.Sequential(
            *(TBlock(dim_emb, num_head, hidden_layer)
              for _ in range(num_transformer))
        )
        self.ln = nn.LayerNorm(dim_emb)
        self.l1 = nn.Linear(dim_emb, len(dico))



    def forward(self, x):
         
        x_emb = self.emb(x)
        x_t = self.tblock(x_emb)
        x_n = self.ln(x_t)
        y = self.l1(x_n)

        return y