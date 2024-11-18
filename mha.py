from torch import nn

class MHA(nn.Module):
     

     def __init__(self, dim_emb, nmbr_head):
        super().__init__()

        self.multiheadattention = nn.MultiheadAttention(dim_emb, nmbr_head)

     def forward(self,x):
         
         # self attention ou cross attention (à décider)
         y = self.multiheadattention(query=x, key=x, value=x, need_weights = False)

         return y