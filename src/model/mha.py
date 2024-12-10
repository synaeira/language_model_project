from torch import nn
import torch 

class MHA(nn.Module):
     

     def __init__(self, dim_emb, nmbr_head):
        super().__init__()
        self.multiheadattention = nn.MultiheadAttention(dim_emb, nmbr_head)

     def forward(self,x):
         
         causal_mask = torch.triu(torch.ones(len(x), len(x)) * float('-inf'), diagonal=1)

         y = self.multiheadattention(query=x, key=x, value=x, need_weights = False, attn_mask = causal_mask)

         return y