from torch import nn
import torch 

class MHA(nn.Module):
     

   def __init__(self, dim_emb, nmbr_head):
      super().__init__()
      self.multiheadattention = nn.MultiheadAttention(dim_emb, nmbr_head, batch_first=True)

   def forward(self,x):
         
      batch_size, seq_len, embed_dim = x.size()

      causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

      y, _ = self.multiheadattention(query=x, key=x, value=x, need_weights = False, attn_mask = causal_mask)

      return y