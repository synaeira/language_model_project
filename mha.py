from torch import nn

class MHA_(nn.Module):
     

     def __init__(self,nmbr_head):
        super().__init__()

        self.multiheadattention = nn.MultiheadAttention(64,nmbr_head)

     def forward(self,x):
         
         # self attention ou cross attention (à décider)
         y = self.multiheadattention(query=x, key=x, value=x, need_weights = False)

         return y