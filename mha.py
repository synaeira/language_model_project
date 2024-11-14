from torch import nn

class MHA_(nn.Module):
     

     def __init__(self,nmbr_head):
        super().__init__()


        self.multiheadattention = nn.MultiheadAttention(64,nmbr_head)

     def forward(self,x):
         
         y = self.multiheadattention(x)

         return y