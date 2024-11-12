from torch import nn


class MLP_(nn.Module):
     

     def __init__(self,hidden_layer):
        super().__init__()

        # 64 car 64 colonne dans ma matrice
        self.firstLinearLayer = nn.Linear(64, hidden_layer)

        # je veux la même dimension en entré qu'en sorti
        self.SecondeLinearLayer = nn.Linear(hidden_layer, 64)


     def forward(self,x):
         # x va être la matrice n X d (avec n le nombre de caractere et d va être 64)
         y = self.firstLinearLayer(x)

         y_2 = self.SecondeLinearLayer(y)


         return y_2