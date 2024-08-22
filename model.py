import torch
import torch.nn as nn

class FCNN(nn.Module):
    def __init__(self, hidden_layer=0, hidden_unit=100, num_control=10):
        self.input_layer = nn.Linear(in_features=12, out_features=hidden_unit)
        self.hidden_layer_list = []
        for _ in range(hidden_layer-1):
            self.hidden_layer_list.append(nn.Linear(in_features=hidden_unit, 
                                          out_features=hidden_unit))
        self.output_layer = nn.Linear(in_features=hidden_unit, 
                                      out_features=6*num_control)
        
    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layer_list:
            x = layer(x)
        
        out = self.output_layer(x)
        
        return out
    
class DeConv(nn.Module):
    pass
    

