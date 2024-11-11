# Adapted from https://github.com/google-research/adapter-bert/blob/1a31fc6e92b1b89a6530f48eb0f9e1f04cc4b750/modeling.py#L321
import torch  
import torch.nn as nn  
import torch.nn.functional as F  


class FeedForwardAdapter(nn.Module):  
    def __init__(self, input_size, hidden_size=64, init_scale=1e-3):  
        super(FeedForwardAdapter, self).__init__() 
        
        # Initialize weights and biases for the two linear transformations 
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, input_size, bias=True)
  
        # Initialize weights using a normal distribution with the given init_scale  
        nn.init.normal_(self.fc1.weight, std=init_scale)
        nn.init.normal_(self.fc2.weight, std=init_scale)
  
        # Initialize biases to zero  
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
  
    def forward(self, x):  
        net = self.fc1(x)
        net = F.gelu(net)
        net = self.fc2(net)
        return net + x 
