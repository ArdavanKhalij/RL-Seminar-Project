#######################################################################################################################
# Libraries
#######################################################################################################################
import torch
import torch.nn as nn
#######################################################################################################################



#######################################################################################################################
# 1. Implement a parametric policy, with parameters θ. We recommend that you use PyTorch and a torch.nn.Module as the
# policy. The module takes as input the state of the environment, and produces an action as output. We suggest that
# it contains a single hidden layer of 128 neurons.
#######################################################################################################################
class Policy(nn.Module):
    def __init__(self, state_size, hidden_layer_size, action_size):
        super(Policy, self).__init__()
        self.func1 = nn.Linear(state_size, hidden_layer_size)
        self.func2 = nn.Linear(hidden_layer_size, action_size)

    def forward(self, state):
        x = self.func1(state)
        actions = torch.tanh(self.func2(x))
        return actions
#######################################################################################################################