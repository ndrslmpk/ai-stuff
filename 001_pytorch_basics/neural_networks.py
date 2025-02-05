#  TUT: https://pytorch.org/tutorials/beginner/nn_tutorial.html
import torch
import torch.nn as nn


# Create own neural network
class MLP(nn.Module):
  def __init__(self, input_size):
    super(MLP, self).__init__()
    self.hidden_layer = nn.Linear(input_size, 64)
    self.output_layer = nn.Linear(64, 2)
    self.activation = nn.ReLU()

  def forward(self, x):
    x = self.activation(self.hidden_layer(x))
    return self.output_layer(x)
  
model = MLP(10)

print(model)


new_model = model.forward(torch.rand(10))
print(new_model)

