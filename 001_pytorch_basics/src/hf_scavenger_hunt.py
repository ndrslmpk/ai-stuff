import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
my_tensor = torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
print(my_tensor)

print(my_tensor.device.type)
assert my_tensor.device.type in {"cuda", "cpu"}
assert my_tensor.shape == (3,3)

print("Success!")


import torch.nn as nn
import torch.nn.functional as F

class MyMLP(nn.Module):
  def __init__(self):
    super(MyMLP, self).__init__()
    self.fc1 = nn.Linear(784,128)
    self.fc2 = nn.Linear(128,10)
    self.relu = nn.ReLU()
    self.softmax = torch.softmax()

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.softmax(x)

    return x
