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


# BY PyTorch official docs
# NN serves image recognition

import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    # convolution helps to make the relation between each pixel clearer  
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)

    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)

    # fc = fully connected layer
    self.fc1 = nn.Linear(9216, 128)
    # outputs 10 labels, as we use the MNIST dataset with all 0-9 digits
    self.fc2 = nn.Linear(128,10)

    # my_nn = Net()
    # print(my_nn)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)

    # Run max Pooling
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x,1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)

    output = F.log_softmax(x, dim=1
                           )
    return output
  
random_data = torch.rand((1,1,28,28))

my_nn = Net()
result = my_nn(random_data)
print(result)



