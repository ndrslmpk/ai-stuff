import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn as nn

class NumberSumData(Dataset):
  def __init__(self, data_range=(1,10)):
    self.numbers = list(range(data_range[0], data_range[1]))
                        
  def __getitem__(self, index):
    number1 = float(self.numbers[index // len(self.numbers)])
    number2 = float(self.numbers[index % len(self.numbers)])
    return torch.tensor([number1, number2]), torch.tensor([number1 + number2])

  def __len__(self):
    return len(self.numbers) ** 2


dataset = NumberSumData(data_range=(1, 100))

for i in range(5):
  print(dataset[i])


class MLP(nn.Module):
  def __init__(self, input_size):
    super(MLP, self).__init__()
    self.hidden_layer = nn.Linear(input_size, 128)
    self.output_layer = nn.Linear(128,1)
    self.activation = nn.ReLU()

  def forward(self,x):
    x = self.activation(self.hidden_layer(x))
    return self.output_layer(x)

dataset = NumberSumData(data_range=(1, 100))
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
model = MLP(input_size=2)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the model for 10 epochs
for epoch in range(10):
    loss = 0.0 
    for number_pairs, sums in dataloader: # Iterate over the batches
      predictions = model(number_pairs) # Compute the model output
      loss = loss_function(predictions, sums) # Compute the loss
      loss.backward() # perform backpropagation
      optimizer.step()
      optimizer.zero_grad() # zero the gradients

      loss += loss.item()

    # Print the loss for this epoch
    print("Epoch {}: Sum of the Batch Losses = {:.5f}".format(epoch, loss))


print(model(torch.tensor([3.0, 7.0])))
