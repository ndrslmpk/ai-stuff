import torch.optim as optim

# Assuming `model` is your defined neural network
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay= 0.1)

# lr = 0.01 sets the learning rate to 0.01
# momentum = 0.9 smooths out updates and can help training



optimizer = optim.Adam(model.parameters(), lr=0.01)
