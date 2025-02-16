import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def load_data(batch_size, data_dir="datda"):
  """Load Fashion-MNIST dataset"""

  # normalizes dataset
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])

  # load testdataset
  trainset = datasets.FashionMNIST(
    data_dir, download=True, train=True, transform=transform
  )
  trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True
  )

  # load traindataset
  testset = datasets.FashionMNIST(
    data_dir, download=True, train=False, transform=transform
  )

  testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True
  )

  return trainloader, testloader

trainloader, testloader = load_data(64)


# Define helper functions 

def get_class_names():
  """returns list of all classes in Fashion-MNIST dataset"""
  return [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
  ]

def get_class_name(class_index):
  """returns name for the index of class names"""
  return get_class_names()[class_index]

def get_class_index(class_name):
  """Returns index for given name"""
  return get_class_names().index(class_name)

for class_index in range(10):
  print(f"class_index={class_index}, class_index={get_class_name(class_index)}")


# Display data to be inspected
import matplotlib.pyplot as plt
import numpy as np

# can make the matplotlib interactive to prevent the window from restarting
# plt.ion()

def imshow(img):
  img = img / 2 + 0.5
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1,2,0)))

images, labels= next(iter(trainloader))

fig=plt.figure(figsize=(15,4))
plot_size = 10

for idx in np.arange(plot_size):
  ax = fig.add_subplot(2,plot_size//2, idx + 1 , xticks=[], yticks=[])
  imshow(images[idx])
  ax.set_title(get_class_name(int(labels[idx])))

# Turn on to show the plot
# plt.show()



# introduce MobileNetV3 architecture

import torchvision.models as models

mobilenet_v3_model = models.mobilenet_v3_small(pretrained=True)
print(mobilenet_v3_model)

# Replace the output layer by the right number of outputs

import torch.nn.functional as F
import torchvision.models as models
from torch import nn

class MobileNetV3(nn.Module):
  def __init__(self):
    super(MobileNetV3, self).__init__()
    self.model = mobilenet_v3_model

    self.model.classifier[3] = nn.Linear(1024, 10)

    self.freeze()

  def forward(self,x):
    # Converts an 1x28x28 input tensor to 3x28x28 to convert image to color image
    x = x.repeat(1,3,1,1)

    if x.shape[2:] != (224,224):
      x = F.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)

    return self.model(x)
  
  def freeze(self):
    for param in self.model.parameters():
      param.requires_grad = False

    for param in self.model.classifier[3].parameters():
      param.requires_grad = True

  def unfreeze(self):
    for param in self.model.parameters():
      param.requires_grad = True

model = MobileNetV3()
print(model)



# Train the model with MNIST data

import torch
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)


# Set the device as GPU, MPS, or CPU according to availability
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# Create a PyTorch training loop

# Move the model weights to the device
model = model.to(device)

epochs = 1
for epoch in range(epochs):
  for batch_num, (images, labels) in enumerate(trainloader):
    # moves tensors to the device
    images = images.to(device)
    labels = labels.to(device)

    # Zero out the optimizer's gradient buffer
    optimizer.zero_grad()

    # Forward pass
    outputs = model(images)

    # Calculate the loss and perform backprop
    loss = loss_fn(outputs, labels)
    loss.backward()

    # Update the weights
    optimizer.step

    # Print the loss for every 100th iteration
    if (batch_num) % 100 == 0:
        print(
            "Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}".format(
                epoch + 1, epochs, batch_num + 1, len(trainloader), loss.item()
            )
        )


# Print the loss and accuracy on the test set
correct = 0
total = 0
loss = 0

for images, labels in testloader:
    # Move tensors to the configured device
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)
    # is added to accumulate the loss
    loss += loss_fn(outputs, labels)

    # torch.max return both max and argmax. We get the argmax here.
    _, predicted = torch.max(outputs.data, 1)

    # Compute the accuracy
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(
    "Test Accuracy of the model on the test images: {} %".format(100 * correct / total)
)
print("Test Loss of the model on the test images: {}".format(loss))


# Plotting a few examples of correct and incorrect predictions

import matplotlib.pyplot as plt
import numpy as np

# Get the first batch of images and labels
images, labels = next(iter(testloader))

# Move tensors to the configured device
images = images.to(device)
labels = labels.to(device)

# Forward pass
outputs = model(images)
_, predicted = torch.max(outputs.data, 1)

# Plot the images with labels, at most 10
fig = plt.figure(figsize=(15, 4))

for idx in np.arange(min(10, len(images))):
    ax = fig.add_subplot(2, 10 // 2, idx + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images.cpu()[idx]))
    ax.set_title(
        "{} ({})".format(get_class_name(predicted[idx]), get_class_name(labels[idx])),
        color=("green" if predicted[idx] == labels[idx] else "red"),
    )
