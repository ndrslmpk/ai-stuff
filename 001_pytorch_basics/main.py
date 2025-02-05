import torch
import numpy as np

data = [[1,2],[3,4]]
x_data = torch.tensor(data)

print(x_data)

print("hello")



images = torch.rand((4,28, 28))

second_image = images[1]


# Displaying images
import matplotlib.pyplot as plt


# plt.imshow(second_image, cmap='gray')
# plt.axis('off')
# plt.show()

# Matrix Multiplication

a = torch.tensor([[1,1],[1,0]])

print(a)

print(torch.matrix_power(a,2))
print(torch.matrix_power(a,3))
print(torch.matrix_power(a,4))



