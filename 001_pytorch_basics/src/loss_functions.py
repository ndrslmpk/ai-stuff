import torch
import torch.nn as nn

loss_function = nn.CrossEntropyLoss()

# The dataset contains a single image of a dog, where
# cat = 0 and dog = 1 (corresponding to index 0 and 1)
target_tensor = torch.tensor([1])
print(target_tensor)

# Prediction: Most likely a dog(index 1 is higher)
predicted_tensor = torch.tensor([[2.0, 5.0]])
loss_value = loss_function(predicted_tensor, target_tensor)
print(loss_value)

# Prediction: Slightly more likely a cat
predicted_tensor = torch.tensor([[1.5, 1.1]])
loss_value = loss_function(predicted_tensor, target_tensor)
print(loss_value)


loss_function2 = nn.MSELoss()


# Prediction: Most likely a dog(index 1 is higher)
predicted_tensor = torch.tensor([23.0])
actual_tensor = torch.tensor([25.0])
mse_loss_value = loss_function2(predicted_tensor, actual_tensor)
print(mse_loss_value.item())

 