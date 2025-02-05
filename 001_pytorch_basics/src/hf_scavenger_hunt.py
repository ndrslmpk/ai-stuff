import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
my_tensor = torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
print(my_tensor)

assert my_tensor.device.type in {"cuda", "cpu"}
assert my_tensor.shape == (3,3)

print("Success!")
