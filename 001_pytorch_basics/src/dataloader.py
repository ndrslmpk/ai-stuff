from torch.utils.data import DataLoader
from dataset import NumberProductDataset

dataset = NumberProductDataset(data_range=(0,5))

dataloader= DataLoader(dataset, batch_size= 3, shuffle=True)

for (num_pairs, products) in dataloader:
  print(num_pairs, products)
