from torch.utils.data import Dataset

class NumberProductDataset(Dataset):
  def __init__(self, data_range=(1,10)):
    self.numbers = list(range(data_range[0], data_range[1]))
    print(self.numbers)

  def __getitem__(self, index):
    number1 = self.numbers[index]
    number2 = self.numbers[index] + 1
    return (number1, number2), number1 * number2
  
  def __len__(self):
    return len(self.numbers)


# dataset = NumberProductDataset(data_range=(0,11))

# data_sample = dataset[3]
# print(data_sample)

# dataset = NumberProductDataset(data_range=(0,100))
# data_sample = dataset[50]
# print(data_sample)
