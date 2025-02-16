from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
  content = requests.get(URL + FILENAME).content
  (PATH / FILENAME).open("wb").write(content)


# DEZIP FILES
import pickle
import gzip
with gzip.open((PATH/FILENAME).as_posix(), "rb") as f:
  ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")


from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
# pyplot.show()
# pyplot.close()



import torch

x_train, y_train, x_valid, y_valid = map(
  torch.tensor, (x_train, y_train, x_valid, y_valid)
)

n,c = x_train.shape

print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())


# TODO: FINISH
# Neural net from scratch

import math

weigths = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias
