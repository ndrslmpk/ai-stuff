import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
my_tensor = torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
print(my_tensor)

print(my_tensor.device.type)
assert my_tensor.device.type in {"cuda", "cpu"}
assert my_tensor.shape == (3,3)

print("Success!")


import torch.nn as nn
import torch.nn.functional as F

class MyMLP(nn.Module):
  def __init__(self):
    super(MyMLP, self).__init__()
    self.fc1 = nn.Linear(784,128)
    self.fc2 = nn.Linear(128,10)
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax() # is dim=10 needed?

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.softmax(x)

    return x

my_mlp = MyMLP()
print(my_mlp)

assert my_mlp.fc1.in_features == 784
assert my_mlp.fc2.out_features == 10
assert my_mlp.fc1.out_features == 128
assert isinstance(my_mlp.fc1, nn.Linear)
assert isinstance(my_mlp.fc2, nn.Linear)

# Loss functions and optimizers

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=my_mlp.parameters(),lr=0.01, momentum=0, weight_decay= 0)


assert isinstance(
    loss_fn, nn.CrossEntropyLoss
), "loss_fn should be an instance of CrossEntropyLoss"
assert isinstance(optimizer, torch.optim.SGD), "optimizer should be an instance of SGD"
assert optimizer.defaults["lr"] == 0.01, "learning rate should be 0.01"
assert optimizer.param_groups[0]["params"] == list(
    my_mlp.parameters()
), "optimizer should be passed the MLP parameters"



def fake_training_loaders():
    for _ in range(30):
        yield torch.randn(64, 784), torch.randint(0, 10, (64,))


for epoch in range(3):
    # Create a training loop
    for i, data in enumerate(fake_training_loaders()):
        # Every data instance is an input + label pair
        x, y = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Forward pass (predictions)
        y_pred = my_mlp(x)

        # Compute the loss and its gradients
        loss = loss_fn(y_pred, y)

        # Adjust learning weights
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {epoch}, batch {i}: {loss.item():.5f}")

assert abs(loss.item() - 2.3) < 0.1, "the loss should be around 2.3 with random data"


#  Hugging Face

# Replace <MASK> with the appropriate code to complete the exercise.

# Get the model and tokenizer

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_prediction(review):
    """Given a review, return the predicted sentiment"""

    # Tokenize the review
    # (Get the response as tensors and not as a list)
    inputs = tokenizer(review, return_tensors="pt")

    # Perform the prediction (get the logits)
    outputs = pt_model(**inputs)

    # Get the predicted class (corresponding to the highest logit)
    predictions = torch.argmax(outputs.logits, dim=-1)

    return "positive" if predictions.item() == 1 else "negative"


review = "This movie is not so great :("

print(f"Review: {review}")
print(f"Sentiment: {get_prediction(review)}")

assert get_prediction(review) == "negative", "The prediction should be negative"


review = "This movie rocks!"

print(f"Review: {review}")
print(f"Sentiment: {get_prediction(review)}")

assert get_prediction(review) == "positive", "The prediction should be positive"
