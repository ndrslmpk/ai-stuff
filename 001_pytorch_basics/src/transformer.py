from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load pre-trained sentiment analysis model
model_name = "textattack/bert-base-uncased-imdb"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize the input sentence
tokenizer = BertTokenizer.from_pretrained(model_name)
inputs = tokenizer("I am pleased when I see Andreas", return_tensors="pt")

# make prediction
with torch.no_grad():
  outputs = model(**inputs).logits
  probabilities = torch.nn.functional.softmax(outputs, dim=1)
  predicted_class = torch.argmax(probabilities)

# Display sentiment result
if predicted_class == 1:
  print(f"Sentiment: Positive ({probabilities[0][1] * 100:.2f} %)")
else:
  print(f"Sentiment: Negative ({probabilities[0][0] * 100:.2f} %)")
