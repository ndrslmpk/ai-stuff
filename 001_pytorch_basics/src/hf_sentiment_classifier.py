from datasets import load_dataset

splits = ["train", "test"]
ds = {split: ds for split, ds in zip(splits, load_dataset("imdb", split=splits))}

for split in splits:
  ds[split] = ds[split].shuffle(seed=42).select(range(500))

print(ds)


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
  """Preprocess the imdb dadtaset by returning tokenized examples"""
  return tokenizer(examples["text"],padding="max_length", truncation=True)

tokenized_ds = {}
for split in splits:
  tokenized_ds[split] = ds[split].map(preprocess_function, batched=True)

assert tokenized_ds["train"][0]["input_ids"][:5] == [101, 2045, 2003, 2053, 7189]

print(tokenized_ds["train"][0]["input_ids"])




from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
  "distilbert-base-uncased",
  num_labels=2,
  id2label={0: "NEGATIVE", 1: "POSITIVE"},  # For converting predictions to strings
  label2id={"NEGATIVE": 0, "POSITIVE": 1},
)

# Freeze all the parameters of the base model by preventing to recalculate the gradient
for param in model.base_model.parameters():
  param.requires_grad = False

model.classifier
