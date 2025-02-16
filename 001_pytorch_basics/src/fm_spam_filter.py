
from datasets import load_dataset

# spam dataset are here https://huggingface.co/datasets

dataset = load_dataset("sms_spam", split=["train"])[0]

for entry in dataset.select(range(3)):
    sms = entry["sms"]
    label = entry["label"]
    print(f"label={label}, sms={sms}")

# mapper: Convenient dictionaries to convert between labels and ids
id2label = {0: "NOT SPAM", 1: "SPAM"}
label2id = {"NOT SPAM": 0, "SPAM": 1}

for entry in dataset.select(range(3)):
    sms = entry["sms"]
    label_id = entry["label"]
    print(f"label={id2label[label_id]}, sms={sms}")


# build email classifier

def get_sms_messages_string(dataset, item_numbers, include_labels=False):
  sms_messages_string = ""
  for item_number, entry in zip(item_numbers, dataset.select(item_numbers)):
    sms = entry["sms"]
    label_id = entry["label"]

    if include_labels:
      sms_messages_string += (
        f"{item_number} (label={id2label[label_id]}) -> {sms}\n)"
      )
    else:
      sms_messages_string += f"{item_number} -> {sms}\n"

  return sms_messages_string


print(get_sms_messages_string(dataset, range(3), include_labels=True))



sms_messages_string = get_sms_messages_string(dataset, range(7,15))

query =  f"""
{sms_messages_string}
-------
Classify the following SMS messages and classify them as either SPAM or NOT SPAM.
Here are the messages:

Please respond in this JSON format:
[
    {{"0": "NOT SPAM"}},
    {{"1": "SPAM"}},
    ...
]
"""
 
print(query)


response = {
  "7": "NOT SPAM",
  "8": "SPAM",
  "9": "SPAM",
  "10": "NOT SPAM",
  "11": "SPAM",
  "12": "SPAM",
  "13": "NOT SPAM",
  "14": "NOT SPAM"
}


# Estimate the accuracy of your classifier by comparing your responses to the labels in the dataset


def get_accuracy(response, dataset, original_indices):
  correct = 0
  total = 0

  for entry_number, prediction in response.items():
    if int(entry_number) not in original_indices:
      continue

    label_id = dataset[int(entry_number)]["label"]
    label = id2label[label_id]

    # If the prediction from the LLM matches the label in the dataset
    # we increment the number of correct predictions.
    # (Since LLMs do not always produce the same output, we use the
    # lower case version of the strings for comparison)
    if prediction.lower() == label.lower():
        correct += 1

    # increment the total number of predictions
    total += 1

  try:
    accuracy = correct / total
  except ZeroDivisionError:
    print("No matching results found!")
    return

  return round(accuracy, 2)


print(f"Accuracy: {get_accuracy(response, dataset, range(7, 15))}")


# Improve LLM with examples on how to complete a task

sms_messages_string_w_labels = get_sms_messages_string(
    dataset, range(54, 60), include_labels=True
)

sms_messages_string_no_labels = get_sms_messages_string(dataset, range(7, 15))




def print_misclassified_messages(response, dataset):
  for entry_number, prediction in response.items():
    label_id = dataset[int(entry_number)]["label"]
    label = id2label[label_id]

    if prediction.lower() != label.lower():
      sms = dataset[int(entry_number)]["sms"]
      print("---")
      print(f"Message: {sms}")
      print(f"Label: {label}")
      print(f"Prediction: {prediction}")


print_misclassified_messages(response, dataset)
