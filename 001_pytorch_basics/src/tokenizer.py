from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print(tokenizer.vocab_size)

tokens = tokenizer.tokenize("I love Moneten")

print(tokens)

print(tokenizer.convert_tokens_to_ids(tokens))
