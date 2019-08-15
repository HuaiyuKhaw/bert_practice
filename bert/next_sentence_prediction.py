import torch
from pytorch_transformers import BertTokenizer, BertForNextSentencePrediction

# Load pre-trained model (weights)
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
model.eval()

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input
text = "[CLS] the man went to the store. [SEP] penguins are flightless. [SEP]"
# text = "[CLS] How old are you? [SEP] The Eiffel Tower is in Paris. [SEP]"
# text = "[CLS] How old are you? [SEP] I am 30 years old. [SEP]"


tokenized_text = tokenizer.tokenize(text)
print("1:", text[:text.index("[SEP]")].replace("[CLS] ", ""))
print("2:", text[text.index("[SEP]"):].replace(" [SEP]", "").replace("[SEP] ", ""))

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
# segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
segments_ids = [0] * (tokenized_text.index('[SEP]') + 1)
segments_ids_next = [1] * (len(tokenized_text) - tokenized_text.index('[SEP]') - 1)
segments_ids = segments_ids + segments_ids_next


# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Predict the next sentence classification logits
with torch.no_grad():
	labels = torch.tensor([1]).unsqueeze(0)
	outputs = model(tokens_tensor, segments_tensors)
	seq_relationship_scores = outputs[0]


print(seq_relationship_scores)





