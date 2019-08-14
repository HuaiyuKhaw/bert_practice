import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Tokenize input
# text = "[CLS] Happy Mother's Day! [SEP] To my dearest mom, all that I am, or hope to be, I owe it all to you. [SEP]"
text = "[CLS] You're the _ mom! I _ you. Happy Mother's Day to the best _ in the _! [SEP]"

tokenized_text = tokenizer.tokenize(text)
print(text.replace("[CLS]", "").replace("[SEP]", ""))

# Preprocess text
mask_positions = []
for i in range(len(tokenized_text)):
    if tokenized_text[i] == '_':
        tokenized_text[i] = '[MASK]'
        mask_positions.append(i)




# Mask a token that we will try to predict back with `BertForMaskedLM`
# masked_index = 9
# tokenized_text[masked_index] = '[MASK]'


# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)


# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
# segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

segments_ids = [0] * len(tokenized_text)

# ## for two lines
# segments_ids = [0] * (tokenized_text.index('[SEP]') + 1)
# segments_ids_next = [1] * (len(tokenized_text) - tokenized_text.index('[SEP]') - 1)
# segments_ids = segments_ids + segments_ids_next

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])


# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()


# Predict missing words from left to right
for mask_pos in mask_positions:

	# Predict all tokens
	with torch.no_grad():
	    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
	    predictions = outputs[0]

	# convert id to token to find out the masked word
	predicted_index = torch.argmax(predictions[0, mask_pos]).item()
	predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

	print(predicted_token)













