import torch
from pytorch_transformers import BertTokenizer, BertForMaskedLM

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertTokenizer', 'bert-base-uncased', do_basic_tokenize=True)


# Tokenize input
# text = "[CLS] To our Team Leader. We appreciate everything you do for the team. Happy Boss's Day! [SEP]"
text = "[CLS] To our _. We appreciate everything you do for the team. Happy _'s Day! [SEP]"
# text = "[CLS] To our Team Leader. We _ everything you do for the _. Happy Boss's Day! [SEP]"

tokenized_text = tokenizer.tokenize(text)
print(text.replace("[CLS] ", "").replace(" [SEP]", ""))

# Mask a token that we will try to predict back with `BertForMaskedLM`
mask_positions = []
for i in range(len(tokenized_text)):
    if tokenized_text[i] == '_':
        tokenized_text[i] = '[MASK]'
        mask_positions.append(i)


# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)


# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
# segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
segments_ids = [0] * len(tokenized_text)


# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])


### Predict missing words from left to right
predicted_tokens = []
for mask_pos in mask_positions:

	### Predict all tokens
	with torch.no_grad():
	    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
	    predictions = outputs[0]


	### convert id tos token to find out the masked word
	predicted_index = torch.argmax(predictions[0, mask_pos]).item()
	predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
	predicted_tokens.append(predicted_token)

print(*predicted_tokens, sep=', ')

