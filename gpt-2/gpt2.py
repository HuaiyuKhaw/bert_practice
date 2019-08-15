import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Encode a text inputs
text = "Good design is not just what looks good. It also needs to perform, convert, astonish, and fulfill its purpose. It can be innovative or"

print(text)

for x in range(50):
	indexed_tokens = tokenizer.encode(text)

	# Convert indexed tokens in a PyTorch tensor
	tokens_tensor = torch.tensor([indexed_tokens])

	# Predict all tokens
	with torch.no_grad():
	    outputs = model(tokens_tensor)
	    predictions = outputs[0]

	# get the predicted next sub-word (in our case, the word 'man')
	predicted_index = torch.argmax(predictions[0, -1, :]).item()
	predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

	# continue to predict next word
	text = predicted_text

print(predicted_text)
