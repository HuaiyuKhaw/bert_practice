
### First, tokenize the input
import torch
tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertTokenizer', 'bert-base-cased', do_basic_tokenize=False)

# Tokenized input
# text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
# tokenized_text = tokenizer.tokenize(text)
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# print(tokenized_text)




#####################################################################
### Get the hidden states computed by `bertModel`
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
# segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

# # Convert inputs to PyTorch tensors
# segments_tensors = torch.tensor([segments_ids])
# tokens_tensor = torch.tensor([indexed_tokens])

# model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertModel', 'bert-base-cased')
# model.eval()


# with torch.no_grad():
#     encoded_layers, _ = model(tokens_tensor, segments_tensors)

#####################################################################
### Predict masked tokens using `bertForMaskedLM`
# # Mask a token that we will try to predict back with `BertForMaskedLM`
# masked_index = 8
# tokenized_text[masked_index] = '[MASK]'
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# tokens_tensor = torch.tensor([indexed_tokens])

# maskedLM_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertForMaskedLM', 'bert-base-cased')
# maskedLM_model.eval()

# with torch.no_grad():
#     predictions = maskedLM_model(tokens_tensor, segments_tensors)

# # Get the predicted token
# predicted_index = torch.argmax(predictions[0, masked_index]).item()
# predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
# assert predicted_token == 'Jim'

#####################################################################
### Classify next sentence using ``bertForNextSentencePrediction``
# Going back to our initial input
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
segments_tensors = torch.tensor([segments_ids])

tokens_tensor = torch.tensor([indexed_tokens])

# nextSent_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertForNextSentencePrediction', 'bert-base-cased')
# nextSent_model.eval()

# # Predict the next sentence classification logits
# with torch.no_grad():
#     next_sent_classif_logits = nextSent_model(tokens_tensor, segments_tensors)

# print(next_sent_classif_logits)


### Question answering using `bertForQuestionAnswering`
questionAnswering_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertForQuestionAnswering', 'bert-base-cased')
print(questionAnswering_model.eval())

# Predict the start and end positions logits
with torch.no_grad():
    start_logits, end_logits = questionAnswering_model(tokens_tensor, segments_tensors)

# Or get the total loss which is the sum of the CrossEntropy loss for the start and end token positions (set model to train mode before if used for training)
start_positions, end_positions = torch.tensor([12]), torch.tensor([14])
multiple_choice_loss = questionAnswering_model(tokens_tensor, segments_tensors, start_positions=start_positions, end_positions=end_positions)

print("start:", start_logits)
print("end:", end_logits)
