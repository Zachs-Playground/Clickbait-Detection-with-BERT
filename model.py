from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
import re

model = BertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = BertTokenizer.from_pretrained('./saved_model')
model.eval()

def clean(text):
    text = re.sub(r'http\S+', " ", text)
    text = re.sub(r'@\w+',' ',text)
    text = re.sub(r'#\w+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'<.*?>',' ', text)
    return text

def classification ():
  # title
  # body
  test_input = tokenizer(test_title,
                    test_body,
                    padding=True,
                    truncation=True,
                    return_tensors="pt")


# Put model in evaluation mode

# Tracking variables 
predictions = []

# Predict 
for batch in bert_test:
  # Add batch to GPU
  batch = tuple(t for t in batch)
  
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  
  # Telling the model not to compute or store gradients, saving memory and 
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Store predictions and true labels
  predictions.append(logits)
  return 0 or 1
