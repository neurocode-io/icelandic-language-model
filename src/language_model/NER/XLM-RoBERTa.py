from transformers import XLMRobertaTokenizer, XLMRobertaForTokenClassification
import torch

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
model = XLMRobertaForTokenClassification.from_pretrained('xlm-roberta-base', return_dict=True)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits
