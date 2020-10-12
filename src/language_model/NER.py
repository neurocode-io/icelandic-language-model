from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, AutoModelWithLMHead
# wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/token-classification/utils_ner.py
from utils_ner import NerDataset, Split


tags = 'load unique tag for the train set'
data_dir = 
# "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter 
# will be padded."
max_seq_length = 40


config = AutoConfig.from_pretrained(
    model,
    num_labels=len(tags),
    id2label={i: tag for i, tag in enumerate(tags)},
    label2id={tag: i for i, tag in enumerate(tags)},
)

tokenizer = AutoTokenizer.from_pretrained("neurocode/IsRoBERTa")

model = AutoModelForTokenClassification.from_pretrained(model, config=config)


train_dataset = NerDataset(
  data_dir=data_dir,
  tokenizer=tokenizer,
  labels=tags,
  model_type=config.model_type,
  max_seq_length=max_seq_length,
  mode=Split.train)

  training_args = {
    'output_dir' : "model_output/",
    'num_train_epochs' : 3,
}