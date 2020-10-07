# Icelandic language-model

## TODO
- [ ] Update Readme with .env setup
- [ ] Add python script for uploading model to azure storage
- [ ] Add docu how to run the container in the VM
- [ ] Add package installation verifications (transformers / CUDA ...)

docker run --gpus all -it --rm --ipc=host -v /tmp:/tmp --name icelandic donchev7/icelandic-model:v72a4c67 bash


**Natural language processing (NLP)** is one of the fields in programming where the natural language is processed by the software. This has many applications, among of them we considered: 
- **Masked language modeling**: predict one or more *masked* (unknown) words given the other words in sentence. 
- **Named-Entity Recognition (NER)**:  locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations,ect.

We trained a model, from scratch, for the icelandic language using the **Huggingface** library. 

Icelandic is the official language of the Republic of Iceland. As one of the Nordic languages, it belongs to the family of the Germanic languages. With a population of only 350 thousand, the language is definitely not wide spread and thus, we are the first to train an icelandic language model.

Before starting you need to install the packages **Tokenizers** and **Transformers**.

## 1. Dataset
The input in NLP is text. We used the Icelandic portion of the OSCAR corpus from INRIA. The Icelandic portion of the dataset is only 1.5G. Thus, as a next step, we will concatenate the portion from OSCAR with the Icelandic sub-corpus of the Leipzig Corpora Collection, which is comprised of text from diverse sources like news, literature, and wikipedia. 

## 2. Tokenization
Tokenization is breaking the raw text into small chunks. Tokenization breaks the raw text into words or subwords, called tokens, which then are converted to ids. These tokens help in understanding the context or developing the model for the NLP. The tokenization helps in interpreting the meaning of the text by analyzing the sequence of the words. 


For example, the text “It is raining” can be tokenized into ‘It’, ‘is’, ‘raining’. Tokenization can be done to either separate words or sentences and there are different methods and libraries available to perform tokenization. We used the **Tokenizers** library from Huggingface, in particular, a **byte-level Byte-pair encoding tokenizer**, with the same special tokens as **RoBERTa** (from the tutorial [Tokenizer summary](https://huggingface.co/transformers/master/tokenizer_summary.html), read the paragraphs [Byte-Pair Encoding](https://huggingface.co/transformers/master/tokenizer_summary.html#byte-pair-encoding) and [Byte-level BPE](https://huggingface.co/transformers/master/tokenizer_summary.html#byte-level-bpe) to get the best overview of a Byte-level BPE i.e. Byte-level Byte-Pair-Encoding). Training a **byte-level Byte-pair encoding tokenizer**, moreover with the same special tokens as [**RoBERTa**](https://huggingface.co/transformers/master/model_doc/roberta.html), enables us to build a vocabulary from an alphabet of single bytes, hence all words will be decomposable into tokens (no more <unk> tokens!). 

### 2.1 Train Tokenizer
Here is tho code to train a byte-level Byte-pair encoding tokenizer. 

```
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path("./is_data/").glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save_model(".", "icelandic")
```
Variables used: 
- **vocab_size**: Total words,; transformers models from Huggingface rarely have a vocabulary size greater than 50,000, especially if they are trained on a single language.We thus
picked, an arbitrary, size of 52,000.
- **min_frequency**: ?
- **special_tokens**: For a list and description of special tokens we refer to [RobertaTokenizer](https://huggingface.co/transformers/model_doc/roberta.html).


Running the above code yields two files; a **vocab.json**, which is a list of the most frequent tokens ranked by frequency, and a **merges.txt** list of merges.

## Train language model for icelandic 
We trained a RoBERTa-like model, which is a BERT-like with a couple of changes (we refer to [RoBERTa Model](https://huggingface.co/transformers/master/model_doc/roberta.html#robertamodel) and [BERT Model](https://huggingface.co/transformers/model_doc/bert.html?highlight=bert) for an introduction to theses models). Since the model is BERT-like, we’ll train it on a task of **Masked language modeling**, i.e. the predict how to fill arbitrary tokens that we randomly mask in the dataset. 

We used the **Trainer** class from Huggingface, which provides an API for feature-complete training in most standard use cases. Here are the required packages and the code:

```
import tokenizers
import argparse
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from language_model.config import Settings


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.", default=-1
)
parser.add_argument("--data_dir", type=str, help="Data directory for storage", default="./data")

argv = parser.parse_args()

local_rank = argv.local_rank
data_dir = argv.data_dir

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizers = RobertaTokenizerFast.from_pretrained(f"{data_dir}/icelandic", max_len=512)
model = RobertaForMaskedLM(config=config)

dataset = LineByLineTextDataset(
    tokenizer=tokenizers,
    file_path=f"{data_dir}/is.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizers, mlm=True, mlm_probability=0.15)

print(f"local rank: {local_rank}")

training_args = TrainingArguments(
    local_rank=local_rank,
    output_dir=f"{data_dir}/icelandic",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=48, # Nvidia K80 99%
    seed=42,
    # save_steps=10_000,
    # save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

trainer.train()

trainer.save_model(f"{data_dir}/icelandic")
```
Let's go over, step by step what is happening.

First we define a RoBERTa-like model:
```
config = RobertaConfig( vocab_size=52_000, max_position_embeddings=514, num_attention_heads=12, num_hidden_layers=6, type_vocab_size=1)
model = RobertaForMaskedLM(config=config)`
```
[**RoBERTaConfig**](https://huggingface.co/transformers/model_doc/roberta.html?highlight=robertaconfig) is the configuration class to store the configuration of a RobertaModel (or a TFRobertaModel). It is used to instantiate a RoBERTa model according to the specified arguments, defining the model architecture. For a description of the variables, see https://huggingface.co/transformers/model_doc/bert.html#transformers.BertConfig.



Training language models is heavy and very time-consuming. As a first attempt we tried to train in Google Colab using GPU's but were unable due to low RAM. We then used the cloud, in particular, Azure.  
