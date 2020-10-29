# IsRoBERTa - icelandic transformer language model


**Natural language processing (NLP)** is one of the fields in AI where software analyzes large amounts of text. This has many applications, among of them we considered: 
- **Masked language modeling**: predict one or more *masked* (unknown) words given the other words in sentence. 
- **Named-Entity Recognition (NER)**:  locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations,ect.

We trained a model, from scratch, for the icelandic language using the **Huggingface** library. 

Icelandic is the official language in Iceland. As one of the Nordic languages, it belongs to the family of the Germanic languages. With a population of only 350 thousand, the language is definitely not wide spread.



## 1. Dataset
The input in NLP is text. We used the Icelandic portion of the OSCAR corpus from INRIA. The Icelandic portion of the dataset is only 1.5G. Thus, as a next step, we will concatenate the portion from OSCAR with the Icelandic sub-corpus of the Leipzig Corpora Collection, which is comprised of text from diverse sources like news, literature, and wikipedia. 

## 2. Tokenization
Tokenization is breaking the raw text into small chunks. Tokenization breaks the raw text into words or subwords, called tokens, which then are converted to ids. These tokens help in understanding the context or developing the model for the NLP. The tokenization helps in interpreting the meaning of the text by analyzing the sequence of the words. 

Great article on tokinizers can be read [here](https://blog.floydhub.com/tokenization-nlp/)


We used the **Tokenizers** library from Huggingface, in particular, a **byte-level Byte-pair encoding tokenizer**, with the same special tokens as **RoBERTa** (from the tutorial [Tokenizer summary](https://huggingface.co/transformers/master/tokenizer_summary.html), read the paragraphs [Byte-Pair Encoding](https://huggingface.co/transformers/master/tokenizer_summary.html#byte-pair-encoding) and [Byte-level BPE](https://huggingface.co/transformers/master/tokenizer_summary.html#byte-level-bpe) to get the best overview of a Byte-level BPE i.e. Byte-level Byte-Pair-Encoding). Training a **byte-level Byte-pair encoding tokenizer**, moreover with the same special tokens as [**RoBERTa**](https://huggingface.co/transformers/master/model_doc/roberta.html), enables us to build a vocabulary from an alphabet of single bytes, hence all words will be decomposable into tokens (no more <unk> tokens!). 


## 3. Training & Infrastructure

Training a language models is heavy and very time-consuming. As a first attempt we tried to train the model in Google Colab using GPU's but were unable due to low RAM. We then used the cloud, in particular, Azure.  

We created a [NC_6_promo machine](https://docs.microsoft.com/en-us/azure/virtual-machines/nc-series?toc=/azure/virtual-machines/linux/toc.json&bc=/azure/virtual-machines/linux/breadcrumb/toc.json) which comes with a K80 Nvidia GPU.

Still the training took 3 days!

For packing the code we used docker. The image lives in [docker hub](https://hub.docker.com/r/donchev7/icelandic-model)


## 5. Do it yourself

If you want to run the code you'll need to have an Azure account in particular an azure storage account. 

If you have access to azure infrastructure you can start with creating an **.env** file:

```env
ACCESS_KEY=<Azure Storage Access Key>
WANDB_API_KEY=<Wand API Key>
WANDB_PROJECT=<Not mandatory>
VM_ADMIN_PASSWD=<If you use our infra scripts>
```

Afterwards you can:

```
make create_machine
```

you'll see the IP address of the machine in your terminal. Use the IP to connect to your machine and run the packaged software:

```
ssh azureuser@<machine_ip>

screen

cat << EOF > .env
ACCESS_KEY=<Mandatory Azure AccessKey>
WANDB_API_KEY=<Optional>
WANDB_PROJECT=<Optional>
EOF

docker run --gpus all -it --rm \
    --env-file=.env \
    --ipc=host \
    -v /tmp:/tmp \
    donchev7/icelandic-model:vc0c9243 python src/train_xml_roberta_large.py --data_dir=/tmp --run_name=xml_roberta_large_malfong_ner


CTRL a + d to detatch from your screen session

exit
```
Check back later:

```
ssh azureuser@<machine_ip>

screen -r
```


Use NER:
```python
from transformers import pipeline
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("neurocode/IsRoBERTa")

nlp = pipeline("ner", model="./data/isroberta_malfong_ner/results", tokenizer=tokenizer)
res = nlp("Eftir að henni lýkur er hægt að gerast áskrifandi að efni vefjarins fyrir 1.290 kr. á mánuði.")

tokens = [r["word"] for r in res]
tokenizer.convert_tokens_to_string(tokens)

```
