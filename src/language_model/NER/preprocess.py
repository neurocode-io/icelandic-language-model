# The script below will split sentences longer than MAX_LENGTH (in terms of tokens) into small ones. Otherwise, 
# long sentences will be truncated when tokenized, causing the loss of training data and some tokens in the test 
# set not being predicted.
# from https://chriskhanhtran.github.io/posts/named-entity-recognition-with-transformers/
# not tried yet!
# wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/token-classification/utils_ner.py

import sys

from transformers import AutoTokenizer

dataset = sys.argv[1]
model_name_or_path = sys.argv[2]
max_len = int(sys.argv[3])

subword_len_counter = 0

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
max_len -= tokenizer.num_special_tokens_to_add()

with open(dataset, "rt") as f_p:
    for line in f_p:
        line = line.rstrip()

        if not line:
            print(line)
            subword_len_counter = 0
            continue

        token = line.split()[0]

        current_subwords_len = len(tokenizer.tokenize(token))

        # Token contains strange control characters like \x96 or \x95
        # Just filter out the complete line
        if current_subwords_len == 0:
            continue

        if (subword_len_counter + current_subwords_len) > max_len:
            print("")
            print(line)
            subword_len_counter = current_subwords_len
            continue

        subword_len_counter += current_subwords_len

        print(line)