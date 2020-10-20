from collections import namedtuple
from typing import List, Tuple
from transformers.tokenization_utils import PreTrainedTokenizer
from language_model.lib.log import get_logger


logger = get_logger(__file__)


def get_labels():
    return [
        "B-Date",
        "B-Location",
        "B-Miscellaneous",
        "B-Money",
        "B-Organization",
        "B-Percent",
        "B-Person",
        "B-Time",
        "I-Date",
        "I-Location",
        "I-Miscellaneous",
        "I-Money",
        "I-Organization",
        "I-Percent",
        "I-Person",
        "I-Time",
        "O",
    ]


def is_sentence_end(sentence: str):
    return not sentence.strip() or len(sentence) == 0 or sentence.startswith("\n") or sentence.startswith(".")


def file_to_sentences(file_path: str) -> List[Tuple[List[str], List[str]]]:
    """
        Converts a file from  CoNLL format to a list of (sentence, labels) tuple
        data[2]
    (['Að', 'sumu', 'leyti', 'líkist', 'Holberg', 'þeirri', 'mynd', 'sem', 'líffræðilega', 'sinnaðir', 'afbrotafræðingar', 'drógu', 'upp', 'af', 'brotamönnum', 'í', 'lok', '19.', 'aldar'], ['O', 'O', 'O', 'O', 'O', 'B-Person', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Date', 'I-Date']
    """
    assert file_path, "File path missing"

    data = []
    sentences = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if is_sentence_end(line):
                if len(sentences) == 0:
                    continue

                data.append((sentences, labels))
                sentences = []
                labels = []
                continue

            splits = line.split()
            assert len(splits) == 2, f"Unsupported line found, line_number: {idx}"

            word, tag = splits
            assert tag in get_labels(), f"Unknown tag found: {tag}"

            sentences.append(word.strip())
            labels.append(tag.strip())

        # In case last line does not end with a .
        if len(sentences) > 0:
            data.append((sentences, labels))

    return data


def convert_examples_to_features(data, tokenizer: PreTrainedTokenizer, label_list, max_seq_length):
    """Converts a set of examples into XLMR compatible format
    * Labels are only assigned to the positions correspoinding to the first BPE token of each word.
    * Other positions are labeled with 0 ("IGNORE")
    """
    InputFeature = namedtuple("InputFeature", ["input_ids", "input_mask", "label_id", "valid_ids", "label_mask"])
    ignored_label = "IGNORE"
    label_map = {label: i for i, label in enumerate(label_list, 1)}
    label_map[ignored_label] = 0  # 0 label is to be ignored

    features = []
    for ex_index, (sentence, labellist) in enumerate(data):
    # for (ex_index, example) in enumerate(examples):

        # textlist = example.text_a.split(" ")
        # labellist = example.label
        labels = []
        valid = []
        label_mask = []
        token_ids = []

        for i, word in enumerate(sentence):
            tokens = tokenizer.tokenize(word.strip())  # word token ids
            token_ids.extend(tokenizer.convert_tokens_to_ids(tokens))  # all sentence token ids
            label_1 = labellist[i]
            for m in range(len(tokens)):
                if m == 0:  # only label the first BPE token of each work
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    labels.append(ignored_label)  # unlabeled BPE token
                    label_mask.append(0)
                    valid.append(0)

        logger.debug("token ids = ")
        logger.debug(token_ids)
        logger.debug("labels = ")
        logger.debug(labels)
        logger.debug("valid = ")
        logger.debug(valid)

        if len(token_ids) >= max_seq_length - 1:  # trim extra tokens
            token_ids = token_ids[0 : (max_seq_length - 2)]
            labels = labels[0 : (max_seq_length - 2)]
            valid = valid[0 : (max_seq_length - 2)]
            label_mask = label_mask[0 : (max_seq_length - 2)]

        # adding <s>
        token_ids.insert(0, 0)
        labels.insert(0, ignored_label)
        label_mask.insert(0, 0)
        valid.insert(0, 0)

        # adding </s>
        token_ids.append(2)
        labels.append(ignored_label)
        label_mask.append(0)
        valid.append(0)

        assert len(token_ids) == len(labels)
        assert len(valid) == len(labels)

        label_ids = []
        for i, _ in enumerate(token_ids):
            label_ids.append(label_map[labels[i]])

        assert len(token_ids) == len(label_ids)
        assert len(valid) == len(label_ids)

        input_mask = [1] * len(token_ids)

        while len(token_ids) < max_seq_length:
            token_ids.append(1)  # token padding idx
            input_mask.append(0)
            label_ids.append(label_map[ignored_label])  # label ignore idx
            valid.append(0)
            label_mask.append(0)

        while len(label_ids) < max_seq_length:
            label_ids.append(label_map[ignored_label])
            label_mask.append(0)

        assert len(token_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in token_ids]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in token_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("label: %s (id = %s)" % (labellist, " ".join(map(str, label_ids))))
            logger.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))
            logger.info("valid mask: %s" % " ".join([str(x) for x in valid]))

        features.append(
            InputFeature(
                input_ids=token_ids, input_mask=input_mask, label_id=label_ids, valid_ids=valid, label_mask=label_mask
            )
        )

    return features
