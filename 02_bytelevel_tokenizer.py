from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path('./data/').glob('*.txt')]
tokenizer = ByteLevelBPETokenizer()


# TODO change to argparse

tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    '<s>',
    '<pad>',
    '</s>',
    '<unk>',
    '<mask>',
])

Path('./data/icelandic').mkdir(parents=True, exist_ok=True)

tokenizer.save_model('./data/icelandic')
