from pathlib import Path
from onnxruntime import InferenceSession
from transformers import AutoTokenizer
import numpy as np
from transformers.convert_graph_to_onnx import convert
from language_model.NER import utils

model_path = "/home/elena/Projects/neurocode/icelandic-language-model/src/data/xmlroberta_malfong_ner/content/Results_xlmroberta/"
onnx_model_name = "onnx_ner/icelandic-NER-base.onnx"

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def convert_for_onnx(model_path, onnx_model_name, tokenizer):
    if path.exists(onnx_model_name):
        print("ONNX input exists")
        return
    convert(
        framework="pt",
        model=model_path,
        tokenizer=tokenizer,
        output=Path(onnx_model_name),
        pipeline_name="ner",
        opset=12
    )  


def predict_onnx(setning, onnx_model_name, tokenizer):
    tokens = tokenizer(setning, return_attention_mask=True, return_tensors="np", truncation=True)
    id2label={i: label for i, label in enumerate(utils.get_labels())}

    session = InferenceSession(onnx_model_name)
    output = session.run(None, tokens.__dict__["data"])
    res = output[0][0]

    score = np.exp(res) / np.exp(res).sum(-1, keepdims=True)
    labels_idx = score.argmax(axis=-1)

    input_ids=tokens.__dict__["data"]["input_ids"][0]

    ignore_labels = ["O"]
    # Filter to labels not in `ignore_labels`
    filtered_labels_idx = [(idx, label_idx) for idx, label_idx in enumerate(labels_idx) if id2label[label_idx] not in ignore_labels]
    
    entities = []
    for idx, label_idx in filtered_labels_idx:
        entity = {
            "word": tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(int(input_ids[idx]))),
            "entity": id2label[label_idx]
        }
        entities += [entity]

    #pred_tokens = [r["word"] for r in entities]
    #pred_words = tokenizer.convert_tokens_to_string(pred_tokens)
    return entities


