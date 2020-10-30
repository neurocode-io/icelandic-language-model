from pathlib import Path
from onnxruntime import InferenceSession
from transformers import AutoTokenizer, pipeline, AutoModelForMaskedLM
import numpy as np
from transformers.convert_graph_to_onnx import convert

model_path = "/home/elena/Projects/neurocode/icelandic-language-model/src/data/isroberta_malfong_ner/tmp/isroberta_malfong_ner/results/"
onnx_model_name = "onnx_mask/icelandic-MASK-base.onnx"

tokenizer = AutoTokenizer.from_pretrained("neurocode/IsRoBERTa")

def convert_for_onnx(model_path, onnx_model_name, tokenizer):
    if path.exists(onnx_model_name):
        print("ONNX input exists")
        return
    convert(
        framework="pt",
        model=model_path,
        tokenizer=tokenizer,
        output=Path(onnx_model_name),
        pipeline_name="fill-mask",
        opset=12
    )   

# convert_for_onnx(model, onnx_model_name, tokenizer)


def fill_mask_onnx(setning, onnx_model_name, tokenizer):
    tokens = tokenizer(setning, return_tensors="np")

    session = InferenceSession(onnx_model_name)
    output = session.run(None,tokens.__dict__['data'])
    token_logits=output[0]

    mask_token_index =np.where(tokens['input_ids'] == tokenizer.mask_token_id)[1]
    mask_token_logits_onnx1 = token_logits[0, mask_token_index, :]
    score = np.exp(mask_token_logits_onnx1) / np.exp(mask_token_logits_onnx1).sum(-1, keepdims=True)  

    top_5_idx = (-score[0]).argsort()[:5]
    top_5_values = score[0][top_5_idx]

    for token, s in zip(top_5_idx.tolist(), top_5_values.tolist()):
        print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])), f"(score: {s})")