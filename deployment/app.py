import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from flask import Flask, request, render_template
from src.modules.ai_model import AIModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_everything():
    cfg = OmegaConf.load("conf/inference.yaml")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_path)
    
    model = AIModel(cfg)
    checkpoint_path = cfg.predict_params.checkpoint_path
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt["state_dict"])
    
    return cfg, model.to(DEVICE), tokenizer

cfg, model, tokenizer = load_everything()

app = Flask(__name__)


def model_inference(text):
    inputs = tokenizer(
        text,
        padding=False,
        truncation=True,
        max_length=cfg.model.max_length,
        add_special_tokens=True,
        return_token_type_ids=False,
    )

    inputs = {
        "input_ids": torch.tensor([inputs["input_ids"]], device=DEVICE),
        "attention_mask": torch.tensor([inputs["attention_mask"]], device=DEVICE),
    }

    with torch.no_grad():
        logits, _ = model(**inputs)

    logits = logits.reshape(-1)
    pred = torch.sigmoid(logits)
    pred = pred.cpu().numpy().tolist()

    pred_report = f"{pred[0] * 100:.2f}% Probability Al generated"
    return pred_report


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():
    input_text = request.form["input_text"]
    inference_result = model_inference(input_text)
    return render_template(
        "result.html", input_text=input_text, result=inference_result
    )


if __name__ == "__main__":
    app.run(debug=False)
