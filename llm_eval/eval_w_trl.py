from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import pandas as pd
from datasets import Dataset
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, accuracy_score, f1_score
from tqdm import tqdm
import yaml

"""
Expects csv with prompt, lmodel_response, rmodel_response
"""


def score_response(prompt, response, model, tokenizer, max_length=512):
    full_text = f"{prompt}\n{response}"
    inputs = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=max_length
    ).to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Assuming the last token's logits represent the overall score
        score = F.softmax(logits[0, -1, :], dim=0)[1].item()

    return score


def evaluate_responses(prompts, responses_model1, responses_model2, model, tokenizer):
    scores_model1 = []
    scores_model2 = []

    print("Evaluating Model 1 responses...")
    for prompt, response in tqdm(zip(prompts, responses_model1)):
        score = score_response(prompt, response, model, tokenizer)
        scores_model1.append(score)

    print("Evaluating Model 2 responses...")
    for prompt, response in tqdm(zip(prompts, responses_model2)):
        score = score_response(prompt, response, model, tokenizer)
        scores_model2.append(score)

    return {
        "raw_scores": {"model1": scores_model1, "model2": scores_model2},
    }


def calculate_statistics(scores):
    mean = np.mean(scores)
    median = np.median(scores)
    std = np.std(scores)
    variance = np.var(scores)
    return {
        "mean": mean,
        "median": median,
        "std": std,
        "variance": variance,
    }


def compare_models(evaluation_results):
    stats_model1 = calculate_statistics(evaluation_results["raw_scores"]["model1"])
    stats_model2 = calculate_statistics(evaluation_results["raw_scores"]["model2"])

    return {"model1": stats_model1, "model2": stats_model2}


if __name__ == "__main__":
    open("eval_config.yaml","r") as f:
        config = yaml.safe_load(f)
    base_model_name = "HuggingFaceTB/SmolLM-1.7B"
    adapter_model_name = "bco_trainer"

    ref_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    rw_model = PeftModel.from_pretrained(ref_model, adapter_model_name).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load data
    df = pd.read_csv(config["eval-data-path"])
    prompts = df[config['prompt-col']].to_list()
    responses_model1 = df[config['lmodel-response-col']].to_list()
    responses_model2 = df[config['rmodel-response-col']].to_list()

    # Evaluate
    evaluation_results = evaluate_responses(
        prompts, responses_model1, responses_model2, rw_model, tokenizer
    )

    metrics = compare_models(evaluation_results)

    for _model, _dict in metrics.items():
        for metric, value in _dict.items():
            print(f"{_model}: {metric}: {value}")
