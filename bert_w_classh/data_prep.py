import pandas as pd
import numpy as np

import numpy as np
from datasets import load_dataset

# Load dataset and convert to pandas DataFrame
ds = load_dataset("openbmb/UltraFeedback")["train"]
ds = ds.shuffle(seed=42)
df = ds.to_pandas().sample(frac=0.02).reset_index(drop=True)


# Initialize output dictionary
dout = {
    "text_seq": [],
    "helpfulness": [],
    "honesty": [],
    "instruction_following": [],
    "truthfulness": [],
    "overall_score": [],  # Initialize this as a list
    "label": [],
}
weights = {
    "helpfulness": 0.3,
    "honesty": 0.1,
    "instruction_following": 0.3,
    "truthfulness": 0.1,
    "texts": 0.2,
}

# Loop through the DataFrame to gather data
for i in range(len(df.index)):
    for j in range(len(df["completions"].iloc[i])):
        print(f"Completion #{i+1}:")
        try:
            if "critique" in df["completions"].iloc[i][j]:
                text_seq = df["completions"].iloc[i][j]["critique"]
            else:
                text_seq = df["completions"].iloc[i][j]["annotations"][
                    "instruction_following"
                ]["Rationale for Rating"]

            # Extract ratings
            truthfulness_rating = int(
                df["completions"].iloc[i][j]["annotations"]["truthfulness"]["Rating"]
            )
            helpfulness_rating = int(
                df["completions"].iloc[i][j]["annotations"]["helpfulness"]["Rating"]
            )
            instruction_following_rating = int(
                df["completions"].iloc[i][j]["annotations"]["instruction_following"][
                    "Rating"
                ]
            )
            honesty_rating = int(
                df["completions"].iloc[i][j]["annotations"]["honesty"]["Rating"]
            )
            # Append ratings (as string) to the output dictionary
            dout["text_seq"].append(text_seq)
            dout["truthfulness"].append(truthfulness_rating)
            dout["helpfulness"].append(helpfulness_rating)
            dout["instruction_following"].append(instruction_following_rating)
            dout["honesty"].append(honesty_rating)
        except Exception as e:
            print(e)

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize_function(text):
    return tokenizer(
        text, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
    )


# Calculate overall score after filtering out "N/A" values
for i in range(len(dout["text_seq"])):
    helpfulness = dout["helpfulness"][i]
    honesty = dout["honesty"][i]
    instruction_following = dout["instruction_following"][i]
    truthfulness = dout["truthfulness"][i]

    log_helpfulness = (
        np.log(helpfulness + 1e-10) * weights["helpfulness"]
    )  # Add a small value to avoid log(0)
    log_honesty = np.log(honesty + 1e-10) * weights["honesty"]
    log_instruction_following = (
        np.log(instruction_following + 1e-10) * weights["instruction_following"]
    )
    log_truthfulness = np.log(truthfulness + 1e-10) * weights["truthfulness"]

    # Calculate the final score
    score = log_helpfulness + log_honesty + log_instruction_following + log_truthfulness

    dout["overall_score"].append(score)  # Append to overall_score list
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
    filename="*q4.gguf",
    verbose=False,
    n_gpu_layers=-1,
    n_ctx=1024,
)
output = lambda x: llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are an helpful assistant.",
        },
        {
            "role": "user",
            "content": f"Rate this feedback as (1,2,3,4,5) as a integer weighted sum of user sentiment, model constructiveness, and model instruction-following: 1 == User thinks poorly of chat && 5 == User thinks highly of the chat. Respond with a single digit integer, no headers or footers or other text. Rate this Chat Feedback: {x}",
        },
    ],
    stop=[
        "\n",
    ],
)
from tqdm import tqdm


def get_output(x, num_attempts=4):
    tries = 0
    while tries < num_attempts:
        try:
            out = output(x)["choices"][0]["message"]["content"]
            return int(out.strip().replace(" ", ""))
        except Exception as e:
            print(out)
            tries += 1
    return 3


import random, math


mean = np.mean(dout["overall_score"])
std = np.std(dout["overall_score"])
score_list = []
for i, score in enumerate(tqdm(dout["overall_score"])):
    out = get_output(dout["text_seq"][i])
    s = np.exp(score + np.log(out * weights["texts"]))
    score_list.append(s)
dout["overall_score"] = score_list


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


for score in dout["overall_score"]:
    print(score, sigmoid(score), 1 - sigmoid(score))
dout["label"] = [
    1 if 1 - sigmoid(score) > 0.5 + random.uniform(-0.1, 0.1) else 0
    for score in dout["overall_score"]
]
pd.DataFrame(dout).to_csv("output_sample.csv", index=False)
print(dout["label"][0], dout["overall_score"][0])

# dout lengths
print("text_seq: ", len(dout["text_seq"]))
print("helpfulness: ", len(dout["helpfulness"]))
print("honesty: ", len(dout["honesty"]))
print("instruction_following: ", len(dout["instruction_following"]))
print("truthfulness: ", len(dout["truthfulness"]))
print("overall_score: ", len(dout["overall_score"]))
print("label: ", len(dout["label"]))
