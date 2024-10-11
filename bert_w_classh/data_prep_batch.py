import pandas as pd
import numpy as np
from transformers import (
    LogitsProcessorList,
    LogitsProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
import numpy as np
from datasets import load_dataset

# Load dataset and convert to pandas DataFrame
ds = load_dataset("openbmb/UltraFeedback")["train"]
ds = ds.shuffle(seed=42)
df = ds.to_pandas().sample(frac=0.1).reset_index(drop=True)


# Initialize output dictionary
dout = {
    "text_seq": [],
    "helpfulness": [],
    "honesty": [],
    "instruction_following": [],
    "truthfulness": [],
    "text_seq_score": [],
    "overall_score": [],  # Initialize this as a list
    "label": [],
}
weights = {
    "helpfulness": 0.2,
    "honesty": 0.1,
    "instruction_following": 0.25,
    "truthfulness": 0.05,
    "texts": 0.3,
}

# Loop through the DataFrame to gather data
for i in range(len(df.index)):
    print(f"Completion #{i+1}:")
    for j in range(len(df["completions"].iloc[i])):
        print(f"\tModel: {j+1}")
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


hf_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

hf_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
hf_tokenizer.pad_token = hf_tokenizer.unk_token
hf_tokenizer.pad_token_id = hf_tokenizer.unk_token_id
hf_tokenizer.padding_side = "left"


valid_responses = ["1", "2", "3", "4", "5"]
valid_response_ids = hf_tokenizer.convert_tokens_to_ids(valid_responses)


def is_convertible_to_int(s):
    try:
        return int(s.strip())
    except ValueError:
        return 6


class SingleTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, valid_token_ids):
        self.valid_token_ids = valid_token_ids

    def __call__(self, input_ids, scores):
        # Mask all tokens except for the valid token IDs (1, 2, 3, 4, 5)
        mask = torch.full_like(
            scores, -1e6
        )  # Set scores for invalid tokens to a very low value
        mask[:, self.valid_token_ids] = scores[
            :, self.valid_token_ids
        ]  # Retain original scores for valid tokens
        return mask


def batch_generate(msgs, batch_size=4, num_attempts=3):
    responses = []
    logits_processor = LogitsProcessorList(
        [SingleTokenLogitsProcessor(valid_response_ids)]
    )

    for i in tqdm(range(0, len(msgs), batch_size)):
        for tries in range(num_attempts):
            try:
                batch_msgs = msgs[i : i + batch_size]
                batch_inputs = [
                    hf_tokenizer.apply_chat_template(msg, return_tensors="pt")
                    for msg in batch_msgs
                ]

                max_length = max(inputs.size(1) for inputs in batch_inputs)
                padded_inputs = [
                    (
                        torch.cat(
                            [
                                inputs,
                                torch.full(
                                    (1, max_length - inputs.size(1)),
                                    hf_tokenizer.pad_token_id,
                                    device=inputs.device,
                                ),
                            ],
                            dim=1,
                        )
                        if inputs.size(1) < max_length
                        else inputs
                    )
                    for inputs in batch_inputs
                ]

                stacked_inputs = torch.cat(padded_inputs, dim=0).to("cuda")

                with torch.no_grad():
                    outputs = hf_model.generate(
                        stacked_inputs,
                        min_new_tokens=1,
                        max_new_tokens=1,
                        do_sample=False,
                        num_return_sequences=1,
                        logits_processor=logits_processor,
                        tokenizer=hf_tokenizer,
                    )

                batch_outputs = hf_tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
                split_outputs = [out.split("[/INST]")[-1] for out in batch_outputs]
                for response in split_outputs:
                    num_response = is_convertible_to_int(response)
                    responses.append(
                        3 if num_response not in [1, 2, 3, 4, 5] else num_response
                    )
                break
            except Exception as e:
                print(f"Attempt {tries + 1} failed: {e})")
    return responses


message_out = lambda x: [
    {
        "role": "user",
        "content": f"Rate this feedback as (1,2,3,4,5) as a integer weighted sum of user sentiment, model constructiveness, and model instruction-following: 1 == User thinks poorly of chat and gives many critiques && 5 == User thinks highly of the chat and gives few critiques. Amount of deviation from score 3 should match correlate to user opinion. Respond with a single digit integer, no headers or footers or other text. Rate this Chat Feedback: {x}",
    },
]
from tqdm import tqdm
import random, math


messages = [message_out(x) for x in dout["text_seq"]]
dout["text_seq_score"] = batch_generate(messages, 16)
score_list = []
for i, score in enumerate(tqdm(dout["overall_score"])):
    out = dout["text_seq_score"][i]
    s = np.exp(score + np.log(out * weights["texts"]))
    score_list.append(s)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


dout["label"] = [
    1 if score >= 3 + random.uniform(-0.15, 0.15) else 0 for score in score_list
]
dout_df = pd.DataFrame(dout)
dout_df.to_csv("output_sample.csv", index=False)
print(dout_df["text_seq_score"].value_counts())

# dout lengths
print("text_seq: ", len(dout["text_seq"]))
print("helpfulness: ", len(dout["helpfulness"]))
print("honesty: ", len(dout["honesty"]))
print("instruction_following: ", len(dout["instruction_following"]))
print("truthfulness: ", len(dout["truthfulness"]))
print("overall_score: ", len(dout["overall_score"]))
print("label: ", len(dout["label"]))
