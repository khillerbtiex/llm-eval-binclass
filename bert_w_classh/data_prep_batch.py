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
from tqdm import tqdm

# Load dataset and convert to pandas DataFrame
ds = load_dataset("openbmb/UltraFeedback")["train"]
ds = ds.shuffle(seed=42)
df = ds.to_pandas().sample(frac=0.1).reset_index(drop=True)

# Initialize output dictionary
din = {
    "text_seq": [],
    "helpfulness": [],
    "honesty": [],
    "instruction_following": [],
    "truthfulness": [],
    "interactions": {"inst": [], "comp": []},
    "text_seq_base_score": [],
    "text_seq_prompt_score": [],
    "overall_score": [],
    "fine_score": [],
    "label": [],
}


weights = {
    "helpfulness": 0.2,
    "honesty": 0.1,
    "instruction_following": 0.25,
    "truthfulness": 0.05,
    "texts": 0.3,
}


def sigmoid_around_3(x):
    return 1 / (1 + math.exp(-(x - 3)))


# Loop through the DataFrame to gather data
for i in range(len(df.index)):
    print(f"Completion #{i+1}:")
    for j in range(len(df["completions"].iloc[i])):
        try:
            if "critique" in df["completions"].iloc[i][j]:
                text_seq = df["completions"].iloc[i][j]["critique"]
            else:
                text_seq = df["completions"].iloc[i][j]["annotations"]["helpfulness"][
                    "Rationale for Rating"
                ]

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
            din["text_seq"].append(text_seq)
            din["truthfulness"].append(truthfulness_rating)
            din["helpfulness"].append(helpfulness_rating)
            din["instruction_following"].append(instruction_following_rating)
            din["honesty"].append(honesty_rating)
            din["interactions"]["inst"].append(df["instruction"].iloc[i])
            din["interactions"]["comp"].append(df["completions"].iloc[i][j]["response"])

            din["fine_score"].append(df["completions"].iloc[i][j]["fine-grained_score"])
        except Exception as e:
            print(e)

del df, ds
# Calculate overall score after filtering out "N/A" values
for i in range(len(din["text_seq"])):
    helpfulness = din["helpfulness"][i]
    honesty = din["honesty"][i]
    instruction_following = din["instruction_following"][i]
    truthfulness = din["truthfulness"][i]

    din["helpfulness"][i] = np.log(helpfulness)  # Add a small value to avoid log(0)
    din["honesty"][i] = np.log(honesty) * weights["honesty"]
    din["instruction_following"][i] = (
        np.log(instruction_following) * weights["instruction_following"]
    )
    din["truthfulness"][i] = np.log(truthfulness) * weights["truthfulness"]

    # Calculate the final score
    score = (
        (din["helpfulness"][i] * (weights["helpfulness"] + weights["texts"]))
        + din["honesty"][i]
        + din["instruction_following"][i]
        + din["truthfulness"][i]
    )

    din["overall_score"].append(score)  # Append to overall_score list


hf_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
)
# hf_model.generation_config.cache_implementation = "static"
# hf_model.forward = torch.compile(
#     hf_model.forward, mode="reduce-overhead", fullgraph=True
# )

hf_tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2", padding_side="left"
)
hf_tokenizer.pad_token = hf_tokenizer.unk_token
hf_tokenizer.pad_token_id = hf_tokenizer.unk_token_id


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

    for i in tqdm(range(0, len(msgs), batch_size), smoothing=0.0):
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
                                torch.full(
                                    (1, max_length - inputs.size(1)),
                                    hf_tokenizer.unk_token_id,
                                    device=inputs.device,
                                ),
                                inputs,
                            ],
                            dim=1,
                        )
                        if inputs.size(1) < max_length
                        else inputs
                    )
                    for inputs in batch_inputs
                ]

                stacked_inputs = torch.cat(padded_inputs, dim=0).to("cuda")
                with torch.nn.attention.sdpa_kernel(
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION
                ):

                    with torch.no_grad():
                        outputs = hf_model.generate(
                            stacked_inputs,
                            min_new_tokens=1,
                            max_new_tokens=1,
                            do_sample=False,
                            num_return_sequences=1,
                            logits_processor=logits_processor,
                            tokenizer=hf_tokenizer,
                            pad_token_id=hf_tokenizer.pad_token_id,
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


message_base_scores = lambda x: [
    {
        "role": "user",
        "content": f"Rate this feedback as (1,2,3,4,5) as a integer weighted sum of user sentiment, model constructiveness, and model instruction-following: 1 == User thinks poorly of chat and gives many critiques && 5 == User thinks highly of the chat and gives few critiques. Amount of deviation from score 3 should match correlate to user opinion. Respond with a single digit integer, no headers or footers or other text. Rate this Chat Feedback: {x}",
    },
]
message_prompt_scores = lambda x, y: [
    {
        "role": "user",
        "content": f"Rate the quality of this chatbot interaction from 1 to 5. Respond with a single digit integer, no headers or footers or other text. User Message: {x}\n\nChatbot Message: {y}",
    }
]
import random, math, pickle


messages_base = [message_base_scores(x) for x in din["text_seq"]]

messages_prompt = [
    message_prompt_scores(x, y)
    for x, y in zip(din["interactions"]["inst"], din["interactions"]["comp"])
]
din["text_seq_base_score"] = batch_generate(messages_base, 16)
din["text_seq_prompt_score"] = batch_generate(messages_prompt, 16)
score_list = []
for i, score in enumerate(tqdm(din["overall_score"])):
    out = np.log(din["text_seq_base_score"][i]) * weights["texts"]
    s = np.exp(
        (din["helpfulness"][i] * weights["helpfulness"])
        + din["instruction_following"][i]
        + din["truthfulness"][i]
        + out
    )
    out = np.average(
        [
            s,
            din["overall_score"][i],
            din["fine_score"][i],
            din["text_seq_base_score"][i],
            din["text_seq_prompt_score"][i],
        ]
    )
    out = sigmoid_around_3(out)
    score_list.append(s)


din["label"] = [
    1 if score >= 0.65 + random.uniform(-0.05, 0.05) else 0 for score in score_list
]

dout = {
    "text_seq": din["text_seq"],
    "helpfulness": din["helpfulness"],
    "honesty": din["honesty"],
    "instruction_following": din["instruction_following"],
    "truthfulness": din["truthfulness"],
    "overall_score": din["overall_score"],
    "label": din["label"],
}

dout_df = pd.DataFrame(dout)
dout_df.to_csv("output_sample.csv", index=False)
print(dout_df["overall_score"].value_counts())

# dout lengths
print("text_seq: ", len(dout["text_seq"]))
print("helpfulness: ", len(dout["helpfulness"]))
print("honesty: ", len(dout["honesty"]))
print("instruction_following: ", len(dout["instruction_following"]))
print("truthfulness: ", len(dout["truthfulness"]))
print("overall_score: ", len(dout["overall_score"]))
print("label: ", len(dout["label"]))
