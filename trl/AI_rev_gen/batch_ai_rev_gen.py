import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import pandas as pd
from typing import List, Dict, Tuple
from tqdm import tqdm

# Load prompts from YAML file
with open("prompts.yaml", "r") as file:
    prompts = yaml.safe_load(file)

# Initialize the LLM
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    trust_remote_code=True,
)
tokenizer.pad_token_id = tokenizer.eos_token_id


def batch_generate(prompts: List[str], batch_size: int = 4) -> List[str]:
    """Generate responses for multiple prompts in batches."""
    responses = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i : i + batch_size]

        # Prepare batch inputs
        batch_messages = [
            [{"role": "user", "content": prompt}] for prompt in batch_prompts
        ]
        batch_inputs = [
            tokenizer.apply_chat_template(msgs, return_tensors="pt")
            for msgs in batch_messages
        ]

        # Pad the batch to the same length
        max_length = max(inputs.shape[1] for inputs in batch_inputs)
        padded_inputs = [
            (
                torch.cat(
                    [
                        inputs,
                        torch.full(
                            (1, max_length - inputs.shape[1]),
                            tokenizer.pad_token_id,
                            device=inputs.device,
                        ),
                    ],
                    dim=1,
                )
                if inputs.shape[1] < max_length
                else inputs
            )
            for inputs in batch_inputs
        ]

        # Stack inputs into a single tensor
        stacked_inputs = torch.cat(padded_inputs, dim=0).to("cuda")

        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                stacked_inputs,
                max_new_tokens=128,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode outputs
        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses.extend(
            [response.split("[/INST]")[-1].strip() for response in batch_responses]
        )

    return responses


def batch_refine_menu_items(
    menuitems: List[str], ingredients_list: List[str], batch_size: int = 4
) -> Dict[str, str]:
    """Refine multiple menu items in batches."""
    stages = ["probe", "expand", "filter", "finalize"]
    current_ingredients = ingredients_list

    for stage in stages:
        prompts_for_stage = [
            prompts["prompts"][stage]["prompt"].format(
                menu_item=menuitem,
                ingredient_list=ingredients,
                init_ingred_list=ingredients_list[i] if stage == "finalize" else "",
            )
            for i, (menuitem, ingredients) in enumerate(
                zip(menuitems, current_ingredients)
            )
        ]

        current_ingredients = batch_generate(prompts_for_stage, batch_size)

    return dict(zip(menuitems, current_ingredients))


# Example usage
if __name__ == "__main__":
    splits = {
        "train": "data/train-00000-of-00001-74da805aa0f7f02c.parquet",
        "test": "data/test-00000-of-00001-b53065eb169d75d2.parquet",
    }
    df = pd.read_parquet(
        "hf://datasets/skadewdl3/recipe-nlg-lite-llama-2/" + splits["train"]
    )
    menuitems = df["name"].tolist()
    ingredients = df["ner"].tolist()

    # Process in batches
    refined_ingredients = batch_refine_menu_items(menuitems, ingredients, batch_size=8)
