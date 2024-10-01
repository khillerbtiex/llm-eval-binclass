import pandas as pd
from trl import BCOTrainer, BCOConfig

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from datasets import Dataset
from peft import get_peft_model, LoraConfig
import yaml, os

with open("train_config.yaml", "r") as f:
    config = yaml.safe_load(f)


def get_data(endpoint, sample_frac=0.05):
    return pd.read_parquet(endpoint).sample(frac=sample_frac)


def fix_msg_data(config, df):
    dialogue_col = config["dialogue-col"]
    chosen_col = config["chosen-col"]
    rejected_col = config["rejected-col"]
    df_true = df[[dialogue_col, chosen_col]]
    df_true["label"] = True
    df_true[dialogue_col] = df_true[dialogue_col].apply(
        lambda x: x.split("Human:")[1].split("Assistant:")[0]
    )

    df_false = df[[dialogue_col, rejected_col]]
    df_false["label"] = False
    df_false[dialogue_col] = df_false[dialogue_col].apply(
        lambda x: x.split("Human:")[1].split("Assistant:")[0]
    )

    df_true.columns = ["prompt", "completion", "label"]
    df_false.columns = ["prompt", "completion", "label"]
    df = pd.concat([df_true, df_false]).sample(frac=1).reset_index(drop=True)

    return Dataset.from_pandas(df)


def main(config, train, test):
    training_args = BCOConfig(
        beta=0.1,
        output_dir=config["rw-model-output-dir"],
        max_length=1024,
        max_prompt_length=256,
        remove_unused_columns=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model-to-train-and-ref"])
    tokenizer.pad_token_id = tokenizer.unk_token_id

    # Load quantized model with 4-bit precision
    model = AutoModelForCausalLM.from_pretrained(
        config["model-to-train-and-ref"],  # Quantize model to 4-bit
        device_map="auto",  # Automatically assigns layers to GPUs if multiple are available
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,  # Use double quantization to reduce VRAM usage further
            bnb_4bit_quant_type="nf4",  # Quantization type, Normal Float 4 is more stable
        ),
    )

    # Set up PEFT for efficient fine-tuning using LoRA (Low-Rank Adaptation)
    peft_config = LoraConfig(
        r=8,  # Low-rank dimension
        lora_alpha=32,  # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Apply LoRA to these modules
        lora_dropout=0.1,  # Dropout for LoRA layers
        bias="none",
    )

    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)

    # Load reference model (quantized)
    model_ref = AutoModelForCausalLM.from_pretrained(
        config["model-to-train-and-ref"],
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
        ),
    )

    bco_trainer = BCOTrainer(
        model,
        model_ref,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=tokenizer,
    )

    bco_trainer.train()

    bco_trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train = get_data(config["train-data-endpoint"])
    test = get_data(config["test-data-endpoint"])
    train_dataset = fix_msg_data(config, train)
    test_dataset = fix_msg_data(config, test)
    main(config, train_dataset, test_dataset)
