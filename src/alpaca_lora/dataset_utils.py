
from .prompter import Prompter
from datasets import load_dataset

def tokenize(prompt, tokenizer, cutoff_len=512, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point, tokenizer, train_on_inputs=True, prompt_template_name="alpaca"):
    prompter = Prompter(prompt_template_name)
    tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token   
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt)
    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt


def generate_alpaca_lora_dataset(data_args, tokenizer):
    if data_args.train_file and (data_args.train_file.endswith(".json") or data_args.train_file.endswith(".jsonl")):
        data = load_dataset("json", data_files=data_args.train_file)
    elif data_args.dataset_name:
        data = load_dataset(data_args.dataset_name)
    else: 
        assert data_args.dataset_name or data_args.train_file, "dataset_name or train_file need to be defined"

    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, fn_kwargs={'tokenizer': tokenizer})
    val_data = None

    return train_data, val_data


