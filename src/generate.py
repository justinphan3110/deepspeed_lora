import os

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)

from typing import Dict

from .torchtracemalloc import TorchTracemalloc, b2mb
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import transformers
from args import (
    ModelArguments,
    DataTrainingArguments,
)
from alpaca_lora.dataset_utils import generate_alpaca_lora_dataset
from alpaca_lora.prompter import Prompter

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

set_seed(0)
def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    
    model_name = model_args.model_name_or_path
    adapter_name = model_args.adapter_name_or_path
    accelerator = Accelerator()

    print("Base Model", model_name)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            # device_map="auto",
    )

    if adapter_name:
        print("PEFT Adapter", adapter_name) 
        model = PeftModel.from_pretrained(
                model,
                adapter_name,
                torch_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    with accelerator.main_process_first():
        processed_datasets, _ = generate_alpaca_lora_dataset(data_args, tokenizer)
    accelerator.wait_for_everyone()

    batch_size = 8
    dataloader = DataLoader(
        processed_datasets, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )

    print("=====Sample data=====")
    print(next(iter(dataloader)))
    prompter = Prompter()
    model.eval()
    test_preds = []

    with TorchTracemalloc() as tracemalloc:
        for _, batch in enumerate(tqdm(dataloader)):
            batch = {k: v for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                outputs = accelerator.unwrap_model(model).generate(
                    **batch, synced_gpus=is_ds_zero_3, max_new_tokens=10
                )  # synced_gpus=True for DS-stage 3
            outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
            preds = accelerator.gather(outputs)
            # preds = preds[:, max_length:].detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            outputs = tokenizer.batch_decode(preds, skip_special_tokens=True)
            texts = [prompter.get_response(o) for o in outputs]
            test_preds.extend(texts)

    accelerator.print("GPU Memory before entering the predict : {}".format(b2mb(tracemalloc.begin)))
    accelerator.print("GPU Memory consumed at the end of the predict (end-begin): {}".format(tracemalloc.used))
    accelerator.print("GPU Peak Memory consumed during the predict (max-begin): {}".format(tracemalloc.peaked))
    accelerator.print(
        "GPU Total Peak Memory consumed during the predict (max): {}".format(
            tracemalloc.peaked + b2mb(tracemalloc.begin)
        )
    )

    print(test_preds)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()