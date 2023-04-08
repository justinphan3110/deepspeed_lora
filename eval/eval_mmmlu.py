# TODO: optimize code

import os
import fire

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
    DataCollatorForSeq2Seq,
    GenerationConfig,
    
)
import numpy as np
from typing import Dict

from torchtracemalloc import TorchTracemalloc, b2mb
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import transformers
from args import (
    ModelArguments,
    DataTrainingArguments,
    TrainingArguments,
)
from alpaca_lora.dataset_utils import generate_alpaca_lora_dataset
from alpaca_lora.prompter import Prompter
from tqdm import tqdm
from datasets import Dataset

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
def main(
    load_8bit: bool = False,
    model_name_or_path: str = "../out/merged_hf_models/llama_alpaca_lora_7B/",
    adapter_name_or_path: str = None,
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
):
    prompter = Prompter(prompt_template)
    device="cuda"
    accelerator = Accelerator()

    print("Base Model", model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
    )

    if adapter_name_or_path:
        print("PEFT Adapter", adapter_name_or_path) 
        model = PeftModel.from_pretrained(
            model,
            adapter_name_or_path,
            torch_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    model = accelerator.prepare(model)
    model.eval()

    print(model)


    def evaluate(
        instructions,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_length=512,
        max_samples = -1,
        **kwargs,
    ):
        questions = [i[0] for i in instructions]
        answers = [i[-1] for i in instructions]

        tokenizer.padding_side = 'left'
        prompt = [prompter.generate_prompt(i, input) for i in questions]
        def __tokenized_text(examples):
            return tokenizer(examples['text'], padding=True, return_tensors="pt", max_length=max_length)

        dataset = Dataset.from_dict({"text": prompt})
        inputs = dataset.map(__tokenized_text, batched=True).remove_columns(['text'])

        print("Total samples", len(inputs))
        dataloader = torch.utils.data.DataLoader(inputs, collate_fn=default_data_collator, batch_size=16)

        preds = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader)):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                # generation_outputs = model.generate(
                #     input_ids=input_ids.to('cuda'),
                #     attention_mask=attention_mask.to('cuda'),
                #     generation_config=generation_config,
                #     return_dict_in_generate=True,
                #     output_scores=True,
                #     max_new_tokens=max_new_tokens,
                # )
                # s = tokenizer.batch_decode(generation_outputs.sequences, skip_special_tokens=True)
                # s = [prompter.get_response(x) for x in s]
                # outputs.extend(s)

                logits = model(
                    input_ids=input_ids.to('cuda'), attention_mask=attention_mask.to('cuda'), output_hidden_states=True
                ).logits[:, -1].to(torch.float32)
                
                options = ["A", "B", "C", "D"]
                option_ids = [tokenizer(option).input_ids[1] for option in options]
                option_ids_tensor = torch.tensor(option_ids, device=logits.device)
                selected_logits = torch.index_select(logits, 1, option_ids_tensor)

                probs = torch.softmax(selected_logits, dim=1).detach().cpu()
                choice = torch.argmax(probs, dim=1).tolist()

                index_to_option = {0: "A", 1: "B", 2: "C", 3: "D"}
                pred = [index_to_option[index] for index in choice]

                preds.extend(pred)

        acc = sum(true.strip().lower()==predict.lower() for true, predict in zip(answers, preds))
        print("Accuracy", acc / len(answers))

    import json
    with open('../data/mmmlu.json', 'r') as file:
        data = json.load(file)
    evaluate(data)

if __name__ == "__main__":
    fire.Fire(main)