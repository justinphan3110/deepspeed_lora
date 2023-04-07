import os

import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402

BASE_MODEL = os.environ.get("BASE_MODEL", None)
LORA_MODEL = os.environ.get("LORA_MODEL", None)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", None)


assert (
    BASE_MODEL
), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=decapoda-research/llama-7b-hf`"  # noqa: E501

assert (
    LORA_MODEL
), "Please specify a value for LORA_MODEL environment variable"  # noqa: E501
assert (
    OUTPUT_DIR
), "Please specify a value for OUTPUT_DIR environment variable" 


print("BASE_MODEL", BASE_MODEL)
print("LORA_MODEL", LORA_MODEL)
print("OUTPUT_DIR", OUTPUT_DIR)

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
base_model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL,
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

lora_weight = lora_model.base_model.model.model.layers[
    0
].self_attn.q_proj.weight

assert torch.allclose(first_weight_old, first_weight)

# merge weights
for layer in lora_model.base_model.model.model.layers:
    layer.self_attn.q_proj.merge_weights = True
    layer.self_attn.v_proj.merge_weights = True

lora_model.train(False)

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

LlamaForCausalLM.save_pretrained(
    base_model, OUTPUT_DIR, state_dict=deloreanized_sd, max_shard_size="400MB"
)
