import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import os
from dotenv import load_dotenv
load_dotenv()

hfToken = os.getenv('HF_TOKEN')

base_model_id = "meta-llama/llama-2-7b-hf"
checkpoint = "./llama2-7b-finetune-brev"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Llama 2 7B, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True, token=hfToken)

from peft import PeftModel

ft_model = PeftModel.from_pretrained(base_model, checkpoint)

eval_prompt = "Artist: "
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

for i in range(10):
	ft_model.eval()
	with torch.no_grad():
	    print(tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=300)[0], skip_special_tokens=False))

