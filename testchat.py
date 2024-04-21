import torch
import transformers
import torch.cuda as cuda
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model
# do not show the warning messages
import warnings
warnings.filterwarnings("ignore")

model_id = "meta-llama/Llama-2-7b-chat-hf"
access_token = "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"
# Bits and bytes config. Parameters explained in the QLoRA paper
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    token=access_token
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    # quantization_config=bnb_config,
    device_map='auto',
    token=access_token
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
)

def get_answer(tst):
  batch = tokenizer(tst, return_tensors='pt')

  with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=90)

  #print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))

  output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
  end_index = output.find("### End")
  output = output[:end_index]

  return  output

import os
from tqdm import tqdm
pbar =tqdm(os.listdir("./test_data/Llamacot/"), total=len(os.listdir("./test_data/Llamacot/")))
for file in pbar:
  with open(f"./test_data/data/{file}", "r") as f:
    tst = f.read()
  answer = get_answer(tst)
#   pbar.set_description(answer)
  with open(f"./test_data/Llamachat/{file}", "w+") as f:
    f.write(answer)