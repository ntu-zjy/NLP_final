import torch
import transformers
import torch.cuda as cuda
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model
# do not show the warning messages
import warnings
warnings.filterwarnings("ignore")
model_fine_tune = "Llama-2-7b-hf"
peft_model_id = f"CoT-{model_fine_tune}"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

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
pbar =tqdm(os.listdir("./test_data/data")[2000:6000], total=4000)
for file in pbar:
  with open(f"./test_data/data/{file}", "r") as f:
    tst = f.read()
  answer = get_answer(tst)
#   pbar.set_description(answer)
  with open(f"./test_data/Llamacot/{file}", "w+") as f:
    f.write(answer)