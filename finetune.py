import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from typing import List
from datasets import load_dataset

import readline
import sys
import time
import textwrap

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

data_prefix = "test"
output_dir = "./peft-lora/"
batch_size = 16
micro_batch_size = 2
num_epochs = 2
learning_rate = 3e-4
cutoff_len = 768
val_set_size = 1000
# lora hyperparams
lora_r = 128
lora_alpha = 256
lora_dropout = 0.05
lora_target_modules: List[str] = [
    # "q_proj",
    # "v_proj",
]
# llm hyperparams
train_on_inputs = True  # if False, masks out inputs in loss
group_by_length = True  # was False faster, but produces an odd training loss curve
resume_from_checkpoint = None #"lora-trained",  # either training checkpoint or final adapter
use_prompt_template = True

output_dir = f"./peft-lora/{data_prefix}-3B-e{num_epochs}-b{batch_size}-m{micro_batch_size}-lr{lora_r}-la{lora_alpha}"

gradient_accumulation_steps = batch_size // micro_batch_size

device_map = "auto"

use_halving = False # set to True if you have 16 GB of GPU RAM, set to False if you have 24 GB or more
model_and_peft_location = "./models/base-3B/"

# Make sure you have downloaded StableLM weights to the models/base-3B/ folder
tokenizer = AutoTokenizer.from_pretrained(model_and_peft_location)
model = AutoModelForCausalLM.from_pretrained(model_and_peft_location)

if use_halving:
    model.half().cuda()
else:
    model.cuda()

tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"  # Allow batched inference

def tokenize(prompt, add_eos_token=True):
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

def generate_and_tokenize_prompt(prompt):
    full_prompt = f'{prompt["instruction"]}. {prompt["output"]}'
    if use_prompt_template:
        full_prompt = f'<|USER|>{prompt["instruction"]}<|ASSISTANT|>{prompt["output"]}'
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt

config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=None,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

data = load_dataset("json", data_files='./data/fake-prompts.json')
train_val = data["train"].train_test_split(
    test_size=val_set_size, shuffle=True, seed=42
)
train_data = (
    train_val["train"].shuffle().map(generate_and_tokenize_prompt)
)
val_data = (
    train_val["test"].shuffle().map(generate_and_tokenize_prompt)
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=use_halving,
        logging_steps=1000,
        optim="adamw_torch",
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=1000 if val_set_size > 0 else None,
        save_steps=1000,
        output_dir=output_dir,
        save_total_limit=2,
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False,
        group_by_length=group_by_length,
        report_to=None,
        run_name=None,
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

trainer.train(resume_from_checkpoint=resume_from_checkpoint)

model.save_pretrained(output_dir)


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

