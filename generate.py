import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

import readline
import sys
import time
import textwrap


use_halving = False # set to True if you have 16 GB of GPU RAM, set to False if you have 24 GB or more
model_and_peft_location = "./models/base-3B/"


tokenizer = AutoTokenizer.from_pretrained(model_and_peft_location, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_and_peft_location, device_map="auto")

if use_halving:
    model.half().cuda()
else:
    model.cuda()

model.config.use_cache = False

model = PeftModel.from_pretrained(
    model,
    model_and_peft_location,
    device_map="auto"
)

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

temperature = 0.7
max_new_tokens = 128

while True:
    my_input = input(f"\ntemp: {temperature} | max: {max_new_tokens} > ")
    should_do_prompt = True

    if my_input == 'exit':
        print("Bye!")
        break

    if my_input[0:4] == 'set ':            
        use_input = my_input[4:].strip()
        if use_input[0:5] == 'temp=':
            t = use_input.split('=')
            if len(t) == 2:
                temperature = float(t[1])
                should_do_prompt = False

        if use_input[0:4] == 'max=':
            t = use_input.split('=')
            if len(t) == 2:
                max_new_tokens = int(t[1])
                should_do_prompt = False

        if use_input[0:6] == 'top_k=':
            t = use_input.split('=')
            if len(t) == 2:
                top_k = int(t[1])
                should_do_prompt = False

    if should_do_prompt:
        # prompt = f"{system_prompt}<|USER|>{my_input}<|ASSISTANT|>"
        # prompt = f"<|USER|>{my_input}<|ASSISTANT|>"
        prompt = f'{my_input}'

        t0 = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        tokens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()])
        )
        answer = tokenizer.decode(tokens[0], skip_special_tokens=True) #[len(my_input):]

        t = time.perf_counter() - t0
        tw = textwrap.wrap(answer)
        print('\n'.join(tw) + '\n')
        print(f"[Time for inference: {t:.02f}s total; Memory in use: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB; type 'help' for help]", file=sys.stderr)
        