# 🐴 🏠 📚 StableLM-Books

This repository is based on my usage of Alpaca-LoRA and StableLM repositories. Purpose of this repo is to generate fake prompts from generic text data (books, textbooks) to be used for fine tuning.

This repository DOES NOT CONTAIN StableLM-alpha weights, download them here:

- 3B model, good for 16 GB (use_halving = True) and 24 GB GPUs, [download](https://huggingface.co/stabilityai/stablelm-base-alpha-3b/tree/main)
- 7B model, good for 24 GB (use_halving = True) and more powerful GPUs [download](https://huggingface.co/stabilityai/stablelm-base-alpha-7b)

If you do not own top consumer GPUs (April 2023), rent a machine at vast.io. RTX 3090 costs less then $0.30 per hour and RTX 4090 costs $0.45-$0.70 per hour. Training time for 15 MB of raw text (70 MB JSON of fake prompts) takes 4-5 hours on RTX 4090.

## Quickstart

Do yourself a favour and use conda (Anaconda, Miniconda) - [download](https://docs.conda.io/en/main/miniconda.html). Google how to install and activate it.

Install the requirements:
```
pip install -r requirements.txt
```

Copy your raw text file to the data folder, rename it to plain-text.txt or change the line in prompt-faker.py
```python
file1 = open('data/plain-text.txt', 'r')    
```

Create fake prompts with
```
python prompt-faker.py
```

To start fine tuning run
```
python finetune.py
```

When done, copy adapter* files from peft-lora/your-training-folder to a folder with base model (./models/base-3B)

Start talking to the model using
```
python generate.py
```

## Demo

I am using a model fine tuned like this with my project [PlotCrafting.com](https://plotcrafting.com). At the moment (Apr 25th 2023) only one model is active, and that is not StableLM based (it is still fine tuning).

## Credits

Credits are due to Stability AI! :)

## Licenses

- Base model checkpoints (`StableLM-Base-Alpha`) are licensed under the Creative Commons license ([CC BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/)). Under the license, you must give [credit](https://creativecommons.org/licenses/by/4.0/#) to Stability AI, provide a link to the license, and [indicate if changes were made](https://creativecommons.org/licenses/by/4.0/#). You may do so in any reasonable manner, but not in any way that suggests the Stability AI endorses you or your use.
