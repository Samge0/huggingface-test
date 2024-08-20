#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author：samge
# date：2024-08-20 10:18
# describe：

"""
SmolLM 是一系列语言模型，有三种大小：135M、360M 和 1.7B 参数。
这些模型在SmolLM-Corpus上进行训练，SmolLM-Corpus 是专为训练 LLM 而设计的精选高质量教育和合成数据集合。
https://huggingface.co/HuggingFaceTB/SmolLM-360M-Instruct
"""

# pip install transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "HuggingFaceTB/SmolLM-360M-Instruct"

device = "cuda" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

while True:
    input_text = input("请输入：")
    messages = [{"role": "user", "content": input_text}]
    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    print(input_text)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
    print(tokenizer.decode(outputs[0]))
