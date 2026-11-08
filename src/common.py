"""Shared pieces: loading the model and data, prompting, and grading answers."""

import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_SUFFIX = "\nSolve step by step, then end with: Answer: <number>"


def load_model(model_id=MODEL_ID):
    """Load the tokenizer and model in float16 on the GPU (if available)."""
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    return tok, model


def load_gsm8k(split="test"):
    """GSM8K from the Hugging Face Hub: 1,319 test problems, ~7,500 train."""
    return load_dataset("openai/gsm8k", "main", split=split)


def build_inputs(tok, model, question):
    """Wrap the question in the chat template and move it to the model's device."""
    msgs = [{"role": "user", "content": question + PROMPT_SUFFIX}]
    return tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)


def generate(tok, model, question, n=1, do_sample=False, temperature=0.7,
             top_p=0.95, max_new_tokens=512):
    """Return a list of n replies. do_sample=False gives the single greedy reply."""
    inputs = build_inputs(tok, model, question)
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        kwargs.update(temperature=temperature, top_p=top_p, num_return_sequences=n)
    out = model.generate(**inputs, **kwargs)
    n_prompt = inputs["input_ids"].shape[1]  # slice off the prompt tokens
    return [tok.decode(o[n_prompt:], skip_special_tokens=True) for o in out]


def last_number(text):
    """The last number in the text, without commas or a trailing period."""
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def extract(text):
    """Prefer the number after 'Answer:'; otherwise fall back to the last number."""
    m = re.search(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if m:
        return m.group(1).replace(",", "")
    return last_number(text)


def gold_answer(ex):
    """GSM8K stores the final answer after '####'."""
    return ex["answer"].split("####")[-1].strip().replace(",", "")


def to_number(p):
    """Turn an extracted answer into a float so '18' and '18.0' match."""
    try:
        return float(p)
    except (TypeError, ValueError):
        return None
