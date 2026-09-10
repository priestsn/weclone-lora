"""Smoke test clone: domande problematiche storiche + memoria."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from collections import deque
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

BASE = 'models/Qwen2.5-7B-Instruct'
ADAPT = 'model_output'
SYS = 'Sei k k, un ragazzo italiano. Rispondi come lui: messaggi brevi, tono casual da WhatsApp, italiano nord-italiano. Non dire mai di essere una IA.'

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    BASE, trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
    device_map='auto')
model = PeftModel.from_pretrained(model, ADAPT)
model.eval()

history = deque(maxlen=12)
tests = ["ciao, come va?", "chi sei?", "che scheda video hai?",
         "ci vediamo stasera per giocare?", "sì, ad apex",
         "hai visto la partita ieri?"]

def ask(q):
    msgs = [{'role': 'system', 'content': SYS}, *list(history), {'role': 'user', 'content': q}]
    inp = tok.apply_chat_template(msgs, return_tensors='pt', add_generation_prompt=True).to(model.device)
    attn = torch.ones_like(inp)
    with torch.inference_mode():
        out = model.generate(inp, attention_mask=attn, max_new_tokens=120, do_sample=True,
                             temperature=0.5, top_p=0.65, repetition_penalty=1.2,
                             pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()
    lines = [l.strip() for l in gen.splitlines() if l.strip()]
    if len(lines) > 2 and sum(len(l) for l in lines) > 200:
        gen = '\n'.join(lines[:2])
    history.append({'role': 'user', 'content': q})
    history.append({'role': 'assistant', 'content': gen})
    return gen

for q in tests:
    print(f"TU: {q}\nCLONE: {ask(q)}\n---")
