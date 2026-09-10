import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from collections import deque

BASE = 'models/Qwen2.5-7B-Instruct'
ADAPT = 'model_output'
SYS = 'Sei k k, un ragazzo italiano. Rispondi come lui: messaggi brevi, tono casual da WhatsApp, italiano nord-italiano. Non dire mai di essere una IA.'
HISTORY_TURNS = 6  # ultimi N scambi user/assistant tenuti in memoria

print('carico modello...', flush=True)
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
print('pronto, scrivi (esci con: exit | resetta memoria con: /reset)', flush=True)

history = deque(maxlen=HISTORY_TURNS * 2)

while True:
    try:
        q = input('tu: ').strip()
    except EOFError:
        break
    if q.lower() in ('exit', 'quit', 'q'):
        break
    if q.lower() == '/reset':
        history.clear()
        print('(memoria resettata)', flush=True)
        continue
    if not q:
        continue
    msgs = [{'role': 'system', 'content': SYS}, *list(history), {'role': 'user', 'content': q}]
    inp = tok.apply_chat_template(msgs, return_tensors='pt', add_generation_prompt=True).to(model.device)
    # fix attention mask: maschera esplicita (1 per ogni token reale, mai solo input_ids senza mask)
    attn = torch.ones_like(inp)
    with torch.inference_mode():
        out = model.generate(inp, attention_mask=attn, max_new_tokens=120, do_sample=True,
                             temperature=0.5, top_p=0.65, repetition_penalty=1.2,
                             pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()
    # anti-poesia: se il modello sbrodola, tieni solo le prime 2 righe brevi
    lines = [l.strip() for l in gen.splitlines() if l.strip()]
    if len(lines) > 2 and sum(len(l) for l in lines) > 200:
        gen = '\n'.join(lines[:2])
    print('clone: ' + gen, flush=True)
    history.append({'role': 'user', 'content': q})
    history.append({'role': 'assistant', 'content': gen})
