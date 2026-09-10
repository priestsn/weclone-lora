import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = 'models/Qwen2.5-7B-Instruct'
ADAPT = 'model_output'
SYS = 'Sei k k, un ragazzo italiano. Rispondi come lui: messaggi brevi, tono casual da WhatsApp, italiano nord-italiano. Non dire mai di essere una IA.'

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE, trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
    device_map='auto')
model = PeftModel.from_pretrained(model, ADAPT)
model.eval()

tests = ['ciao, come va?', 'cosa facciamo stasera?', 'hai visto la partita ieri?']
with open('clone_test_out.txt', 'w', encoding='utf-8') as f:
    for t in tests:
        msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': t}]
        inp = tok.apply_chat_template(msgs, return_tensors='pt', add_generation_prompt=True).to(model.device)
        with torch.inference_mode():
            out = model.generate(inp, max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9,
                                 repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()
        print('Q:', t, '\nA:', gen, '\n' + '-' * 40, flush=True)
        f.write(f'Q: {t}\nA: {gen}\n{"-" * 40}\n')
print('WROTE clone_test_out.txt')
