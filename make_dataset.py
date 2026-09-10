"""Rigenera dataset SFT dai CSV grezzi: multi-turno, senza token cinesi, senza junk."""
import csv, json, glob, os, random
from datetime import datetime

random.seed(42)
CHATS = ['dataset/csv/chat_a.csv',
         'dataset/csv/chat_b.csv',
         'dataset/csv/chat_c.csv']
GAP_NEW_CONV_MIN = 10     # gap > 10 min = nuova conversazione
MAX_CTX_TURNS = 6         # turni di contesto per campione
MAX_MSG_CHARS = 800
CONTAM = ['你应该说', '<begin_chat>', '</begin_chat>']

def parse_t(s):
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')

def load_rows(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['type_name'] != 'text':
                continue
            msg = (r['msg'] or '').strip()
            if not msg or any(c in msg for c in CONTAM):
                continue
            try:
                t = parse_t(r['CreateTime'])
            except Exception:
                continue
            rows.append((t, r['is_sender'] == '1', msg))
    rows.sort(key=lambda x: x[0])
    return rows

def merge_consecutive(rows):
    """unisce messaggi consecutivi dello stesso speaker in un solo turno"""
    turns = []
    for t, is_me, msg in rows:
        if turns and turns[-1][1] == is_me and (t - turns[-1][0]).total_seconds() < 120:
            turns[-1] = (turns[-1][0], is_me, turns[-1][2] + '\n' + msg)
        else:
            turns.append((t, is_me, msg))
    return turns

def split_conversations(turns):
    convs, cur = [], []
    for t, is_me, msg in turns:
        if cur and (t - cur[-1][0]).total_seconds() > GAP_NEW_CONV_MIN * 60:
            convs.append(cur); cur = []
        cur.append((t, is_me, msg))
    if cur:
        convs.append(cur)
    return convs

def is_junk(msg):
    m = msg.strip().lower()
    if len(m) > MAX_MSG_CHARS:
        return True
    # solo emoji/sticker/url -> poco segnale stilistico
    if m.startswith('http') or m.startswith('www.'):
        return True
    return False

samples = []
for path in CHATS:
    rows = load_rows(path)
    turns = merge_consecutive(rows)
    for conv in split_conversations(turns):
        msgs = []
        for t, is_me, msg in conv:
            if is_junk(msg):
                # un url da solo non rompe la conversazione, lo saltiamo
                continue
            role = 'assistant' if is_me else 'user'
            msgs.append({'role': role, 'content': msg})
        # ruoli alternati: se due turni adiacenti hanno lo stesso ruolo, unisci
        clean = []
        for m in msgs:
            if clean and clean[-1]['role'] == m['role']:
                clean[-1]['content'] += '\n' + m['content']
            else:
                clean.append(dict(m))
        # ogni turno assistant genera un campione con contesto
        for i, m in enumerate(clean):
            if m['role'] != 'assistant':
                continue
            start = max(0, i - MAX_CTX_TURNS)
            seg = clean[start:i + 1]
            if seg[0]['role'] != 'user':  # deve iniziare con user
                seg = seg[1:]
            if len(seg) < 2:
                continue
            resp = seg[-1]['content']
            if len(resp.strip()) < 2:
                continue
            samples.append({'id': f'{os.path.basename(path)}_{i}_{len(samples)}',
                            'messages': seg})

# dedup esatto
seen, out = set(), []
for s in samples:
    key = (s['messages'][-2]['content'], s['messages'][-1]['content'])
    if key in seen:
        continue
    seen.add(key)
    out.append(s)
samples = out

# ribilancia: tieni risposte brevi (stile reale) ma non farle dominare
random.shuffle(samples)
print('campioni totali:', len(samples))
resp_lens = [len(s['messages'][-1]['content']) for s in samples]
print('risposta media:', sum(resp_lens)//len(resp_lens), 'chars')

# cap a 14000
if len(samples) > 14000:
    samples = samples[:14000]

os.makedirs('dataset/sft', exist_ok=True)
with open('dataset/sft/sft-my.json', 'w', encoding='utf-8') as f:
    json.dump(samples, f, ensure_ascii=False, indent=1)
print('scritto dataset/sft/sft-my.json ->', len(samples), 'campioni')

# sanity: nessun token cinese residuo
bad = [s for s in samples if any(c in json.dumps(s, ensure_ascii=False) for c in CONTAM)]
print('contaminati residui:', len(bad))
multi = [s for s in samples if len(s['messages']) > 2]
print('campioni multi-turno:', len(multi), '/', len(samples))