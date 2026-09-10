# Clone vocale stile WhatsApp — fine-tuning LoRA

Fine-tuning di un LLM open-weight perché risponda come me su WhatsApp: messaggi brevi, tono casual nord-italiano.

## Metodo

1. **Dataset** (`make_dataset.py`) — dai CSV grezzi delle chat: merge dei messaggi consecutivi, split in conversazioni (gap > 10 min), finestre multi-turno a 6 turni, dedup esatto, ribilanciamento risposte brevi. ~14.000 campioni, zero token residui dal tool di export.
2. **Identità** (`add_identity.py`) — 35 Q&A (chi sono, hobby, setup, gusti) per risposte coerenti su di me.
3. **Training** (`settings.jsonc`) — base Qwen2.5-7B-Instruct, LoRA r=32 su tutti i layer, 3 epoche, 4-bit NF4 + double quantization, su singola RTX 3060 12GB. Loss 3.08 → 1.56 in 2610 step (~10h).
4. **Inferenza** (`chat_clone.py`) — CLI con memoria degli ultimi 6 scambi, maschera di attenzione esplicita, anti-sbrodolamento. `smoke_test.py` / `clone_test.py` per verifiche rapide.

## Risultati

Su 4 domande di verifica lo stile è quello giusto nel 50% dei casi al primo colpo (2/4): risposte brevi e tono corretto; i fallimenti sono allucinazioni da poco contesto, non da training.

## Privacy

Il dataset (chat reali) e i pesi dell'adapter **non** sono in questa repo: solo codice e configurazione. I nomi nei path sono placeholder.

## Requisiti

torch, transformers, peft, bitsandbytes — GPU NVIDIA con ≥12GB VRAM per il training.
