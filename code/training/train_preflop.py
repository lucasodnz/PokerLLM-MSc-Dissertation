import os, json, torch, random
from typing import List, Dict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ===== Config =====
MODEL_ID   = "Qwen/Qwen3-14B"
TRAIN_PATH = "/home/lucasdiniz/llm_workspace/data/preflop_train_4class.json"
VAL_PATH   = "/home/lucasdiniz/llm_workspace/data/preflop_val_4class.json"
OUT_DIR    = "/home/lucasdiniz/llm_workspace/out/qwen3_14b_preflop_lora4bit_ZS_full"
SEED = 42

MAX_LEN = 1024
LR = 1e-4
EPOCHS = 2
BATCH = 2
GRAD_ACCUM = 8
WARMUP_RATIO = 0.03
LOG_STEPS = 50
SMOKE = False        # << coloque True para validar pipeline rápido; depois mude para False
SMOKE_TRAIN = 3000   # ~3k exemplos para teste rápido
SMOKE_VAL   = 300
SMOKE_LEN   = 512    # reduz contexto no smoke test


random.seed(SEED)
torch.manual_seed(SEED)

# ===== Prompt zero-shot (mesmo preâmbulo do seu teste, sem few-shots) =====
def build_zeroshot_prompt(target_instruction: str) -> str:
    prompt = (
        "You are a specialist in playing 6-handed No Limit Texas Hold'em.\n"
        "For each hand history below, decide the optimal preflop action. "
        "Respond with only one word: fold, call, raise, or check.\n"
        "Do not explain your answer.\n\n"
        "Now consider the following scenario:\n"
    )
    prompt += target_instruction.strip() + "\n"
    prompt += "Your optimal action is:"
    return prompt

def load_list(path): 
    with open(path,"r",encoding="utf-8") as f: 
        return json.load(f)

# ===== Dados =====
train_data = load_list(TRAIN_PATH)
val_data   = load_list(VAL_PATH)
print("train:", len(train_data), "val:", len(val_data))

# ===== Oversampling leve no full-train (para evitar colapso em 'fold') =====
if not SMOKE:
    from collections import defaultdict
    import random

    # fatores de multiplicação (ajuste fino possível depois)
    OS_FACTORS = {"check": 30, "raise": 2}  # 'check' é raríssimo; 'raise' tem muitos sub-rótulos na origem
    random.seed(42)

    by = defaultdict(list)
    for ex in train_data:
        y = (ex.get("output") or "").strip().lower()
        if y in {"fold","call","raise","check"}:
            by[y].append(ex)

    boosted = []
    for y, items in by.items():
        if y in OS_FACTORS:
            k = OS_FACTORS[y]
            # amostragem com reposição para multiplicar os raros
            boosted.extend(random.choices(items, k=len(items)*(k-1)))
    train_data = train_data + boosted
    random.shuffle(train_data)
    print("[OVERSAMPLE] +", len(boosted), "itens | novo train:", len(train_data))

BAL = "/home/lucasdiniz/llm_workspace/data/preflop_train_4class_SMOKE_BAL.json"
if SMOKE:
    import json as _json
    train_data = _json.load(open(BAL, encoding="utf-8"))
    EPOCHS     = 2          # 1–2 épocas é rápido nesse tamanho
    MAX_LEN    = 512
    BATCH      = 2
    GRAD_ACCUM = 8
    print(f"[SMOKE-BAL] train={len(train_data)} val={len(val_data)} MAX_LEN={MAX_LEN} BATCH={BATCH} GRAD_ACCUM={GRAD_ACCUM}")




# ===== Tokenizer/Modelo 4-bit + LoRA =====
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
)

base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=bnb_cfg,
    device_map="auto",
    dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
)

base = prepare_model_for_kbit_training(base)
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(base, lora_cfg)
model.print_trainable_parameters()

# ===== Construção dos exemplos (loss só na resposta) =====
def build_records(items: List[Dict]):
    recs = []
    for ex in items:
        instr = ex["instruction"].strip()
        ans   = ex["output"].strip().lower()

        # prompt SEM resposta
        prompt = build_zeroshot_prompt(instr)      # termina em "Your optimal action is:"
        # prompt COM resposta
        full   = prompt + " " + ans

        ids_p  = tok(prompt, truncation=True, max_length=MAX_LEN, add_special_tokens=False)["input_ids"]
        ids_f  = tok(full,   truncation=True, max_length=MAX_LEN, add_special_tokens=False)["input_ids"]
        # labels: -100 no prefixo -> loss só nos tokens da resposta
        labels = [-100]*len(ids_p) + ids_f[len(ids_p):]

        if len(labels)==0 or all(t==-100 for t in labels):
            continue

        # truncate/pad manual
        ids  = ids_f[:MAX_LEN]
        labs = labels[:MAX_LEN]
        att  = [1]*len(ids)

        if len(ids)<MAX_LEN:
            pad = MAX_LEN - len(ids)
            ids += [tok.pad_token_id]*pad
            att += [0]*pad
            labs+= [-100]*pad

        recs.append({
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(att, dtype=torch.long),
            "labels": torch.tensor(labs, dtype=torch.long),
        })
    return recs

train_ds = build_records(train_data)
val_ds   = build_records(val_data)
print("built train:", len(train_ds), "val:", len(val_ds))

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

# ===== Treinamento =====
args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=50,              # logs mais frequentes
    save_strategy="steps",         # salva por passo
    save_steps=500,                # checkpoint a cada 500 passos
    eval_strategy="no",            # sem validação automática por passo (simples e compatível)
    save_total_limit=3,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    gradient_checkpointing=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    weight_decay=0.0,
    report_to="none",
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ListDataset(train_ds),
    eval_dataset=ListDataset(val_ds),
)


from transformers.trainer_utils import get_last_checkpoint

last_ckpt = get_last_checkpoint(OUT_DIR) if os.path.isdir(OUT_DIR) else None
if last_ckpt:
    print(f"[RESUME] Encontrado checkpoint: {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)
else:
    print("[RESUME] Nenhum checkpoint encontrado — iniciando do zero.")
    trainer.train()

# ===== Salvar adapter =====
trainer.model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
print(f"✅ Adapter LoRA salvo em: {OUT_DIR}")
11
