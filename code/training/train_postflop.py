import torch
import json
import random
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import Dataset
import os
import gc
import psutil
import signal
import sys

# ============== CONFIGURAÇÕES OTS PARA MEMÓRIA ==============
@dataclass
class TrainingConfig:
    # Modelo
    model_name: str = "Qwen/Qwen3-14B"
    
    # Dados
    train_path: str = "/home/lucasdiniz/llm_workspace/datasets/postflop_500k_train_set_prompt_and_label.json"
    test_path: str = "/home/lucasdiniz/llm_workspace/datasets/postflop_10k_test_set_prompt_and_label.json"
    output_dir: str = "/home/lucasdiniz/llm_workspace/ft_optimized_memory"
    
    # Treino (OTIMIZADO E SEGURO para RTX 5090 32GB)
    max_train_samples: int = 0  # 0 = usar todos (500k)
    max_eval_samples: int = 5000  # Mais amostras para avaliação robusta
    num_train_epochs: float = 3.0  # Aumentado para melhor convergência
    learning_rate: float = 2e-4  # Aumentado levemente
    per_device_train_batch_size: int = 4  # AUMENTADO (32GB suporta)
    gradient_accumulation_steps: int = 4  # REDUZIDO (batch maior)
    warmup_ratio: float = 0.05  # Aumentado para dataset grande
    
    # Modelo (CONSERVADOR após testes de segfault)
    max_length: int = 1024  # RESTAURADO para contexto completo
    lora_r: int = 16  # MANTIDO em 16 (estável, 64M params treináveis)
    lora_alpha: int = 32  # Proporcional ao r
    lora_dropout: float = 0.1  # Aumentado para regularização
    
    # Avaliação e Checkpoints (CRÍTICO para não perder progresso)
    eval_steps: int = 5000  # Avaliar menos frequentemente (mais estável)
    save_steps: int = 1000  # Salvar A CADA 1000 steps (~1 hora) - SEGURANÇA!
    
    # Semente
    seed: int = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clear_cuda_memory():
    """Limpa memória CUDA completamente"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print("🧹 Memória CUDA limpa")

def print_memory_status():
    """Monitora status de memória GPU e RAM"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {allocated:.2f}GB alocado / {reserved:.2f}GB reservado / {total:.2f}GB total")
            
            # AVISO se memória > 90%
            if reserved / total > 0.9:
                print(f"  ⚠️  AVISO: GPU {i} com {(reserved/total)*100:.1f}% de uso!")
    
    memory = psutil.virtual_memory()
    print(f"  RAM: {memory.used / 1024**3:.2f}GB / {memory.total / 1024**3:.2f}GB ({memory.percent}%)")

def setup_signal_handlers():
    """Configura handlers para sinais de crash"""
    def signal_handler(signum, frame):
        print(f"\n⚠️  Sinal recebido: {signum}")
        print("💾 Tentando salvar estado...")
        print_memory_status()
        sys.exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

def load_and_prepare_data(config: TrainingConfig):
    """Carrega dados com validação"""
    print("📂 Carregando dados...")
    
    try:
        with open(config.train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        with open(config.test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        raise
    
    # Limitar amostras
    if config.max_train_samples > 0:
        train_data = random.sample(train_data, min(config.max_train_samples, len(train_data)))
    
    if config.max_eval_samples > 0:
        test_data = random.sample(test_data, min(config.max_eval_samples, len(test_data)))
    
    print(f"📊 Dados: {len(train_data)} treino, {len(test_data)} teste")
    
    # Formatar
    def format_example(example):
        instruction = example["instruction"].strip()
        output = example["output"].strip()
        
        # Garantir formato correto
        if not instruction.endswith("Your optimal action is:"):
            instruction = instruction.rstrip()
            if not instruction.endswith(":"):
                instruction = instruction + "\nYour optimal action is:"
        
        text = instruction + " " + output
        return {"text": text, "instruction": instruction, "output": output}
    
    train_formatted = [format_example(ex) for ex in train_data]
    test_formatted = [format_example(ex) for ex in test_data]
    
    return {
        "train": Dataset.from_list(train_formatted),
        "test": Dataset.from_list(test_formatted)
    }

def tokenize_dataset(dataset, tokenizer, max_length):
    """Tokenização eficiente"""
    def tokenize_function(examples):
        # Tokenizar sem padding - data collator criará labels automaticamente
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,  # IMPORTANTE: sem padding aqui
        )
        
        # NÃO criar labels aqui - DataCollatorForLanguageModeling faz isso!
        return tokenized
    
    return dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

def load_model_for_training(config: TrainingConfig):
    """Carrega modelo otimizado para memória"""
    print("🧠 Carregando modelo com QLoRA (otimizado)...")
    
    clear_cuda_memory()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Carregar modelo com configurações de memória
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_cache=False  # IMPORTANTE para gradient checkpointing
    )
    
    # Preparar para treino
    model = prepare_model_for_kbit_training(model)
    
    # LoRA com MAIS parâmetros (32GB VRAM)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # TODOS os módulos principais
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    clear_cuda_memory()
    return model, tokenizer

def train_model(config: TrainingConfig):
    """Treinamento otimizado para memória"""
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 1. Carregar modelo
    model, tokenizer = load_model_for_training(config)
    
    # 2. Carregar dados
    datasets = load_and_prepare_data(config)
    
    # 3. Tokenizar
    print("🔢 Tokenizando...")
    train_dataset = tokenize_dataset(datasets["train"], tokenizer, config.max_length)
    eval_dataset = tokenize_dataset(datasets["test"], tokenizer, config.max_length)
    
    # 4. Configurações OTIMIZADAS PARA MEMÓRIA
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=4,  # AUMENTADO (32GB suporta)
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=True,  # REDUZ MEMÓRIA (mais lento)
        gradient_checkpointing_kwargs={"use_reentrant": True},  # MUDADO para True (mais estável após segfault)
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        warmup_ratio=config.warmup_ratio,
        logging_steps=20,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=False,  # Desabilitado para menos memória
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,  # Reduz memória
        remove_unused_columns=True,
        eval_accumulation_steps=2,  # Acumular durante avaliação
    )
    
    # 5. Data collator com padding dinâmico
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # Novo parâmetro para transformers 4.57.1
    )
    
    # 7. Treinar com monitoramento de memória
    print("🚀 Iniciando treinamento (otimizado para memória)...")
    
    # Detectar checkpoint existente para retomar
    checkpoint_path = None
    if os.path.exists(config.output_dir):
        checkpoints = [d for d in os.listdir(config.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint_path = os.path.join(config.output_dir, latest_checkpoint)
            print(f"🔄 Checkpoint detectado! Retomando de: {checkpoint_path}")
        else:
            print("📍 Nenhum checkpoint encontrado. Iniciando do zero.")
    
    print("\n📊 Memória antes do treino:")
    print_memory_status()
    
    try:
        # Treinar (retomando se houver checkpoint)
        trainer.train(resume_from_checkpoint=checkpoint_path)
        
        print("\n📊 Memória após treino:")
        print_memory_status()
        
        # Salvar modelo
        trainer.save_model(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        
        # Salvar config
        config_dict = config._dict_
        with open(os.path.join(config.output_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✅ Modelo salvo em: {config.output_dir}")
        return model, tokenizer
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ Out of memory mesmo com otimizações: {e}")
        print("💡 Tente reduzir ainda mais max_length ou batch_size")
        return None, None

class SimplePokerEvaluator:
    """Avaliador simplificado para menos memória"""
    
    def _init_(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        
        self.action_tokens = {
            "fold": tokenizer.encode(" fold", add_special_tokens=False)[0],
            "call": tokenizer.encode(" call", add_special_tokens=False)[0],
            "check": tokenizer.encode(" check", add_special_tokens=False)[0],
            "bet": tokenizer.encode(" bet", add_special_tokens=False)[0],
            "raise": tokenizer.encode(" raise", add_special_tokens=False)[0]
        }
    
    def extract_action(self, output):
        output = output.strip().lower()
        if output == "fold": return "fold"
        elif output == "call": return "call"
        elif output == "check": return "check"
        elif output.startswith("bet"): return "bet"
        elif output.startswith("raise"): return "raise"
        return "unknown"
    
    def predict(self, prompt):
        """Predição com limpeza de memória"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Log-probs do último token
        next_token_logits = logits[0, -1, :]
        log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        
        scores = {action: log_probs[token_id].item() for action, token_id in self.action_tokens.items()}
        return max(scores, key=scores.get)
    
    def evaluate_quick(self, test_data, n=500):
        """Avaliação rápida com limpeza periódica"""
        print(f"📊 Avaliação rápida ({n} amostras)...")
        
        test_data = random.sample(test_data, min(n, len(test_data)))
        correct = 0
        
        for i, item in enumerate(test_data):
            prompt = item["instruction"].strip()
            if not prompt.endswith("Your optimal action is:"):
                prompt = prompt.rstrip() + "\nYour optimal action is:"
            
            true_action = self.extract_action(item["output"])
            pred_action = self.predict(prompt)
            
            if true_action == pred_action:
                correct += 1
            
            # Limpar memória a cada 50 amostras
            if i % 50 == 0:
                clear_cuda_memory()
        
        accuracy = correct / len(test_data)
        print(f"✅ Acurácia rápida: {accuracy:.4f} ({accuracy*100:.1f}%)")
        return accuracy

def main():
    """Pipeline principal com otimizações de memória"""
    
    # Configurar handlers de sinal
    setup_signal_handlers()
    
    # Configurar variáveis de ambiente
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Verificar espaço em disco
    disk = psutil.disk_usage('/')
    print(f"💾 Espaço em disco: {disk.free / 1024**3:.1f}GB livre / {disk.total / 1024**3:.1f}GB total")
    if disk.free / 1024**3 < 50:
        print("⚠️  AVISO: Menos de 50GB livres - checkpoints podem falhar!")
    
    print("=" * 60)
    print("🎯 FINE-TUNING OTIMIZADO PARA MEMÓRIA")
    print("=" * 60)
    
    # Config - TREINAMENTO COMPLETO (RTX 5090 32GB)
    config = TrainingConfig(
        max_train_samples=0,  # 0 = TODOS os 500k
        max_eval_samples=5000,  # Avaliação robusta
        num_train_epochs=3.0,  # 3 épocas completas
        output_dir="/home/lucasdiniz/llm_workspace/ft_qwen14b_full_500k"
    )
    
    # 1. Treinar
    model, tokenizer = train_model(config)
    
    if model is None:
        print("❌ Falha no treinamento por falta de memória")
        return
    
    # 2. Avaliação rápida
    print("\n" + "=" * 60)
    print("📊 AVALIAÇÃO RÁPIDA")
    print("=" * 60)
    
    with open(config.test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    evaluator = SimplePokerEvaluator(model, tokenizer)
    accuracy = evaluator.evaluate_quick(test_data, n=500)
    
    # 3. Salvar resultados
    results = {
        "accuracy": accuracy,
        "config": config._dict_,
        "samples_evaluated": 500
    }
    
    results_path = os.path.join(config.output_dir, "quick_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # 4. Comparação
    print("\n📊 COMPARAÇÃO COM BASELINE (47.7% few-shot):")
    if accuracy > 0.477:
        print(f"  🎉 MELHORIA: +{(accuracy - 0.477)*100:.1f}%")
    else:
        print(f"  ⚠️  REGRESSÃO: -{(0.477 - accuracy)*100:.1f}%")
        print(f"  💡 Considere: Aumentar dados ou ajustar hyperparameters")

if __name__ == "__main__":
    main()