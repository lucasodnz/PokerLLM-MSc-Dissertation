#!/usr/bin/env python3
"""
Avaliação HÍBRIDA: LOGPROB (ação) + GENERATE (sizing)

MÉTODO HÍBRIDO:
1. Usar LOGPROB para classificar ação (fold/call/check/bet/raise)
2. Usar GENERATE apenas para sizing quando gold OU pred é {bet, raise}
3. Rastrear parse_fail_rate quando parser não consegue extrair valor
4. Implementar PokerBench EM: 1.0 (exato), 0.5 (ação OK, size errado), 0.0 (ação errada)

Este método combina o melhor dos dois mundos:
- Logprob para ação (92% AA esperado, como evaluate_model.py original)
- Generate apenas quando necessário para sizing (economiza tempo)
"""

import argparse, json, gc, os, random, re, sys, time
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Pipeline version
PIPELINE_VERSION = "hybrid_logprob_action_generate_sizing_v1_metricsfix_allpreds"

# Configuração DEFAULT
DEFAULT_MODEL = "Qwen/Qwen3-14B"
ADAPTER_PATH = "modelos_bons/qwen3_14b_preflop_lora4bit_ZS_full"

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="lora", 
                    choices=["lora", "fewshot"],
                    help="lora: fine-tuned model | fewshot: base model")
parser.add_argument("--dataset", type=str, default="datasets/preflop_1k_test_set_prompt_and_label.json")
parser.add_argument("--test_file", type=str, default=None,
                    help="Test dataset file (alias for --dataset)")
parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
parser.add_argument("--seed", type=int, default=111)
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument("--max_new_tokens_sizing", type=int, default=10,
                    help="Max tokens for sizing generation (8-12 recommended)")
parser.add_argument("--output_file", type=str, default=None)
parser.add_argument("--adapter", type=str, default=ADAPTER_PATH)
parser.add_argument("--model_revision", type=str, default=None,
                    help="Specific model revision/commit hash to use")
parser.add_argument("--save_predictions", type=str, default=None,
                    help="CSV file to save per-sample predictions")
parser.add_argument("--save_pairs_csv", type=str, default=None,
                    help="CSV file to save per-sizing-sample audit pairs (new in metricsfix)")
args = parser.parse_args()

# Handle test_file alias
if args.test_file:
    args.dataset = args.test_file

if args.adapter:
    ADAPTER_PATH = args.adapter

# Get absolute paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

# Auto-detect stage
if "postflop" in args.dataset.lower():
    STAGE = "postflop"
    TRAIN_FILE = os.path.join(PARENT_DIR, "datasets/postflop_500k_train_set_prompt_and_label.json")
    BASE_ACTIONS = ["fold", "call", "check", "bet", "raise"]
else:
    STAGE = "preflop"
    TRAIN_FILE = os.path.join(PARENT_DIR, "datasets/preflop_60k_train_set_prompt_and_label.json")
    BASE_ACTIONS = ["fold", "call", "check", "raise"]

MODEL_NAME = args.model
TEST_FILE = args.dataset

print("="*80)
print(f"🎯 AVALIAÇÃO HÍBRIDA: LOGPROB (ação) + GENERATE (sizing)")
print(f"🎮 Stage: {STAGE.upper()}")
print(f"📁 Mode: {args.mode.upper()}")
print(f"🤖 Model: {MODEL_NAME}")
print(f"🌱 Seed: {args.seed}")
if args.max_samples:
    print(f"📊 Samples: {args.max_samples}")
else:
    print(f"📊 Samples: ALL")
print(f"🔢 Max tokens (sizing): {args.max_new_tokens_sizing}")
print("="*80)

# Set seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def extract_base_action(output: str) -> str:
    """Extract base action from output (e.g., 'raise 12.5' → 'raise')"""
    output = output.strip().lower()
    valid_actions = ["fold", "call", "check", "bet", "raise"]
    first_word = output.split()[0] if output.split() else ""
    if first_word in valid_actions:
        return first_word
    for action in valid_actions:
        if output.startswith(action):
            return action
    return first_word

def parse_label(label: str) -> Tuple[str, Optional[float]]:
    """Parse ground truth label"""
    label = label.strip().lower()
    value = None
    if label.startswith("raise ") or label.startswith("bet "):
        parts = label.split()
        if len(parts) >= 2:
            try:
                value = float(parts[1])
            except:
                value = None
    action = extract_base_action(label)
    return action, value

def parse_action_and_value(generated_text: str) -> Tuple[str, Optional[float]]:
    """Parse generated text to extract action and value"""
    text = generated_text.strip().lower()
    
    # Clean: take only first line
    if 'scenario' in text or 'sc' in text[:10]:
        text = text.split('scenario')[0].split('sc')[0].strip()
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        text = lines[0]
    
    # Extract first word
    words = text.split()
    if not words:
        return "unknown", None
    
    first_word = words[0].strip('.,!?;:')
    
    # Map to valid actions
    action = None
    if first_word in ["fold", "folding"]:
        action = "fold"
    elif first_word in ["call", "calling"]:
        action = "call"
    elif first_word in ["check", "checking"]:
        action = "check"
    elif first_word in ["raise", "raising"]:
        action = "raise"
    elif first_word in ["bet", "betting"]:
        action = "bet"
    else:
        if "fold" in text: action = "fold"
        elif "call" in text: action = "call"
        elif "check" in text: action = "check"
        elif "raise" in text: action = "raise"
        elif "bet" in text: action = "bet"
        else: action = "unknown"
    
    # Extract value for raise/bet
    value = None
    if action in ["raise", "bet"]:
        patterns = [
            r"(?:raise|bet|bets?|raising|betting)\s*:?\s*(?:to\s+)?(\d+(?:\.\d+)?)",
            r"(?:raise|bet)\s+(\d+(?:\.\d+)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    break
                except:
                    continue
        # Fallback: any number
        if value is None:
            num_pattern = r"(\d+(?:\.\d+)?)"
            matches = re.findall(num_pattern, text)
            if matches:
                try:
                    value = float(matches[0])
                except:
                    pass
    
    return action, value

def parse_sizing_value(generated_text: str) -> Optional[float]:
    """
    Parse sizing generation to extract ONLY the numeric value.
    Used when we already know the action from logprob.
    
    Returns:
        float value if found, None otherwise
    """
    text = generated_text.strip()
    
    # Extract first number (int or float) from text
    num_pattern = r"(\d+(?:\.\d+)?)"
    match = re.search(num_pattern, text)
    
    if match:
        try:
            return float(match.group(1))
        except:
            return None
    
    return None

def extract_position_from_instruction(instruction: str) -> str:
    """
    Extract position from instruction text using strict regex.
    Looks for "your position is X" pattern.
    
    Returns:
        One of: UTG, HJ, CO, BTN, SB, BB, or "unknown"
    """
    # Strict regex: "your position is (UTG|HJ|CO|BTN|SB|BB)"
    pattern = r"your position is\s+(UTG|HJ|CO|BTN|SB|BB)"
    match = re.search(pattern, instruction, re.IGNORECASE)
    
    if match:
        return match.group(1).upper()
    
    return "unknown"

def extract_street_from_instruction(instruction: str) -> str:
    """
    Extract street from instruction by finding last occurrence of street indicators.
    
    Strategy:
    1. Cut text before "Now it is your turn"
    2. Find LAST occurrence among: "The river comes", "The turn comes", "The flop comes"
    3. Priority order: river > turn > flop (to get latest street)
    
    Returns:
        One of: "preflop", "flop", "turn", "river", or "unknown"
    """
    # Cut text before "Now it is your turn"
    cut_marker = "Now it is your turn"
    if cut_marker in instruction:
        text = instruction.split(cut_marker)[0]
    else:
        text = instruction
    
    # Find last occurrence of each street indicator
    street_patterns = {
        "river": r"[Tt]he river comes",
        "turn": r"[Tt]he turn comes",
        "flop": r"[Tt]he flop comes",
    }
    
    last_positions = {}
    for street, pattern in street_patterns.items():
        matches = list(re.finditer(pattern, text))
        if matches:
            last_positions[street] = matches[-1].start()
    
    # If any found, return the one with latest position (rightmost)
    if last_positions:
        latest_street = max(last_positions, key=last_positions.get)
        return latest_street
    
    # Check if preflop (before any flop/turn/river)
    if "before the flop" in text.lower() or "preflop" in text.lower():
        return "preflop"
    
    return "unknown"

def infer_legal_actions(instruction: str, base_actions: List[str]) -> List[str]:
    """Infer legal actions from instruction (postflop only)"""
    inst_lower = instruction.lower()
    
    # If explicit legal actions, use them
    if "legal actions:" in inst_lower:
        legal_str = inst_lower.split("legal actions:")[-1].split('\n')[0].strip()
        legal_actions = []
        for action in base_actions:
            if action in legal_str:
                legal_actions.append(action)
        if legal_actions:
            return legal_actions
    
    # Heuristic: if pot is 0 and no betting, check/bet are likely
    # For simplicity, return all actions if not specified
    return base_actions

def get_best_action_logprob(prompt: str, model, tokenizer, actions: List[str], legal_actions: List[str] = None) -> Tuple[str, Dict[str, float]]:
    """Classify action using log-probabilities of next token"""
    if legal_actions is None:
        legal_actions = actions
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
    
    # Token IDs for each legal action (with leading space: " fold", " call", etc.)
    action_tokens = {}
    for action in legal_actions:
        token_with_space = f" {action}"
        token_ids = tokenizer.encode(token_with_space, add_special_tokens=False)
        if len(token_ids) > 0:
            action_tokens[action] = token_ids[0]
        else:
            token_ids = tokenizer.encode(action, add_special_tokens=False)
            if len(token_ids) > 0:
                action_tokens[action] = token_ids[0]
    
    scores = {}
    for action, token_id in action_tokens.items():
        scores[action] = log_probs[token_id].item()
    
    best_action = max(scores, key=scores.get)
    return best_action, scores

def build_prompt_for_logprob(instruction: str, stage: str) -> str:
    """Build prompt for logprob method (ends with 'Your optimal action is:')"""
    if stage == "postflop":
        return (
            f"{instruction.strip()}\n"
            "Your optimal action is:"
        )
    else:
        return (
            f"{instruction.strip()}\n"
            "Your optimal action is:"
        )

def build_prompt_for_sizing(instruction: str, action: str, stage: str) -> str:
    """Build prompt for sizing generation (when action is bet/raise)"""
    return (
        f"{instruction.strip()}\n"
        f"Your optimal action is: {action}\n"
        f"Return only the size in chips as a number (e.g., 10 or 12.5). No words.\n"
        f"Size:"
    )

def sample_fewshot_examples(train_data: List[dict], seed: int, stage: str) -> Tuple[List[dict], List[int]]:
    """Sample K examples (1 from each action)"""
    rng = random.Random(seed)
    actions = BASE_ACTIONS
    examples = []
    indices = []
    
    for action in actions:
        candidates = [(i, ex) for i, ex in enumerate(train_data) 
                     if ex["output"].strip().lower().startswith(action)]
        if candidates:
            idx, ex = rng.choice(candidates)
            examples.append(ex)
            indices.append(idx)
        else:
            print(f"⚠️  Warning: No examples for '{action}'")
    
    return examples, indices

def build_prompt_fewshot_logprob(instruction: str, few_shot_examples: List[dict]) -> str:
    """Build few-shot prompt for logprob (ends with 'Your optimal action is:')"""
    prompt = ""
    for ex in few_shot_examples:
        output_action = extract_base_action(ex["output"])
        prompt += ex["instruction"].strip() + "\n"
        prompt += f"Your optimal action is: {output_action}\n\n"
    
    prompt += instruction.strip() + "\n"
    prompt += "Your optimal action is:"
    return prompt

def build_prompt_fewshot_sizing(instruction: str, action: str, few_shot_examples: List[dict]) -> str:
    """Build few-shot prompt for sizing generation"""
    # For sizing, use examples that have same action
    relevant_examples = [ex for ex in few_shot_examples if ex["output"].strip().lower().startswith(action)]
    
    prompt = ""
    for ex in relevant_examples[:2]:  # Max 2 examples
        ex_output = ex["output"].strip().lower()
        # Extract just the numeric value from example
        parts = ex_output.split()
        if len(parts) >= 2:
            ex_value = parts[1]  # e.g., "raise 12.5" -> "12.5"
        else:
            ex_value = "10"  # fallback
        
        prompt += ex["instruction"].strip() + "\n"
        prompt += f"Your optimal action is: {action}\n"
        prompt += f"Size: {ex_value}\n\n"
    
    prompt += instruction.strip() + "\n"
    prompt += f"Your optimal action is: {action}\n"
    prompt += f"Return only the size in chips as a number (e.g., 10 or 12.5). No words.\n"
    prompt += f"Size:"
    return prompt

print("\n🔧 Carregando modelo...")
clear_cuda()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True,
    revision=args.model_revision
)

if args.mode == "lora":
    print(f"📦 Carregando LoRA: {ADAPTER_PATH}")
    if args.model_revision:
        print(f"   Usando revision: {args.model_revision}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        revision=args.model_revision,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
else:
    print(f"📦 Carregando base model: {MODEL_NAME}")
    if args.model_revision:
        print(f"   Usando revision: {args.model_revision}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        revision=args.model_revision,
    )

model.eval()
print("✅ Modelo carregado!")

print(f"\n📂 Carregando dataset: {TEST_FILE}")
with open(TEST_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"✅ {len(data)} exemplos")

# Few-shot
fewshot_examples = []
fewshot_indices = []
if args.mode == "fewshot":
    print(f"\n📚 Carregando training data: {TRAIN_FILE}")
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    print(f"✅ {len(train_data)} exemplos de treino")
    
    k = len(BASE_ACTIONS)
    print(f"🎲 Amostrando K={k} exemplos (seed={args.seed})")
    fewshot_examples, fewshot_indices = sample_fewshot_examples(train_data, args.seed, STAGE)
    print(f"✅ Exemplos selecionados:")
    for i, (ex, idx) in enumerate(zip(fewshot_examples, fewshot_indices), 1):
        action = ex["output"].strip().lower()
        print(f"   [{i}] idx={idx}, action={action}")

samples_to_process = data[:args.max_samples] if args.max_samples else data
print(f"\n🚀 Processando {len(samples_to_process)} samples...")

# Coletar predições
y_true_action = []
y_pred_action = []
y_true_exact = []
y_pred_exact = []

# Metadados extraídos do texto da instruction
sample_positions = []  # Position de cada sample (UTG, HJ, CO, BTN, SB, BB, unknown)
sample_streets = []    # Street de cada sample (preflop, flop, turn, river, unknown)

sizing_comparisons = []  # Para bet/raise: {true_value, pred_value, action_match, exact_match, parse_fail}
sizing_generation_examples = []  # Debug: primeiros 20 exemplos

# Contadores
n_logprob_calls = 0
n_generate_calls = 0
n_parse_fails = 0  # Quando parser não consegue extrair valor

t0 = time.time()

for i, item in enumerate(tqdm(samples_to_process, desc="Evaluating")):
    instruction = item["instruction"]
    gold_label = item["output"].strip().lower()
    
    # Parse ground truth
    action_true, value_true = parse_label(gold_label)
    
    # Extract metadata from instruction text
    position = extract_position_from_instruction(instruction)
    street = extract_street_from_instruction(instruction)
    sample_positions.append(position)
    sample_streets.append(street)
    
    # ============================================================================
    # STEP 1: PREDICT ACTION via LOGPROB
    # ============================================================================
    
    # Infer legal actions (postflop only)
    if STAGE == "postflop":
        legal_actions = infer_legal_actions(instruction, BASE_ACTIONS)
    else:
        legal_actions = BASE_ACTIONS
    
    # Build prompt for logprob
    if args.mode == "lora":
        prompt_logprob = build_prompt_for_logprob(instruction, STAGE)
    else:
        prompt_logprob = build_prompt_fewshot_logprob(instruction, fewshot_examples)
    
    # Predict action
    action_pred, logprob_scores = get_best_action_logprob(prompt_logprob, model, tokenizer, BASE_ACTIONS, legal_actions)
    n_logprob_calls += 1
    
    # ============================================================================
    # STEP 2: PREDICT SIZING via GENERATE (only if bet/raise)
    # ============================================================================
    
    value_pred = None
    generated_text = None
    parse_fail = False
    
    # Only generate if gold OR pred is bet/raise
    needs_sizing = (action_true in ["bet", "raise"]) or (action_pred in ["bet", "raise"])
    
    if needs_sizing:
        # Build prompt for sizing
        if args.mode == "lora":
            prompt_sizing = build_prompt_for_sizing(instruction, action_pred, STAGE)
        else:
            prompt_sizing = build_prompt_fewshot_sizing(instruction, action_pred, fewshot_examples)
        
        # Generate
        inputs = tokenizer(prompt_sizing, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens_sizing,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[tokenizer.eos_token_id],
            )
        
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        n_generate_calls += 1
        
        # Parse value (ONLY number, we already have action from logprob)
        value_pred = parse_sizing_value(generated_text)
        
        # Check parse fail
        if action_pred in ["bet", "raise"] and value_pred is None:
            parse_fail = True
            n_parse_fails += 1
    
    # ============================================================================
    # STEP 3: STORE RESULTS
    # ============================================================================
    
    y_true_action.append(action_true)
    y_pred_action.append(action_pred)
    
    # Reconstruct exact labels
    if action_true in ["bet", "raise"] and value_true is not None:
        y_true_exact.append(f"{action_true} {value_true}")
    else:
        y_true_exact.append(action_true)
    
    if action_pred in ["bet", "raise"] and value_pred is not None:
        y_pred_exact.append(f"{action_pred} {value_pred}")
    else:
        y_pred_exact.append(action_pred)
    
    # Store sizing comparison (for bet/raise)
    if action_true in ["bet", "raise"]:
        sizing_comparisons.append({
            "sample_idx": i,  # Add index for later lookup
            "street": street,
            "position": position,
            "true_action": action_true,
            "true_value": value_true,
            "pred_action": action_pred,
            "pred_value": value_pred,
            "action_match": action_pred == action_true,
            "exact_match": (action_pred == action_true and value_true == value_pred),
            "parse_fail": parse_fail,
            "generated_text": generated_text,
            "gold_label": gold_label,
        })
        
        # Save examples
        if len(sizing_generation_examples) < 20:
            sizing_generation_examples.append({
                "gold": f"{action_true} {value_true}",
                "generated_text": generated_text.strip() if generated_text else None,
                "pred_action": action_pred,
                "pred_value": value_pred,
                "parse_fail": parse_fail,
            })
    
    # Progress
    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        avg_time = elapsed / (i + 1)
        eta = avg_time * (len(samples_to_process) - i - 1)
        print(f"Progress: {i+1}/{len(samples_to_process)} | Avg: {avg_time:.3f}s/sample | ETA: {eta/60:.1f}min", flush=True)
    
    if (i + 1) % 500 == 0:
        clear_cuda()

total_runtime = time.time() - t0
avg_time_per_sample = total_runtime / len(samples_to_process)

print(f"\n⏱️  Runtime: {total_runtime:.2f}s ({total_runtime/60:.2f}min)")
print(f"⏱️  Avg per sample: {avg_time_per_sample:.3f}s")
print(f"📊 Logprob calls: {n_logprob_calls}")
print(f"📊 Generate calls: {n_generate_calls} ({n_generate_calls/n_logprob_calls*100:.1f}%)")
print(f"📊 Parse fails: {n_parse_fails} ({n_parse_fails/max(1,n_generate_calls)*100:.1f}% of generates)")

# ============================================================================
# MÉTRICAS
# ============================================================================

print("\n📊 Calculando métricas...")

# Action Accuracy (AA)
acc_action = accuracy_score(y_true_action, y_pred_action)

# Exact Match (EM) - string-based
acc_exact = accuracy_score(y_true_exact, y_pred_exact)

# AM (Action-Match) metric with ±1bb tolerance
# AM_i = 0 if action wrong
# AM_i = 1 if action correct and NOT bet/raise
# AM_i = 1 if action correct and IS bet/raise and size within ±1bb
# AM_i = 0.5 if action correct and IS bet/raise and (size outside ±1bb OR missing/parse fail)
pb_em = 0  # This is now AM metric
pb_aa = 0

# Build lookup for sizing values
sizing_values_lookup = {}
for comp in sizing_comparisons:
    idx = comp["sample_idx"]
    sizing_values_lookup[idx] = {
        "true_value": comp["true_value"],
        "pred_value": comp["pred_value"],
        "pred_action": comp["pred_action"]
    }

for i in range(len(y_true_exact)):
    true_action = y_true_action[i]
    pred_action = y_pred_action[i]
    
    # Action Accuracy (always counts correct actions)
    if pred_action == true_action:
        pb_aa += 1.0
    
    # AM (Approximate Match) metric calculation with ±1bb tolerance
    if pred_action != true_action:
        # Action wrong: AM_i = 0
        pb_em += 0.0
    elif true_action not in ["bet", "raise"]:
        # Action correct and NOT bet/raise: AM_i = 1
        pb_em += 1.0
    else:
        # Action correct and IS bet/raise: check sizing tolerance
        true_value = None
        pred_value = None
        
        # Get values from sizing_comparisons
        if i in sizing_values_lookup:
            true_value = sizing_values_lookup[i]["true_value"]
            pred_value = sizing_values_lookup[i]["pred_value"]
        
        if pred_value is None or true_value is None:
            # Size missing or parse failure: AM_i = 0.5
            pb_em += 0.5
        else:
            # Check tolerance
            size_error = abs(pred_value - true_value)
            if size_error <= 1.0:  # Within ±1bb
                pb_em += 1.0
            else:  # Outside ±1bb
                pb_em += 0.5

pb_em_acc = pb_em / len(y_true_exact)
pb_aa_acc = pb_aa / len(y_true_exact)
gap = pb_aa_acc - pb_em_acc

# Sizing metrics (bet/raise)
sizing_action_correct = sum(1 for s in sizing_comparisons if s["action_match"])
sizing_exact_correct = sum(1 for s in sizing_comparisons if s["exact_match"])
sizing_parse_fails = sum(1 for s in sizing_comparisons if s["parse_fail"])

if sizing_comparisons:
    sizing_action_acc = sizing_action_correct / len(sizing_comparisons)
    sizing_exact_acc = sizing_exact_correct / len(sizing_comparisons)
    parse_fail_rate = sizing_parse_fails / len(sizing_comparisons)
else:
    sizing_action_acc = 0.0
    sizing_exact_acc = 0.0
    parse_fail_rate = 0.0

# MAE (only when both have values and action matches)
sizing_errors = []
for s in sizing_comparisons:
    if s["true_value"] is not None and s["pred_value"] is not None and s["action_match"]:
        error = abs(s["true_value"] - s["pred_value"])
        sizing_errors.append(error)

sizing_mae = np.mean(sizing_errors) if sizing_errors else None
sizing_median_ae = np.median(sizing_errors) if sizing_errors else None

# Tolerances (sizing only - when both have values and action matches)
sizing_within_0_5bb = 0
sizing_within_1bb = 0
sizing_within_2bb = 0

for s in sizing_comparisons:
    if s["true_value"] is not None and s["pred_value"] is not None and s["action_match"]:
        error = abs(s["true_value"] - s["pred_value"])
        if error <= 0.5:
            sizing_within_0_5bb += 1
        if error <= 1.0:
            sizing_within_1bb += 1
        if error <= 2.0:
            sizing_within_2bb += 1

n_sizing_with_values = len(sizing_errors)
if n_sizing_with_values > 0:
    sizing_within_0_5bb_prop = sizing_within_0_5bb / n_sizing_with_values
    sizing_within_1bb_prop = sizing_within_1bb / n_sizing_with_values
    sizing_within_2bb_prop = sizing_within_2bb / n_sizing_with_values
else:
    sizing_within_0_5bb_prop = 0.0
    sizing_within_1bb_prop = 0.0
    sizing_within_2bb_prop = 0.0

# Overall tolerances (all samples)
overall_within_0_5bb = 0
overall_within_1bb = 0
overall_within_2bb = 0

for i in range(len(y_true_action)):
    true_action = y_true_action[i]
    pred_action = y_pred_action[i]
    
    if true_action not in ["bet", "raise"]:
        if true_action == pred_action:
            overall_within_0_5bb += 1
            overall_within_1bb += 1
            overall_within_2bb += 1
    else:
        if pred_action == true_action:
            sizing_idx = sum(1 for j in range(i) if y_true_action[j] in ["bet", "raise"])
            if sizing_idx < len(sizing_comparisons):
                s = sizing_comparisons[sizing_idx]
                if s["true_value"] is not None and s["pred_value"] is not None:
                    error = abs(s["true_value"] - s["pred_value"])
                    if error <= 0.5:
                        overall_within_0_5bb += 1
                    if error <= 1.0:
                        overall_within_1bb += 1
                    if error <= 2.0:
                        overall_within_2bb += 1

overall_within_0_5bb_prop = overall_within_0_5bb / len(y_true_action)
overall_within_1bb_prop = overall_within_1bb / len(y_true_action)
overall_within_2bb_prop = overall_within_2bb / len(y_true_action)

# Classification report
report_action = classification_report(
    y_true_action, y_pred_action, 
    labels=BASE_ACTIONS, zero_division=0, output_dict=True
)

cm = confusion_matrix(y_true_action, y_pred_action, labels=BASE_ACTIONS)

# ============================================================================
# BREAKDOWNS: Extract position/street from instruction text
# ============================================================================

print("\n📊 Calculando breakdowns por posição e street...")

# FIX: Save ALL global sizing metrics before breakdown loops overwrite the
# shared variable names (sizing_errors, sizing_mae, sizing_within_1bb_prop).
# The original code already saved within_0_5bb and within_2bb correctly,
# but sizing_mae, sizing_mae_n (len(sizing_errors)), and sizing_within_1bb_prop
# were NOT saved and got overwritten by the last breakdown-group iteration (flop).
saved_sizing_mae = sizing_mae
saved_sizing_median_ae = sizing_median_ae
saved_sizing_mae_n = len(sizing_errors)
saved_sizing_within_0_5bb_prop = sizing_within_0_5bb_prop
saved_sizing_within_1bb_prop = sizing_within_1bb_prop
saved_sizing_within_2bb_prop = sizing_within_2bb_prop

breakdowns = {}

# Count unknowns
n_unknown_positions = sample_positions.count("unknown")
n_unknown_streets = sample_streets.count("unknown")
breakdowns["unknown_position_count"] = n_unknown_positions
breakdowns["unknown_position_rate"] = n_unknown_positions / len(sample_positions)
breakdowns["unknown_street_count"] = n_unknown_streets
breakdowns["unknown_street_rate"] = n_unknown_streets / len(sample_streets)

# Group by position
position_groups = defaultdict(lambda: {
    "true_action": [], "pred_action": [], 
    "true_exact": [], "pred_exact": [],
    "sizing_comparisons": []
})

for idx in range(len(sample_positions)):
    pos = sample_positions[idx]
    position_groups[pos]["true_action"].append(y_true_action[idx])
    position_groups[pos]["pred_action"].append(y_pred_action[idx])
    position_groups[pos]["true_exact"].append(y_true_exact[idx])
    position_groups[pos]["pred_exact"].append(y_pred_exact[idx])
    
    # Add sizing comparison if bet/raise
    if y_true_action[idx] in ["bet", "raise"]:
        sizing_idx = sum(1 for j in range(idx) if y_true_action[j] in ["bet", "raise"])
        if sizing_idx < len(sizing_comparisons):
            position_groups[pos]["sizing_comparisons"].append(sizing_comparisons[sizing_idx])

# Calculate metrics by position
breakdowns["by_position"] = {}
for pos, data in position_groups.items():
    if len(data["true_action"]) == 0:
        continue
    
    # AA
    aa = accuracy_score(data["true_action"], data["pred_action"])
    
    # PokerBench EM
    pb_em_pos = 0
    for true_label, pred_label, true_action, pred_action in zip(
        data["true_exact"], data["pred_exact"], data["true_action"], data["pred_action"]
    ):
        if true_label == pred_label:
            pb_em_pos += 1.0
        elif true_action == pred_action:
            pb_em_pos += 0.5
    pb_em_acc_pos = pb_em_pos / len(data["true_action"])
    
    # Sizing metrics (bet/raise)
    sizing_data = data["sizing_comparisons"]
    sizing_errors = []
    sizing_within_1bb = 0
    
    for s in sizing_data:
        if s["true_value"] is not None and s["pred_value"] is not None and s["action_match"]:
            error = abs(s["true_value"] - s["pred_value"])
            sizing_errors.append(error)
            if error <= 1.0:
                sizing_within_1bb += 1
    
    sizing_mae = np.mean(sizing_errors) if sizing_errors else None
    sizing_within_1bb_prop = sizing_within_1bb / len(sizing_errors) if sizing_errors else None
    
    breakdowns["by_position"][pos] = {
        "n_samples": len(data["true_action"]),
        "AA": aa,
        "AM": pb_em_acc_pos,
        "sizing_n": len(sizing_errors),
        "sizing_MAE": sizing_mae,
        "sizing_within_1bb": sizing_within_1bb_prop,
    }

# Group by street
street_groups = defaultdict(lambda: {
    "true_action": [], "pred_action": [], 
    "true_exact": [], "pred_exact": [],
    "sizing_comparisons": []
})

for idx in range(len(sample_streets)):
    street = sample_streets[idx]
    street_groups[street]["true_action"].append(y_true_action[idx])
    street_groups[street]["pred_action"].append(y_pred_action[idx])
    street_groups[street]["true_exact"].append(y_true_exact[idx])
    street_groups[street]["pred_exact"].append(y_pred_exact[idx])
    
    # Add sizing comparison if bet/raise
    if y_true_action[idx] in ["bet", "raise"]:
        sizing_idx = sum(1 for j in range(idx) if y_true_action[j] in ["bet", "raise"])
        if sizing_idx < len(sizing_comparisons):
            street_groups[street]["sizing_comparisons"].append(sizing_comparisons[sizing_idx])

# Calculate metrics by street
breakdowns["by_street"] = {}
for street, data in street_groups.items():
    if len(data["true_action"]) == 0:
        continue
    
    # AA
    aa = accuracy_score(data["true_action"], data["pred_action"])
    
    # PokerBench EM
    pb_em_street = 0
    for true_label, pred_label, true_action, pred_action in zip(
        data["true_exact"], data["pred_exact"], data["true_action"], data["pred_action"]
    ):
        if true_label == pred_label:
            pb_em_street += 1.0
        elif true_action == pred_action:
            pb_em_street += 0.5
    pb_em_acc_street = pb_em_street / len(data["true_action"])
    
    # Sizing metrics (bet/raise)
    sizing_data = data["sizing_comparisons"]
    sizing_errors = []
    sizing_within_1bb = 0
    
    for s in sizing_data:
        if s["true_value"] is not None and s["pred_value"] is not None and s["action_match"]:
            error = abs(s["true_value"] - s["pred_value"])
            sizing_errors.append(error)
            if error <= 1.0:
                sizing_within_1bb += 1
    
    sizing_mae = np.mean(sizing_errors) if sizing_errors else None
    sizing_within_1bb_prop = sizing_within_1bb / len(sizing_errors) if sizing_errors else None
    
    breakdowns["by_street"][street] = {
        "n_samples": len(data["true_action"]),
        "AA": aa,
        "AM": pb_em_acc_street,
        "sizing_n": len(sizing_errors),
        "sizing_MAE": sizing_mae,
        "sizing_within_1bb": sizing_within_1bb_prop,
    }

# ============================================================================
# PRINT RESULTS
# ============================================================================

print("\n" + "="*80)
print("📊 RESULTADOS")
print("="*80)
print(f"\n🎯 Action Accuracy (AA): {acc_action:.4f} ({acc_action*100:.2f}%)")
print(f"🎯 Exact Match (EM string): {acc_exact:.4f} ({acc_exact*100:.2f}%)")
print(f"\n📊 Metrics:")
print(f"   AA (Action Accuracy): {pb_aa_acc:.4f} ({pb_aa_acc*100:.2f}%)")
print(f"   EM (Exact Match): {acc_exact:.4f} ({acc_exact*100:.2f}%)")
print(f"   AM (Approximate Match ±1bb): {pb_em_acc:.4f} ({pb_em_acc*100:.2f}%)")
print(f"   Gap (AA-EM): {(pb_aa_acc - acc_exact):.4f} ({(pb_aa_acc - acc_exact)*100:.2f}%)")
print(f"   Gap (AA-AM): {gap:.4f} ({gap*100:.2f}%)")

print(f"\n🎲 Sizing Metrics (Bet/Raise, n={len(sizing_comparisons)}):")
print(f"   Action Accuracy: {sizing_action_acc:.4f} ({sizing_action_acc*100:.2f}%) - {sizing_action_correct}/{len(sizing_comparisons)}")
print(f"   Exact Match: {sizing_exact_acc:.4f} ({sizing_exact_acc*100:.2f}%) - {sizing_exact_correct}/{len(sizing_comparisons)}")
print(f"   Parse Fail Rate: {parse_fail_rate:.4f} ({parse_fail_rate*100:.2f}%) - {sizing_parse_fails}/{len(sizing_comparisons)}")
if sizing_mae is not None:
    print(f"   MAE: {sizing_mae:.2f} BB (n={len(sizing_errors)})")
if sizing_median_ae is not None:
    print(f"   Median AE: {sizing_median_ae:.2f} BB")

print(f"\n📏 Sizing Tolerance (n={n_sizing_with_values}):")
if saved_sizing_within_0_5bb_prop is not None:
    print(f"   Within ±0.5 BB: {saved_sizing_within_0_5bb_prop:.4f} ({saved_sizing_within_0_5bb_prop*100:.2f}%)")
else:
    print(f"   Within ±0.5 BB: N/A (no samples with values)")
if saved_sizing_within_1bb_prop is not None:
    print(f"   Within ±1.0 BB: {saved_sizing_within_1bb_prop:.4f} ({saved_sizing_within_1bb_prop*100:.2f}%)")
else:
    print(f"   Within ±1.0 BB: N/A (no samples with values)")
if saved_sizing_within_2bb_prop is not None:
    print(f"   Within ±2.0 BB: {saved_sizing_within_2bb_prop:.4f} ({saved_sizing_within_2bb_prop*100:.2f}%)")
else:
    print(f"   Within ±2.0 BB: N/A (no samples with values)")

print(f"\n📏 Overall Tolerance (All samples):")
print(f"   Within ±0.5 BB: {overall_within_0_5bb_prop:.4f} ({overall_within_0_5bb_prop*100:.2f}%)")
print(f"   Within ±1.0 BB: {overall_within_1bb_prop:.4f} ({overall_within_1bb_prop*100:.2f}%)")
print(f"   Within ±2.0 BB: {overall_within_2bb_prop:.4f} ({overall_within_2bb_prop*100:.2f}%)")

print(f"\n📋 Classification Report (Action):")
print(classification_report(y_true_action, y_pred_action, labels=BASE_ACTIONS, zero_division=0))

# Print sizing examples
if sizing_generation_examples:
    print("\n" + "="*80)
    print("🔍 SIZING GENERATION EXAMPLES (primeiros 10)")
    print("="*80)
    for idx, ex in enumerate(sizing_generation_examples[:10], 1):
        gold_parts = ex['gold'].split()
        if len(gold_parts) >= 2:
            gold_action = gold_parts[0]
            gold_value_str = gold_parts[1]
            try:
                gold_value = float(gold_value_str)
            except (ValueError, TypeError):
                gold_value = None
        else:
            gold_action = gold_parts[0] if gold_parts else "unknown"
            gold_value_str = "N/A"
            gold_value = None
        
        print(f"\n[{idx}] Gold: {ex['gold']} (action={gold_action}, value={gold_value_str})")
        if ex['generated_text']:
            print(f"    Generated text: '{ex['generated_text']}'")
        print(f"    Pred: action={ex['pred_action']} (logprob), value={ex['pred_value']} (parsed)")
        if ex['parse_fail']:
            print(f"    ⚠️  PARSE FAIL (could not extract numeric value)")
        elif ex['pred_action'] == gold_action and ex['pred_value'] is not None and gold_value is not None:
            if gold_value == ex['pred_value']:
                print(f"    ✅ EXACT MATCH")
            else:
                diff = abs(gold_value - ex['pred_value'])
                print(f"    ⚠️  Diff: {diff:.2f} BB")

# Error samples
error_samples = []
for i in range(min(20, len(y_true_exact))):
    if y_true_action[i] != y_pred_action[i]:
        error_samples.append({
            "true": y_true_exact[i],
            "pred": y_pred_exact[i],
            "true_action": y_true_action[i],
            "pred_action": y_pred_action[i],
        })

# ============================================================================
# PRINT BREAKDOWNS
# ============================================================================

if breakdowns:
    print("\n" + "="*80)
    print("📊 BREAKDOWNS POR POSIÇÃO E STREET")
    print("="*80)
    
    # Unknown counts
    print(f"\n📍 Unknown Position: {breakdowns['unknown_position_count']}/{len(sample_positions)} ({breakdowns['unknown_position_rate']*100:.2f}%)")
    print(f"📍 Unknown Street: {breakdowns['unknown_street_count']}/{len(sample_streets)} ({breakdowns['unknown_street_rate']*100:.2f}%)")
    
    # Breakdowns by position
    if "by_position" in breakdowns and breakdowns["by_position"]:
        print("\n🎯 BY POSITION:")
        for pos in ["UTG", "HJ", "CO", "BTN", "SB", "BB", "unknown"]:
            if pos in breakdowns["by_position"]:
                data = breakdowns["by_position"][pos]
                print(f"\n   {pos}:")
                print(f"      Samples: {data['n_samples']}")
                print(f"      AA: {data['AA']:.4f} ({data['AA']*100:.2f}%)")
                print(f"      AM: {data['AM']:.4f} ({data['AM']*100:.2f}%)")
                if data['sizing_n'] > 0:
                    print(f"      Sizing (n={data['sizing_n']}): MAE={data['sizing_MAE']:.2f}BB, Within±1BB={data['sizing_within_1bb']*100:.1f}%")
    
    # Breakdowns by street
    if "by_street" in breakdowns and breakdowns["by_street"]:
        print("\n🃏 BY STREET:")
        for street in ["preflop", "flop", "turn", "river", "unknown"]:
            if street in breakdowns["by_street"]:
                data = breakdowns["by_street"][street]
                print(f"\n   {street.upper()}:")
                print(f"      Samples: {data['n_samples']}")
                print(f"      AA: {data['AA']:.4f} ({data['AA']*100:.2f}%)")
                print(f"      AM: {data['AM']:.4f} ({data['AM']*100:.2f}%)")
                if data['sizing_n'] > 0:
                    print(f"      Sizing (n={data['sizing_n']}): MAE={data['sizing_MAE']:.2f}BB, Within±1BB={data['sizing_within_1bb']*100:.1f}%")

# ============================================================================
# SAVE RESULTS
# ============================================================================

result = {
    "pipeline_version": PIPELINE_VERSION,
    "model": ADAPTER_PATH if args.mode == "lora" else MODEL_NAME,
    "mode": args.mode,
    "stage": STAGE,
    "dataset": TEST_FILE,
    "seed": args.seed,
    "total_samples": len(samples_to_process),
    "runtime_seconds": total_runtime,
    "avg_seconds_per_sample": avg_time_per_sample,
    "n_logprob_calls": n_logprob_calls,
    "n_generate_calls": n_generate_calls,
    "parse_fail_count": n_parse_fails,
    "parse_fail_rate": parse_fail_rate,
    
    # Main metrics
    "accuracy_action": acc_action,
    "accuracy_exact_match": acc_exact,
    "am_score": pb_em_acc,  # AM (Approximate Match) with ±1bb tolerance
    "gap_aa_em": pb_aa_acc - acc_exact,
    "gap_aa_am": gap,
    
    # Sizing metrics
    "sizing_samples": len(sizing_comparisons),
    "sizing_action_accuracy": sizing_action_acc,
    "sizing_exact_match_accuracy": sizing_exact_acc,
    # FIX: use saved_* variables; the plain sizing_mae / sizing_errors / sizing_within_1bb_prop
    # are overwritten by the last breakdown-group iteration (street=flop) before this point.
    "sizing_mae": saved_sizing_mae,
    "sizing_median_ae": saved_sizing_median_ae,
    "sizing_mae_n": saved_sizing_mae_n,
    "sizing_parse_fail_rate": parse_fail_rate,
    
    # Tolerance metrics - sizing only
    "sizing_within_0_5bb": saved_sizing_within_0_5bb_prop,
    "sizing_within_1bb": saved_sizing_within_1bb_prop,
    "sizing_within_2bb": saved_sizing_within_2bb_prop,
    
    # Tolerance metrics - overall
    "overall_within_0_5bb": overall_within_0_5bb_prop,
    "overall_within_1bb": overall_within_1bb_prop,
    "overall_within_2bb": overall_within_2bb_prop,
    
    # Classification
    "confusion_matrix_action": cm.tolist(),
    "labels_action": BASE_ACTIONS,
    "classification_report_action": report_action,
    
    # Breakdowns
    "breakdowns": breakdowns,
    
    # Samples
    "error_samples": error_samples[:50],
    "sizing_examples": sizing_generation_examples[:10],
}

# Few-shot info
if args.mode == "fewshot":
    result["fewshot_k"] = len(fewshot_examples)
    result["fewshot_train_indices"] = fewshot_indices
    result["fewshot_examples"] = [
        {
            "train_idx": idx,
            "instruction": ex["instruction"][:100] + "...",
            "output": ex["output"]
        }
        for idx, ex in zip(fewshot_indices, fewshot_examples)
    ]

# ============================================================================
# SAVE PAIRS CSV (metricsfix: audit CSV for sizing pairs)
# ============================================================================
if args.save_pairs_csv:
    import csv as _csv_mod
    print(f"\n💾 Salvando pairs CSV de auditoria em: {args.save_pairs_csv}")
    fieldnames = [
        "sample_index", "street", "position",
        "true_label", "true_action", "true_size",
        "pred_action", "generated_text", "pred_size_raw",
        "parse_fail", "action_correct",
        "size_abs_error", "within_0_5bb", "within_1bb", "within_2bb",
        "included_in_sizing_metrics", "exclusion_reason",
    ]
    with open(args.save_pairs_csv, "w", newline="", encoding="utf-8") as _f:
        writer = _csv_mod.DictWriter(_f, fieldnames=fieldnames)
        writer.writeheader()
        for s in sizing_comparisons:
            true_size = s["true_value"]
            pred_size = s["pred_value"]
            action_correct = s["action_match"]
            parse_fail = s["parse_fail"]

            # Determine inclusion in sizing_metrics
            included = (true_size is not None and pred_size is not None and action_correct)
            if parse_fail:
                excl_reason = "parse_fail"
            elif not action_correct:
                excl_reason = "action_mismatch"
            elif true_size is None:
                excl_reason = "true_value_none"
            elif pred_size is None:
                excl_reason = "pred_value_none"
            else:
                excl_reason = ""

            if included:
                err = abs(true_size - pred_size)
                w05 = int(err <= 0.5)
                w1  = int(err <= 1.0)
                w2  = int(err <= 2.0)
            else:
                err = w05 = w1 = w2 = ""

            writer.writerow({
                "sample_index": s["sample_idx"],
                "street": s.get("street", ""),
                "position": s.get("position", ""),
                "true_label": s.get("gold_label", ""),
                "true_action": s["true_action"],
                "true_size": true_size if true_size is not None else "",
                "pred_action": s["pred_action"],
                "generated_text": (s.get("generated_text") or "").replace("\n", "\\n"),
                "pred_size_raw": pred_size if pred_size is not None else "",
                "parse_fail": int(parse_fail),
                "action_correct": int(action_correct),
                "size_abs_error": f"{err:.4f}" if err != "" else "",
                "within_0_5bb": w05,
                "within_1bb": w1,
                "within_2bb": w2,
                "included_in_sizing_metrics": int(included),
                "exclusion_reason": excl_reason,
            })
    print(f"\u2705 {len(sizing_comparisons)} sizing samples salvos em {args.save_pairs_csv}")

# Save per-sample predictions CSV (if requested)
if args.save_predictions:
    import csv
    print(f"\n💾 Salvando predições em: {args.save_predictions}")
    
    with open(args.save_predictions, "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = ["sample_idx", "gold_action", "pred_action", "gold_size", "pred_size", "abs_error"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Track sizing index for bet/raise actions
        sizing_idx = 0
        
        for i in range(len(y_true_action)):
            gold_action = y_true_action[i]
            pred_action = y_pred_action[i]
            
            # Parse gold_size and pred_size from exact labels
            gold_label = y_true_exact[i]
            pred_label = y_pred_exact[i]
            
            # Extract sizes (None for non-sizing actions)
            gold_size = None
            pred_size = None
            abs_error = None
            
            if gold_action in ["bet", "raise"]:
                # Parse gold size from label
                parts = gold_label.split()
                if len(parts) >= 2:
                    try:
                        gold_size = float(parts[1])
                    except:
                        pass
                
                # Get predicted size from sizing_comparisons
                if sizing_idx < len(sizing_comparisons):
                    s = sizing_comparisons[sizing_idx]
                    pred_size = s["pred_value"]
                    
                    # Calculate absolute error only if both values exist and action matches
                    if gold_size is not None and pred_size is not None and s["action_match"]:
                        abs_error = abs(gold_size - pred_size)
                
                sizing_idx += 1
            
            writer.writerow({
                "sample_idx": i,
                "gold_action": gold_action,
                "pred_action": pred_action,
                "gold_size": gold_size if gold_size is not None else "",
                "pred_size": pred_size if pred_size is not None else "",
                "abs_error": f"{abs_error:.2f}" if abs_error is not None else ""
            })
    
    print(f"✅ {len(y_true_action)} predições salvas em {args.save_predictions}")

# Output filename
if args.output_file:
    output_file = args.output_file
else:
    model_tag = MODEL_NAME.split("/")[-1].replace("-", "").lower()[:10]
    n_samples = args.max_samples if args.max_samples else len(samples_to_process)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    output_file = f"diagnostico_{STAGE}_{model_tag}_hybrid_seed{args.seed}_n{n_samples}_{timestamp}.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n💾 Resultados salvos em: {output_file}")
print("\n✅ Avaliação concluída!")
