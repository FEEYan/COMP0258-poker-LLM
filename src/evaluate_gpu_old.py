import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from typing import Optional
from datetime import datetime
from datasets import load_dataset  # Added import
from peft import PeftModel  # Added import
import os  # Added import
from safetensors.torch import load_file  # Added import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate language model generation')
    parser.add_argument(
        '--model-name',
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help='Model name or path'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=50,
        help='Maximum length of generated text'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        default="test.jsonl",
        help='Path to test data file'
    )
    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Force CPU usage even if GPU is available'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='./',
        help='Path to save evaluation metrics (timestamp will be added automatically)'
    )
    parser.add_argument(
        '--prompt-key',
        type=str,
        default="instruction",
        help='Key for the prompt in the dataset'
    )
    parser.add_argument(
        '--output-key',
        type=str,
        default="output",
        help='Key for the output in the dataset'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Name of the Hugging Face dataset to load instead of a local file'
    )
    parser.add_argument(
        '--data-source',
        type=str,
        choices=['local', 'hf'],
        default='local',
        help='Data source to use: "local" for a local file, "hf" for Hugging Face dataset'
    )
    parser.add_argument(
        '--adapter-path',
        type=str,
        default=None,
        help='Path to LoRA adapter folder'
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=50,
        help='Maximum number of new tokens to generate'
    )
    parser.add_argument('--iter', type=str, help='Iteration number for adapter weights file selection')
    return parser


def get_device(force_cpu: bool = False):
    if force_cpu:
        return torch.device("cpu")
    # Priority: CUDA first, then Apple Silicon MPS, fallback to CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            messages = item.get("messages", [])
            prompt = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            output = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
            data.append({"instruction": prompt, "output": output})
    return data


def exact_match(prediction, truth):
    return prediction.strip() == truth.strip()


def compute_em(predictions, ground_truths):
    total = len(predictions)
    matches = sum(exact_match(p, t)
                  for p, t in zip(predictions, ground_truths))
    return matches / total if total > 0 else 0


def compute_aa(predictions, ground_truths):
    accuracies = []
    for pred, truth in zip(predictions, ground_truths):
        # Convert both strings to lowercase before comparison
        pred_lower = pred.strip().lower()
        truth_lower = truth.strip().lower()
        common = sum(1 for a, b in zip(pred_lower, truth_lower) if a == b)
        max_len = max(len(pred_lower), len(truth_lower))
        accuracies.append(common / max_len if max_len > 0 else 0)
    return sum(accuracies) / len(accuracies) if accuracies else 0


def generate_text(prompt, tokenizer, model, device, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
       
        temperature=0.7, 
        top_p=0.9,      
        do_sample=True 
    )
    
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated.strip().split()[0]


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    device = get_device(args.force_cpu)
    print("Using device:", device)

    # Load base model
    print(f"Loading base model {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load LoRA adapter
    if args.adapter_path:
        print(f"Loading adapter from {args.adapter_path}")
        # Load custom config
        with open(f"{args.adapter_path}/adapter_config.json") as f:
            lora_config = json.load(f)
        
        # Create PEFT config from custom config
        from peft import LoraConfig, PeftModel
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=True,
            r=lora_config["lora_parameters"]["rank"],
            lora_alpha=lora_config["lora_parameters"]["scale"],
            lora_dropout=lora_config["lora_parameters"]["dropout"]
        )
        
        # Find latest adapter weights file
        if args.iter is not None:
            adapter_file = f"{args.iter}_adapters.safetensors"
            adapter_path = os.path.join(args.adapter_path, adapter_file)
            if not os.path.exists(adapter_path):
                raise FileNotFoundError(f"Adapter file not found: {adapter_path}")
        else:
            adapter_files = sorted([f for f in os.listdir(args.adapter_path) if f.endswith('adapters.safetensors')])
            latest_adapter = adapter_files[-1] if adapter_files else 'adapters.safetensors'
            adapter_path = os.path.join(args.adapter_path, latest_adapter)
        
        print(f"Using adapter weights from: {adapter_path}")
        # Load the adapter weights using safetensors
        model = PeftModel(
            model,
            peft_config,
        )
        state_dict = load_file(adapter_path)
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    model.to(device)

    # Load test data based on --data-source flag
    if args.data_source == 'hf' and args.dataset:
        dataset = load_dataset(args.dataset)
        # Assumes using the 'train' split; adjust if necessary
        test_data = dataset["train"]
    else:
        test_data = load_data(args.test_file)
    
    ground_truths = [item[args.output_key] for item in test_data]
    prompts = [item[args.prompt_key] for item in test_data]

    predictions = []
    for prompt, truth in zip(prompts, ground_truths):
        # Generate text based on prompt
        generated = generate_text(
            prompt, tokenizer, model, device, max_new_tokens=args.max_new_tokens)
        predictions.append(generated)
        # print("Instruction:", prompt)
        print("Ground Truth:", truth)
        print("." * 50)
        print("Generated:", generated) 
        print("-" * 100)

    # Calculate evaluation metrics
    em_score = compute_em(predictions, ground_truths)
    aa_score = compute_aa(predictions, ground_truths)
    print("Exact Match (EM):", em_score)
    print("Answer Accuracy (AA):", aa_score)

    # Save evaluation metrics to file
    metrics = {
        "Exact Match (EM)": em_score,
        "Answer Accuracy (AA)": aa_score
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{args.output_path}evaluation_metrics_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {output_file}")


if __name__ == "__main__":
    # Entry point of the script
    main()
