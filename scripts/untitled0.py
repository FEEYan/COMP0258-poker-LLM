import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# --- Load the PokerBench dataset ---
# Assume the JSON files contain 'instruction' (poker scenario prompt) and 'output' (optimal decision)
dataset = load_dataset("RZ412/PokerBench", split="train")  # You can also load the test split for evaluation
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # small batch size for demonstration

# --- Load the pre-trained Llama 3B model and tokenizer ---
model_name = "llama-3b"  # Replace with the actual model repository name
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a frozen reference model (used for KL regularization)
ref_model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

# --- Define hyperparameters ---
learning_rate = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
beta = 0.04           # KL divergence coefficient
epsilon = 0.2         # Clipping parameter
group_size = 4        # Number of samples per prompt

# --- Define a simple reward function ---
def compute_reward(generated_text: str, target_text: str) -> float:
    # For illustration, we give a reward of 1 if the generated decision (case-insensitive) matches the target.
    return 1.0 if generated_text.strip().lower() == target_text.strip().lower() else 0.0

# --- Helper function to compute log-likelihood of generated output ---
def get_log_prob(model, input_ids):
    # Get logits and compute log probabilities for the sequence (sum over tokens)
    outputs = model(input_ids, labels=input_ids)
    # outputs.loss is the mean negative log-likelihood; multiply by token count to get the total
    seq_length = input_ids.shape[1]
    return -outputs.loss * seq_length

# --- GRPO Training Loop ---
num_epochs = 3  # For demonstration; adjust as needed

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}")
    for batch in dataloader:
        instructions = batch["instruction"]
        targets = batch["output"]

        # Tokenize the prompts (for generation)
        prompt_inputs = tokenizer(instructions, return_tensors="pt", padding=True, truncation=True).to("cuda")

        # For each prompt, generate a group of candidate responses
        group_outputs = []   # List[List[str]]
        group_rewards = []   # List[List[float]]
        group_log_probs = [] # List[List[float]] for current model and old policy

        for i, prompt in enumerate(instructions):
            samples = []
            rewards = []
            log_probs = []
            # Generate multiple candidate responses
            for _ in range(group_size):
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
                generated_ids = model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 50,  # generate up to 50 tokens beyond prompt
                    do_sample=True,
                    top_p=0.95,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                samples.append(generated_text)
                
                # Compute reward by comparing with the target decision
                r = compute_reward(generated_text, targets[i])
                rewards.append(r)
                
                # Compute log probability (using teacher forcing on the generated text)
                full_input = tokenizer.encode(generated_text, return_tensors="pt").to("cuda")
                lp = get_log_prob(model, full_input)
                log_probs.append(lp.item())
            group_outputs.append(samples)
            group_rewards.append(rewards)
            group_log_probs.append(log_probs)

        # Compute group mean rewards and relative advantages for each prompt
        group_advantages = []
        for rewards in group_rewards:
            mean_r = sum(rewards) / len(rewards)
            advantages = [r - mean_r for r in rewards]
            group_advantages.append(advantages)

        # Now compute the GRPO loss over the entire batch
        total_loss = 0.0
        total_count = 0
        for i in range(len(instructions)):
            prompt = instructions[i]
            for j in range(group_size):
                # Re-compute log probabilities for current and old policy (using reference model as proxy for old policy)
                input_seq = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
                # Append generated tokens (this simple example assumes generation is concatenated)
                gen_part = tokenizer.encode(group_outputs[i][j][len(prompt):], return_tensors="pt").to("cuda")
                full_input = torch.cat([input_seq, gen_part], dim=1)
                
                # Log probabilities under current policy
                current_logp = get_log_prob(model, full_input)
                # Log probabilities under reference (old) policy
                with torch.no_grad():
                    ref_logp = get_log_prob(ref_model, full_input)
                
                # Importance ratio
                ratio = torch.exp(current_logp - ref_logp)
                # Clip the ratio
                clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
                advantage = group_advantages[i][j]
                # PPO-style loss (we minimize negative objective)
                loss_policy = -torch.min(ratio * advantage, clipped_ratio * advantage)
                
                # For simplicity, we approximate the KL divergence as zero or use a simple penalty if needed.
                # In practice, you would compute token-level KL divergence between current and reference model.
                kl_penalty = 0.0

                total_loss += loss_policy + beta * kl_penalty
                total_count += 1

        total_loss = total_loss / total_count

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} loss: {total_loss.item():.4f}")

print("Training complete.")
