import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_name = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

test_prompts = [
    "The Golden Gate Bridge is located in",
    "San Francisco's most famous landmark is the",
    "The city famous for the Golden Gate Bridge is"
]

for prompt in test_prompts:
    print(f"\nPrompt: '{prompt}'")
    tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        logits = model(tokens.input_ids)[0]
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        top_tokens = torch.topk(probs, 10)
    
    print("Top 10 predictions:")
    for prob, token_id in zip(top_tokens.values, top_tokens.indices):
        token = tokenizer.decode([token_id]).strip()
        print(f"  '{token}' -> {float(prob):.4f}")
        
    # Check tokenization of San Francisco
    sf_tokens = tokenizer.encode("San Francisco", add_special_tokens=False)
    print(f"\n'San Francisco' tokenizes to: {sf_tokens}")
    for token_id in sf_tokens:
        decoded = tokenizer.decode([token_id])
        print(f"  {token_id} -> '{decoded}'")
