import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import os

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)

# Load model (Save memory by applying 4-bit quantization)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,  # 4-bit quantization
    device_map="auto",
    use_auth_token=HF_TOKEN
)

# LoRA settings (set to work on CPU as well)
lora_config = LoraConfig(
    r=8,  # Rank (The lower the memory, the less memory used)
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Check LoRA applied
print("LoRA applied (CPU optimization)")