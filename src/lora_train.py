import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import os

MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

os.environ["HF_HOME"] = "/tmp/hg_cache"

#bnb_config = BitsAndBytesConfig(load_in_8bit = False, load_in_4bit = True) # Enable to run on CPU

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

# Load model
model = AutoModel.from_pretrained(MODEL_NAME, token=HF_TOKEN)

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