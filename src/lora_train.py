from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model
import os

MODEL_NAME = "mosaicml/mpt-1b-redpajama-200b"
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

os.environ["HF_HOME"] = "/tmp/hg_cache"

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
    )

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    token=HF_TOKEN,
    trust_remote_code=True,
    low_cpu_mem_usage=True
    ).to(device)

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