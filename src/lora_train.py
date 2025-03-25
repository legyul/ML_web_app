from dotenv import load_dotenv
import os
import boto3
from botocore.exceptions import NoCredentialsError
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import shutil
from utils.download_utils import download_llm_model_from_s3

load_dotenv()

# Set S3 & Local path
S3_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_MODEL_PATH = "models/tinyllama_model/"
LOCAL_MODEL_DIR = "./tmp/tinyllama_model"
HF_CACHE = "/tmp/hf_cache"

s3 = boto3.client('s3', region_name=S3_REGION, config=boto3.session.Config(signature_version='s3v4'))

REQUIRED_FILES = {
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "generation_config.json",
    "model.safetensors"
}

download_llm_model_from_s3(S3_REGION=S3_REGION,
                           S3_BUCKET_NAME=S3_BUCKET_NAME,
                           s3_model_path=S3_MODEL_PATH,
                           local_dir=LOCAL_MODEL_DIR,
                           required_files=REQUIRED_FILES)

# Load Hugging Face model
try:
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, cache_dir=HF_CACHE)
    base_model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_DIR, cache_dir=HF_CACHE, torch_dtype=torch.float32)
    print("✅ Complete the loading Model & tokenizer!")

except Exception as e:
    print(f"❌ Fail loading model: {e}")

base_model = prepare_model_for_kbit_training(base_model)

# LoRA settings (set to work on CPU as well)
lora_config = LoraConfig(
    r=8,  # Rank (The lower the memory, the less memory used)
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(base_model, lora_config)

# Check LoRA applied
print("LoRA applied (CPU optimization)")

# -------------- Test ----------
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")
labels = inputs.input_ids.clone()
inputs["labels"] = labels

# Train loop
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(3):
    optimizer.zero_grad()
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} - Loss: {loss.item(): .4f}")

# Save trained model
SAVE_PATH = "./tmp/lora_finetuned_model"
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

# Upload the trained model to S3
for file in os.listdir(SAVE_PATH):
    s3.upload_file(os.path.join(SAVE_PATH, file), S3_BUCKET_NAME, f"models/lora_finetuned/{file}")
print("LoRA fine-tuned model uploaded to S3.")