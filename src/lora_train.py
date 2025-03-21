from dotenv import load_dotenv
import os
import boto3
from botocore.exceptions import NoCredentialsError
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

load_dotenv()

# Set S3 & Local path
S3_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_MODEL_PATH = "models/tinyllama_model/"
DOWNLOAD_DIR = "./tmp/tinyllama_model"

REQUIRED_FILES = {
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "generation_config.json",
    "model.safetensors"
}

# Set S3 client
s3 = boto3.client('s3', region_name=S3_REGION, config=boto3.session.Config(signature_version='s3v4'))

# If the model is not in the local, download from S3
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Import model file list
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_MODEL_PATH)

        if "Contents" in response:
            for obj in response["Contents"]:
                filename = os.path.basename(obj["Key"])
                if filename in REQUIRED_FILES:
                    dest_path = os.path.join(DOWNLOAD_DIR, filename)
                    print(f"Downloading: {filename}")
                    s3.download_file(S3_BUCKET_NAME, obj["Key"], dest_path)

        else:
            print(f"🚨 Model files are not in S3 '{S3_MODEL_PATH}'!")

    except NoCredentialsError:
        print("❌ No AWS authentication information! Authentication setting required with 'aws configure'")

# Check model directory local files
downloaded_files = os.listdir(DOWNLOAD_DIR)
print(f"The list of downloaded files: {downloaded_files}")

# Load Hugging Face model
try:
    tokenizer = AutoTokenizer.from_pretrained(DOWNLOAD_DIR, cache_dir="/tmp/hf_cache")
    model = AutoModelForCausalLM.from_pretrained(DOWNLOAD_DIR, cache_dir="/tmp/hf_cache")
    print("✅ Complete the loading Model & tokenizer!")

except Exception as e:
    print(f"❌ Fail loading model: {e}")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")


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