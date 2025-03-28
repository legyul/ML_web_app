from dotenv import load_dotenv
import os
import pandas as pd
import boto3
from botocore.exceptions import NoCredentialsError
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from torch.utils.data import Dataset, DataLoader
import shutil
from utils.download_utils import download_llm_model_from_s3
from models.common import load_file
from utils.logger_utils import logger

load_dotenv()

device = torch.device("cpu")

def get_prompts_from_s3_dataset(s3_key: str) -> list[str]:
    """
    Convert user uploaded data to training sentence
    """
    logger.debug(f"[DEBUG] Loading prompts from dataset key: {s3_key}")
    df, _ = load_file(s3_key)
    prompts = []

    IGNORE_COLUMNS = ["ID", "Timestamp"]

    for _, row in df.iterrows():
        text = "\n".join([f"{col}: {row[col]}" for col in df.columns if col not in IGNORE_COLUMNS])
        prompts.append(text)
    
    return prompts

class PromptDataset(Dataset):
    """
    Dataset for training LoRA
    """
    def __init__(self, prompts, tokenizer):
        self.encodings = tokenizer(prompts, truncation=True, padding=True, return_tensors="pt")
        self.labels = self.encodings["input_ids"].clone()
        self.encodings["labels"] = self.labels
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.labels)
    
def train_lora_from_user_data(s3_dataset_key: str):
    logger.debug("[DEBUG] Entered train_lora_from_user_data()")
    logger.debug("[DEBUG] Step 1: Starting model download")

    S3_REGION = os.getenv("AWS_REGION")
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
    S3_MODEL_PATH = "models/tinyllama_model/"
    LOCAL_MODEL_DIR = "/tmp/tinyllama_model"
    HF_CACHE = "/tmp/hf_cache"
    SAVE_PATH = "/tmp/lora_finetuned_model"

    REQUIRED_FILES = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "generation_config.json",
        "model.safetensors"
    ]

    # Download model
    download_llm_model_from_s3(S3_REGION=S3_REGION,
                            S3_BUCKET_NAME=S3_BUCKET_NAME,
                            s3_model_path=S3_MODEL_PATH,
                            local_dir=LOCAL_MODEL_DIR,
                            required_files=REQUIRED_FILES)
    logger.debug("[DEBUG] Step 2: Model download complete")

    logger.debug("[DEBUG] Step 3: Loading tokenizer and base model")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, cache_dir=HF_CACHE)
    base_model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_DIR, cache_dir=HF_CACHE, torch_dtype=torch.float32).to(device)

    logger.debug("[DEBUG] Step 4: Preparing LoRA config")
    # Apply LoRA setting
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)

    # Prepare the data
    prompts = get_prompts_from_s3_dataset(s3_dataset_key)
    dataset = PromptDataset(prompts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Train
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(3):
        total_loss = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        logger.info(f"Epoch {epoch+1} - Loss: {total_loss: .4f}")

    # Save and upload to S3
    model.eval()

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        logger.debug(f"[DEBUG] Created directory: {SAVE_PATH}")
    else:
        logger.debug(f"[DEBUG] Directory already exists: {SAVE_PATH}")

    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    logger.info("Saving base model weights...")
    model.base_model.save_pretrained(SAVE_PATH)

    base_config_path = os.path.join(LOCAL_MODEL_DIR, "config.json")
    target_config_path = os.path.join(SAVE_PATH, "config.json")

    logger.debug(f"[DEBUG] Base config path: {base_config_path}")
    logger.debug(f"[DEBUG] Target config path: {target_config_path}")
    logger.debug(f"[DEBUG] Does base config exist? {os.path.exists(base_config_path)}")

    if os.path.exists(base_config_path):
        import shutil
        try:
            shutil.copy(base_config_path, target_config_path)
            logger.debug("[DEBUG] config.json copied.")

            # If there is no model_type, add it
            import json
            try:
                with open(target_config_path, "r") as f:
                    config_data = json.load(f)

                    if "model_type" not in config_data:
                        config_data["model_type"] = "llama"
                    
                    with open(target_config_path, "w") as f:
                        json.dump(config_data, f, indent=2)
                    
                    logger.debug("config.json copied and updated.")
            
            except Exception as json_err:
                logger.error(f"[ERROR] Failed to read or update config.json: {json_err}")
        
        except Exception as copy_err:
            logger.error(f"[ERROR] Failed to copy config.json: {copy_err}")
    
    else:
        logger.error("Base model config.json not found.")


    s3 = boto3.client('s3', region_name=S3_REGION, config=boto3.session.Config(signature_version='s3v4'))
    for file in os.listdir(SAVE_PATH):
        s3.upload_file(os.path.join(SAVE_PATH, file), S3_BUCKET_NAME, f"models/lora_finetuned/{file}")

