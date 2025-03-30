from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import boto3
from utils.logger_utils import logger
from models.common import load_file

device = torch.device("cpu")


class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer):
        self.encodings = tokenizer(prompts, truncation=True, padding=True, return_tensors="pt")
        self.labels = self.encodings["input_ids"].clone()
        self.encodings["labels"] = self.labels
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.labels)

def get_prompts_from_s3_dataset(s3_key: str) -> list[str]:
    df, _ = load_file(s3_key)
    prompts = []
    IGNORE_COLUMNS = ["ID", "Timestamp"]
    for _, row in df.iterrows():
        text = "\n".join([f"{col}: {row[col]}" for col in df.columns if col not in IGNORE_COLUMNS])
        prompts.append(text)
    return prompts

def get_finedtuned_model_path(upload_filename: str, selected_model: str) -> str:
    """
    Create a unique path with the file name uploaded by the user and the classification model name selected
    """
    filename_no_ext = os.path.splitext(os.path.basename(upload_filename))[0]
    model_folder_name = f"{filename_no_ext}_{selected_model.lower()}"

    return os.path.join("/tmp/lora_finetuned_model", model_folder_name)

def train_lora_from_user_data(s3_dataset_key: str, filename: str, selected_model: str):
    logger.debug("[DEBUG] Entered train_lora_from_user_data()")

    try:
        SAVE_PATH = get_finedtuned_model_path(filename, selected_model)
        HF_CACHE = "/tmp/hf_cache"
        BASE_MODEL_DIR = "/tmp/tinyllama_model"
        logger.debug(f"[DEBUG] BASE_MODEL_DIR = {BASE_MODEL_DIR}")
        logger.debug(f"[DEBUG] BASE_MODEL_DIR contents = {os.listdir(BASE_MODEL_DIR)}")
        
        filename_no_ext = os.path.splitext(os.path.basename(filename))[0]
        model_folder_name = f"{filename_no_ext}_{selected_model.lower()}"
        s3_model_path = f"models/lora_finetuned_model/{model_folder_name}"
        logger.info(f"train_lora_from: {selected_model}")

        config_path = os.path.join(BASE_MODEL_DIR, "config.json")
        with open(config_path, "r") as f:
            config_data = json.load(f)

        config_data["torch_dtype"] = "float32"

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        # ✅ Step 1: Load tokenizer and base model from pre-downloaded path
        try:
            logger.debug("[DEBUG] Trying to load tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, cache_dir=HF_CACHE, use_fast=False, local_files_only=True)
            logger.debug("✅ Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load tokenizer: {e}")
            return
        try:
            logger.debug("[DEBUG] Trying to load base model...")
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_DIR,
                cache_dir=HF_CACHE,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True
            ).to("cpu")
            logger.debug("✅ Base model loaded successfully")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load base model: {e}")
            return
        logger.debug("Loaded tokenizer successfully")

        # ✅ Step 2: Apply LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
            )
        model = get_peft_model(base_model, lora_config)

        # ✅ Step 3: Prepare Data
        prompts = get_prompts_from_s3_dataset(s3_dataset_key)
        dataset = PromptDataset(prompts, tokenizer)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # ✅ Step 4: Training
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
            logger.info(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

        # ✅ Step 5: Save
        os.makedirs(SAVE_PATH, exist_ok=True)
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)

        # ✅ Step 6: Add model_type to config.json
        config_path = os.path.join(SAVE_PATH, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = json.load(f)
            
            config_data["model_type"] = "llama"
            config_data["architectures"] = ["LlamaForCausalLM"]

            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)
            
            logger.debug("Done edit config.json")
        
        else:
            logger.debug("config.json not found")

        # ✅ Step 7: Upload to S3
        s3 = boto3.client('s3', region_name=os.getenv("AWS_REGION"), config=boto3.session.Config(signature_version='s3v4'))
        for file in os.listdir(SAVE_PATH):
            local_path = os.path.join(SAVE_PATH, file)
            s3_key = f"{s3_model_path}/{file}"
            s3.upload_file(local_path, os.getenv("S3_BUCKET_NAME"), s3_key)
    except Exception as e:
        logger.error(f"[ERROR] train_lora_from_user_data() Exception: {str(e)}")
        print(f"[ERROR] train_lora_from_user_data() Exception: {str(e)}")

def run_train_thread(s3_path, filename, model_choice):
    try:
        logger.debug("Inside run_train_thread")
        print("Inside run_train_thread")
        train_lora_from_user_data(s3_path, filename, model_choice)
    except Exception as e:
        logger.error(f"Threaded training failed: {str(e)}")
        print(f"Threaded training failed: {str(e)}")