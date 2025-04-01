from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import boto3
from utils.logger_utils import logger
from models.common import load_file
import shutil
import time
from pathlib import Path

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

def sanitize_model_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

def get_finedtuned_model_path(upload_filename: str, selected_model: str) -> str:
    safe_model = sanitize_model_name(selected_model)
    filename_no_ext = os.path.splitext(os.path.basename(upload_filename))[0]
    model_folder_name = f"{filename_no_ext}_{safe_model}"
    abs_path = Path("/tmp/lora_finetuned_model") / model_folder_name
    return abs_path.resolve().as_posix()

def train_lora_from_user_data(s3_dataset_key: str, filename: str, selected_model: str):
    logger.debug("[DEBUG] Entered train_lora_from_user_data()")

    try:
        SAVE_PATH = get_finedtuned_model_path(filename, selected_model)
        HF_CACHE = "/tmp/hf_cache"
        BASE_MODEL_DIR = "/tmp/distilgpt2"
        logger.debug(f"[DEBUG] BASE_MODEL_DIR = {BASE_MODEL_DIR}")
        logger.debug(f"[DEBUG] BASE_MODEL_DIR contents = {os.listdir(BASE_MODEL_DIR)}")
        
        safe_model = sanitize_model_name(selected_model)
        filename_no_ext = os.path.splitext(os.path.basename(filename))[0]
        model_folder_name = f"{filename_no_ext}_{safe_model}"
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

            # ✅ fix padding issue
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

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
        logger.info("Applied lora_config")
        logger.info(f"✅ LoRA Model object: {model}")
        logger.info(f"✅ Using device: {device}")

        # ✅ Step 3: Prepare Data
        prompts = get_prompts_from_s3_dataset(s3_dataset_key)

        if len(prompts) < 10:
            prompts *= (10 // len(prompts)) + 1
            prompts = prompts[:10] 

        prompt_count = len(prompts)

        if prompt_count < 50:
            num_epochs = 15
        elif prompt_count < 200:
            num_epochs = 13
        else:
            num_epochs = 10

        dataset = PromptDataset(prompts, tokenizer)

        # ✅ Step 4: Training
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

        seen_batches = 0
        for epoch in range(num_epochs):
            total_loss = 0
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=0)
            assert len(dataloader) < 1000, "Dataloader length is unexpectedly large. Check dataset logic."
            
            for step, batch in enumerate(dataloader):
                seen_batches += 1
                if seen_batches > len(dataloader):
                    logger.error("Too many steps, something's worng with dataloader")
                    break
                try:
                    
                    logger.debug("🧠 Moving batch to device...")
                    batch = {k: v.to(device) for k, v in batch.items()}
                    logger.debug("✅ Batch moved to device")
                    logger.debug("📥 Forward pass start")
                    outputs = model(**batch)
                    logger.debug("📤 Forward pass complete")
                    loss = outputs.loss
                    logger.debug(f"🧮 Epoch {epoch+1} | Step {step+1} | Loss: {loss.item()}")
                    
                    if torch.isnan(loss):
                        logger.error("❌ NaN loss detected! Stopping training.")
                        break

                    start = time.time()
                    loss.backward()
                    logger.debug(f"✅ Done Backward (time: {time.time() - start:.2f}seconds)")
                    logger.debug("💾 Start Optimizer step")
                    optimizer.step()
                    logger.debug("💾 Done Optimizer step")
                    optimizer.zero_grad()
                    total_loss += loss.item()
                except Exception as e:
                    logger.error(f"❌ Error during training step: {e}")
            scheduler.step()
            logger.info(f"Epoch {epoch+1} Finished - Loss: {total_loss:.4f}")
        
        logger.info("✅ Finished all epochs. Proceeding to save model...")

        # ✅ Step 5: Save
        try:
            os.makedirs(SAVE_PATH, exist_ok=True)
            model.save_pretrained(SAVE_PATH)
        except Exception as save_err:
            logger.error(f"Model saving failed: {save_err}")
            raise

        # ✅ After model.save_pretrained(SAVE_PATH)
        # Copy base model config into SAVE_PATH
        # base_config_path = os.path.join(BASE_MODEL_DIR, "config.json")
        # target_config_path = os.path.join(SAVE_PATH, "config.json")
        # shutil.copyfile(base_config_path, target_config_path)
        
        # ✅ Step 6: Add model_type to config.json
        config_path = os.path.join(SAVE_PATH, "config.json")
        config_dict = model.config.to_dict()

        if "model_type" not in config_dict:
            config_dict["model_type"] = "gpt2"
        
        if "architectures" not in config_dict:
            config_dict["architectures"] = ["GPT2LMHeadModel"]

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        with open(config_path, "r") as f:
            conf = json.load(f)
        
        tokenizer.save_pretrained(os.path.join(SAVE_PATH, "_tokenizer"))
            
            # logger.info("Done edit config.json")
        
        # else:
        #     logger.info("config.json not found")

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