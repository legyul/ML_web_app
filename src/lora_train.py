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
    """
    Create a unique path with the file name uploaded by the user and the classification model name selected
    """
    logger.info(f"get_finedtuned_model: {selected_model}")
    safe_model = sanitize_model_name(selected_model)
    filename_no_ext = os.path.splitext(os.path.basename(upload_filename))[0]
    model_folder_name = f"{filename_no_ext}_{safe_model}"

    path = os.path.join("/tmp/lora_finetuned_model", model_folder_name)
    logger.info(f"finetuned model path: {path}")

    return path

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
        logger.debug(f"📄 Number of prompts: {len(prompts)}")

        if len(prompts) < 10:
            logger.warning(f"📉 Prompt 개수 너무 적음: {len(prompts)} → 데이터 증강 시작")

            # row를 복제해서 prompt 수 늘리기
            prompts *= (10 // len(prompts)) + 1
            prompts = prompts[:10]  # 10개까지만 사용

        logger.info(f"✅ 최종 prompt 개수: {len(prompts)}")

        prompt_count = len(prompts)

        if prompt_count < 50:
            num_epochs = 5
        elif prompt_count < 200:
            num_epochs = 3
        else:
            num_epochs = 2

        dataset = PromptDataset(prompts, tokenizer)
        logger.debug(f"📦 Dataset length: {len(dataset)}")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=0)

        # ✅ Step 4: Training
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for epoch in range(num_epochs):
            total_loss = 0
            logger.debug(f"💡 Epoch {epoch+1} 시작 - 총 배치 수: {len(dataloader)}")
            for step, batch in enumerate(dataloader):
                try:
                    logger.debug(f"💡 Epoch {epoch+1} 시작 - 총 배치 수: {len(dataloader)}")
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

                    logger.debug("🌀 Backward 시작")
                    start = time.time()
                    loss.backward()
                    logger.debug(f"✅ Backward 끝 (소요시간: {time.time() - start:.2f}초)")
                    logger.debug("💾 Optimizer step 시작")
                    optimizer.step()
                    logger.debug("💾 Optimizer step 끝")
                    optimizer.zero_grad()
                    total_loss += loss.item()
                except Exception as e:
                    logger.error(f"❌ Error during training step: {e}")
            logger.info(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")
        
        logger.info("✅ Finished all epochs. Proceeding to save model...")

        # ✅ Step 5: Save
        os.makedirs(SAVE_PATH, exist_ok=True)
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)

        # ✅ After model.save_pretrained(SAVE_PATH)
        # Copy base model config into SAVE_PATH
        base_config_path = os.path.join(BASE_MODEL_DIR, "config.json")
        target_config_path = os.path.join(SAVE_PATH, "config.json")
        shutil.copyfile(base_config_path, target_config_path)
        
        # ✅ Step 6: Add model_type to config.json
        config_path = os.path.join(SAVE_PATH, "config.json")
        logger.info("DEBUG - config.json exists =", os.path.exists(os.path.join(SAVE_PATH, "config.json")))
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = json.load(f)
            
            config_data.append({"model_type": "llama"})
            config_data.append({"architectures": "LlamaForCausalLM"})

            # with open(config_path, "w") as f:
            #     json.dump(config_data, f, indent=2)
            
            logger.info("Done edit config.json")
        
        else:
            logger.info("config.json not found")

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