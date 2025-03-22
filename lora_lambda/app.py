import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset

MODEL_DIR = "tmp/tinyllama_model"

def lambda_handler(event, context):
    prompt = event.get("prompt", "Sample text")
    s3_output_path = event.get("output_s3_path", "models/fine_tuned")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)

    base_model = prepare_model_for_kbit_training(base_model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(base_model, lora_config)

    # Generate training dataset (Will be edited to fine tuning)
    inputs = tokenizer(prompt, return_tensors="pt")
    labels = inputs.input_ids.clone()
    inputs["labels"] = labels

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for _ in range(3):
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
    # Save
    save_path = "/tmp/lora_finetuned_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Upload to S3
    import boto3

    S3_REGION = "us-east-2"
    S3_BUCKET_NAME = "ml-platform-service"

    s3 = boto3.client('s3', region_name=S3_REGION, config=boto3.session.Config(signature_version='s3v4'))

    for file in os.listdir(save_path):
        s3.upload_file(os.path.join(save_path, file), S3_BUCKET_NAME, f"{s3_output_path}/{file}")

    return {
        "statusCode": 200,
        "body": json.dumps({"message": "LoRA fine-tuning complete and uploaded to S3"})
    }
