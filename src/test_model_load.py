from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import os
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "debug"

model_path = Path("/tmp/lora_finetuned_model/Iris_naive_bayes").resolve().as_posix()
tokenizer_path = os.path.join(model_path, "_tokenizer")

print(f"Model path: {model_path}")
print("▶ Trying to load config...")
config = AutoConfig.from_pretrained(model_path, local_files_only=True)
print("✅ config.model_type:", config.model_type)

print("▶ Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    local_files_only=True,
    trust_remote_code=True,
)

print("▶ Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

print("🎉 ALL LOADED SUCCESSFULLY 🎉")