import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import huggingface_pipeline
from langchain.chains import retrieval_qa
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig, GPT2Config
from lora_train import get_finedtuned_model_path
import json
from peft import PeftModel, PeftConfig
from pathlib import Path

load_dotenv()
# Lazy-load cache
_qa_pipeline = None

# def safe_load_model(model_path: str):
#     # ✅ Step 1: config.json 수정 확인 및 보완
#     config_path = os.path.join(model_path, "config.json")
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"config.json not found in {model_path}")
#     with open(config_path, "r") as f:
#         config_data = json.load(f)

#     # ✅ Step 2: model_type 자동 보완 (예: distilgpt2 → gpt2)
#     if "model_type" not in config_data:
#         print("🔧 'model_type' not found in config.json. Adding it manually...")
#         # 아래는 사용자 선택에 따라 고칠 수 있음
#         config_data["model_type"] = "gpt2"  # 사용 중인 모델에 따라 변경
#         with open(config_path, "w") as f:
#             json.dump(config_data, f)
#         print("✅ 'model_type' successfully inserted into config.json.")

#     # ✅ Step 3: config와 model 함께 로드
#     config = AutoConfig.from_pretrained(model_path)
#     model = AutoModelForCausalLM.from_pretrained(model_path, config=config)
#     return model

def get_qa_pipeline(filename: str, model_choice: str):
    global _qa_pipeline
    if _qa_pipeline is not None:
        return _qa_pipeline

    try:
        print("[DEBUG] Loading RAG pipeline")

        # model_path = get_finedtuned_model_path(filename, model_choice)
        model_path = model_path = Path(f"/tmp/lora_finetuned_model/{filename}_naive_bayes").resolve().as_posix()
        tokenizer_path = os.path.join(model_path, "_toeknizer")
        base_model_path = "/tmp/distilgpt2"
        HF_CACHE = "/tmp/hf_cache"

        print("[DEBUG] Loading tokenizer...")
        
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=HF_CACHE, config=config, local_files_only=True, trust_remote_code=True)
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=HF_CACHE, use_fast=False, local_files_only=True)
        
        # Attach LoRA adapter
        # model = PeftModel.from_pretrained(base_model, model_path, local_files_only=True)
        # model.config.model_type = "gpt2"
        # model.to("cpu")

        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )

        EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
        CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
        embedding_function = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        llm = huggingface_pipeline(pipeline=llm_pipeline)
        _qa_pipeline = retrieval_qa.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

        print("✅ QA Pipeline loaded successfully.")
        return _qa_pipeline

    except Exception as e:
        print(f"❌ Failed to load QA pipeline: {e}")
        return None
    
def run_qa(query: str, filename: str, model_choice: str) -> str:
    """
    Create a RAG QA response to a user's question
    """
    qa = get_qa_pipeline(filename, model_choice)
    response = qa.run(query)
    return response