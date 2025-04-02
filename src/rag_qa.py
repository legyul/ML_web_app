import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import huggingface_pipeline
from langchain.chains import retrieval_qa
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel 
from lora_train import get_finedtuned_model_path
import json
from peft import PeftModel, PeftConfig
from pathlib import Path

load_dotenv()
# Lazy-load cache
_qa_pipeline = None

def get_qa_pipeline(filename: str, model_choice: str):
    global _qa_pipeline
    if _qa_pipeline is not None:
        return _qa_pipeline

    try:
        print("[DEBUG] Loading RAG pipeline")

        model_path = get_finedtuned_model_path(filename, model_choice)
        tokenizer_path = os.path.join(model_path, "_tokenizer")
        
        HF_CACHE = "/tmp/hf_cache"

        print("[DEBUG] Loading tokenizer...")
        AutoModelForCausalLM.register("gpt2", GPT2LMHeadModel)
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config_dict = json.load(f)
        
        if "model_type" not in config_dict:
            config_dict["model_type"] = "gpt2"
        if "architectures" not in config_dict:
            config_dict["architectures"] = ["GPT2LMHeadModel"]
        
        config = GPT2Config.from_dict(config_dict)
        
        model = GPT2LMHeadModel.from_pretrained(  # ✅ AutoModel → 직접 명시
            model_path,
            cache_dir=HF_CACHE,
            local_files_only=True,
            trust_remote_code=True,
            use_safetensors=True
        )

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