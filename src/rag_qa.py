import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import huggingface_pipeline
from langchain.chains import retrieval_qa
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from lora_train import get_finedtuned_model_path
from peft import PeftModel, PeftConfig

load_dotenv()

# Lazy-load cache
_qa_pipeline = None

def get_qa_pipeline(filename: str, model_choice: str):
    global _qa_pipeline
    if _qa_pipeline is not None:
        return _qa_pipeline

    try:
        print("[DEBUG] Loading RAG pipeline with PEFT")
        model_path = get_finedtuned_model_path(filename, model_choice)
        HF_CACHE = "/tmp/hf_cache"

        # 🔧 1. PEFT 설정 로드
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_path = peft_config.base_model_name_or_path

        # 🔧 2. Base model + tokenizer 로드
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, cache_dir=HF_CACHE, use_fast=False)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, cache_dir=HF_CACHE)

        # 🔧 3. LoRA 적용된 모델 로드
        model = PeftModel.from_pretrained(base_model, model_path).to("cpu")

        # 🔧 4. Text-generation 파이프라인 구성
        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )

        # 🔧 5. Vector DB 로드
        EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
        CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

        embedding_function = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        llm = huggingface_pipeline(pipeline=llm_pipeline)

        _qa_pipeline = retrieval_qa.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

        print("✅ QA Pipeline loaded successfully.")
        return _qa_pipeline

    except Exception as e:
        print(f"❌ Failed to load LoRA model with PEFT: {e}")
        return None
    
def run_qa(query: str, filename: str, model_choice: str) -> str:
    """
    Create a RAG QA response to a user's question
    """
    qa = get_qa_pipeline(filename, model_choice)
    response = qa.run(query)
    return response