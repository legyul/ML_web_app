import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import huggingface_pipeline
from langchain.chains import retrieval_qa
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from utils.download_utils import download_llm_model_from_s3


# Lazy-load cache
_qa_pipeline = None

def get_qa_pipeline():
    """
    Initialize a system that finds similar documents when asked and causes LLM to generate answers
    """
    global _qa_pipeline
    if _qa_pipeline is not None:
        return _qa_pipeline
    
    # Set the vector DB path
    CHROMA_PATH = os.path.abspath(os.getenv("CHROMA_PATH", "./chroma_db"))
    # Embedding model path
    EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    # LLM model path
    LLM_MODEL_PATH = "/tmp/tinyllama_model"
    HF_CACHE = "/tmp/hf_cache"

    S3_REGION = os.getenv("AWS_REGION")
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
    S3_MODEL_PATH = "models/tinyllama_model/"

    REQUIRED_FILES = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "generation_config.json",
        "model.safetensors"
    ]
    
    print(f"[DEBUG] Downloading + Loading RAG pipeline...")
    download_llm_model_from_s3(
        S3_REGION=S3_REGION,
        S3_BUCKET_NAME=S3_BUCKET_NAME,
        s3_model_path=S3_MODEL_PATH,
        local_dir=LLM_MODEL_PATH,
        required_files=REQUIRED_FILES
    )

    embedding_function = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, cache_dir=HF_CACHE, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, cache_dir=HF_CACHE)

    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )

    llm = huggingface_pipeline(pipeline=llm_pipeline)
    _qa_pipeline = retrieval_qa.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

    print("QA Pipeline fully loaded.")
    
    return _qa_pipeline

def run_qa(query: str) -> str:
    """
    Create a RAG QA response to a user's question
    """
    qa = get_qa_pipeline()
    response = qa.run(query)

    return response