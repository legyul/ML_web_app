import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import huggingface_pipeline
from langchain.chains import retrieval_qa
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

load_dotenv()

# Lazy-load cache
_qa_pipeline = None

def get_qa_pipeline():
    """
    Initialize a system that finds similar documents when asked and causes LLM to generate answers
    """
    global _qa_pipeline
    if _qa_pipeline is not None:
        return _qa_pipeline
    
    print("[DEBUG] Loading RAG pipeline (no model download)")

    # Set the vector DB path
    CHROMA_PATH = os.path.abspath(os.getenv("CHROMA_PATH", "./chroma_db"))
    # Embedding model path
    EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    # LLM model path (already downloaded)
    LLM_MODEL_PATH = "/tmp/tinyllama_model"
    HF_CACHE = "/tmp/hf_cache"

    # Embedding & Vector DB
    embedding_function = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Load tokenizer & model only
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, cache_dir=HF_CACHE, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        cache_dir=HF_CACHE,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True
    ).to("cpu")

    # Pipeline Configuration
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )

    # LLM Rapper to be used in lanchain
    llm = huggingface_pipeline(pipeline=llm_pipeline)

    # Create QA Pipeline
    _qa_pipeline = retrieval_qa.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever()
    )

    print("✅ QA Pipeline fully loaded.")
    return _qa_pipeline

def run_qa(query: str) -> str:
    """
    Create a RAG QA response to a user's question
    """
    qa = get_qa_pipeline()
    response = qa.run(query)
    return response