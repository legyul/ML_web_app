import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, GPT2LMHeadModel,TextGenerationPipeline, GPT2Config
from lora_train import get_finedtuned_model_path
import json
from peft import PeftModel, PeftConfig
from pathlib import Path

load_dotenv()
# Lazy-load cache
_qa_pipeline = {}


def get_qa_pipeline(filename: str, model_choice: str):
    global _qa_pipeline
    key = f"{filename}_{model_choice}"
    if key in _qa_pipeline:
        return _qa_pipeline[key]

    try:
        print("[DEBUG] Loading RAG pipeline")

        model_path = get_finedtuned_model_path(filename, model_choice)
        tokenizer_path = os.path.join(model_path, "_tokenizer")
        
        if not os.path.isdir(model_path):
            raise ValueError(f"Model path {model_path} is not a directory. Cannot load locally.")
        
        HF_CACHE = "/tmp/hf_cache"

        print("[DEBUG] Loading tokenizer...")
        
        model = GPT2LMHeadModel.from_pretrained( 
            pretrained_model_name_or_path=model_path,
            cache_dir=HF_CACHE,
            local_files_only=True,
            trust_remote_code=True,
            use_safetensors=True
        )
        model.to("cpu")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=HF_CACHE, use_fast=False, local_files_only=True)

        class SimpleTextGenWrapper:
            def __init__(self, pipeline):
                self.pipeline = pipeline
                self.model = pipeline.model
                self.task = pipeline.task or "text-generation"
            
            def __call__(self, prompt, **kwargs):
                outputs = self.pipeline(prompt, **kwargs)
                if isinstance(outputs, list) and "generated_text" in outputs[0]:
                    return outputs[0]["generated_text"]
                return outputs

        llm_pipeline = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )

        wrapped_pipeline = SimpleTextGenWrapper(llm_pipeline)

        EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
        CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
        embedding_function = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        llm = HuggingFacePipeline(pipeline=wrapped_pipeline)
        _qa_pipeline[key] = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

        print("✅ QA Pipeline loaded successfully.")
        return _qa_pipeline[key]

    except Exception as e:
        print(f"❌ Failed to load QA pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def run_qa(query: str, filename: str, model_choice: str) -> str:
    """
    Create a RAG QA response to a user's question
    """
    qa = get_qa_pipeline(filename, model_choice)

    if qa is None:
        return "QA pipeline is not ready"
    
    response = qa.run(query)

    # Respond if LangChain returns to list
    if isinstance(response, list):
        try:
            # Extract text from list format returned by huggingface pipeline
            return response[0]["generated_text"]
        except (KeyError, TypeError):
            return str(response[0])
    
    # Huggingface pipeline is just returned to the string
    elif isinstance(response, str):
        return response.strip()
        
    return str(response)