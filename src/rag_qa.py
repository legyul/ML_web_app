import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import huggingface_pipeline
from langchain_community.chains import retrieval_qa
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Set the vector DB path
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

# Embedding model path
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM model path
LLM_MODEL_PATH = "/tmp/tinyllama_model"

def load_qa_pipeline():
    """
    Initialize a system that finds similar documents when asked and causes LLM to generate answers
    """

    # Import embeddings and vector stores
    embedding_function = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Load LLM
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH)

    # Configuring Generation Pipeline
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )

    # Retrieval QA Chain Configuration
    llm = huggingface_pipeline(pipeline=llm_pipeline)
    qa = retrieval_qa.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

    return qa

def run_qa(query: str) -> str:
    """
    Create a RAG QA response to a user's question
    """
    qa = load_qa_pipeline()
    response = qa.run(query)

    return response