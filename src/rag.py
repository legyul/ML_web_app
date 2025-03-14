import openai
import chromadb
import os
import json
import pickle
import io
import zipfile
import boto3
import torch
import subprocess
import sys  
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Setting Environmental Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key is not set.")

S3_BUCKET_NAME = "ml-platform-service"
S3_REGION = "us-east-2"

# S3 Client
s3 = boto3.client('s3', region_name=S3_REGION, config=boto3.session.Config(signature_version='s3v4'))

# OpenAI Client
client = openai.Client(api_key=OPENAI_API_KEY)

# Load SentenceTransformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create ChromaDB Client (vector DB)
chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
collection = chroma_client.get_or_create_collection(name="ml_data")

os.environ["HF_HOME"] = "/tmp/huggingface"

def load_model_from_s3(zip_s3_key):
    '''
    Download the saved model from S3 and unzip it to load it.
    '''
    try:
        # Download the saved model
        model_obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=zip_s3_key)
        zip_buffer = io.BytesIO(model_obj["Body"].read())

        with zipfile.ZipFile(zip_buffer, "r") as zipf:
            # Load the model information
            model_info_filename = [name for name in zipf.namelist() if name.endswith("_model_info.json")][0]
            model_filename = [name for name in zipf.namelist() if name.endswith("_model.pkl")][0]

            with zipf.open(model_info_filename) as f:
                model_info = json.load(f)
            
            # Install required packages
            required_packages = model_info.get("required_packages", [])
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            
            # Load model
            with zipf.open(model_filename) as f:
                model = pickle.load(f)

        return model
    
    except Exception as e:
        print(f"Error loading model from S3: {e}")
        return None

def add_document(doc_id, text, metadata=None):
    '''
    Vectorize the dataset and store it in ChromaDB
    '''
    embedding = embedding_model.encode(text).tolist()
    collection.add(ids=[doc_id], documents=[text], embeddings=[embedding], metadatas=[metadata or {}])
    print(f"Document is added: {doc_id}")

def search_documents(query, top_k=3):
    '''
    Search for documents most similar to the input question
    '''
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    if results['documents']:
        return results["documents"][0]
    
    return []

def ask_model_rag(question, input_data=None, model_s3_key=None):
    '''
    1. RAG-based search: Find relevant data
    2. GPT-3.5-turbo to generate answers
    3. When new input data is present, use the trained model to make predictions
    '''
    # 1. RAG-based search
    relevant_docs = search_documents(question)
    context = "\n".join(relevant_docs) if relevant_docs else "No relevant data found."

    # 2. Performing model predictions (if the user has provided new data)
    prediction = None
    if model_s3_key and input_data:
        model = load_model_from_s3(model_s3_key)
        if model:
            prediction = model.predict([input_data])[0]
            context += f"\nPredicted Value: {prediction}"
    
    # GPT-3.5-turbo to generate responses
    response = client.chat.completions.creat(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant that answers based on provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )

    final_response = response.choices[0].message.content
    if prediction is not None:
        final_response += f"\n\n **Predicted Value:** {prediction}"

    return final_response
