import chromadb
from sentence_transformers import SentenceTransformer
import openai
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key is not set.")

# Use LLM based on GPT-3.5-turbo
client = openai.Client(api_key=OPENAI_API_KEY)

# Load SentenceTransformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create ChromaDB Client (vector DB)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="ml_data")

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

def ask_gpt_rag(question):
    '''
    RAG-Based GPT-3.5-turbo Call
    '''
    relevant_docs = search_documents(question)
    context = "\n".join(relevant_docs) if relevant_docs else "No relevant data found."

    response = client.chat.completions.creat(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant that answers based on provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content
