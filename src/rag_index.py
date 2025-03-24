import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from models.common import load_file

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def df_to_docs(df: pd.DataFrame) -> list[Document]:
    """
    Converting row including numeric, categorical, and date types into natural language text
    """
    docs = []
    for i, row in df.iterrows():
        # Converting column names and their values to natural language text
        content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append(Document(page_content=content))
    
    return docs

def create_vectorstore_from_s3(file_key: str):
    """
    Import the dataset, convert it into a sentence, and embed it in Chroma
    """
    df, _ = load_file(file_key)
    documents = df_to_docs(df)

    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectordb = Chroma.from_documents(documents, embedding=embedding_function, persist_directory=CHROMA_PATH)
    vectordb.persist()

    return vectordb
