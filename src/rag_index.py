import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document
from models.common import load_file

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def df_to_docs(df: pd.DataFrame) -> list[Document]:
    """
    Converting row including numeric, categorical, and date types into natural language text
    """
    docs = []

    # Add dataset-level summary at the beginning
    summary_parts = [
        "This dataset appears to contain structured information.",
        f"It has {df.shape[0]} rows and {df.shape[1]} columns.",
        "Here are the columns: " + ", ".join(df.columns)
    ]

    # Add sample row preview
    try:
        preview = df.head(2).to_string(index=False)
        summary_parts.append(f"Here are a few sample rows:\n{preview}")
    except:
        pass

    summary_text = "\n".join(summary_parts)
    docs.append(Document(page_content=summary_text))

    for _, row in df.iterrows():
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
