from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
import os

def setup_rag_chain(chroma_path):
    load_dotenv()

    # Initialize Hugging Face embeddings (no API required)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load Chroma vector store
    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings
    )

    # Use retriever directly
    retriever = db.as_retriever()

    return retriever
