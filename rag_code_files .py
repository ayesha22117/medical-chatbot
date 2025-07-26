import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
PDF_FOLDER = r"E:\chatbot\project files"
CHROMA_PATH = "db_chroma"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize embeddings (no API needed)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# PDF loader
def load_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return pages
    except Exception as e:
        print(f"Error with PyPDFLoader, trying UnstructuredPDFLoader: {e}")
        try:
            loader = UnstructuredPDFLoader(file_path)
            pages = loader.load()
            return pages
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            return []

# Gather all PDF files
pdf_files = [os.path.join(PDF_FOLDER, f) for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
print(f"\n Found {len(pdf_files)} PDFs to process\n")

# Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

all_chunks = []
total_pages = 0

for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
    pages = load_pdf(pdf_file)
    if pages:
        print(f" {os.path.basename(pdf_file)}: {len(pages)} pages loaded")
        total_pages += len(pages)

        chunks = text_splitter.split_documents(pages)
        print(f" {os.path.basename(pdf_file)}: {len(chunks)} chunks created\n")

        all_chunks.extend(chunks)
    else:
        print(f" Skipped {os.path.basename(pdf_file)} (no pages loaded)\n")

print(f"\n Total pages loaded: {total_pages}")
print(f" Total chunks created: {len(all_chunks)} from {len(pdf_files)} PDFs\n")

# Create and save Chroma DB
if os.path.exists(CHROMA_PATH):
    print(" Loading existing ChromaDB")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
else:
    print(" Creating new ChromaDB and saving embeddings...")
    db = Chroma.from_documents(documents=all_chunks, embedding=embeddings, persist_directory=CHROMA_PATH)

print("\n Vector store ready for your RAG chatbot!")
