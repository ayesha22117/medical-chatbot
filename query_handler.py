from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import re

def clean_text(text):
    # Remove multiple spaces, newlines, phone numbers, page numbers, addresses
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\bPage\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bFax:.*?(?=\s|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b', '', text)  # phone numbers
    text = re.sub(r'http\S+', '', text)  # URLs
    text = re.sub(r'\bFigure\s*\d+.*?(?=\s[A-Z]|\s\d+|$)', '', text, flags=re.IGNORECASE)  # Figure captions
    return text.strip()

def main():
    load_dotenv()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory="db_chroma",
        embedding_function=embeddings
    )
    retriever = db.as_retriever()

    print("\n RAG chatbot ready. Type your question below ('exit' to quit):")
    print(f" Total chunks in DB: {db._collection.count()}")

    while True:
        query = input("\n Question: ").strip()
        if query.lower() in ["exit", "quit"]:
            print(" Goodbye!")
            break

        try:
            docs = retriever.invoke(query)
            if docs:
                for i, doc in enumerate(docs, start=1):
                    content_preview = clean_text(doc.page_content.strip())[:350]
                    print(f"\n{i}. {content_preview}...")
            else:
                print("\n Sorry, no relevant information found for your query.")
        except Exception as e:
            print(f"\n Error: {str(e)}")

if __name__ == "__main__":
    main()
