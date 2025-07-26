from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import re

def clean_text(text):
   
    text = re.sub(r'\s+', ' ', text)


    text = re.sub(r'\bPage\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b', '', text)  # phone numbers
    text = re.sub(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', '', text)   # dates
    text = re.sub(r'\b[A-Z][a-z]+,?\s+\d{4}\b', '', text)             # e.g., February 2012
    text = re.sub(r'http\S+', '', text)                               # URLs

    
    text = re.sub(r'[•*►]', '', text)

    return text.strip()

def main():
    load_dotenv()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma(
        persist_directory="db_chroma",
        embedding_function=embeddings
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})

    print("\n RAG chatbot ready. Type your question below ('exit' to quit):")
    print(f" Total chunks in DB: {db._collection.count()}")

    while True:
        query = input("\n Question : ").strip()
        if query.lower() in ["exit", "quit"]:
            print(" Goodbye!")
            break

        try:
            docs = retriever.invoke(query)

            if docs:
                print("\n BOT (short, clean, sequenced):\n")
                for i, doc in enumerate(docs, start=1):
                    content = doc.page_content.strip().replace("\n", " ")
                    content = clean_text(content)
                    content = content[:300]  
                    print(f"{i}. {content}...\n")
            else:
                print("\n BOT: Sorry, no relevant information found for your query.\n")

        except Exception as e:
            print(f"\n Error: {str(e)}")

if __name__ == "__main__":
    main()
