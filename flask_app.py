from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import re

app = Flask(__name__)
CORS(app)

# Clean incoming text
def clean_text(text):
    text = re.sub(r'\bPage\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', '', text)
    text = re.sub(r'\b[A-Z][a-z]+,?\s+\d{4}\b', '', text)
    text = re.sub(r'\bFax.*|University.*|Figure.*|Diagram.*|Table.*|GINA.*|Acknowledgements.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', '', text)  # names like John Smith
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[•*►]', '', text)
    text = re.sub(r'\([^)]*\)', '', text)  # remove content in brackets
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Format into short bullet points
def format_response(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    filtered = [s.strip() for s in sentences if 30 < len(s.strip()) < 160 and not s.strip().endswith(":")]
    bullets = [f"• {s}" for s in filtered[:6]]
    return "\n".join(bullets)

# Embedding setup
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="db_chroma", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        query = data.get('question', '')

        if not query:
            return jsonify({'answer': 'No question received.'})

        docs = retriever.get_relevant_documents(query)
        if not docs:
            return jsonify({'answer': 'Sorry, no relevant information found for your query.'})

        all_cleaned_text = ""
        for doc in docs:
            raw = doc.page_content.strip().replace("\n", " ")
            cleaned = clean_text(raw)
            all_cleaned_text += " " + cleaned

        final_answer = format_response(all_cleaned_text)
        return jsonify({'answer': final_answer})

    except Exception as e:
        return jsonify({'answer': f'Error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
