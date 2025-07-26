from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import re

app = Flask(__name__)
CORS(app)

# Clean incoming text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\bPage\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b', '', text)
    text = re.sub(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', '', text)
    text = re.sub(r'\b[A-Z][a-z]+,?\s+\d{4}\b', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[•*►]', '', text)
    return text.strip()

# Format final bullet points
def format_response(text):
    text = re.sub(r'\bPage\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', '', text)
    text = re.sub(r'\b[A-Z][a-z]+ \d{4}\b', '', text)
    text = re.sub(r'\bFax.*|University.*|Figure.*|Diagram.*|Table.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()

    # Split into bullet-worthy lines
    points = re.split(r'(?<=\d\.)\s+', text)

    # Filter and bullet them
    bullet_points = [f"• {point.strip()}" for point in points if len(point.strip()) > 10]
    return "\n".join(bullet_points[:6])

# Set up embeddings
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

        cleaned_answers = []
        for doc in docs:
            content = doc.page_content.strip().replace("\n", " ")
            content = clean_text(content)
            # Take first 2-3 complete sentences
            sentences = re.split(r'(?<=[.!?])\s+', content)
            snippet = " ".join(sentences[:3])
            cleaned_answers.append(snippet)

        final_answer = format_response(" ".join(cleaned_answers))
        return jsonify({'answer': final_answer})

    except Exception as e:
        return jsonify({'answer': f'Error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
