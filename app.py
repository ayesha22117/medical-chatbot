from query_handler import QueryHandler
import gradio as gr  # or streamlit/flask for web interface

def initialize_chatbot():
    return QueryHandler()

def chat_interface(query, history):
    handler = initialize_chatbot()
    response = handler.process_query(query)
    
    if "error" in response:
        return response["error"]
    
    answer = response["answer"]
    sources = "\n".join(f"• {src}" for src in response["sources"])
    
    return f"{answer}\n\nSources:\n{sources}"

# Gradio interface
iface = gr.ChatInterface(
    fn=chat_interface,
    title="PDF Chatbot",
    description="Ask questions about your PDF documents"
)

if __name__ == "__main__":
    iface.launch()