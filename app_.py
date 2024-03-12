import gradio as gr

import os, json
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from gemini_async import embed

# DOCUMENT IO
def read_documents(filepath):
    
    if filepath is None:
        exit_status = '-1'
        return "Please select a file", exit_status
    
    try:
        loader = PyMuPDFLoader(filepath)
        documents = loader.load()
        
        pages = [page.page_content for page in documents]

        exit_status = '0'
        return '\n\n'.join(pages), exit_status

    except Exception as e:
        raise ValueError(f"Error loading file: {e}")


def split_text(text, chunk_size=384):
    text_splitter = RecursiveCharacterTextSplitter('\n\n', chunk_size=chunk_size, chunk_overlap=0)
    return text_splitter.split_text(text)


# APIs
def get_embdeddings(text, exit_status):
    if exit_status == '0':      # Everything's fine
        chunks = split_text(text)
        embeddings, time_taken = embed(chunks)
        snippet = json.dumps(embeddings[0])
        return snippet, time_taken
    else:
        return "No embeddings to display", "Execution time 0 seconds"


with gr.Blocks() as demo:   #Named demo to allow easy debug mode with $gradio app.py

    # Components
    file_input = gr.File()
    file_status = gr.Markdown(visible=False)
    duration_output = gr.Markdown()
    with gr.Row(equal_height=True):
        document_output = gr.Textbox(label="Document Contents")
        embedding_output = gr.Textbox(label="Embeddings")

    # Event Listeners
    file_input.change(fn=read_documents, inputs=file_input, outputs=[document_output, file_status])
    document_output.change(fn=get_embdeddings, inputs=[document_output, file_status], outputs=[embedding_output, duration_output])

demo.launch()