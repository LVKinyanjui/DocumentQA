import gradio as gr

import os, json, re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from gemini_async import embed
from pinecone_client import batch_upsert

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
    
def clean_filename(pathname):
    """Extracts the filename from a given path without artifacts."""
    filename = os.path.basename(pathname)
    filename = re.sub(r'[^a-zA-Z0-9]', '', filename)
    filename.lower()
    return filename


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
    
def get_upsert_embeddings(text, exit_status, filepath=None):
    if exit_status == '0':      # Everything's fine
        chunks = split_text(text)
        embeddings, time_taken = embed(chunks)
        snippet = json.dumps(embeddings[0])

        if filepath is not None:
            namespace = clean_filename(filepath)
        else:
            raise ValueError("Filepath is None. Cannot get value from file input component.")

        upsert_results = batch_upsert(chunks, embeddings, namespace)

        return snippet, time_taken, str(upsert_results)
    else:
        return "No embeddings to display", "Execution time 0 seconds", str(None)


with gr.Blocks() as demo:   #Named demo to allow easy debug mode with $gradio app.py

    # Components
    file_input = gr.File()
    file_status = gr.Markdown(visible=False)

    with gr.Row(equal_height=True):
        duration_output = gr.Markdown()
        upsert_output = gr.Markdown()

    with gr.Row(equal_height=True):
        document_output = gr.Textbox(label="Document Contents")
        embedding_output = gr.Textbox(label="Embeddings")

    # Event Listeners
    file_input.change(fn=read_documents, inputs=file_input, outputs=[document_output, file_status])
    # document_output.change(fn=get_embdeddings, inputs=[document_output, file_status], outputs=[embedding_output, duration_output])
    document_output.change(fn=get_upsert_embeddings, inputs=[document_output, file_status, file_input], outputs=[embedding_output, duration_output, upsert_output])

demo.launch()