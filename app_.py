import gradio as gr

import os, json
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from gemini_async import embed

# DOCUMENT IO
def read_documents(filepath):
    
    if filepath is None:
        return "Please select a file"
    
    try:
        loader = PyMuPDFLoader(filepath)
        documents = loader.load()
        
        pages = [page.page_content for page in documents]
        return '\n\n'.join(pages)

    except Exception as e:
        raise ValueError(f"Error loading file: {e}")


def split_text(text, chunk_size=384):
    text_splitter = RecursiveCharacterTextSplitter('\n\n', chunk_size=chunk_size, chunk_overlap=0)
    return text_splitter.split_text(text)


# APIs
def get_embdeddings(text):
    chunks = split_text(text)
    embeddings = embed(chunks)
    return '\n\n'.join([json.dumps(vect) for vect in embeddings])


with gr.Blocks() as app:

    # Components
    file_input = gr.File()
    document_output = gr.Markdown()
    retrieval_output = gr.Markdown()

    # Event Listeners
    file_input.change(fn=read_documents, inputs=file_input, outputs=document_output)
    document_output.change(fn=get_embdeddings, inputs=document_output, outputs=retrieval_output)

app.launch()