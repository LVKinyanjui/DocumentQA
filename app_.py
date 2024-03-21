import gradio as gr

import os, json, re
from pypdf import PdfReader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from gemini_async import embed
from gemini import generate_content
from pinecone_client import batch_upsert, retrieve, detect_namespace

# DOCUMENT IO
def read_documents(filepath):
    
    if filepath is None:
        exit_status = '-1'
        return "Please select a file", exit_status
    
    try:
        loader = PyMuPDFLoader(filepath)
        documents = loader.load()
        pdf_texts = [document.page_content for document in documents]
        
        exit_status = '0'
        return '\n\n'.join(pdf_texts), exit_status

    except Exception as e:
        raise ValueError(f"Error loading file: {e}")
    
def clean_filename(pathname):
    """Extracts the filename from a given path without artifacts."""
    filename = os.path.basename(pathname)
    filename = re.sub(r'[^a-zA-Z0-9]', '', filename)
    filename.lower()
    return filename


def split_text(text, chunk_size=384):
    separators = ['\n\n# ', '\n\n## ', '\n\n### ', '\n\n', '\n', '. ', ', ', '; ', ': ', '? ', '! ']
    text_splitter = RecursiveCharacterTextSplitter(separators, chunk_size=chunk_size, chunk_overlap=0)
    return text_splitter.split_text(text)

    
def get_upsert_embeddings(text, exit_status, filepath=None, chunk_size=384):
    if exit_status == '0':      # Filepath is not none: document uploaded in file component

        if filepath is None:
            raise ValueError("Filepath is None. Cannot get value from file input component.")
        
        namespace = clean_filename(filepath)     
        namespace_exists = detect_namespace(namespace)

        if not namespace_exists:
            chunks = split_text(text, chunk_size=chunk_size)
            embeddings, time_taken = embed(chunks)
            snippet = json.dumps(embeddings[0])
            upsert_results = batch_upsert(dense_vectors=embeddings, 
                                          namespace=namespace)

            return snippet, time_taken, str(upsert_results)
        
        else:
            return str(None), "Execution time 0 seconds", "File already Present in Database. Ask Away!"
    else:
        return "No embeddings to display", "Execution time 0 seconds", str(None)


def visible_component(input_text):
    return gr.update(visible=True)


def chatbot_answer(original_query, history, namespace):
    
    # Expand with generation
    hypothetical_answer = generate_content(original_query)
    joint_query = f"{original_query} {hypothetical_answer}"

    query_vector, _ = embed(joint_query)
    context = retrieve(query_vector, namespace)

    prompt = f"""
        You are a helpfull user assistance
        Given a user Query, summarize the information Context and provide an accurate response.
        
        Do not format your response unnecesarily such as with unnecessary titlea.
        Simply respond to the user's question as if you were having a conversation.

        You are quite capable of this task and I am sure you will be able to generate a worthwhile response.
        Please do your best as this is quite important for the success of this project. 

        Query: {original_query}
        
        Context: {context}

        """
    
    ## Final generation
    return generate_content(prompt)



with gr.Blocks() as demo:   #Named demo to allow easy debug mode with $gradio app.py

    # Components
    file_input = gr.File()
    file_status = gr.Markdown(visible=False)
    file_namespace= gr.Textbox(visible=False)

    with gr.Column(visible=False) as query_input:
        # query_box = gr.Textbox(label="Query Text")
        # query_button = gr.Button("Query")

        gr.ChatInterface(fn=chatbot_answer, additional_inputs=[file_namespace])


    with gr.Row(equal_height=True):
        duration_output = gr.Markdown()
        upsert_output = gr.Markdown()

    with gr.Accordion("See Document Details", open=False):
        with gr.Row(equal_height=True):
            document_output = gr.Textbox(label="Document Contents")
            embedding_output = gr.Textbox(label="Embeddings")

    # Event Listeners
    file_input.change(fn=read_documents, inputs=file_input, outputs=[document_output, file_status])
    file_input.change(fn=clean_filename, inputs=file_input, outputs=file_namespace)
    document_output.change(fn=get_upsert_embeddings, inputs=[document_output, file_status, file_input], outputs=[embedding_output, duration_output, upsert_output])
    upsert_output.change(fn=visible_component, inputs=[upsert_output], outputs=[query_input])

    # gr.update(query_input, visible=True)
demo.launch()