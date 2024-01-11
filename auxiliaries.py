import os
import re
import json
import uuid
import requests
import itertools
from tqdm import tqdm

import pinecone
import google.generativeai as palm

import time
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

palm_key = 'AIzaSyAv775lnDC5XMibOJgMntsfR7MouNYxpUU'
# os.getenv("PALM_API_KEY")
pinecone_key = '2face206-ee83-4167-bc38-c6f319ebb8c6'
# os.getenv("PINECONE_API_KEY")

palm.configure(api_key=palm_key)

def read_split_pdf(file, chunk_size=512, chunk_overlap=0):
    start_time = time.time()

    loader = PyPDFLoader(file)
    # pages = loader.load_and_split()

    # contents = [page.page_content for page in pages]
    # text = ' '.join(contents)
    # return text

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Execution time in seconds: {total_time}")

    return texts

def upsert(records, api_key):
    request_url = "https://url-from-console.svc.us-east1-gcp.pinecone.io/vectors/upsert"
    response = requests.post(request_url,
                                headers={
                                    'Api-Key': api_key,
                                    'accept': 'application/json',
                                    'content-type': 'application/json'
                                },
                                data=json.dumps(records))


    if response.status_code == 200:
        print("Documents upserted succesfully!")


def batch_upsert(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


def embed_upsert(filepath, verbose=False):
    """
    Takes a read-in file gets its embeddings and upserts the embeddings to pinecone vector database
    """
    documents = read_split_pdf(filepath)

    def clean_filename(pathname):
        filename = os.path.basename(pathname)
        filename = re.sub(r'[^a-zA-Z0-9]', '', filename)
        return filename

    namespace = clean_filename(filepath)

    pinecone.init(
        api_key=pinecone_key,
        environment='gcp-starter'
        )
    
    index = pinecone.Index(pinecone.list_indexes()[0]) 

    for document in tqdm(documents):

        embedding = palm.generate_embeddings(model='models/embedding-gecko-001', text=document.page_content)

        vector = [{
            'id': str(uuid.uuid4().int),
            'values': embedding['embedding'],
            'metadata': {
                'text': document.page_content
            }
        }]

        index.upsert(vectors=vector, namespace=namespace)

        if verbose:
            print(index.describe_index_stats())
    
    return ["File embedded and upserted succesfully!", namespace] # Return to multiple components


def retrieve(query, history, namespace='', verbose=False):
        pinecone.init(
                api_key=pinecone_key,
                environment='gcp-starter'
                )

        index = pinecone.GRPCIndex(pinecone.list_indexes()[0])
        index.describe_index_stats()

        xq = palm.generate_embeddings(model='models/embedding-gecko-001', text=query)
        res = index.query(xq['embedding'], top_k=5, include_metadata=True, namespace=namespace)
        context = '\n\n'.join([match['metadata']['text'] for match in res['matches']])


        prompt_template = """
                You are a consultant \
                Your job is to ingest data from documents and come up with concise ways of expressing it \
                In other words, you will summarize documents for audiences \
                You will target a novive audience who may not have prior knowledge of the text \
                Whenever you do not have enough information to summarize say explicitly:
                "I do not have enough information"
                Otherwise provide a well annotated summary \
                If there are points use numbered or bulleted lists \
                Highlight important points \
                Provide an introduction and conclusion whenever necessary \
                
                You are provided with the user query in follow up messages
        """

        message = f"""
                {prompt_template}

                {context}

                Given the above context, answer the following question to the best of your ability: \

                {xq}
        """
        res = palm.chat(prompt=message)

        if verbose:
                print(context)
        return res.last


# # Batch Upsert
# for batch in batch_upsert(vectors, batch_size=100):
#     index.upsert(vectors=batch, namespace=clean_filename(filepath)) 
