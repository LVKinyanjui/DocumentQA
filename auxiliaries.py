import os
import re
import json
import uuid
import time
import requests
import itertools
from tqdm import tqdm

from pinecone import Pinecone
import google.generativeai as genai

import time
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

genai_key = 'AIzaSyAv775lnDC5XMibOJgMntsfR7MouNYxpUU'
# os.getenv("genai_API_KEY")
pinecone_key = '2face206-ee83-4167-bc38-c6f319ebb8c6'
# os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_key)

genai.configure(api_key=genai_key)

def read_split_pdf(file, chunk_size=512, chunk_overlap=0):
    start_time = time.time()

    loader = PyPDFLoader(file)
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
        filename.lower()
        return filename

    namespace = clean_filename(filepath)
    
    # Ensure index available (Any)
    indexes = pc.list_indexes()
    if len(indexes) > 0:
        index_name = indexes[0]['name']
    else:
         pc.create_index('general', dimension=768)

    index = pc.Index(index_name)                          # Initialize index

    # Check if Namespace already availabe, if so terminate.
    stats = index.describe_index_stats()
    if namespace in stats['namespaces'].keys():
         return ["File already Present in Database. Ask Away!", namespace]

    index.delete(delete_all=True, namespace=namespace)          # Delete namespace if exists

    for document in tqdm(documents):

        embedding = genai.generate_embeddings(model='models/embedding-gecko-001', text=document.page_content)

        vector = [{
            'id': str(uuid.uuid4().int),
            'values': embedding['embedding'],
            'metadata': {
                'text': document.page_content
            }
        }]

        index.upsert(vectors=vector, namespace=namespace)       # Upsert to target namespace

        if verbose:
            print(index.describe_index_stats())
    
    return ["File embedded and upserted succesfully!", namespace] # Return to multiple gradio components


def retrieve(query, history, namespace='', temperature=0.0, verbose=False):

        # Ensure index available (Any)
        indexes = pc.list_indexes()
        if len(indexes) > 0:
            index_name = indexes[0]['name']
            index = pc.Index(index_name)
        else:
            raise NameError(f"The index {index_name} does not exist. Please make sure the index is present before attempting to connect to it.") 

        # Check for availability of vectors
        try:
            stats = index.describe_index_stats()
            vector_count = stats['namespaces'][namespace]['vector_count']

            for retry in range(3):
                if vector_count > 0:
                    # Index namespace populated; safe to begin querying
                    break
                else:
                    time.sleep(10)
                    continue
        except KeyError:
             print(f"Unable to retrieve index stats for {namespace}")

        xq = genai.generate_embeddings(model='models/embedding-gecko-001', text=query)

        res = index.query(
             top_k=5,
             vector=xq['embedding'], 
             include_metadata=True, 
             namespace=namespace
             )
        
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
        # res = genai.chat(prompt=message, temperature=temperature)

        # if verbose:
        #         print(context)
        # return res.last

        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(message, stream=True)

        partial_message = ""
        for chunk in response:
            partial_message = partial_message + chunk.text
            yield partial_message


# # Batch Upsert
# for batch in batch_upsert(vectors, batch_size=100):
#     index.upsert(vectors=batch, namespace=clean_filename(filepath)) 
