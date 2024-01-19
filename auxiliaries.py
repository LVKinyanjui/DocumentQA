import os
import re
import uuid
import time
from tqdm import tqdm

from pinecone import Pinecone
import google.generativeai as genai

import time
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

genai_key = 'AIzaSyAv775lnDC5XMibOJgMntsfR7MouNYxpUU'
# os.getenv("genai_API_KEY")
pinecone_key = '2face206-ee83-4167-bc38-c6f319ebb8c6'
# os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_key)

genai.configure(api_key=genai_key)



def summarize(filepath, chunk_size=16000, api_call_limit=20, verbose=True):
    """
    Triggered by change event on file upload,
        to summarize file contents
    """

    loader = PyMuPDFLoader(filepath)
    documents = loader.load()
    docs = "\n\n".join([document.page_content for document in documents])

    
    llm_calls = 0
    while True:

        if llm_calls > api_call_limit:
            break

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_size//8)
        contexts = text_splitter.split_text(docs)

        responses = []
        for context in contexts:

            prompt = f"""
            I will provide you with text enclosed in triple quote marks (```)
            Your goal is to summarize very briefly, what is contained in the text \
            It should be like the introduction to a book or an article abstract \
            You will inform the user:
                What the text is about in general
                The key points made in the text
                Some key words and terminology, if they stand out

            Here is the text:


            ```
            {context}
            ```

            The output of this will be used for a subsequent summarization.
            """

            model = genai.GenerativeModel('gemini-pro')
            res = model.generate_content(prompt)
            responses.append(res.text)

            llm_calls += 1
            if verbose:
                print(f"LLM called {llm_calls} times")

            time.sleep(1)

        docs = "\n\n".join(responses)

        # Simulates do while loop
        # Critical. If contexts have only one chunk end the inference.
        if len(contexts) <= 1:
            break

    if docs is not None:
        return docs
    else:
        return "API call Limit exceeded. Document May be too long."



def read_split_pdf(file, chunk_size=512, chunk_overlap=0):
    start_time = time.time()

    loader = PyMuPDFLoader(file)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Execution time in seconds: {total_time}")

    return texts

def clean_filename(pathname):
    filename = os.path.basename(pathname)
    filename = re.sub(r'[^a-zA-Z0-9]', '', filename)
    filename.lower()
    return filename


def embed_upsert(filepath, verbose=False):
    """
    Takes a read-in file gets its embeddings and upserts the embeddings to pinecone vector database
    """
    



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
    

    documents = read_split_pdf(filepath)

    # index.delete(delete_all=True, namespace=namespace)          # Delete namespace if exists

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
                If the context section (usually enclosed in triple backticks ```) is empty say: \
                "No Documents were provided to summarize" \
                If there are points use numbered or bulleted lists \
                Highlight important points \
                Provide an introduction and conclusion whenever necessary \
                
                You are provided with the user query in follow up messages
        """

        message = f"""
                {prompt_template}

                ```
                {context}
                ```

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
