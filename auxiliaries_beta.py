# %%
import os, re, uuid, time
from tqdm import tqdm

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import ConversationChain
from langchain.chat_models import ChatGooglePalm
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

genai_key = 'AIzaSyAv775lnDC5XMibOJgMntsfR7MouNYxpUU'
# os.getenv("genai_API_KEY")
pinecone_key = '18e4d60d-209a-43ed-8e2c-0f3a8d1ffcbb'
# os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_key)

genai.configure(api_key=genai_key)

bm25 = BM25Encoder()

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
            result = model.generate_content(prompt)
            try:
                responses.append(result.text)
            except ValueError:
                # responses.append(result.parts[0])
                print("Prompt response probably blocked")
                pass

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



def read_split_pdf(file, chunk_size=256, chunk_overlap=0):
    start_time = time.time()

    loader = PyMuPDFLoader(file)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)

    # Fit BM2
    doc_texts = [document.page_content for document in docs]       # Type to list[str]
    bm25.fit(corpus=doc_texts)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Execution time in seconds: {total_time}")

    return docs

def clean_filename(pathname):
    filename = os.path.basename(pathname)
    filename = re.sub(r'[^a-zA-Z0-9]', '', filename)
    filename.lower()
    return filename



def embed_upsert(filepath, verbose=False):
    """
    Takes a read-in file gets its embeddings and upserts the embeddings to pinecone vector database
    """
    documents = read_split_pdf(filepath)

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
    
    # index.delete(delete_all=True, namespace=namespace)          # Delete namespace if exists

    # EMBEDDING MODELS

    ## Dense
    model = SentenceTransformer(
        'multi-qa-MiniLM-L6-cos-v1',
        device='cpu'
    )

    for document in tqdm(documents):
        texts = document.page_content

        dense_vector = model.encode(texts).tolist()
        sparse_vector = bm25.encode_documents(texts)
        
        records = [{
            'id': str(uuid.uuid4().int),
            'values': dense_vector,
            'sparse_values': sparse_vector,
            'metadata': {
                'text': texts
            }
        }]

        index.upsert(records, namespace=namespace)

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

        ## Dense
        model = SentenceTransformer(
            'multi-qa-MiniLM-L6-cos-v1',
            device='cpu'
        )


        sparse_vector = bm25.encode_documents(query)
        dense_vector = model.encode(query).tolist()

        # Check for presence of records before querying
        if index.describe_index_stats()['total_vector_count'] == 0:
            print("No Records Found. Query may not retrieve any matches.")

        res = index.query(
            top_k=2,
            vector=dense_vector,
            sparse_vector=sparse_vector,
            include_metadata=True,
            namespace=namespace
        )
        
        context = '\n\n'.join([match['metadata']['text'] for match in res['matches']])


        prompt_template = """
                You are a teacher \
                The human in this case is the student \
                Your job is to create A SINGLE qustion from a textbook context, 
                enclosed in triple backticks ``` \
                You are to take the student's response to your question \
                You will then assess it based on the context
                If it is correct, tell them 'correct', plus any additional feedback \
                If wrong, tell them kindly what the right answer is \
                If the context section (usually enclosed in triple backticks ```) is empty say: \
                "No Documents were provided for question answer" \
                
                The question should be asked simply  in the following format:\
                "
                what is this thing called?
                what does that thing do?
                How do we do this thing
                " etc.

                The student responds with text enclosed in triple hyphens ---


                This is what you will evaluate based on the context
                
                You are provided with the textbook context and the student response below
        """

        message = f"""
                {prompt_template}

                ```
                {context}
                ```
                
                ---
                {query}
                ---

        """

        # Model with memory
        llm = ChatGooglePalm(
            google_api_key=genai_key
        )
        conversation_chain = ConversationChain(
            llm=llm,
            memory=ConversationBufferWindowMemory(k=5)
        )

        result = conversation_chain(message)

        return result['response']

        # gemini = genai.GenerativeModel('gemini-pro')
        # response = gemini.generate_content(message, stream=True)

        # partial_message = ""
        # for chunk in response:
        #     partial_message = partial_message + chunk.text
        #     yield partial_message
