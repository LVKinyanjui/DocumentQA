from pinecone import Pinecone
import os, uuid, tqdm

api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key, environment='gcp-starter')

def get_index():

    indexes = pc.list_indexes()
    index_name = indexes[0]['name']
    index = pc.Index(index_name)
    return index


def detect_namespace(namespace):

    index = get_index()
    stats = index.describe_index_stats()
    return namespace in stats['namespaces'].keys()


def batch_upsert(dense_vectors, namespace, batch_size=100):

    index = get_index()

    # Check whether document already exists in vector store
    if detect_namespace(namespace):
         return "File already Present in Database. Ask Away!"


    records = []
    for dense_vector in dense_vectors:
        
        records = [{
            'id': str(uuid.uuid4().int),
            'values': dense_vector['embeddings']['embedding']['values'],
            # 'sparse_values': sparse_vector,
            'metadata': {
                'text': dense_vector['text_metadata']
            }
        }]

        ## Synchronous upsert: Slowwer
        # index.upsert(records, namespace=namespace)

    # Asynchronous upsert: Faster
    def chunker(seq, batch_size):
        return (seq[pos:pos + batch_size] for pos in range(0, len(seq), batch_size))

    async_results = [
    index.upsert(vectors=chunk, namespace=namespace, async_req=True)
    for chunk in chunker(records, batch_size=batch_size)
    ]

    return "File uploaded to Database"


def retrieve(query_vector, namespace):

    index = get_index()

    res = index.query(
        top_k=10,
        vector=query_vector[0]['embeddings']['embedding']['values'],
        include_metadata=True,
        namespace=namespace
    )
    
    return '\n\n'.join([match['metadata']['text'] for match in res['matches']])