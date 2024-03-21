from pinecone import Pinecone
import os, uuid

from gemini_async import embed

api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key, environment='gcp-starter')

def get_index():

    indexes = pc.list_indexes()
    index_name = indexes[0]['name']
    index = pc.Index(index_name)
    return index


def get_namespaces():

    index = get_index()
    stats = index.describe_index_stats()
    namespace_keys = stats['namespaces'].keys()
    return list(namespace_keys)


def detect_namespace(namespace):

    return namespace in get_namespaces()


def batch_upsert(dense_vectors, namespace, batch_size=100):

    index = get_index()

    records = []
    for dense_vector in dense_vectors:
        
        record = {
            'id': str(uuid.uuid4().int),
            'values': dense_vector['embeddings']['embedding']['values'],
            'metadata': {
                'text': dense_vector['text_metadata']
            }
        }

        records.append(record)

    # Asynchronous upsert: Faster
    def chunker(seq, batch_size):
        return (seq[pos:pos + batch_size] for pos in range(0, len(seq), batch_size))

    async_results = [
    index.upsert(vectors=chunk, namespace=namespace, async_req=True)
    for chunk in chunker(records, batch_size=batch_size)
    ]

    return "File uploaded to Database"


def retrieve(query_vector, namespace, top_k=20):

    index = get_index()

    res = index.query(
        top_k=top_k,
        vector=query_vector[0]['embeddings']['embedding']['values'],
        include_metadata=True,
        namespace=namespace
    )
    
    # return '\n\n'.join([match['metadata']['text'] for match in res['matches']])

    documents = [match['metadata']['text'] for match in res['matches']]

    # Remove duplicates the retrieved documents
    unique_documents = set()
    for document in documents:
        unique_documents.add(document)

    return list(unique_documents)


if __name__ == "__main__":

    # Pick a random namespace to test
    namespace = get_namespaces()[0]

    # Query a document
    query_vector, _ = embed("What is life?")

    res = retrieve(query_vector, namespace)
    print(res)
    print(type(res))
