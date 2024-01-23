# %%
import requests
from tqdm.auto import tqdm

from transformers import BertTokenizerFast              # Sparse Embeddings
from sentence_transformers import SentenceTransformer   # Dense Embeddings
from collections import Counter


# %%
class HybridPinecone:
    # initializes the HybridPinecone object
    def __init__(self, api_key, environment):
        # make environment, headers and project_id available across all the function within the class
        self.environment = environment
        self.headers = {
            'Api-Key': api_key,
            'Content-Type': 'application/json'
            }
        # get project_id
        res = requests.get(
            f"https://controller.{self.environment}.pinecone.io/actions/whoami",
            headers=self.headers
        )
        self.project_id = res.json()['project_name']
        self.host = None

    # creates an index in pinecone vector database
    def create_index(self, index_name, dimension, metric, pod_type):
        # index specification
        params = {
            'name': index_name,
            'dimension': dimension,
            'metric': metric,
            'pod_type': pod_type
        }
        # sent a post request with the headers and parameters to pinecone database
        res = requests.post(
            f"https://controller.{self.environment}.pinecone.io/databases",
            headers=self.headers,
            json=params
        )
        # return the creation status
        if res.status_code == 201:
            print("Index created.")

        return res
    
    # get the project_id for the index and update self.host variable
    def connect_index(self, index_name):
        # set the self.host variable
        self.host = f"{index_name}-{self.project_id}.svc.{self.environment}.pinecone.io"
        res = self.describe_index_stats()
        # return index related information as json
        return res
    
    def describe_index(self, index_name):
        # send a get request to pinecone database to get index description
        res = requests.get(
            f"https://controller.{self.environment}.pinecone.io/databases/{index_name}",
            headers=self.headers
        )
        return res.json()

    # returns description of the index
    def describe_index_stats(self):
        # send a get request to pinecone database to get index description
        res = requests.get(
            f"https://{self.host}/describe_index_stats",
            headers=self.headers
        )
        # return the index description as json
        return res.json()

    # uploads the documents to pinecone database
    def upsert(self, vectors):
        # send a post request with vectors to pinecone database
        res = requests.post(
            f"https://{self.host}/hybrid/vectors/upsert",
            headers=self.headers,
            json={'vectors': vectors}
        )
        # return the http response status
        return res

    # searches pinecone database with the query
    def query(self, top_k, vector, sparse_vector, include_metadata):
        # sends a post request to hybrib vector index with the query dict
        params = {
            "includeValues": True,
            "includeMetadata": include_metadata,
            "vector": vector,
            "sparseVector": sparse_vector,
            "topK": top_k,
            "namespace": ""
        }

        res = requests.post(
            f"https://{self.host}/query",
            headers=self.headers,
            json=params
        )
        # returns the result as json
        if res.status_code == 200:
            return res.json()
        else:
            return res

    # deletes an index in pinecone database
    def delete_index(self, index_name):
        # sends a delete request
        res = requests.delete(
            f"https://controller.{self.environment}.pinecone.io/databases/{index_name}",
            headers=self.headers
        )
        # returns the http response status
        return res
    





    def build_dict(self, input_batch: dict):
        """
        Convert sparse embeddings into proper format required by pinecone
        Parameters:
            input_batch: sparse embeddings as a key and value pair (dictionary)
        """
        sparse_emb = []
        # iterate through input batch
        for token_ids in input_batch:
            indices = []
            values = []
            # convert the input_ids list to a dictionary of key to frequency values
            d = dict(Counter(token_ids))
            for idx in d:
                    indices.append(idx)
                    values.append(float(d[idx]))                        # Extremely important to cast values as float
                                                                        # Otherwise you get: SparseValuesMissingKeysError: Missing required keys in data in column `sparse_values`.
            sparse_emb.append({'indices': indices, 'values': values})
        # return sparse_emb list
        return sparse_emb
    

    def generate_sparse_vectors(self, context_batch: list, tokenizer_name='bert-base-uncased'):
        """
        Generate sparse embeddings in a format suitable for pinecone.
        Parameters:
            contents: texts to be tokenized
            name: pretrained tokenizer, from hugging face
        """
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)     # To be replaced with a higher order function

        # Generate input IDs
        inputs = tokenizer(
                context_batch, padding=True,
                truncation=True,
                max_length=512
        )['input_ids']

        # create sparse dictionaries
        sparse_embeds = self.build_dict(inputs)
        return sparse_embeds


    def generate_dense_vectors(self, context_batch: list, model_name='multi-qa-MiniLM-L6-cos-v1', device='cpu'):
        """
        Generate dense embeddings. Uses a Sentence Transformer model
        Parameters
            context_batch: the texts to get embeddings of
            model_name: model name to use with sentence transformer, hugging face
            device: where to run model on, cpu or cuda
        """
        model = SentenceTransformer(
            model_name,
            device=device                   # or cuda, if available
        )

        dense_embeds = model.encode(context_batch).tolist()
        return dense_embeds
    
    
    def hybrid_embed_upsert(self, contexts: list, batch_size=100):
        """
        Upsert both dense and sparse vectors to the index, after generating embeddings
        Parameters
            contexts: Text to be encoded and upserted
        """

        for i in tqdm(range(0, len(contexts), batch_size)):
            # find end of batch
            i_end = min(i+batch_size, len(contexts))
            # extract batch
            context_batch = contexts[i:i_end]
            # create unique IDs
            ids = [str(x) for x in range(i, i_end)]
            # add context passages as metadata
            meta = [{'context': context} for context in context_batch]
            # create dense vectors
            dense_embeds = self.generate_dense_vectors(context_batch)
            # create sparse vectors
            sparse_embeds = self.generate_sparse_vectors(context_batch)

            vectors = []
            # loop through the data and create dictionaries for uploading documents to pinecone index
            for _id, sparse, dense, metadata in zip(ids, sparse_embeds, dense_embeds, meta):
                vectors.append({
                    'id': _id,
                    'sparse_values': sparse,
                    'values': dense,
                    'metadata': metadata
                })

            # upload the documents to the new hybrid index
            self.upsert(vectors)


    def hybrid_scale(self, dense: list, sparse: list, alpha: float):
        """
        Scale sparse and dense vectors to create hybrid search vecs
            while enusring alpha value is between 0 and 1
        Parameters
            dense: dense vector list
            sparse: sparse vector list
            alpha: a value weighting dense or sparse search
                1, full dense search    0, full sparse search
        """

        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")
        
        # Scale
        hsparse = {
            'indices': sparse['indices'],
            'values':  [v * (1 - alpha) for v in sparse['values']]
        }
        hdense = [v * alpha for v in dense]
        return hdense, hsparse


    def hybrid_query(self, question: str, top_k=5, alpha=0.5):
        """
        Query pinecone index using a sparse and dense vector representation of the question
        Parameters:
            question: string query for pinecone
            top-k: number of results to return from the query
            alpha: a value weighting dense or sparse search
                1, full dense search    0, full sparse search
        """
        # convert the question into a sparse vector
        sparse_vec = self.generate_sparse_vectors([question])[0]
        # convert the question into a dense vector
        dense_vec = self.generate_dense_vectors([question])[0]
        # scale alpha with hybrid_scale
        dense_vec, sparse_vec = self.hybrid_scale(
            dense_vec, sparse_vec, alpha
        )
        # query pinecone with the query parameters
        result = self.query(top_k, dense_vec, sparse_vec, include_metadata=True)
        # return search results as json
        return result
    
# %%
