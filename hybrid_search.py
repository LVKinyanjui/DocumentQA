# %%
import requests

from transformers import BertTokenizerFast


# %%
class HybridPinecone:
    # initializes the HybridPinecone object
    def __init__(self, api_key, environment):
        # make environment, headers and project_id available across all the function within the class
        self.environment = environment
        self.headers = {'Api-Key': api_key}
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
    def query(self, query):
        # sends a post request to hybrib vector index with the query dict
        res = requests.post(
            f"https://{self.host}/hybrid/query",
            headers=self.headers,
            json=query
        )
        # returns the result as json
        return res.json()

    # deletes an index in pinecone database
    def delete_index(self, index_name):
        # sends a delete request
        res = requests.delete(
            f"https://controller.{self.environment}.pinecone.io/databases/{index_name}",
            headers=self.headers
        )
        # returns the http response status
        return res




# load bert tokenizer from huggingface
tokenizer = BertTokenizerFast.from_pretrained(
    'bert-base-uncased'
)

inputs = tokenizer(
    contexts[0], padding=True, truncation=True,
    max_length=512
)
inputs.keys()

# %%
# extract the input ids
input_ids = inputs['input_ids']
input_ids

# %%
from collections import Counter

# convert the input_ids list to a dictionary of key to frequency values
sparse_vec = dict(Counter(input_ids))
sparse_vec

# %% [markdown]
# Let's write a function to do this in batches. Notice that we are removing some keys from the dictionary. These are special tokens from the tokenizer which we do not really need when creating sparse vectors.

# %%
def build_dict(input_batch):
  # store a batch of sparse embeddings
    sparse_emb = []
    # iterate through input batch
    for token_ids in input_batch:
        # convert the input_ids list to a dictionary of key to frequency values
        d = dict(Counter(token_ids))
        # remove special tokens and append sparse vectors to sparse_emb list
        sparse_emb.append({key: d[key] for key in d if key not in [101, 102, 103, 0]})
    # return sparse_emb list
    return sparse_emb

# %% [markdown]
# Let's write another function to help us generate sparse vectors in batches.

# %%
def generate_sparse_vectors(context_batch):
    # create batch of input_ids
    inputs = tokenizer(
            context_batch, padding=True,
            truncation=True,
            max_length=512
    )['input_ids']
    # create sparse dictionaries
    sparse_embeds = build_dict(inputs)
    return sparse_embeds

# %% [markdown]
# ## Dense Vectors

# %% [markdown]
# Alongside sparse vectors we must also add dense vectors (as usual). We do this like so:

# %%
from sentence_transformers import SentenceTransformer

# load a sentence transformer model from huggingface
model = SentenceTransformer(
    'multi-qa-MiniLM-L6-cos-v1',
    device='cuda'
)
model

# %% [markdown]
# The model gives us a `384` dimensional dense vector.

# %% [markdown]
# ## Upsert Documents

# %% [markdown]
# Now we can go ahead and generate sparse and dense vectors for the full dataset and upsert them along with the metadata to the new hybrid index. We can do that easily as follows:

# %%
from tqdm.auto import tqdm

batch_size = 32

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
    dense_embeds = model.encode(context_batch).tolist()
    # create sparse vectors
    sparse_embeds = generate_sparse_vectors(context_batch)

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
    pinecone.upsert(vectors)

# show index description after uploading the documents
pinecone.describe_index_stats()

# %% [markdown]
# ## Querying

# %% [markdown]
# Now we can query the index, providing the sparse and dense vectors of a question, along with a weight for keyword relevance (“alpha”). `Alpha=1` will provide a purely semantic-based search result and `alpha=0` will provide a purely keyword-based result equivalent to BM25. The default value is `0.5`.

# %% [markdown]
# Let's write a helper function to execute queries and after that run some queries.

# %%
def hybrid_query(question, top_k, alpha):
    # convert the question into a sparse vector
    sparse_vec = generate_sparse_vectors([question])
    # convert the question into a dense vector
    dense_vec = model.encode([question]).tolist()
    # set the query parameters to send to pinecone
    query = {
      "topK": top_k,
      "vector": dense_vec,
      "sparseVector": sparse_vec[0],
      "alpha": alpha,
      "includeMetadata": True
    }
    # query pinecone with the query parameters
    result = pinecone.query(query)
    # return search results as json
    return result

# %%
question = "Can clinicians use the PHQ-9 to assess depression in people with vision loss?"

# %% [markdown]
# First, we will do a pure semantic search by setting the alpha value as 1.

# %%
hybrid_query(question, top_k=3, alpha=1)

# %% [markdown]
# The most relevant result from above is the second document with id 711. Now let's try with an alpha value of 0.3.

# %%
hybrid_query(question, top_k=3, alpha=0.3)

# %% [markdown]
# The most relevant document is now ranked the highest.

# %% [markdown]
# # Delete the Index

# %%
pinecone.delete_index("hybrid-test")

# %% [markdown]
# ---


