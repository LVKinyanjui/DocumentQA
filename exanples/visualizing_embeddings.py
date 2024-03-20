# %% [markdown]
# ## Visualizing Embeddings

# %% [markdown]
# ### umap
# A model that fits a manifold to data in order to project it into two dimensions.

# %%
import umap.umap_ as umap
import numpy as np
from tqdm import tqdm

# %% [markdown]
# ### Load Embeddings
# We now retrieve all the embeddings from an index and namespace in our vectorstore.

# %%
from pinecone import Pinecone
import google.generativeai as genai
import os

# %%
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'), environment='gcp-starter')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# %%
index_name = pc.list_indexes()[0]['name']
index = pc.Index(index_name)

# %%
query = "What is life?"
query_vector = genai.embed_content('models/embedding-001', query)['embedding']

# %%
res = index.query(top_k=20, vector=query_vector, include_values=True)

# %% [markdown]
# ### UMAP

# %%
embeddings = [result['values'] for result in res['matches']]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)

# %%
def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings),2))
    for i, embedding in enumerate(tqdm(embeddings)): 
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings  

# %%
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

# %% [markdown]
# ### Plot

# %%
import matplotlib.pyplot as plt

# %%
query_embedding = query_vector
retrieved_embeddings = embeddings

projected_query_embedding = project_embeddings([query_embedding], umap_transform)
projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)

# %%
# Plot the projected query and retrieved documents in the embedding space
plt.figure()
plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10, color='gray')
plt.scatter(projected_query_embedding[:, 0], projected_query_embedding[:, 1], s=150, marker='X', color='r')
plt.scatter(projected_retrieved_embeddings[:, 0], projected_retrieved_embeddings[:, 1], s=100, facecolors='none', edgecolors='g')

plt.gca().set_aspect('equal', 'datalim')
plt.title(f'{query}')
plt.axis('off')

# %% [markdown]
# ### Conclusions
# We have succesfully visualized our query vector along with its retrieved embeddings.


