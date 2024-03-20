# %% [markdown]
# ### Text Loaders with filtering blank pages
# 

# %%
from pypdf import PdfReader

# %%
reader = PdfReader("../data/E1. ExngTextOnly.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filter the empty strings
pdf_texts = [text for text in pdf_texts if text]

# %% [markdown]
# ### Sentence Splitters

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

# %%
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0
)
character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))

# %%
token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)


# %% [markdown]
# #### Remarks
# `SentenceTransformersTokenTextSplitter` is extremely slow. Unless optimized, it cannot immediately be used in production.

# %%



