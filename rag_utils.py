
from sentence_transformers import CrossEncoder

import numpy as np

from typing import List

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rank_documents_relevance(documents: List[str], original_query: str, top_k: int = 10) -> List[str]:

    pairs: List[List[str]] = [[original_query, doc] for doc in documents]
    scores: np.ndarray = cross_encoder.predict(pairs)

    print("Scores:")
    for score in scores:
        print(score)

    # Reranking
    top_k_idx: List[int] = [o for o in np.argsort(scores)[::-1]][:top_k]

    return [documents[i] for i in top_k_idx]

