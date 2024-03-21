import unittest

from gemini_async import embed

from pinecone_client import retrieve, get_namespaces
from rag_utils import rank_documents_relevance

class TestRankDocumentsRelevance(unittest.TestCase):

    def setUp(cls) -> None:
        query = "What is artificial intelligence or AI"
        cls.query_vector, _ = embed(query)
        cls.namespace = get_namespaces()[0]

    def test_ranking(self):
        # 1. hypothetical question
        query = "What are some pros and cons of adopting a pet dog versus a pet cat?" 

        # 2. Detailed plausible answers
        answers = [
        "Dogs require more physical activity like walks but can be very loyal and trainable. Cats are lower maintenance but can be more aloof.",
        "Dogs tend to be more energetic, requiring lots of exercise, while cats are usually calmer. Dogs are often more affectionate and eager to please their owners. Cats are more independent.",
        "Pros for dogs: Very social, fun to play/exercise with, protective. Cons: Need lots of training, require more daily care, some breeds prone to health issues. Pros for cats: Self-reliant, low maintenance, clean. Cons: Less affectionate, destructive scratching behavior.", 
        "Dogs provide companionship, can be trained to follow commands, and enjoy activities with humans. Cats are self-sufficient, less costly, and don't require as much attention.",
        "Dogs are better for people looking for unconditional love, playing outside, and protection. Cats are better for those wanting an independent pet and don't mind litter boxes."
        ]

        res = rank_documents_relevance(answers, query, top_k=2)

        assert isinstance(res, list)
        assert len(res) > 0


    def test_retrieve_returns_list(cls, self):
        
        
        result = retrieve(cls.query_vector, cls.namespace)
        self.assertIsInstance(result, list)

    def test_retrieve_no_duplicates(cls, self):
         
        
        result = retrieve(cls.query_vector, cls.namespace)
        self.assertEqual(len(result), len(set(result)))

    def test_retrieve_top_k(cls, self):
        
         
        top_k = 2
        result = retrieve(cls.query_vector, cls.namespace, top_k=top_k)
        self.assertLessEqual(len(result), top_k)
    

if __name__ == "__main__":
    unittest.main()

