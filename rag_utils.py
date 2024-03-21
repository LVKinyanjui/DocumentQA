import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

def expand_with_generation(original_query):
    """Expand the user query by returning a plausible answer with which to retrieve documents"""
    
    hypothetical_answer = model.generate_content(original_query).text
    joint_query = f"{original_query} {hypothetical_answer}"

    query_vector = genai.embed_content('models/embedding-001', joint_query)
    texts, retrieved_embeddings = retrieve(query_vector, 
            index=get_index(),
            namespace=namespace
            )
