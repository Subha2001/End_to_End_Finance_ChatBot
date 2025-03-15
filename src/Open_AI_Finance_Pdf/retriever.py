import numpy as np
from embedder import generate_sentence_transformer_embeddings  # Assuming you have your embedder.py for generating embeddings

def retrieve_query(vector_store, query, k=4):
    """
    Retrieves the top 'k' documents from the FAISS vector store based on cosine similarity.

    Args:
        vector_store: The FAISS vector store dictionary (containing 'index' and 'texts').
        query: The user's query string.
        k: The number of top results to retrieve (default: 4).

    Returns:
        A list of matching text results, or None if an error occurs.
    """
    try:
        embeddings = generate_sentence_transformer_embeddings([query])
        D, I = vector_store['index'].search(np.array(embeddings).astype('float32'), k)
        matching_results = [vector_store['texts'][i] for i in I[0]]
        return matching_results
    except Exception as e:
        print(f"Error retrieving query results: {e}")
        return None

'''
# For Example
if __name__ == "__main__":
    from vector_store import init_faiss_vector_store
    # Initialize the FAISS vector store (replace with your actual initialization logic)
    vector_store = init_faiss_vector_store()  # Call the init_faiss_vector_store function

    if vector_store: #Check if the vector store was created.
        query = "What is a trading account?"  # Example query
        results = retrieve_query(vector_store, query)  # Retrieve results

        if results:
            print("Retrieved Results:")
            for result in results:
                print(f"- {result}")
        else:
            print("No matching results found.")
    else:
        print("Failed to initialize vector store")'
        '''