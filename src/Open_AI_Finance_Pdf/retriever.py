import numpy as np
from embedder import generate_sentence_transformer_embeddings  # Assuming you have your embedder.py for generating embeddings

def retrieve_query(vector_store, query, k=2):
    """
    Retrieves the top 'k' documents from the FAISS vector store based on cosine similarity.

    Args:
        vector_store: The FAISS vector store dictionary (containing 'index' and 'texts').
        query: The user's query string.
        k: The number of top results to retrieve (default: 2).

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

# Example Usage (optional):
if __name__ == "__main__":
    # Assuming you have already created a vector_store using vector_store.py
    # and have a query string
    from vector_store import init_faiss_vector_store
    from pdf_loader import load_pdf_with_pypdf2
    from text_splitter import split_text_with_nltk
    import nltk
    import os
    from dotenv import load_dotenv

    load_dotenv()

    pdf_path = r"C:\End_to_End_Finance_ChatBot\data\ICICI-direct-FAQ.pdf"
    text = load_pdf_with_pypdf2(pdf_path)
    splitted_texts = split_text_with_nltk(text)
    vectors = generate_sentence_transformer_embeddings(splitted_texts)

    vector_store = init_faiss_vector_store(splitted_texts, vectors)

    if vector_store:
        query = "What is the process to open a trading account?"
        results = retrieve_query(vector_store, query, k=2)

        if results:
            print("Retrieved Results:")
            for result in results:
                print(f"- {result}")
        else:
            print("No results found.")
    else:
        print("Vector store not initialized correctly.")