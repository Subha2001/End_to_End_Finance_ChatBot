import os
import faiss
import numpy as np
import nltk
import embedder
import pdf_loader
import text_splitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

def init_faiss_vector_store(texts, embeddings):
    """
    Initializes FAISS and creates a vector store from the provided texts and embeddings.
    """
    try:
        # Convert embeddings to a format suitable for FAISS
        embeddings_array = np.array(embeddings).astype('float32')

        # Initialize FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        # Create a mapping between texts and embeddings
        vector_store = {'index': index, 'texts': texts}
        return vector_store
    except Exception as e:
        print(f"Error creating FAISS vector store: {e}")
        return None

''''def retrieve_query(vector_store, query, k=2):
    """
    Retrieves the top 'k' documents from the vector store based on cosine similarity.
    """
    try:
        embeddings = generate_sentence_transformer_embeddings([query])
        D, I = vector_store['index'].search(np.array(embeddings).astype('float32'), k)
        matching_results = [vector_store['texts'][i] for i in I[0]]
        return matching_results
    except Exception as e:
        print(f"Error retrieving query results: {e}")
        return None'''

if __name__ == "__main__":
    pdf_path = r"C:\End_to_End_Finance_ChatBot\data\ICICI-direct-FAQ.pdf"
    output_path = r"C:\End_to_End_Finance_ChatBot\data\Embeddings\embeddings.pkl"

    try:
        text = pdf_loader.load_pdf_with_pypdf2(pdf_path)

        if text:
            splitted_texts = text_splitter.split_text_with_nltk(text)

            if splitted_texts:
                vectors = embedder.generate_sentence_transformer_embeddings(splitted_texts)

                if vectors is not None:
                    if embedder.save_embeddings(vectors, output_path):
                        print("Sentence Transformer embeddings generated and saved.")
                    else:
                        print("Embeddings generation successful, but saving failed.")

                    # Connect to existing FAISS vector store
                    vector_store = init_faiss_vector_store(splitted_texts, vectors)

                    if vector_store:
                        print("Connected to existing FAISS vector store and populated.")
                    else:
                        print("Failed to connect to FAISS vector store.")
                else:
                    print("Embeddings generation failed.")
            else:
                print("Text splitting failed.")
        else:
            print("PDF loading failed.")

    except Exception as outer_e:
        print(f"An outer error occurred: {outer_e}")
