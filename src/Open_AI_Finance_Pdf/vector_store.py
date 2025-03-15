import faiss
import numpy as np
from dotenv import load_dotenv
from text_splitter import split_text_with_nltk
from embedder import generate_sentence_transformer_embeddings

load_dotenv()

def init_faiss_vector_store():
    """
    Initializes FAISS and creates a vector store from the fixed texts and embeddings.
    """
    try:
        texts = split_text_with_nltk() #Get the texts.
        embeddings = generate_sentence_transformer_embeddings([texts]) # Get the embeddings.
        if texts is None or embeddings is None:
            return None #return none if either is none.

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

'''
# For Example
if __name__ == "__main__":
    
    vector_store = init_faiss_vector_store()

    if vector_store:
        print("FAISS vector store initialized successfully.")
        print(f"Number of texts: {len(vector_store['texts'])}") #Number of texts: 504
        print(f"Index Dimensions: {vector_store['index'].d}") #Index Dimensions: 384
    else:
        print("Failed to initialize FAISS vector store.")
'''