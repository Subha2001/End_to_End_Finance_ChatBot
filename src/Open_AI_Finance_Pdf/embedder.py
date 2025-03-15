from sentence_transformers import SentenceTransformer
import pickle

def generate_sentence_transformer_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    """Generates embeddings using Sentence Transformers."""
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts)
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def save_embeddings(embeddings, output_path = r"C:\End_to_End_Finance_ChatBot\data\embeddings.pkl"):
    """Saves embeddings to a file."""
    try:
        with open(output_path, "wb") as f:
            pickle.dump(embeddings, f)
        return True
    except Exception as e:
        print(f"Error saving embeddings: {e}")
        return False

'''
# For Example
if __name__ == "__main__":
    from text_splitter import split_text_with_nltk

    texts = split_text_with_nltk() #Get the texts.

    if texts is None:
        print("Failed to load texts.")
    else:
        embeddings = generate_sentence_transformer_embeddings(texts) #generate the embeddings.

        if embeddings is not None:
            if save_embeddings(embeddings):
                print(f"Embeddings saved")
            else:
                print("Failed to save embeddings.")

        else:
            print("Failed to generate embeddings.")'
            '''
