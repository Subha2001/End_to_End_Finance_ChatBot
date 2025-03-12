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

def save_embeddings(embeddings, file_path):
    """Saves embeddings to a file."""
    try:
        with open(file_path, "wb") as f:
            pickle.dump(embeddings, f)
        return True
    except Exception as e:
        print(f"Error saving embeddings: {e}")
        return False

if __name__ == "__main__":
    from pdf_loader import load_pdf_with_pypdf2
    from text_splitter import split_text_with_nltk

    pdf_path = r"C:\End_to_End_Finance_ChatBot\data\ICICI-direct-FAQ.pdf"
    output_path = r"C:\End_to_End_Finance_ChatBot\data\Embeddings\embeddings.pkl"

    try:
        text = load_pdf_with_pypdf2(pdf_path)
        if text:
            splitted_texts = split_text_with_nltk(text)
            if splitted_texts:
                vectors = generate_sentence_transformer_embeddings(splitted_texts)
                if vectors is not None:
                    if save_embeddings(vectors, output_path):
                        print("Sentence Transformer embeddings generated and saved.")
                    else:
                        print("Embeddings generation successful, but saving failed.")
                else:
                    print("Embeddings generation failed.")
            else:
                print("Text splitting failed.")
        else:
            print("PDF loading failed.")

    except Exception as outer_e:
        print(f"An outer error occurred: {outer_e}")