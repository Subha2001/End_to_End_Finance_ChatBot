import nltk
from nltk.tokenize import PunktSentenceTokenizer
from pdf_loader import load_pdf_with_pypdf2

def split_text_with_nltk(chunk_size=1000, chunk_overlap=200):
    """Splits text from PDF into chunks using nltk, handling potential errors."""
    # Ensure punkt tokenizer is downloaded
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK punkt_tab resource...")
        nltk.download('punkt_tab')

    try:
        nltk.download('punkt')
        print("Download of 'punkt' completed successfully!")
        
        tokenizer = PunktSentenceTokenizer()
        print("'punkt_tab' resource is now generated and ready!")

    except Exception as e:
        print(f"An error occurred: {e}")

    text = load_pdf_with_pypdf2()  # Load the text from the PDF
    if text is None:
        return []  # Return an empty list if the text is None

    try:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                # Add some overlap
                current_chunk = current_chunk[-chunk_overlap:] + sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    except Exception as e:
        print(f"An error occurred while splitting text: {e}")
        return []

'''
# For Example
if __name__ == "__main__":
    chunks = split_text_with_nltk()
    
    if chunks:
        print(f"Number of chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:5]): #print the first 5 chunks. [We are getting 504 chunks]
            print(f"Chunk {i+1}:\n{chunk}\n{'-'*20}")
    else:
        print("Text splitting failed.")'
        '''