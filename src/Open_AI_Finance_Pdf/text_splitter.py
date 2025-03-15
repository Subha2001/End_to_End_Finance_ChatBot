import nltk
from pdf_loader import load_pdf_with_pypdf2

def split_text_with_nltk(chunk_size=1000, chunk_overlap=200):
    """Splits text from PDF into chunks using nltk, handling potential errors."""
    text = load_pdf_with_pypdf2() #load the text from the pdf.
    if text is None:
        return [] #return empty list if the text is none.
    try:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                #add some overlap
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