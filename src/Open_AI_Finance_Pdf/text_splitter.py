import nltk
from PyPDF2 import PdfReader

def split_text_with_nltk(text, chunk_size=1000, chunk_overlap=200):
    """Splits text into chunks using nltk, handling potential errors."""
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

if __name__ == "__main__":
    pdf_path = "C:\End_to_End_Finance_ChatBot\data\ICICI-direct-FAQ.pdf"  # Replace with the path to your PDF file
    try:
        nltk.download('punkt')  # Download the sentence tokenizer model
        with open(pdf_path, 'rb') as file: #open the file in read binary mode.
            reader = PdfReader(file)
            pdf_text = ""
            for page in reader.pages:
                pdf_text += page.extract_text() or "" #extract_text can return none.

        if pdf_text:
            chunks = split_text_with_nltk(pdf_text)
            if chunks:
                print(f"Split into {len(chunks)} chunks.")
                print(chunks[0][:100])
            else:
                print("Text splitting failed.")
        else:
            print("PDF loading failed.")

    except Exception as outer_e:
        print(f"An outer error occurred: {outer_e}")