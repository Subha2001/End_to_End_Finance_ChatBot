from PyPDF2 import PdfReader
#from exception import CustomException
#from logger import logging

def load_pdf_with_pypdf2(file_path):
    """Loads text from a PDF document using PyPDF2."""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or "" #extract_text can return none.
            #logging.info('Data loaded from the PDF file')
            return text
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    
    except Exception as e:
        print(f"An error occurred while loading the PDF: {e}")
        return None

if __name__ == "__main__":
    pdf_path = "C:\End_to_End_Finance_ChatBot\data\ICICI-direct-FAQ.pdf"  # Replace with the path to your PDF file
    pdf_text = load_pdf_with_pypdf2(pdf_path)

    if pdf_text:
        print("PDF loaded successfully. Here is the first 200 characters:")
        print(pdf_text[:200])  # Print the first 200 characters of the extracted text
    else:
        print("PDF loading failed.")