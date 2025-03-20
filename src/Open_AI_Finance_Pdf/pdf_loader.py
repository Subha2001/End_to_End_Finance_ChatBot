from PyPDF2 import PdfReader

def load_pdf_with_pypdf2():
    """Loads text from a fixed PDF document using PyPDF2."""
    file_path = "/app/data/ICICI-direct-FAQ.pdf"  # file path based on Docker
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""  # extract_text can return None.
            return text

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

    except Exception as e:
        print(f"An error occurred while loading the PDF: {e}")
        return None

'''
# For Example
if __name__ == "__main__":
    pdf_text = load_pdf_with_pypdf2()
    if pdf_text:
        print(pdf_text[:500]) #print first 500 characters to test.
    else:
        print("PDF loading failed.")'
        '''