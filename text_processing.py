import fitz  # PyMuPDF

def pdf_to_text(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path (str): The path to the PDF file.
    
    Returns:
        str: The extracted text from the PDF.
    """
    # Open the PDF file
    document = fitz.open(pdf_path)
    
    # Initialize a variable to store the text
    text = ""
    
    # Iterate over each page in the PDF
    for page_num in range(len(document)):
        # Get the page
        page = document.load_page(page_num)
        
        # Extract text from the page
        page_text = page.get_text()
        
        # Append the text to the variable
        text += page_text
    
    return text

# Example usage:
# pdf_path = "path_to_your_pdf_file.pdf"
# extracted_text = pdf_to_text(pdf_path)
# print(extracted_text)
