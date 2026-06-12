import os
from pypdf import PdfReader

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an in-memory PDF file object."""
    text = ""
    try:
        # PdfReader can read directly from the file-like object
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        print(f"Error reading PDF {uploaded_file.name}: {e}")
    return text

def extract_text_from_txt(uploaded_file):
    """Extracts text from an in-memory text/markdown file object."""
    try:
        # Read the bytes and decode to a UTF-8 string
        return uploaded_file.read().decode('utf-8')
    except Exception as e:
        print(f"Error reading TXT {uploaded_file.name}: {e}")
        return ""

def load_course_materials(uploaded_files):
    """
    Parses a list of Streamlit UploadedFile objects.
    Returns a dictionary: { "filename": "extracted_text" }
    """
    documents = {}
    
    for file in uploaded_files:
        # Handle PDFs
        if file.name.endswith('.pdf'):
            documents[file.name] = extract_text_from_pdf(file)
            
        # Handle Markdown, Text, and CSV
        elif file.name.endswith(('.md', '.txt', '.csv')):
            documents[file.name] = extract_text_from_txt(file)
            
        # [YOUR EXTENSION HERE] - e.g., docx, pptx parsing
            
    return documents

def chunk_text(text, chunk_size=1000):
    """
    A basic utility to split large texts into smaller chunks for the agent.
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]