import os
from pypdf import PdfReader
import re

def clean_text(text):
    """Basic text cleaning."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text_with_metadata(text, metadata, chunk_size=1000, overlap=200):
    """
    Splits text into overlapping chunks and attaches metadata.
    """
    chunks = []
    if not text:
        return chunks
        
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append({
            "text": chunk,
            "metadata": metadata.copy()
        })
        start += (chunk_size - overlap)
        
    return chunks

def extract_chunks_from_pdf(uploaded_file):
    """Extracts chunks from an in-memory PDF file object, tracking pages."""
    chunks = []
    try:
        reader = PdfReader(uploaded_file)
        for i, page in enumerate(reader.pages):
            extracted = page.extract_text()
            if extracted:
                cleaned = clean_text(extracted)
                metadata = {"file": uploaded_file.name, "page": f"Page {i+1}"}
                # Chunk per page to keep page metadata accurate
                page_chunks = chunk_text_with_metadata(cleaned, metadata)
                chunks.extend(page_chunks)
    except Exception as e:
        print(f"Error reading PDF {uploaded_file.name}: {e}")
    return chunks

def extract_chunks_from_txt(uploaded_file):
    """Extracts chunks from an in-memory text/markdown file object."""
    chunks = []
    try:
        text = uploaded_file.read().decode('utf-8')
        cleaned = clean_text(text)
        metadata = {"file": uploaded_file.name, "page": "N/A"}
        chunks = chunk_text_with_metadata(cleaned, metadata)
    except Exception as e:
        print(f"Error reading TXT {uploaded_file.name}: {e}")
    return chunks

def load_course_materials(uploaded_files):
    """
    Parses a list of Streamlit UploadedFile objects.
    Returns a list of chunks: [{"text": "...", "metadata": {"file": "...", "page": "..."}}]
    """
    all_chunks = []
    
    for file in uploaded_files:
        if file.name.endswith('.pdf'):
            all_chunks.extend(extract_chunks_from_pdf(file))
        elif file.name.endswith(('.md', '.txt', '.csv')):
            all_chunks.extend(extract_chunks_from_txt(file))
            
    return all_chunks