import re
from pypdf import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed


def clean_text(text):
    """Basic text cleaning."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text_with_metadata(text, metadata, chunk_size=1000, overlap=200):
    """Splits text into overlapping chunks and attaches metadata."""
    chunks = []
    if not text:
        return chunks

    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunks.append({
            "text": text[start:end],
            "metadata": metadata.copy()
        })
        start += (chunk_size - overlap)

    return chunks


def extract_chunks_from_pdf(uploaded_file):
    """Extracts chunks from an in-memory PDF file object, tracking pages."""
    chunks = []
    try:
        reader = PdfReader(uploaded_file)

        def parse_page(args):
            i, page = args
            extracted = page.extract_text()
            if not extracted:
                return []
            cleaned = clean_text(extracted)
            metadata = {"file": uploaded_file.name, "page": f"Page {i + 1}"}
            return chunk_text_with_metadata(cleaned, metadata)

        # Parallelize page extraction within a single PDF
        with ThreadPoolExecutor() as page_executor:
            futures = {
                page_executor.submit(parse_page, (i, page)): i
                for i, page in enumerate(reader.pages)
            }
            # Collect results in page order
            ordered = {}
            for future in as_completed(futures):
                page_index = futures[future]
                ordered[page_index] = future.result()

        for i in sorted(ordered):
            chunks.extend(ordered[i])

    except Exception as e:
        print(f"Error reading PDF '{uploaded_file.name}': {e}")

    return chunks


def extract_chunks_from_txt(uploaded_file):
    """Extracts chunks from an in-memory text/markdown file object."""
    try:
        text = uploaded_file.read().decode('utf-8')
        cleaned = clean_text(text)
        metadata = {"file": uploaded_file.name, "page": "N/A"}
        return chunk_text_with_metadata(cleaned, metadata)
    except Exception as e:
        print(f"Error reading TXT '{uploaded_file.name}': {e}")
        return []


def _parse_single_file(file):
    """Dispatcher for a single file — used by the thread pool."""
    if file.name.endswith('.pdf'):
        return extract_chunks_from_pdf(file)
    elif file.name.endswith(('.md', '.txt', '.csv')):
        return extract_chunks_from_txt(file)
    return []


def load_course_materials(uploaded_files, max_workers=None):
    """
    Parses a list of Streamlit UploadedFile objects concurrently.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.
        max_workers: Max threads. Defaults to min(32, cpu_count + 4) via ThreadPoolExecutor.

    Returns:
        List of chunks: [{"text": "...", "metadata": {"file": "...", "page": "..."}}]
    """
    all_chunks = []

    # One thread per file — each PDF also spawns its own inner pool for pages
    with ThreadPoolExecutor(max_workers=max_workers) as file_executor:
        future_to_file = {
            file_executor.submit(_parse_single_file, file): file.name
            for file in uploaded_files
        }

        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                chunks = future.result()
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Unexpected error processing '{filename}': {e}")

    return all_chunks