import os
import PyPDF2
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Try to import PyMuPDF, fallback to alternatives if not available
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("PyMuPDF not available, using alternative PDF processing methods")

# Try to import pdfplumber as alternative
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("pdfplumber not available, using PyPDF2 only")

def extract_text_from_txt(txt_path: str) -> str:
    """Extract text from a text file."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading text file {txt_path}: {e}")
        return ""

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using multiple methods for better coverage."""
    text = ""
    
    # Method 1: Try pdfplumber first (most reliable for text extraction)
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return text
        except Exception as e:
            print(f"pdfplumber failed for {pdf_path}: {e}")
    
    # Method 2: Try PyMuPDF (fitz) if available - better for complex layouts
    if PYMUPDF_AVAILABLE:
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            if text.strip():
                return text
        except Exception as e:
            print(f"PyMuPDF failed for {pdf_path}: {e}")
    
    # Method 3: Fallback to PyPDF2
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"PyPDF2 failed for {pdf_path}: {e}")
        return ""
    
    return text

def process_pdf_directory(pdf_directory: str) -> List[Document]:
    """Process all PDF files in a directory and return as LangChain Documents."""
    documents = []
    
    if not os.path.exists(pdf_directory):
        print(f"PDF directory {pdf_directory} does not exist")
        return documents
    
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}")
        return documents
    
    print(f"Processing {len(pdf_files)} PDF files...")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        print(f"Processing: {pdf_file}")
        
        try:
            # Extract text from PDF
            text = extract_text_from_pdf(pdf_path)
            source_type = "pdf"
            
            if text.strip():
                # Create a document with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": pdf_file,
                        "source_type": source_type,
                        "file_path": pdf_path
                    }
                )
                documents.append(doc)
                print(f"Successfully processed {pdf_file} ({len(text)} characters)")
            else:
                print(f"Warning: No text extracted from {pdf_file}")
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
    
    return documents

def chunk_pdf_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Split PDF documents into smaller chunks for better retrieval."""
    if not documents:
        return documents
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunked_docs = []
    for doc in documents:
        chunks = splitter.split_documents([doc])
        chunked_docs.extend(chunks)
    
    print(f"Split {len(documents)} PDF documents into {len(chunked_docs)} chunks")
    return chunked_docs
