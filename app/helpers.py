import os
import requests
import markdown
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from config import CHROMA_DB_PATH
from app.pdf_processor import process_pdf_directory, chunk_pdf_documents
from app.excel_processor import process_excel_directory, chunk_excel_documents
from app.doc_processor import process_doc_directory, chunk_doc_documents


def load_webpage(url: str):
    """Fetch posts from WordPress API and return raw text."""
    response = requests.get(url)
    data = response.json()
    texts = []
    for post in data:
        if "content" in post and "rendered" in post["content"]:
            texts.append(post["content"]["rendered"])
    return "\n\n".join(texts)

def strip_markdown(md_text: str) -> str:
    """Convert Markdown/HTML to plain text."""
    html = markdown.markdown(md_text)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

def build_vectorstore(url: str):
    """Build and persist embeddings for web documents."""
    raw_text = load_webpage(url)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([raw_text])
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DB_PATH)
    return vectorstore

def build_combined_vectorstore(url: str, pdf_directory: str, excel_directory: str = None, doc_directory: str = None):
    """Build and persist embeddings for web content, PDF documents, Excel files, and Word documents."""
    print("Loading web content...")
    raw_text = load_webpage(url)
    web_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    web_docs = web_splitter.create_documents([raw_text])
    
    # Add source metadata to web docs
    for doc in web_docs:
        doc.metadata["source_type"] = "web"
        doc.metadata["source"] = "cloudfuze_blog"
    
    print("Processing PDF documents...")
    pdf_docs = process_pdf_directory(pdf_directory)
    pdf_chunks = chunk_pdf_documents(pdf_docs, chunk_size=1000, chunk_overlap=200)
    
    # Process Excel files if directory exists
    excel_chunks = []
    if excel_directory and os.path.exists(excel_directory):
        print("Processing Excel documents...")
        excel_docs = process_excel_directory(excel_directory)
        excel_chunks = chunk_excel_documents(excel_docs, chunk_size=1000, chunk_overlap=200)
    else:
        print("Excel directory not found or not specified, skipping Excel processing...")
    
    # Process Word documents if directory exists
    doc_chunks = []
    if doc_directory and os.path.exists(doc_directory):
        print("Processing Word documents...")
        doc_docs = process_doc_directory(doc_directory)
        doc_chunks = chunk_doc_documents(doc_docs, chunk_size=1000, chunk_overlap=200)
    else:
        print("Word documents directory not found or not specified, skipping Word processing...")
    
    # Combine all documents
    all_docs = web_docs + pdf_chunks + excel_chunks + doc_chunks
    print(f"Total documents to process: {len(all_docs)}")
    print(f"  - Web documents: {len(web_docs)}")
    print(f"  - PDF documents: {len(pdf_chunks)}")
    print(f"  - Excel documents: {len(excel_chunks)}")
    print(f"  - Word documents: {len(doc_chunks)}")
    
    # Create embeddings and vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(all_docs, embeddings, persist_directory=CHROMA_DB_PATH)
    
    print("Combined knowledge base created successfully!")
    return vectorstore