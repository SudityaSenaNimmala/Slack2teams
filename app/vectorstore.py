from app.helpers import build_vectorstore, build_combined_vectorstore
from app.pdf_processor import process_pdf_directory, chunk_pdf_documents
from config import url, CHROMA_DB_PATH
import os
import shutil
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def rebuild_vectorstore_if_needed():
    """Rebuild vectorstore to ensure it includes all current PDFs, Excel files, and web content."""
    print("=" * 60)
    print("INITIALIZING CF-CHATBOT KNOWLEDGE BASE")
    print("=" * 60)
    print("Fetching data from all available sources...")
    
    # Always rebuild to ensure latest data
    pdf_directory = "./pdfs"
    excel_directory = "./excel"
    doc_directory = "./docs"
    
    # Check what sources are available
    sources_found = []
    if os.path.exists(pdf_directory):
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        if pdf_files:
            sources_found.append(f"PDFs ({len(pdf_files)} files)")
    
    if os.path.exists(excel_directory):
        excel_files = [f for f in os.listdir(excel_directory) if f.lower().endswith(('.xlsx', '.xls'))]
        if excel_files:
            sources_found.append(f"Excel files ({len(excel_files)} files)")
    
    if os.path.exists(doc_directory):
        doc_files = [f for f in os.listdir(doc_directory) if f.lower().endswith(('.docx', '.doc'))]
        if doc_files:
            sources_found.append(f"Word documents ({len(doc_files)} files)")
    
    sources_found.append("Web content (CloudFuze blog)")
    
    print(f"Sources found: {', '.join(sources_found)}")
    print("Building comprehensive knowledge base...")
    
    # Try to remove old vectorstore, but don't fail if it's in use
    if os.path.exists(CHROMA_DB_PATH):
        try:
            print("Removing old vectorstore to ensure fresh build...")
            shutil.rmtree(CHROMA_DB_PATH)
        except PermissionError:
            print("Warning: Could not remove old vectorstore (in use), but will rebuild anyway...")
    
    # Build the combined vectorstore
    if os.path.exists(pdf_directory) or os.path.exists(excel_directory) or os.path.exists(doc_directory):
        vectorstore = build_combined_vectorstore(url, pdf_directory, excel_directory, doc_directory)
    else:
        vectorstore = build_vectorstore(url)
    
    total_docs = vectorstore._collection.count()
    print(f"Knowledge base built successfully!")
    print(f"Total documents indexed: {total_docs}")
    print("Chatbot is ready to answer questions from all sources!")
    print("=" * 60)
    
    return vectorstore

def manage_vectorstore_backup_and_rebuild():
    """Manage vectorstore backup and rebuild with proper versioning."""
    import shutil
    from datetime import datetime
    import time
    
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    backup_path = "./data/chroma_db_backup"
    current_path = CHROMA_DB_PATH
    
    print("=" * 60)
    print("VECTORSTORE BACKUP AND REBUILD MANAGEMENT")
    print("=" * 60)
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current vectorstore: {current_path}")
    print(f"Backup vectorstore: {backup_path}")
    print("-" * 60)
    
    # Step 1: Create backup of existing vectorstore (if it exists)
    if os.path.exists(current_path):
        try:
            # Remove old backup if it exists
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
                print("[OK] Removed old backup vectorstore")
            
            # Create backup of current vectorstore
            shutil.copytree(current_path, backup_path)
            print("[OK] Created backup of existing vectorstore")
            print(f"  Backup created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"[WARNING] Could not create backup: {e}")
    else:
        print("[INFO] No existing vectorstore found - this will be the first build")
    
    # Step 2: Remove current vectorstore to force fresh rebuild
    if os.path.exists(current_path):
        try:
            shutil.rmtree(current_path)
            print("[OK] Removed current vectorstore for fresh rebuild")
        except Exception as e:
            print(f"[WARNING] Could not remove current vectorstore: {e}")
    
    # Step 3: Rebuild vectorstore with latest data
    print("[OK] Starting fresh vectorstore rebuild...")
    print("-" * 60)
    return rebuild_vectorstore_if_needed()

# Execute backup and rebuild process
vectorstore = manage_vectorstore_backup_and_rebuild()

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 15}  # Optimized for semantic search
)