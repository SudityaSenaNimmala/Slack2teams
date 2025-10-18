from app.helpers import build_vectorstore, build_combined_vectorstore
from app.pdf_processor import process_pdf_directory, chunk_pdf_documents
from config import url, CHROMA_DB_PATH
import os
import shutil
import json
import hashlib
from datetime import datetime
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

METADATA_FILE = "./data/vectorstore_metadata.json"

def get_file_hash(file_path):
    """Get MD5 hash of a file for change detection."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def get_directory_hash(directory):
    """Get combined hash of all files in a directory."""
    if not os.path.exists(directory):
        return None
    
    hashes = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_hash = get_file_hash(file_path)
            if file_hash:
                hashes.append(f"{file_path}:{file_hash}")
    
    if not hashes:
        return None
    
    combined = "|".join(sorted(hashes))
    return hashlib.md5(combined.encode()).hexdigest()

def get_current_metadata():
    """Get current metadata of all source files and directories."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "url": url,
        "pdfs": get_directory_hash("./pdfs"),
        "excel": get_directory_hash("./excel"),
        "docs": get_directory_hash("./docs"),
        "vectorstore_exists": os.path.exists(CHROMA_DB_PATH)
    }
    return metadata

def load_stored_metadata():
    """Load previously stored metadata."""
    if not os.path.exists(METADATA_FILE):
        return None
    
    try:
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    except:
        return None

def save_metadata(metadata):
    """Save current metadata."""
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def should_rebuild_vectorstore():
    """Check if vectorstore needs to be rebuilt based on file changes."""
    print("🔍 Checking if vectorstore rebuild is needed...")
    
    # If vectorstore doesn't exist, we need to rebuild
    if not os.path.exists(CHROMA_DB_PATH):
        print("❌ Vectorstore not found - rebuild needed")
        return True
    
    # Load stored metadata
    stored_metadata = load_stored_metadata()
    if not stored_metadata:
        print("❌ No stored metadata found - rebuild needed")
        return True
    
    # Get current metadata
    current_metadata = get_current_metadata()
    
    # Check if any source has changed
    sources_to_check = ["pdfs", "excel", "docs", "url"]
    for source in sources_to_check:
        if stored_metadata.get(source) != current_metadata.get(source):
            print(f"❌ {source} has changed - rebuild needed")
            print(f"   Stored: {stored_metadata.get(source)}")
            print(f"   Current: {current_metadata.get(source)}")
            return True
    
    print("✅ No changes detected - using existing vectorstore")
    return False

def load_existing_vectorstore():
    """Load existing vectorstore without rebuilding."""
    print("📂 Loading existing vectorstore...")
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        
        # Test if vectorstore is working
        total_docs = vectorstore._collection.count()
        print(f"✅ Loaded existing vectorstore with {total_docs} documents")
        return vectorstore
    except Exception as e:
        print(f"❌ Failed to load existing vectorstore: {e}")
        return None

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
    
    # Save metadata after successful build
    current_metadata = get_current_metadata()
    save_metadata(current_metadata)
    print("💾 Saved vectorstore metadata for future change detection")
    
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

def initialize_vectorstore():
    """Smart vectorstore initialization that only rebuilds when needed."""
    print("=" * 60)
    print("🚀 INITIALIZING CF-CHATBOT KNOWLEDGE BASE")
    print("=" * 60)
    
    # Check if rebuild is needed
    if should_rebuild_vectorstore():
        print("🔨 Rebuilding vectorstore...")
        vectorstore = rebuild_vectorstore_if_needed()
    else:
        # Try to load existing vectorstore
        vectorstore = load_existing_vectorstore()
        if vectorstore is None:
            print("⚠️  Failed to load existing vectorstore, rebuilding...")
            vectorstore = rebuild_vectorstore_if_needed()
    
    print("✅ Vectorstore initialization complete!")
    print("=" * 60)
    return vectorstore

# Initialize vectorstore smartly
vectorstore = initialize_vectorstore()

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 15}  # Optimized for semantic search
)