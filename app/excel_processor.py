import os
import pandas as pd
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Try to import openpyxl for Excel support
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    print("openpyxl not available, Excel processing may be limited")

# Try to import xlrd for older Excel formats
try:
    import xlrd
    XLRD_AVAILABLE = True
except ImportError:
    XLRD_AVAILABLE = False
    print("xlrd not available, older Excel formats (.xls) may not be supported")

def extract_text_from_excel(excel_path: str) -> str:
    """Extract text content from an Excel file, including all sheets."""
    text_content = []
    
    try:
        # Read all sheets from the Excel file
        excel_file = pd.ExcelFile(excel_path)
        
        # Add file-level metadata for better searchability
        text_content.append(f"Excel file: {os.path.basename(excel_path)}")
        text_content.append(f"Contains structured data and information")
        
        for sheet_name in excel_file.sheet_names:
            try:
                # Read the sheet
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                
                # Add sheet header
                text_content.append(f"\n--- Sheet: {sheet_name} ---\n")
                
                # Convert DataFrame to text
                if not df.empty:
                    # Add column headers
                    if len(df.columns) > 0:
                        headers = " | ".join([str(col) for col in df.columns])
                        text_content.append(f"Columns: {headers}\n")
                    
                    # Add data rows
                    for index, row in df.iterrows():
                        row_data = []
                        for col in df.columns:
                            cell_value = str(row[col]) if pd.notna(row[col]) else ""
                            row_data.append(cell_value)
                        
                        if any(cell.strip() for cell in row_data):  # Only add non-empty rows
                            row_text = " | ".join(row_data)
                            text_content.append(f"Row {index + 1}: {row_text}\n")
                    
                    # Add summary statistics for numeric columns
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        text_content.append(f"\nSummary for numeric columns:\n")
                        for col in numeric_cols:
                            if not df[col].isna().all():
                                stats = df[col].describe()
                                text_content.append(f"{col}: mean={stats.get('mean', 'N/A'):.2f}, "
                                                  f"min={stats.get('min', 'N/A'):.2f}, "
                                                  f"max={stats.get('max', 'N/A'):.2f}\n")
                
            except Exception as e:
                print(f"Error processing sheet '{sheet_name}' in {excel_path}: {e}")
                text_content.append(f"\nError reading sheet '{sheet_name}': {str(e)}\n")
        
        return "\n".join(text_content)
        
    except Exception as e:
        print(f"Error reading Excel file {excel_path}: {e}")
        return ""

def process_excel_directory(excel_directory: str) -> List[Document]:
    """Process all Excel files in a directory and return as LangChain Documents."""
    documents = []
    
    if not os.path.exists(excel_directory):
        print(f"Excel directory {excel_directory} does not exist")
        return documents
    
    # Supported Excel file extensions
    excel_extensions = ['.xlsx', '.xls']
    excel_files = []
    
    for file in os.listdir(excel_directory):
        if any(file.lower().endswith(ext) for ext in excel_extensions):
            excel_files.append(file)
    
    if not excel_files:
        print(f"No Excel files found in {excel_directory}")
        return documents
    
    print(f"Processing {len(excel_files)} Excel file(s)...")
    
    for excel_file in excel_files:
        excel_path = os.path.join(excel_directory, excel_file)
        print(f"Processing: {excel_file}")
        
        try:
            # Extract text from Excel file
            text = extract_text_from_excel(excel_path)
            source_type = "excel"
            
            if text.strip():
                # Create a document with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": excel_file,
                        "source_type": source_type,
                        "file_path": excel_path,
                        "file_format": excel_file.split('.')[-1].lower(),
                        "content_type": "excel_data",
                        "searchable_terms": " ".join(text.split()[:20])  # Add first 20 words for better searchability
                    }
                )
                documents.append(doc)
                print(f"Successfully processed {excel_file} ({len(text)} characters)")
            else:
                print(f"Warning: No text extracted from {excel_file}")
                
        except Exception as e:
            print(f"Error processing {excel_file}: {e}")
    
    return documents

def chunk_excel_documents(documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 150) -> List[Document]:
    """Split Excel documents into smaller chunks for better retrieval."""
    if not documents:
        return documents
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " | ", " ", ""]  # Better separators for Excel data
    )
    
    chunked_docs = []
    for doc in documents:
        chunks = splitter.split_documents([doc])
        # Add metadata to each chunk for better searchability
        for chunk in chunks:
            chunk.metadata.update({
                "chunk_type": "excel_data",
                "searchable_content": " ".join(chunk.page_content.split()[:20])  # First 20 words for search
            })
        chunked_docs.extend(chunks)
    
    print(f"Split {len(documents)} Excel documents into {len(chunked_docs)} chunks")
    return chunked_docs

def get_excel_summary(excel_path: str) -> Dict:
    """Get a summary of an Excel file's structure."""
    try:
        excel_file = pd.ExcelFile(excel_path)
        summary = {
            "file_name": os.path.basename(excel_path),
            "sheet_count": len(excel_file.sheet_names),
            "sheet_names": excel_file.sheet_names,
            "total_rows": 0,
            "total_columns": 0
        }
        
        for sheet_name in excel_file.sheet_names:
            try:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                summary["total_rows"] += len(df)
                summary["total_columns"] = max(summary["total_columns"], len(df.columns))
            except:
                continue
                
        return summary
    except Exception as e:
        return {"error": str(e)}
