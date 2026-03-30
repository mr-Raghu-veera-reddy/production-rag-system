"""
Document Loader Module
Loads and extracts text from PDF documents
"""

import PyPDF2
from typing import List, Dict
import os

class DocumentLoader:
    """
    Load and extract text from PDF documents
    """
    
    def __init__(self):
        """Initialize document loader"""
        self.documents = []
        print("📂 Document Loader initialized")
    
    def load_pdf(self, pdf_path: str) -> Dict:
        """
        Extract text from a single PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with filename, text, and metadata
        """
        
        print(f"\n📄 Loading: {pdf_path}")
        
        try:
            # Open PDF file
            with open(pdf_path, 'rb') as file:
                # Create PDF reader
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Get number of pages
                num_pages = len(pdf_reader.pages)
                print(f"   Pages: {num_pages}")
                
                # Extract text from all pages
                text = ""
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    text += page_text
                
                # Get filename without path
                filename = os.path.basename(pdf_path)
                
                # Calculate statistics
                word_count = len(text.split())
                char_count = len(text)
                
                print(f"   ✅ Extracted {char_count:,} characters, {word_count:,} words")
                
                return {
                    'filename': filename,
                    'filepath': pdf_path,
                    'text': text,
                    'num_pages': num_pages,
                    'word_count': word_count,
                    'char_count': char_count
                }
                
        except FileNotFoundError:
            print(f"   ❌ Error: File not found - {pdf_path}")
            return None
            
        except Exception as e:
            print(f"   ❌ Error loading {pdf_path}: {e}")
            return None
    
    def load_directory(self, directory_path: str) -> List[Dict]:
        """
        Load all PDF files from a directory
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of document dictionaries
        """
        
        print("=" * 70)
        print(f"📁 Loading PDFs from: {directory_path}")
        print("=" * 70)
        
        # Check if directory exists
        if not os.path.exists(directory_path):
            print(f"❌ Error: Directory not found - {directory_path}")
            return []
        
        # Get all PDF files
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"❌ No PDF files found in {directory_path}")
            return []
        
        print(f"Found {len(pdf_files)} PDF files")
        
        # Load each PDF
        documents = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory_path, pdf_file)
            doc = self.load_pdf(pdf_path)
            
            if doc:
                documents.append(doc)
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 LOADING SUMMARY")
        print("=" * 70)
        print(f"Total documents loaded: {len(documents)}")
        
        if documents:
            total_pages = sum(doc['num_pages'] for doc in documents)
            total_words = sum(doc['word_count'] for doc in documents)
            print(f"Total pages: {total_pages}")
            print(f"Total words: {total_words:,}")
        
        print("=" * 70)
        
        self.documents = documents
        return documents
    
    def get_document_info(self) -> None:
        """Print information about loaded documents"""
        
        if not self.documents:
            print("No documents loaded yet")
            return
        
        print("\n📋 LOADED DOCUMENTS")
        print("=" * 70)
        
        for i, doc in enumerate(self.documents, 1):
            print(f"\n[{i}] {doc['filename']}")
            print(f"    Pages: {doc['num_pages']}")
            print(f"    Words: {doc['word_count']:,}")
            print(f"    Characters: {doc['char_count']:,}")
            print(f"    Preview: {doc['text'][:100]}...")
        
        print("=" * 70)


# Test code - runs when you execute this file directly
if __name__ == "__main__":
    print("\n🧪 TESTING DOCUMENT LOADER")
    print("=" * 70)
    
    # Create loader
    loader = DocumentLoader()
    
    # Load documents from data folder
    documents = loader.load_directory("./data")
    
    # Show document info
    if documents:
        loader.get_document_info()
        
        print("\n✅ Document Loader Test PASSED!")
    else:
        print("\n❌ Document Loader Test FAILED!")
        print("Make sure you have PDF files in ./data folder")