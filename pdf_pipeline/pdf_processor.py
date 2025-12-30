"""
PDF Processor usando Docling para extrair e processar PDFs
"""
import os
from typing import List, Dict, Optional
from pathlib import Path

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat

from embedding_manager.embedding_manager import EmbeddingManager


class PDFProcessor:
    """Process PDFs using Docling and store in Qdrant"""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Initialize the PDF processor
        
        Args:
            embedding_manager: EmbeddingManager instance
        """
        self.embedding_manager = embedding_manager
        self._converter = None  # Lazy loading
    
    @property
    def converter(self):
        """Lazy loading DocumentConverter to avoid unnecessary resource consumption"""
        if self._converter is None:
            print("🔧 Initializing Docling DocumentConverter (first time)...")
            from docling.document_converter import DocumentConverter
            self._converter = DocumentConverter()
            print("✅ Docling initialized!")
        return self._converter
    
    def process_pdf(
        self,
        pdf_path: str,
        max_tokens: int = 500,
    ) -> Dict:
        """
        Process a PDF and store in Qdrant
        
        Args:
            pdf_path: Path to the PDF file
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap tokens (not used in Docling)
            
        Returns:
            Dictionary with processing statistics
        """
        print(f"📄 Processing PDF: {pdf_path}")
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")
        
        pdf_file_name = Path(pdf_path).stem
        
        # Convert PDF with Docling
        print("🔄 Converting PDF with Docling...")
        result = self.converter.convert(source=pdf_path)
        doc = result.document
        
        # Create hybrid chunker
        print(f"✂️  Creating chunks (max {max_tokens} tokens)...")
        chunker = HybridChunker(
            tokenizer="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=max_tokens
        )
        
        # Generate chunks
        chunks_list = []
        metadatas = []
        
        for i, chunk in enumerate(chunker.chunk(dl_doc=doc)):
            # Use chunk.text directly (do not contextualize that causes error with DocItem)
            chunk_text = chunk.text
            chunks_list.append(chunk_text)
            
            # Create basic metadata
            metadata = {
                "source": pdf_file_name,
                "chunk_id": f"{pdf_file_name}_chunk_{i:04d}",
                "total_chunks": -1,  # Will be updated later
                "chunk_index": i,
                "doc_type": "pdf"
            }
            metadatas.append(metadata)
        
        total_chunks = len(chunks_list)
        
        # Update total_chunks in all metadatas
        for metadata in metadatas:
            metadata["total_chunks"] = total_chunks
        
        print(f"✅ Created {total_chunks} chunks")
        
        # Add to Qdrant
        print("💾 Adding chunks to Qdrant...")
        self.embedding_manager.add_documents(chunks_list, metadatas)
        
        # Calculate statistics
        # Note: Docling does not expose token count directly,
        # estimates based on text size
        total_chars = sum(len(chunk) for chunk in chunks_list)
        avg_chars = total_chars / total_chunks if total_chunks > 0 else 0
        
        # Estimate: ~4 chars per token (approximation)
        estimated_total_tokens = total_chars // 4
        estimated_avg_tokens = avg_chars / 4
        
        stats = {
            "pdf_file": pdf_file_name,
            "total_chunks": total_chunks,
            "total_tokens": estimated_total_tokens,
            "avg_tokens_per_chunk": estimated_avg_tokens,
            "max_tokens_per_chunk": max_tokens,
            "min_tokens_per_chunk": min(len(chunk) // 4 for chunk in chunks_list) if chunks_list else 0
        }
        
        print(f"📊 Processing completed: {stats}")
        
        return stats
