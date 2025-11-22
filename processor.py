# processor.py
"""Enhanced text processing with better chunking"""

import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextProcessor:
    """Enhanced text processor with smart chunking"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        )
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"]', '', text)
        
        # Remove multiple punctuation
        text = re.sub(r'([.,!?;:]){2,}', r'\1', text)
        
        return text.strip()
    
    def process_file(self, file_content: str) -> List[str]:
        """Process uploaded file and return chunks"""
        # Clean text
        cleaned_text = self.clean_text(file_content)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(cleaned_text)
        
        # Filter and clean chunks
        processed_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()
            # Only keep chunks with sufficient content
            if len(chunk) > 50 and len(chunk.split()) > 10:
                processed_chunks.append(chunk)
        
        return processed_chunks
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess user query"""
        query = query.strip()
        query = re.sub(r'\s+', ' ', query)
        return query
    
    def extract_keywords(self, text: str, min_length: int = 4) -> List[str]:
        """Extract important keywords from text"""
        # Remove common words
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 
            'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 
            'does', 'did', 'will', 'would', 'should', 'could', 'may', 
            'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + r',}\b', text.lower())
        
        # Filter stop words and get unique
        keywords = list(set(w for w in words if w not in stop_words))
        
        return keywords
    
    def calculate_text_statistics(self, text: str) -> dict:
        """Calculate text statistics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0
        }