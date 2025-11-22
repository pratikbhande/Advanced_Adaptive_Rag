# indexing.py
"""Enhanced vector store with better error handling"""

import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict
import os
import logging

class VectorStore:
    """Enhanced vector store with ChromaDB"""
    
    def __init__(self, user_id: str, openai_api_key: str):
        self.user_id = user_id
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-3-small"
        )
        
        # Create directory
        os.makedirs("./chroma_db", exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Collection name
        self.collection_name = f"docs_{user_id}"
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_documents(self, texts: List[str]) -> None:
        """Add document chunks to vector store"""
        if not texts:
            return
        
        try:
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)
            
            # Prepare data
            ids = [f"doc_{self.user_id}_{i}_{hash(text)}" for i, text in enumerate(texts)]
            metadatas = [{"chunk_id": i, "user_id": self.user_id, "text_length": len(text)} 
                        for i, text in enumerate(texts)]
            
            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                end_idx = min(i + batch_size, len(texts))
                self.collection.add(
                    embeddings=embeddings[i:end_idx],
                    documents=texts[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )
        except Exception as e:
            logging.error(f"Error adding documents: {str(e)}")
            raise
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Format results
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    retrieved_docs.append({
                        'content': doc,
                        'id': results['ids'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else 0.0,
                        'metadata': results['metadatas'][0][i] if 'metadatas' in results else {}
                    })
            
            return retrieved_docs
        except Exception as e:
            logging.error(f"Error searching documents: {str(e)}")
            return []
    
    def clear_collection(self) -> None:
        """Clear all documents for this user"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logging.error(f"Error clearing collection: {str(e)}")
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                'document_count': count,
                'collection_name': self.collection_name
            }
        except:
            return {
                'document_count': 0,
                'collection_name': self.collection_name
            }