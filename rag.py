# rag.py (Updated imports and usage)
"""Enhanced Adaptive RAG with Neural-RL Integration"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import Dict, List, Tuple
from indexing import VectorStore
from reinforcement_learning import EnhancedRLAgent
from clustering import LLMQueryClusterer
from processor import TextProcessor
from logger import SystemLogger
from config import *
import time

# Import all prompt templates
from prompt_template import (
    STRATEGY_PROMPTS,
    QUERY_COMPLEXITY_PROMPT,
    STRATEGY_DESCRIPTIONS
)


class EnhancedAdaptiveRAG:
    """Enhanced Adaptive RAG with Neural-RL"""
    
    def __init__(self, user_id: str, openai_api_key: str):
        self.user_id = user_id
        self.openai_api_key = openai_api_key
        
        # LLM models
        self.llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=CHAT_TEMPERATURE,
            openai_api_key=openai_api_key
        )
        
        self.analyzer_llm = ChatOpenAI(
            model=ANALYZER_MODEL,
            temperature=ANALYZER_TEMPERATURE,
            openai_api_key=openai_api_key
        )
        
        # Core components
        self.vector_store = VectorStore(user_id, openai_api_key)
        self.rl_agent = EnhancedRLAgent(user_id)
        self.clusterer = LLMQueryClusterer(user_id, openai_api_key)
        self.processor = TextProcessor()
        self.logger = SystemLogger(user_id)
        
        self.logger.log_session_start()
        
        # Current query context (for feedback)
        self.current_context = None
    
    def index_document(self, file_content: str) -> int:
        """Process and index document"""
        try:
            start_time = time.time()
            chunks = self.processor.process_file(file_content)
            self.vector_store.add_documents(chunks)
            duration = time.time() - start_time
            
            self.logger.log_indexing(len(chunks))
            return len(chunks)
        except Exception as e:
            self.logger.log_error(e, "index_document")
            raise
    
    def analyze_query_complexity(self, query: str) -> str:
        """Analyze query complexity using detailed prompt"""
        prompt = PromptTemplate.from_template(QUERY_COMPLEXITY_PROMPT)
        formatted_prompt = prompt.format(query=query)
        
        try:
            response = self.analyzer_llm.invoke(formatted_prompt)
            complexity = response.content.strip().lower()
            
            # Validate response
            valid_complexities = ["simple", "moderate", "complex"]
            for valid in valid_complexities:
                if valid in complexity:
                    return valid
            
            return "moderate"  # Default fallback
        except Exception as e:
            self.logger.log_error(e, "analyze_query_complexity")
            return "moderate"
    
    def query(self, user_query: str) -> Tuple[str, Dict]:
        """Process query with neural-RL strategy selection"""
        try:
            start_time = time.time()
            
            # Preprocess query
            processed_query = self.processor.preprocess_query(user_query)
            
            # Step 1: Assign to cluster
            cluster_start = time.time()
            cluster_name, is_new_cluster, cluster_info = self.clusterer.assign_cluster(processed_query)
            cluster_time = time.time() - cluster_start
            
            # Step 2: Analyze complexity
            complexity = self.analyze_query_complexity(processed_query)
            
            # Step 3: Get cluster's best strategy
            cluster_best_strategy = self.clusterer.get_best_strategy_for_cluster(cluster_name)
            
            # Step 4: Extract features for neural network
            features = self.rl_agent.extract_features(
                processed_query, 
                cluster_info['embedding'], 
                complexity
            )
            
            # Step 5: Select strategy using Neural Contextual Bandit
            strategy_start = time.time()
            strategy, top_k, selection_info = self.rl_agent.select_strategy(
                processed_query,
                cluster_info['embedding'],
                complexity,
                cluster_best_strategy
            )
            strategy_time = time.time() - strategy_start
            
            self.logger.log_query(processed_query, strategy, complexity)
            
            # Step 6: Retrieve documents
            retrieval_start = time.time()
            retrieved_docs = self.vector_store.search(processed_query, top_k=top_k)
            retrieval_time = time.time() - retrieval_start
            
            self.logger.log_retrieval(processed_query, len(retrieved_docs), top_k)
            
            if not retrieved_docs:
                total_time = time.time() - start_time
                return "I couldn't find relevant information to answer your question.", {
                    "strategy": strategy,
                    "strategy_description": STRATEGY_DESCRIPTIONS.get(strategy, ""),
                    "retrieved_docs": [],
                    "complexity": complexity,
                    "cluster_name": cluster_name,
                    "cluster_info": cluster_info,
                    "is_new_cluster": is_new_cluster,
                    "selection_info": selection_info,
                    "timing": {
                        "total": total_time,
                        "clustering": cluster_time,
                        "strategy_selection": strategy_time,
                        "retrieval": retrieval_time,
                        "generation": 0
                    }
                }
            
            # Step 7: Generate response using strategy-specific prompt
            generation_start = time.time()
            context = "\n\n".join([doc['content'] for doc in retrieved_docs])
            
            # Get strategy-specific prompt from templates
            prompt_template = STRATEGY_PROMPTS[strategy]
            prompt = PromptTemplate.from_template(prompt_template)
            formatted_prompt = prompt.format(context=context, question=processed_query)
            
            response = self.llm.invoke(formatted_prompt)
            answer = response.content
            generation_time = time.time() - generation_start
            
            total_time = time.time() - start_time
            
            # Store context for feedback
            self.current_context = {
                'query': processed_query,
                'strategy': strategy,
                'response': answer,
                'retrieved_docs': [doc['content'][:100] for doc in retrieved_docs],
                'cluster_name': cluster_name,
                'features': features
            }
            
            metadata = {
                "strategy": strategy,
                "strategy_description": STRATEGY_DESCRIPTIONS.get(strategy, ""),
                "top_k": top_k,
                "retrieved_docs": [doc['content'][:100] for doc in retrieved_docs],
                "complexity": complexity,
                "cluster_name": cluster_name,
                "cluster_info": cluster_info,
                "is_new_cluster": is_new_cluster,
                "selection_info": selection_info,
                "used_cluster_strategy": strategy == cluster_best_strategy,
                "timing": {
                    "total": round(total_time * 1000, 2),  # ms
                    "clustering": round(cluster_time * 1000, 2),
                    "strategy_selection": round(strategy_time * 1000, 2),
                    "retrieval": round(retrieval_time * 1000, 2),
                    "generation": round(generation_time * 1000, 2)
                }
            }
            
            return answer, metadata
        
        except Exception as e:
            self.logger.log_error(e, "query")
            raise
    
    def submit_feedback(self, feedback: int) -> Dict:
        """Submit user feedback"""
        if self.current_context is None:
            return {"error": "No active query to provide feedback for"}
        
        # Record feedback in RL agent
        drift_detected = self.rl_agent.record_feedback(
            query=self.current_context['query'],
            strategy=self.current_context['strategy'],
            response=self.current_context['response'],
            feedback=feedback,
            retrieved_docs=self.current_context['retrieved_docs'],
            cluster_name=self.current_context['cluster_name'],
            features=self.current_context['features']
        )
        
        # Record in clusterer
        reward = POSITIVE_REWARD if feedback > 0 else NEGATIVE_REWARD
        self.clusterer.record_strategy_performance(
            self.current_context['cluster_name'],
            self.current_context['strategy'],
            reward
        )
        
        self.logger.log_feedback(
            self.current_context['query'],
            feedback,
            self.current_context['strategy']
        )
        
        return {
            "recorded": True,
            "drift_detected": drift_detected,
            "strategy": self.current_context['strategy']
        }
    
    def get_metrics(self) -> Dict:
        """Get comprehensive system metrics"""
        base_metrics = self.rl_agent.get_performance_metrics()
        
        # Add cluster information
        clusters_info = self.clusterer.get_all_clusters()
        
        base_metrics['clusters'] = clusters_info
        base_metrics['total_clusters'] = len(clusters_info)
        
        return base_metrics
    

    # rag.py - Add this method to EnhancedAdaptiveRAG class

    def generate_training_queries(self, text: str, num_queries: int = 20) -> List[str]:
        """Generate training queries from text using LLM"""
        from langchain_core.prompts import PromptTemplate
        from prompt_template import QUERY_GENERATION_PROMPT
        
        # Truncate text if too long
        if len(text) > 3000:
            text = text[:3000]
        
        prompt = PromptTemplate.from_template(QUERY_GENERATION_PROMPT)
        formatted_prompt = prompt.format(text=text, num_queries=num_queries)
        
        try:
            response = self.llm.invoke(formatted_prompt)
            generated_text = response.content.strip()
            
            # Parse queries
            queries = []
            for line in generated_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering
                    query = line.lstrip('0123456789.-) ').strip()
                    if query and len(query) > 10:
                        queries.append(query)
            
            return queries[:num_queries]
            
        except Exception as e:
            self.logger.log_error(e, "generate_training_queries")
            # Fallback queries
            return [
                "What is this about?",
                "Explain the main concept",
                "How does this work?",
                "What are the key points?",
                "Why is this important?"
            ][:num_queries]
    
    def clear_documents(self):
        """Clear all indexed documents"""
        self.vector_store.clear_collection()