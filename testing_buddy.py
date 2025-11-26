# testing_buddy.py
"""Intelligent testing framework with document-based query generation"""

from typing import List, Dict, Optional
import random


class TestingBuddy:
    """Document-based testing framework for neural network training"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.test_history = []
        self.openai_api_key = openai_api_key
        self.generated_queries = []
        self.current_document = None
    
    def generate_queries_from_document(self, rag_system, document_text: str, num_queries: int = 20) -> List[str]:
        """Generate diverse test queries from document using LLM"""
        try:
            queries = rag_system.generate_training_queries(document_text, num_queries)
            self.generated_queries = queries
            self.current_document = document_text[:500]  # Store snippet
            return queries
        except Exception as e:
            print(f"Error generating queries: {e}")
            return []
    
    def get_warmup_sequence(self, num_queries: int = 30) -> List[Dict[str, any]]:
        """Get warmup sequence for neural network training
        
        Neural network needs diverse training examples to learn properly.
        Minimum 20-30 queries recommended for initial learning.
        """
        if not self.generated_queries:
            return []
        
        # Use all generated queries for warmup
        sequence = []
        for idx, query in enumerate(self.generated_queries[:num_queries]):
            # Alternate feedback for variety
            if idx < num_queries // 3:
                feedback = 1  # First third: positive
            elif idx < 2 * num_queries // 3:
                feedback = random.choice([1, -1])  # Middle: mixed
            else:
                feedback = 1  # Last third: mostly positive
            
            sequence.append({
                'step': idx + 1,
                'query': query,
                'feedback_suggestion': feedback,
                'phase': self._get_phase(idx, num_queries)
            })
        
        return sequence
    
    def _get_phase(self, idx: int, total: int) -> str:
        """Determine training phase"""
        progress = idx / total
        if progress < 0.33:
            return "Initial Exploration"
        elif progress < 0.67:
            return "Learning Phase"
        else:
            return "Refinement"
    
    def get_clustering_demo_queries(self) -> List[Dict[str, str]]:
        """Get queries that demonstrate intelligent clustering"""
        if len(self.generated_queries) < 6:
            return []
        
        # Pick diverse queries
        demos = []
        for i in range(0, min(6, len(self.generated_queries)), 2):
            demos.append({
                'query': self.generated_queries[i],
                'type': 'Query Type Detection',
                'feedback_suggestion': 1
            })
        
        return demos
    
    def record_test(self, query: str, cluster_name: str, strategy: str, feedback: int, 
                   cluster_info: Dict = None):
        """Record test interaction"""
        test_entry = {
            'query': query,
            'cluster_name': cluster_name,
            'strategy': strategy,
            'feedback': feedback
        }
        
        if cluster_info:
            test_entry.update({
                'intent': cluster_info.get('intent', 'unknown'),
                'depth': cluster_info.get('depth', 'unknown'),
                'scope': cluster_info.get('scope', 'unknown')
            })
        
        self.test_history.append(test_entry)
    
    def get_test_report(self) -> Dict:
        """Generate test report"""
        if not self.test_history:
            return {'total': 0}
        
        total = len(self.test_history)
        positive = sum(1 for t in self.test_history if t['feedback'] > 0)
        
        strategy_counts = {}
        cluster_counts = {}
        intent_counts = {}
        
        for test in self.test_history:
            strategy_counts[test['strategy']] = strategy_counts.get(test['strategy'], 0) + 1
            cluster_counts[test['cluster_name']] = cluster_counts.get(test['cluster_name'], 0) + 1
            if 'intent' in test:
                intent_counts[test['intent']] = intent_counts.get(test['intent'], 0) + 1
        
        return {
            'total_tests': total,
            'positive_feedback': positive,
            'negative_feedback': total - positive,
            'success_rate': positive / total if total > 0 else 0,
            'strategy_distribution': strategy_counts,
            'cluster_distribution': cluster_counts,
            'intent_distribution': intent_counts
        }
    
    def clear_history(self):
        """Clear test history"""
        self.test_history = []
    
    def clear_generated_queries(self):
        """Clear generated queries"""
        self.generated_queries = []
        self.current_document = None