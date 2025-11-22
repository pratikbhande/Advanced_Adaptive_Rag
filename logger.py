# logger.py
"""Enhanced logging with structured output"""

import logging
import os
from datetime import datetime
from typing import Dict, Any
import json

class SystemLogger:
    """Enhanced system logger with structured logging"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.log_dir = "./logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create log files
        self.log_file = f"{self.log_dir}/system_{user_id}.log"
        self.json_log_file = f"{self.log_dir}/system_{user_id}_structured.jsonl"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(f"RAG_System_{user_id}")
        self.logger.setLevel(logging.INFO)
    
    def _log_structured(self, event_type: str, data: Dict[Any, Any]):
        """Log structured data to JSONL file"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'user_id': self.user_id,
                'event_type': event_type,
                'data': data
            }
            
            with open(self.json_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write structured log: {str(e)}")
    
    def log_query(self, query: str, strategy: str, complexity: str) -> None:
        """Log user query"""
        msg = f"QUERY | Strategy={strategy} | Complexity={complexity} | Query='{query[:50]}...'"
        self.logger.info(msg)
        
        self._log_structured('query', {
            'query': query,
            'strategy': strategy,
            'complexity': complexity
        })
    
    def log_feedback(self, query: str, feedback: int, strategy: str) -> None:
        """Log user feedback"""
        feedback_type = "POSITIVE" if feedback > 0 else "NEGATIVE"
        msg = f"FEEDBACK | Type={feedback_type} | Strategy={strategy} | Query='{query[:30]}...'"
        self.logger.info(msg)
        
        self._log_structured('feedback', {
            'query': query,
            'feedback': feedback,
            'strategy': strategy
        })
    
    def log_retrieval(self, query: str, num_docs: int, top_k: int) -> None:
        """Log document retrieval"""
        msg = f"RETRIEVAL | Retrieved={num_docs}/{top_k} | Query='{query[:30]}...'"
        self.logger.info(msg)
        
        self._log_structured('retrieval', {
            'query': query,
            'num_docs': num_docs,
            'top_k': top_k
        })
    
    def log_indexing(self, num_chunks: int) -> None:
        """Log document indexing"""
        msg = f"INDEXING | Chunks={num_chunks} | User={self.user_id}"
        self.logger.info(msg)
        
        self._log_structured('indexing', {
            'num_chunks': num_chunks
        })
    
    def log_strategy_update(self, strategy: str, new_stats: Dict[str, Any]) -> None:
        """Log strategy performance update"""
        msg = f"STRATEGY_UPDATE | Strategy={strategy} | Stats={new_stats}"
        self.logger.info(msg)
        
        self._log_structured('strategy_update', {
            'strategy': strategy,
            'stats': new_stats
        })
    
    def log_neural_network_training(self, batch_num: int, loss: float, buffer_size: int) -> None:
        """Log neural network training"""
        msg = f"NN_TRAINING | Batch={batch_num} | Loss={loss:.4f} | Buffer={buffer_size}"
        self.logger.info(msg)
        
        self._log_structured('nn_training', {
            'batch_number': batch_num,
            'loss': loss,
            'buffer_size': buffer_size
        })
    
    def log_concept_drift(self, z_score: float) -> None:
        """Log concept drift detection"""
        msg = f"CONCEPT_DRIFT | Z-Score={z_score:.2f} | DRIFT DETECTED!"
        self.logger.warning(msg)
        
        self._log_structured('concept_drift', {
            'z_score': z_score,
            'action': 'drift_detected'
        })
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log errors"""
        msg = f"ERROR | Context='{context}' | Error={str(error)}"
        self.logger.error(msg)
        
        self._log_structured('error', {
            'context': context,
            'error': str(error),
            'error_type': type(error).__name__
        })
    
    def log_session_start(self) -> None:
        """Log session start"""
        msg = f"SESSION_START | User={self.user_id} | Time={datetime.now()}"
        self.logger.info(msg)
        
        self._log_structured('session_start', {})
    
    def log_session_end(self) -> None:
        """Log session end"""
        msg = f"SESSION_END | User={self.user_id} | Time={datetime.now()}"
        self.logger.info(msg)
        
        self._log_structured('session_end', {})