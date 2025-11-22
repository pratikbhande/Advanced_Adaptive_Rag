# reinforcement_learning.py
"""Enhanced RL Agent with Neural Network Integration"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer
from neural_bandit import NeuralBandit
from config import *

class EnhancedRLAgent:
    """Reinforcement Learning Agent with Neural Contextual Bandit"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.data_dir = RL_DATA_PATH
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.feedback_file = f"{self.data_dir}/feedback_{user_id}.json"
        self.strategy_file = f"{self.data_dir}/strategy_{user_id}.json"
        self.suppression_file = f"{self.data_dir}/suppression_{user_id}.json"
        
        # Load data
        self.strategy_stats = self._load_strategy_stats()
        self.feedback_history = self._load_feedback_history()
        self.suppression_info = self._load_suppression_info()
        
        # Neural bandit
        self.neural_bandit = NeuralBandit(user_id, learning_rate=NEURAL_LEARNING_RATE)
        
        # Sentence transformer
        self.embedder = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        
        # Adaptive epsilon
        self.epsilon = EPSILON_START
        self.query_count = len(self.feedback_history)
    
    def _load_strategy_stats(self) -> Dict:
        """Load strategy statistics"""
        if os.path.exists(self.strategy_file):
            with open(self.strategy_file, 'r') as f:
                return json.load(f)
        return {s: {"wins": 0, "total": 0, "reward_sum": 0.0, "consecutive_negatives": 0} 
                for s in STRATEGIES}
    
    def _save_strategy_stats(self):
        """Save strategy statistics"""
        with open(self.strategy_file, 'w') as f:
            json.dump(self.strategy_stats, f, indent=2)
    
    def _load_feedback_history(self) -> List[Dict]:
        """Load feedback history"""
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_feedback_history(self):
        """Save feedback history"""
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_history, f, indent=2)
    
    def _load_suppression_info(self) -> Dict:
        """Load suppression info"""
        if os.path.exists(self.suppression_file):
            with open(self.suppression_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_suppression_info(self):
        """Save suppression info"""
        with open(self.suppression_file, 'w') as f:
            json.dump(self.suppression_info, f, indent=2)
    
    def extract_features(self, query: str, cluster_embedding: List[float], 
                        complexity: str) -> np.ndarray:
        """Extract feature vector for neural network"""
        # Query embedding
        query_embedding = self.embedder.encode(query)
        
        # Complexity score
        complexity_map = {"simple": 0.0, "moderate": 0.5, "complex": 1.0}
        complexity_score = complexity_map.get(complexity, 0.5)
        
        # User history features
        total_queries = len(self.feedback_history)
        avg_feedback = np.mean([f['reward'] for f in self.feedback_history]) if self.feedback_history else 0.0
        
        # Get preferred strategy
        preferred_idx = 0
        if self.strategy_stats:
            max_reward = max(s['reward_sum'] / max(s['total'], 1) for s in self.strategy_stats.values())
            for idx, (name, stats) in enumerate(self.strategy_stats.items()):
                if stats['total'] > 0 and stats['reward_sum'] / stats['total'] == max_reward:
                    preferred_idx = idx
                    break
        
        # Recent performance
        recent_window = self.feedback_history[-10:] if len(self.feedback_history) >= 10 else self.feedback_history
        recent_positive_rate = np.mean([1 if f['reward'] > 0 else 0 for f in recent_window]) if recent_window else 0.5
        
        user_features = np.array([
            min(total_queries / 100.0, 1.0),  # Normalized query count
            (avg_feedback + 1) / 2.0,  # Normalized avg feedback [-1,1] -> [0,1]
            preferred_idx / len(STRATEGIES),  # Normalized strategy ID
            len(recent_window) / 50.0,  # Session length
            datetime.now().hour / 24.0,  # Time of day
            recent_positive_rate,
            0.15,  # Follow-up rate (placeholder)
            1.8,  # Avg response time (placeholder)
            self.epsilon,  # Current exploration rate
            min(total_queries / 365.0, 1.0)  # Days active (approximation)
        ])
        
        # Time features
        now = datetime.now()
        time_features = np.array([
            now.hour / 24.0,
            now.weekday() / 7.0,
            1.0 if now.weekday() >= 5 else 0.0,  # Weekend
            0.5,  # Hours since last query (placeholder)
            len(recent_window)  # Queries this session
        ])
        
        # Concatenate all features
        feature_vector = np.concatenate([
            query_embedding,
            np.array(cluster_embedding),
            np.array([complexity_score]),
            user_features,
            time_features
        ])
        
        return feature_vector
    
    def select_strategy(self, query: str, cluster_embedding: List[float], 
                       complexity: str, cluster_best_strategy: Optional[str] = None) -> Tuple[str, int, Dict]:
        """Select strategy using Neural Contextual Bandit + UCB"""
        
        # Extract features
        features = self.extract_features(query, cluster_embedding, complexity)
        
        # Get neural network scores
        neural_scores = self.neural_bandit.predict(features)
        
        # Calculate UCB scores
        total_pulls = sum(s['total'] for s in self.strategy_stats.values())
        ucb_scores = {}
        
        for idx, strategy in enumerate(STRATEGIES):
            stats = self.strategy_stats[strategy]
            n = stats['total']
            
            # Neural score (learned value)
            neural_value = neural_scores[idx]
            
            # Exploration bonus
            if n == 0:
                exploration_bonus = float('inf')
            else:
                exploration_bonus = UCB_BETA * np.sqrt(np.log(total_pulls + 1) / n)
            
            ucb_scores[strategy] = neural_value + exploration_bonus
        
        # Adaptive epsilon based on confidence
        confidence = self._calculate_confidence()
        self.epsilon = max(EPSILON_MIN, EPSILON_START * (1 - confidence))
        
        # Epsilon-greedy selection
        if np.random.random() < self.epsilon:
            selected_strategy = np.random.choice(STRATEGIES)
        else:
            # Consider cluster bias
            if cluster_best_strategy and np.random.random() < CLUSTER_STRATEGY_BIAS:
                selected_strategy = cluster_best_strategy
            else:
                selected_strategy = max(ucb_scores, key=ucb_scores.get)
        
        # Check suppression
        if self._is_suppressed(selected_strategy):
            available = {k: v for k, v in ucb_scores.items() if not self._is_suppressed(k)}
            if available:
                selected_strategy = max(available, key=available.get)
        
        # Get top_k
        top_k = STRATEGY_K_VALUES.get(selected_strategy, 4)
        
        selection_info = {
            'neural_scores': {STRATEGIES[i]: float(neural_scores[i]) for i in range(len(STRATEGIES))},
            'ucb_scores': {k: float(v) if v != float('inf') else 999.0 for k, v in ucb_scores.items()},
            'epsilon': self.epsilon,
            'confidence': confidence,
            'cluster_bias_used': selected_strategy == cluster_best_strategy
        }
        
        return selected_strategy, top_k, selection_info
    
    def record_feedback(self, query: str, strategy: str, response: str, feedback: int,
                       retrieved_docs: List[str], cluster_name: str, features: np.ndarray):
        """Record feedback and update models"""
        reward = POSITIVE_REWARD if feedback > 0 else NEGATIVE_REWARD
        
        # Update strategy statistics
        self.strategy_stats[strategy]['total'] += 1
        self.strategy_stats[strategy]['reward_sum'] += reward
        if reward > 0:
            self.strategy_stats[strategy]['wins'] += 1
            self.strategy_stats[strategy]['consecutive_negatives'] = 0
        else:
            self.strategy_stats[strategy]['consecutive_negatives'] += 1
        
        # Check for suppression
        if self.strategy_stats[strategy]['consecutive_negatives'] >= CONSECUTIVE_NEGATIVES_THRESHOLD:
            self._suppress_strategy(strategy)
        
        # Add to feedback history
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'strategy': strategy,
            'response': response[:200],
            'feedback': feedback,
            'reward': reward,
            'retrieved_docs': retrieved_docs[:2],
            'cluster': cluster_name
        }
        self.feedback_history.append(feedback_entry)
        
        # Add experience to neural bandit
        strategy_idx = STRATEGIES.index(strategy)
        self.neural_bandit.add_experience(features, strategy_idx, reward)
        
        # Train neural network if buffer is full
        if len(self.neural_bandit.training_buffer) >= BATCH_SIZE:
            train_result = self.neural_bandit.train_batch(BATCH_SIZE)
            if train_result['trained']:
                self.neural_bandit.save_model()
        
        # Check concept drift
        drift_detected = self._detect_concept_drift()
        if drift_detected:
            self._handle_drift()
        
        # Save everything
        self._save_strategy_stats()
        self._save_feedback_history()
        self._save_suppression_info()
        
        self.query_count += 1
        
        return drift_detected
    
    def _calculate_confidence(self) -> float:
        """Calculate system confidence based on variance"""
        if not self.feedback_history:
            return 0.0
        
        recent = self.feedback_history[-20:]
        rewards = [f['reward'] for f in recent]
        
        if len(rewards) < 5:
            return 0.0
        
        variance = np.var(rewards)
        confidence = 1.0 / (1.0 + variance)  # Inverse variance
        
        return min(confidence, 1.0)
    
    def _is_suppressed(self, strategy: str) -> bool:
        """Check if strategy is suppressed"""
        if strategy not in self.suppression_info:
            return False
        
        if self.query_count >= self.suppression_info[strategy]['until_query']:
            del self.suppression_info[strategy]
            return False
        
        return True
    
    def _suppress_strategy(self, strategy: str):
        """Suppress strategy temporarily"""
        self.suppression_info[strategy] = {
            'until_query': self.query_count + SUPPRESSION_DURATION,
            'reason': 'consecutive_negatives',
            'timestamp': datetime.now().isoformat()
        }
    
    def _detect_concept_drift(self) -> bool:
        """Detect concept drift using statistical test"""
        if len(self.feedback_history) < DRIFT_WINDOW_SIZE + DRIFT_REFERENCE_SIZE:
            return False
        
        recent_window = self.feedback_history[-DRIFT_WINDOW_SIZE:]
        reference_window = self.feedback_history[-(DRIFT_WINDOW_SIZE + DRIFT_REFERENCE_SIZE):-DRIFT_WINDOW_SIZE]
        
        recent_error_rate = sum(1 for f in recent_window if f['reward'] < 0) / len(recent_window)
        reference_error_rate = sum(1 for f in reference_window if f['reward'] < 0) / len(reference_window)
        
        if reference_error_rate == 0 or reference_error_rate == 1:
            return False
        
        # Z-test
        z_score = abs((recent_error_rate - reference_error_rate) / 
                     np.sqrt(reference_error_rate * (1 - reference_error_rate) / DRIFT_WINDOW_SIZE))
        
        return z_score > DRIFT_Z_THRESHOLD
    
    def _handle_drift(self):
        """Handle detected concept drift"""
        # Increase exploration
        self.epsilon = min(0.4, self.epsilon * 2)
        
        # Partially reset statistics (50% retention)
        for strategy in STRATEGIES:
            self.strategy_stats[strategy]['total'] = int(self.strategy_stats[strategy]['total'] * 0.5)
            self.strategy_stats[strategy]['reward_sum'] *= 0.5
            self.strategy_stats[strategy]['wins'] = int(self.strategy_stats[strategy]['wins'] * 0.5)
            self.strategy_stats[strategy]['consecutive_negatives'] = 0
        
        # Clear suppression
        self.suppression_info.clear()
    
    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        metrics = {
            'total_interactions': len(self.feedback_history),
            'positive_feedback': sum(1 for f in self.feedback_history if f['reward'] > 0),
            'negative_feedback': sum(1 for f in self.feedback_history if f['reward'] < 0),
            'current_epsilon': self.epsilon,
            'confidence': self._calculate_confidence(),
            'strategy_performance': {}
        }
        
        for strategy in STRATEGIES:
            stats = self.strategy_stats[strategy]
            total = stats['total']
            if total > 0:
                win_rate = stats['wins'] / total
                avg_reward = stats['reward_sum'] / total
            else:
                win_rate = 0.0
                avg_reward = 0.0
            
            metrics['strategy_performance'][strategy] = {
                'total_uses': total,
                'win_rate': round(win_rate, 3),
                'avg_reward': round(avg_reward, 3),
                'consecutive_negatives': stats['consecutive_negatives'],
                'suppressed': self._is_suppressed(strategy)
            }
        
        # Add neural network metrics
        metrics['neural_network'] = self.neural_bandit.get_training_metrics()
        
        return metrics