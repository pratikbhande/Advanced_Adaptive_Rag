# reinforcement_learning.py
"""Enhanced RL Agent with multi-dimensional features"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer
from neural_bandit import NeuralBandit
from config import *


class EnhancedRLAgent:
    """Enhanced Reinforcement Learning Agent with Neural Contextual Bandit"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        
        # Files
        self.feedback_file = RL_DATA_PATH / f"feedback_{user_id}.json"
        self.strategy_file = RL_DATA_PATH / f"strategy_{user_id}.json"
        
        # Load data
        self.strategy_stats = self._load_strategy_stats()
        self.feedback_history = self._load_feedback_history()
        
        # Neural bandit
        self.neural_bandit = NeuralBandit(user_id, learning_rate=NEURAL_LEARNING_RATE)
        
        # Sentence transformer for query embeddings
        self.embedder = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        
        # Adaptive epsilon
        self.epsilon = EPSILON_START
        self.query_count = len(self.feedback_history)
    
    def _load_strategy_stats(self) -> Dict:
        """Load strategy statistics"""
        if self.strategy_file.exists():
            with open(self.strategy_file, 'r') as f:
                return json.load(f)
        return {s: {"wins": 0, "total": 0, "reward_sum": 0.0} for s in STRATEGIES}
    
    def _save_strategy_stats(self):
        """Save strategy statistics"""
        with open(self.strategy_file, 'w') as f:
            json.dump(self.strategy_stats, f, indent=2)
    
    def _load_feedback_history(self) -> List[Dict]:
        """Load feedback history"""
        if self.feedback_file.exists():
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_feedback_history(self):
        """Save feedback history (keep last 1000)"""
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_history[-1000:], f, indent=2)
    
    def extract_features(self, query: str, cluster_feature_embedding: List[float]) -> np.ndarray:
        """Extract feature vector for neural network
        
        Feature components:
        - Query embedding (384 dim)
        - Cluster features: intent(6) + depth(3) + scope(2) = 11 dim
        - User history features (10 dim)  # REDUCED FROM 15
        - Time features (5 dim)
        Total: 410 dimensions (MATCHES CONFIG)
        """
        # Query embedding
        query_embedding = self.embedder.encode(query)
        
        # User history features
        total_queries = len(self.feedback_history)
        
        # Recent feedback stats
        recent_window = self.feedback_history[-50:] if len(self.feedback_history) >= 50 else self.feedback_history
        avg_feedback = np.mean([f['reward'] for f in recent_window]) if recent_window else 0.0
        positive_rate = np.mean([1 if f['reward'] > 0 else 0 for f in recent_window]) if recent_window else 0.5
        
        # Strategy usage patterns (5 strategies)
        strategy_usage = [0.0] * len(STRATEGIES)
        for feedback in recent_window:
            strategy_idx = STRATEGIES.index(feedback['strategy'])
            strategy_usage[strategy_idx] += 1
        if sum(strategy_usage) > 0:
            strategy_usage = [u / sum(strategy_usage) for u in strategy_usage]
        
        # Session features
        session_length = min(len(recent_window), 50) / 50.0
        
        # Experience level
        experience = min(total_queries / 100.0, 1.0)
        
        # User features: 10 dimensions (2 + 3 + 5 strategy usage)
        user_features = np.array([
            experience,
            (avg_feedback + 1) / 2.0,  # Normalize [-1,1] to [0,1]
            positive_rate,
            session_length,
            self.epsilon,
            *strategy_usage  # 5 dimensions
        ])
        
        # Time features (5 dimensions)
        now = datetime.now()
        time_features = np.array([
            now.hour / 24.0,
            now.weekday() / 7.0,
            1.0 if now.weekday() >= 5 else 0.0,
            (now.hour >= 9 and now.hour <= 17) * 1.0,
            len(recent_window) / 50.0
        ])
        
        # Concatenate all features
        feature_vector = np.concatenate([
            query_embedding,  # 384
            np.array(cluster_feature_embedding),  # 11 (intent + depth + scope)
            user_features,  # 10
            time_features  # 5
        ])
        
        # Verify dimension
        assert len(feature_vector) == 410, f"Feature dimension mismatch: {len(feature_vector)} != 410"
        
        return feature_vector.astype(np.float32)
    
    def select_strategy(self, query: str, cluster_feature_embedding: List[float], 
                       cluster_name: str, best_strategy_for_cluster: Optional[str] = None) -> Tuple[str, int, Dict]:
        """Select strategy using Neural Network with epsilon-greedy exploration"""
        
        # Extract features
        features = self.extract_features(query, cluster_feature_embedding)
        
        # Get Q-values from neural network
        q_values = self.neural_bandit.predict(features)
        
        # Update epsilon
        confidence = self._calculate_confidence()
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        
        # Epsilon-greedy selection
        if np.random.random() < self.epsilon:
            selected_strategy = np.random.choice(STRATEGIES)
            selection_method = 'exploration'
        else:
            if best_strategy_for_cluster and np.random.random() < 0.3 and self.query_count > 10:
                selected_strategy = best_strategy_for_cluster
                selection_method = 'cluster_best'
            else:
                strategy_idx = np.argmax(q_values)
                selected_strategy = STRATEGIES[strategy_idx]
                selection_method = 'neural_network'
        
        top_k = STRATEGY_K_VALUES.get(selected_strategy, 4)
        
        selection_info = {
            'method': selection_method,
            'q_values': {STRATEGIES[i]: float(q_values[i]) for i in range(len(STRATEGIES))},
            'epsilon': float(self.epsilon),
            'confidence': float(confidence),
            'cluster_name': cluster_name,
            'best_strategy_for_cluster': best_strategy_for_cluster,
            'selected_strategy': selected_strategy
        }
        
        return selected_strategy, top_k, selection_info
    
    def record_feedback(self, query: str, strategy: str, response: str, feedback: int,
                       cluster_name: str, features: np.ndarray) -> Dict:
        """Record feedback and train neural network"""
        
        reward = POSITIVE_REWARD if feedback > 0 else NEGATIVE_REWARD
        
        # Update strategy statistics
        self.strategy_stats[strategy]['total'] += 1
        self.strategy_stats[strategy]['reward_sum'] += reward
        if reward > 0:
            self.strategy_stats[strategy]['wins'] += 1
        
        # Add to feedback history
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query[:200],
            'strategy': strategy,
            'cluster_name': cluster_name,
            'response': response[:100],
            'feedback': feedback,
            'reward': reward
        }
        self.feedback_history.append(feedback_entry)
        
        # Add experience to neural bandit
        strategy_idx = STRATEGIES.index(strategy)
        self.neural_bandit.add_experience(features, strategy_idx, reward)
        
        # Train neural network
        train_result = {'trained': False}
        if len(self.neural_bandit.replay_buffer) >= BATCH_SIZE:
            train_result = self.neural_bandit.train_batch(BATCH_SIZE)
            if train_result.get('trained', False):
                if train_result['training_step'] % 10 == 0:
                    self.neural_bandit.save_model()
        
        # Check for concept drift
        drift_detected = self._detect_concept_drift()
        if drift_detected:
            self._handle_drift()
        
        # Save statistics
        self._save_strategy_stats()
        self._save_feedback_history()
        
        self.query_count += 1
        
        return {
            'drift_detected': drift_detected,
            'training_result': train_result,
            'new_epsilon': self.epsilon,
            'total_experiences': len(self.neural_bandit.replay_buffer)
        }
    
    def _calculate_confidence(self) -> float:
        """Calculate system confidence"""
        if len(self.feedback_history) < 10:
            return 0.0
        
        recent = self.feedback_history[-20:]
        rewards = [f['reward'] for f in recent]
        
        positive_rate = sum(1 for r in rewards if r > 0) / len(rewards)
        variance = np.var(rewards)
        
        confidence = positive_rate * (1.0 / (1.0 + variance))
        
        return min(confidence, 1.0)
    
    def _detect_concept_drift(self) -> bool:
        """Detect concept drift"""
        if len(self.feedback_history) < DRIFT_WINDOW_SIZE + DRIFT_REFERENCE_SIZE:
            return False
        
        recent_window = self.feedback_history[-DRIFT_WINDOW_SIZE:]
        reference_window = self.feedback_history[-(DRIFT_WINDOW_SIZE + DRIFT_REFERENCE_SIZE):-DRIFT_WINDOW_SIZE]
        
        recent_error_rate = sum(1 for f in recent_window if f['reward'] < 0) / len(recent_window)
        reference_error_rate = sum(1 for f in reference_window if f['reward'] < 0) / len(reference_window)
        
        error_increase = recent_error_rate - reference_error_rate
        
        return error_increase > DRIFT_THRESHOLD
    
    def _handle_drift(self):
        """Handle detected concept drift"""
        self.epsilon = min(0.4, self.epsilon * 1.5)
        
        for strategy in STRATEGIES:
            self.strategy_stats[strategy]['total'] = int(self.strategy_stats[strategy]['total'] * 0.7)
            self.strategy_stats[strategy]['reward_sum'] *= 0.7
            self.strategy_stats[strategy]['wins'] = int(self.strategy_stats[strategy]['wins'] * 0.7)
        
        print(f"⚠️ Concept drift detected! Epsilon increased to {self.epsilon:.3f}")
    
    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        metrics = {
            'total_interactions': len(self.feedback_history),
            'positive_feedback': sum(1 for f in self.feedback_history if f['reward'] > 0),
            'negative_feedback': sum(1 for f in self.feedback_history if f['reward'] < 0),
            'success_rate': 0.0,
            'current_epsilon': self.epsilon,
            'confidence': self._calculate_confidence(),
            'strategy_performance': {}
        }
        
        if metrics['total_interactions'] > 0:
            metrics['success_rate'] = metrics['positive_feedback'] / metrics['total_interactions']
        
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
                'wins': stats['wins'],
                'win_rate': round(win_rate, 3),
                'avg_reward': round(avg_reward, 3)
            }
        
        metrics['neural_network'] = self.neural_bandit.get_training_metrics()
        
        if len(self.feedback_history) >= 20:
            recent = self.feedback_history[-20:]
            recent_positive = sum(1 for f in recent if f['reward'] > 0)
            metrics['recent_success_rate'] = recent_positive / 20
        else:
            metrics['recent_success_rate'] = metrics['success_rate']
        
        return metrics
    
    def get_learning_history(self) -> Dict:
        """Get learning history for visualization"""
        history = {
            'timestamps': [],
            'rewards': [],
            'strategies': [],
            'cluster_names': [],
            'cumulative_reward': []
        }
        
        cumulative = 0
        for feedback in self.feedback_history[-100:]:
            history['timestamps'].append(feedback['timestamp'])
            history['rewards'].append(feedback['reward'])
            history['strategies'].append(feedback['strategy'])
            history['cluster_names'].append(feedback.get('cluster_name', 'unknown'))
            cumulative += feedback['reward']
            history['cumulative_reward'].append(cumulative)
        
        return history