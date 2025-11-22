# neural_bandit.py
"""Enhanced Neural Network with Advanced Visualization Metrics"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import json
import os
from datetime import datetime


class StrategyScorer(nn.Module):
    """Neural network that scores strategies based on context"""
    
    def __init__(self, input_dim: int = 784, hidden_dims: List[int] = [256, 128, 64], 
                 output_dim: int = 5, dropout: float = 0.2):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        current_dim = input_dim
        
        # Build hidden layers with separate components for tracking
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            self.activations.append(nn.ReLU())
            self.dropouts.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(current_dim, output_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor:
        """Forward pass through network"""
        if return_intermediates:
            intermediates = {}
            current = x
            intermediates['input'] = current.detach().cpu().numpy()
            
            # Hidden layers
            for i, (layer, activation, dropout) in enumerate(zip(self.layers, self.activations, self.dropouts)):
                current = layer(current)
                intermediates[f'layer_{i}_pre_activation'] = current.detach().cpu().numpy()
                
                current = activation(current)
                intermediates[f'layer_{i}_post_activation'] = current.detach().cpu().numpy()
                
                current = dropout(current)
                intermediates[f'layer_{i}_output'] = current.detach().cpu().numpy()
            
            # Output layer
            output = self.output_layer(current)
            intermediates['output'] = output.detach().cpu().numpy()
            
            return output, intermediates
        else:
            current = x
            for layer, activation, dropout in zip(self.layers, self.activations, self.dropouts):
                current = dropout(activation(layer(current)))
            return self.output_layer(current)
    
    def get_weight_statistics(self) -> Dict:
        """Get detailed weight statistics for each layer"""
        stats = {}
        
        for i, layer in enumerate(self.layers):
            weight = layer.weight.detach().cpu().numpy()
            bias = layer.bias.detach().cpu().numpy()
            
            stats[f'layer_{i}'] = {
                'weight_mean': float(np.mean(weight)),
                'weight_std': float(np.std(weight)),
                'weight_min': float(np.min(weight)),
                'weight_max': float(np.max(weight)),
                'weight_norm': float(np.linalg.norm(weight)),
                'bias_mean': float(np.mean(bias)),
                'weight_shape': list(weight.shape)
            }
        
        # Output layer
        weight = self.output_layer.weight.detach().cpu().numpy()
        bias = self.output_layer.bias.detach().cpu().numpy()
        
        stats['output_layer'] = {
            'weight_mean': float(np.mean(weight)),
            'weight_std': float(np.std(weight)),
            'weight_min': float(np.min(weight)),
            'weight_max': float(np.max(weight)),
            'weight_norm': float(np.linalg.norm(weight)),
            'bias_mean': float(np.mean(bias)),
            'weight_shape': list(weight.shape)
        }
        
        return stats


class NeuralBandit:
    """Enhanced Neural Contextual Bandit with Advanced Visualization"""
    
    def __init__(self, user_id: str, learning_rate: float = 0.001):
        self.user_id = user_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize neural network
        self.model = StrategyScorer().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Training buffer
        self.training_buffer = deque(maxlen=500)
        
        # Enhanced metrics for visualization
        self.training_history = {
            'losses': [],
            'timestamps': [],
            'batch_numbers': [],
            'learning_rates': [],
            'gradient_norms': [],
            'weight_norms': [],
            'prediction_confidence': [],
            'strategy_predictions': {i: [] for i in range(5)}
        }
        
        # Per-query learning tracking
        self.query_learning_log = []
        
        # Weight evolution tracking
        self.weight_snapshots = []
        self.weight_snapshot_frequency = 5
        
        self.batch_count = 0
        
        # Load if exists
        self.model_path = f"./rl_data/neural_model_{user_id}.pt"
        self.history_path = f"./rl_data/training_history_{user_id}.json"
        self.query_log_path = f"./rl_data/query_learning_{user_id}.json"
        self._load_model()
    
    def predict(self, features: np.ndarray, return_confidence: bool = False):
        """Predict strategy scores with optional confidence"""
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            if len(features_tensor.shape) == 1:
                features_tensor = features_tensor.unsqueeze(0)
            
            scores = self.model(features_tensor)
            scores_np = scores.cpu().numpy().squeeze()
            
            if return_confidence:
                # Calculate confidence as entropy
                probs = torch.softmax(scores, dim=-1).cpu().numpy().squeeze()
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                max_entropy = np.log(len(probs))
                confidence = 1 - (entropy / max_entropy)
                
                return scores_np, confidence
            
            return scores_np
    
    def predict_with_intermediates(self, features: np.ndarray) -> Tuple:
        """Predict with intermediate activations for visualization"""
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            if len(features_tensor.shape) == 1:
                features_tensor = features_tensor.unsqueeze(0)
            
            scores, intermediates = self.model(features_tensor, return_intermediates=True)
            scores_np = scores.cpu().numpy().squeeze()
            
            return scores_np, intermediates
    
    def add_experience(self, features: np.ndarray, strategy_idx: int, reward: float,
                      query: str = "", cluster: str = ""):
        """Add experience to training buffer with metadata"""
        self.training_buffer.append({
            'features': features,
            'strategy': strategy_idx,
            'reward': reward,
            'query': query[:100],
            'cluster': cluster,
            'timestamp': datetime.now().isoformat()
        })
    
    def train_batch(self, batch_size: int = 10) -> Dict:
        """Train on batch with comprehensive metrics"""
        if len(self.training_buffer) < batch_size:
            return {'trained': False, 'reason': 'insufficient_data'}
        
        # Sample batch
        indices = np.random.choice(len(self.training_buffer), batch_size, replace=False)
        batch = [self.training_buffer[i] for i in indices]
        
        # Store pre-training predictions
        pre_predictions = []
        for sample in batch:
            with torch.no_grad():
                pred_scores = self.predict(sample['features'])
                pre_predictions.append(pred_scores.copy())
        
        # Prepare tensors
        features = torch.FloatTensor([b['features'] for b in batch]).to(self.device)
        strategies = torch.LongTensor([b['strategy'] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b['reward'] for b in batch]).to(self.device)
        
        # Training
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        predicted_scores = self.model(features)
        predicted_values = predicted_scores[range(len(batch)), strategies]
        
        # Compute loss
        loss = self.criterion(predicted_values, rewards)
        
        # Backward pass
        loss.backward()
        
        # Calculate gradient norm
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        gradient_norm = total_norm ** 0.5
        
        self.optimizer.step()
        
        # Calculate weight norm
        total_weight_norm = 0.0
        for p in self.model.parameters():
            param_norm = p.data.norm(2)
            total_weight_norm += param_norm.item() ** 2
        weight_norm = total_weight_norm ** 0.5
        
        # Get post-training predictions
        self.model.eval()
        post_predictions = []
        with torch.no_grad():
            for sample in batch:
                pred_scores = self.predict(sample['features'])
                post_predictions.append(pred_scores.copy())
        
        # Calculate prediction changes
        prediction_changes = []
        for pre, post, sample in zip(pre_predictions, post_predictions, batch):
            strategy_idx = sample['strategy']
            change = post[strategy_idx] - pre[strategy_idx]
            prediction_changes.append({
                'query': sample['query'],
                'strategy': strategy_idx,
                'reward': sample['reward'],
                'pre_score': float(pre[strategy_idx]),
                'post_score': float(post[strategy_idx]),
                'change': float(change),
                'all_pre_scores': pre.tolist(),
                'all_post_scores': post.tolist()
            })
        
        # Record metrics
        self.batch_count += 1
        self.training_history['losses'].append(loss.item())
        self.training_history['timestamps'].append(len(self.training_buffer))
        self.training_history['batch_numbers'].append(self.batch_count)
        self.training_history['gradient_norms'].append(gradient_norm)
        self.training_history['weight_norms'].append(weight_norm)
        self.training_history['learning_rates'].append(
            self.optimizer.param_groups[0]['lr']
        )
        
        # Calculate prediction confidence
        avg_confidence = np.mean([
            1 - (np.std(pred) / (np.max(pred) - np.min(pred) + 1e-6))
            for pred in post_predictions
        ])
        self.training_history['prediction_confidence'].append(avg_confidence)
        
        # Track strategy-specific predictions
        for pre, post in zip(pre_predictions, post_predictions):
            for strategy_idx in range(5):
                self.training_history['strategy_predictions'][strategy_idx].append(
                    float(post[strategy_idx])
                )
        
        # Log query-level learning
        self.query_learning_log.extend(prediction_changes)
        
        # Keep only recent logs (last 100)
        if len(self.query_learning_log) > 100:
            self.query_learning_log = self.query_learning_log[-100:]
        
        # Save weight snapshot periodically
        if self.batch_count % self.weight_snapshot_frequency == 0:
            self.weight_snapshots.append({
                'batch': self.batch_count,
                'weights': self.model.get_weight_statistics()
            })
            # Keep only last 20 snapshots
            if len(self.weight_snapshots) > 20:
                self.weight_snapshots = self.weight_snapshots[-20:]
        
        return {
            'trained': True,
            'loss': loss.item(),
            'batch_size': batch_size,
            'buffer_size': len(self.training_buffer),
            'batch_number': self.batch_count,
            'gradient_norm': gradient_norm,
            'weight_norm': weight_norm,
            'prediction_changes': prediction_changes,
            'avg_prediction_change': float(np.mean([abs(c['change']) for c in prediction_changes]))
        }
    
    def get_training_metrics(self) -> Dict:
        """Get comprehensive training metrics"""
        return {
            'total_batches': self.batch_count,
            'buffer_size': len(self.training_buffer),
            'recent_losses': self.training_history['losses'][-20:] if self.training_history['losses'] else [],
            'all_losses': self.training_history['losses'],
            'batch_numbers': self.training_history['batch_numbers'],
            'gradient_norms': self.training_history['gradient_norms'],
            'weight_norms': self.training_history['weight_norms'],
            'learning_rates': self.training_history['learning_rates'],
            'prediction_confidence': self.training_history['prediction_confidence'],
            'strategy_predictions': self.training_history['strategy_predictions'],
            'query_learning_log': self.query_learning_log[-20:],  # Recent 20
            'weight_snapshots': self.weight_snapshots
        }
    
    def get_layer_activations(self, features: np.ndarray) -> Dict:
        """Get activation patterns for visualization"""
        scores, intermediates = self.predict_with_intermediates(features)
        
        activation_stats = {}
        for key, values in intermediates.items():
            if 'post_activation' in key:
                flat_values = values.flatten()
                activation_stats[key] = {
                    'mean': float(np.mean(flat_values)),
                    'std': float(np.std(flat_values)),
                    'sparsity': float(np.mean(flat_values == 0)),  # ReLU zeros
                    'max': float(np.max(flat_values)),
                    'sample': flat_values[:50].tolist()
                }
        
        return activation_stats
    
    def save_model(self):
        """Save model and training history"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'batch_count': self.batch_count
        }, self.model_path)
        
        # Save history
# Save history (convert numpy types to native Python types)
        with open(self.history_path, 'w') as f:
            json.dump(self.training_history, f, default=lambda x: float(x) if hasattr(x, 'item') else x)
        
        # Save query log
        with open(self.query_log_path, 'w') as f:
            json.dump(self.query_learning_log, f)
    
    def _load_model(self):
        """Load model if exists"""
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.batch_count = checkpoint.get('batch_count', 0)
            except:
                pass
        
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    self.training_history = json.load(f)
            except:
                pass
        
        if os.path.exists(self.query_log_path):
            try:
                with open(self.query_log_path, 'r') as f:
                    self.query_learning_log = json.load(f)
            except:
                pass
    
    def batch_train_synthetic(self, synthetic_data: List[Dict]) -> Dict:
        """Train on synthetic data batch"""
        if not synthetic_data:
            return {'trained': False}
        
        # Add all to buffer
        for sample in synthetic_data:
            self.training_buffer.append(sample)
        
        # Train multiple batches
        total_loss = 0.0
        batches_trained = 0
        
        while len(self.training_buffer) >= 10:
            result = self.train_batch(10)
            if result['trained']:
                total_loss += result['loss']
                batches_trained += 1
        
        avg_loss = total_loss / batches_trained if batches_trained > 0 else 0.0
        
        return {
            'trained': True,
            'batches': batches_trained,
            'avg_loss': avg_loss
        }