# neural_bandit.py
"""Enhanced Neural Contextual Bandit with Proper Training"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from config import *


class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network for strategy selection"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super(DuelingDQN, self).__init__()
        
        # Shared feature extractor
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(NEURAL_DROPOUT),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages: Q = V + (A - mean(A))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
    
    def add(self, experience: Tuple, priority: float = 1.0):
        """Add experience with priority"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample batch with prioritization"""
        if len(self.buffer) == 0:
            return [], [], []
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), 
                                   size=min(batch_size, len(self.buffer)),
                                   p=probabilities,
                                   replace=False)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        samples = [self.buffer[idx] for idx in indices]
        
        return samples, indices, weights
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)


class NeuralBandit:
    """Enhanced Neural Contextual Bandit"""
    
    def __init__(self, user_id: str, learning_rate: float = 0.001):
        self.user_id = user_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = DuelingDQN(
            NEURAL_INPUT_DIM,
            NEURAL_OUTPUT_DIM,
            NEURAL_HIDDEN_DIMS
        ).to(self.device)
        
        self.target_net = DuelingDQN(
            NEURAL_INPUT_DIM,
            NEURAL_OUTPUT_DIM,
            NEURAL_HIDDEN_DIMS
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(TRAINING_BUFFER_SIZE)
        
        # Training metrics
        self.training_step = 0
        self.loss_history = deque(maxlen=100)
        self.q_value_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        
        # Beta annealing for importance sampling
        self.beta = REPLAY_BETA_START
        self.beta_increment = (1.0 - REPLAY_BETA_START) / REPLAY_BETA_FRAMES
        
        # Load existing model
        self.model_path = MODELS_PATH / f"neural_bandit_{user_id}.pt"
        self.load_model()
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict Q-values for all actions"""
        self.policy_net.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            q_values = self.policy_net(features_tensor)
            return q_values.cpu().numpy()[0]
    
    def add_experience(self, features: np.ndarray, action: int, reward: float, 
                      next_features: Optional[np.ndarray] = None):
        """Add experience to replay buffer"""
        # Calculate TD error as initial priority
        current_q = self.predict(features)[action]
        priority = abs(reward - current_q) + 1e-6
        
        experience = (features, action, reward, next_features)
        self.replay_buffer.add(experience, priority)
        
        self.reward_history.append(reward)
    
    def train_batch(self, batch_size: int) -> Dict:
        """Train on a batch from replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return {'trained': False, 'reason': 'insufficient_data'}
        
        # Sample batch
        experiences, indices, weights = self.replay_buffer.sample(batch_size, self.beta)
        
        if not experiences:
            return {'trained': False, 'reason': 'sampling_failed'}
        
        # Prepare batch
        states = torch.FloatTensor([exp[0] for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp[1] for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp[2] for exp in experiences]).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        self.policy_net.train()
        current_q_values = self.policy_net(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values (using target network)
        with torch.no_grad():
            target_q_values = self.target_net(states)
            # Double DQN: use policy net to select action, target net to evaluate
            next_actions = self.policy_net(states).argmax(dim=1)
            target_q = target_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + NEURAL_GAMMA * target_q
        
        # Compute loss with importance sampling weights
        td_errors = target - current_q
        loss = (weights_tensor * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities in replay buffer
        priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, priorities.tolist())
        
        # Update target network
        self.training_step += 1
        if self.training_step % TARGET_UPDATE_FREQUENCY == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Store metrics
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        self.q_value_history.append(current_q.mean().item())
        
        return {
            'trained': True,
            'loss': loss_value,
            'avg_q_value': current_q.mean().item(),
            'training_step': self.training_step,
            'buffer_size': len(self.replay_buffer),
            'beta': self.beta
        }
    
    def save_model(self):
        """Save model to disk"""
        torch.save({
            'policy_net_state': self.policy_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'beta': self.beta
        }, self.model_path)
    
    def load_model(self):
        """Load model from disk"""
        if self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['policy_net_state'])
                self.target_net.load_state_dict(checkpoint['target_net_state'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                self.training_step = checkpoint.get('training_step', 0)
                self.beta = checkpoint.get('beta', REPLAY_BETA_START)
                print(f"Model loaded: step {self.training_step}")
            except Exception as e:
                print(f"Failed to load model: {e}")
    
    def get_training_metrics(self) -> Dict:
        """Get training metrics"""
        return {
            'training_steps': self.training_step,
            'buffer_size': len(self.replay_buffer),
            'recent_loss': np.mean(self.loss_history) if self.loss_history else 0.0,
            'recent_q_value': np.mean(self.q_value_history) if self.q_value_history else 0.0,
            'recent_reward': np.mean(self.reward_history) if self.reward_history else 0.0,
            'loss_history': list(self.loss_history),
            'q_value_history': list(self.q_value_history),
            'reward_history': list(self.reward_history),
            'beta': self.beta
        }