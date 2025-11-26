# clustering.py
"""Intelligent multi-dimensional query clustering"""

import json
import numpy as np
from typing import Dict, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from config import *
from prompt_template import INTELLIGENT_QUERY_ANALYSIS_PROMPT


class IntelligentQueryClusterer:
    """Multi-dimensional query clustering for precise intent understanding"""
    
    def __init__(self, user_id: str, openai_api_key: str):
        self.user_id = user_id
        self.data_dir = RL_DATA_PATH / f"clusters_{user_id}"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.clusters_file = self.data_dir / "query_clusters.json"
        self.clusters = self._load_clusters()
        
        # LLM for classification
        self.llm = ChatOpenAI(
            model=ANALYZER_MODEL,
            temperature=ANALYZER_TEMPERATURE,
            openai_api_key=openai_api_key
        )
        
        self.prompt_template = PromptTemplate.from_template(INTELLIGENT_QUERY_ANALYSIS_PROMPT)
    
    def _load_clusters(self) -> Dict:
        """Load existing clusters"""
        if self.clusters_file.exists():
            with open(self.clusters_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_clusters(self):
        """Save clusters"""
        with open(self.clusters_file, 'w') as f:
            json.dump(self.clusters, f, indent=2)
    
    def classify_query(self, query: str) -> Tuple[str, Dict]:
        """
        Classify query across multiple dimensions
        Returns: (cluster_name, cluster_info)
        """
        formatted_prompt = self.prompt_template.format(query=query)
        
        try:
            response = self.llm.invoke(formatted_prompt)
            content = response.content.strip()
            
            # Parse response
            intent = None
            depth = None
            scope = None
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('INTENT:'):
                    intent = line.split(':', 1)[1].strip().lower()
                elif line.startswith('DEPTH:'):
                    depth = line.split(':', 1)[1].strip().lower()
                elif line.startswith('SCOPE:'):
                    scope = line.split(':', 1)[1].strip().lower()
            
            # Validate and fallback
            if intent not in INTENT_TYPES:
                intent = 'explanation'
            if depth not in DEPTH_LEVELS:
                depth = 'moderate'
            if scope not in SCOPE_TYPES:
                scope = 'specific'
            
            # Create cluster name
            cluster_name = f"{intent}_{depth}_{scope}"
            
            # Initialize cluster if new
            if cluster_name not in self.clusters:
                self.clusters[cluster_name] = {
                    'queries': [],
                    'strategy_performance': {},
                    'intent': intent,
                    'depth': depth,
                    'scope': scope
                }
            
            # Add query
            if query not in self.clusters[cluster_name]['queries']:
                self.clusters[cluster_name]['queries'].append(query)
                # Keep last 50 queries per cluster
                if len(self.clusters[cluster_name]['queries']) > 50:
                    self.clusters[cluster_name]['queries'] = self.clusters[cluster_name]['queries'][-50:]
            
            self._save_clusters()
            
            cluster_info = {
                'name': cluster_name,
                'intent': intent,
                'depth': depth,
                'scope': scope,
                'query_count': len(self.clusters[cluster_name]['queries']),
                'feature_embedding': self._get_feature_embedding(intent, depth, scope)
            }
            
            return cluster_name, cluster_info
            
        except Exception as e:
            print(f"Classification error: {e}")
            # Simple fallback with defaults
            intent = 'explanation'
            depth = 'moderate'
            scope = 'specific'
            cluster_name = f"{intent}_{depth}_{scope}"
            
            if cluster_name not in self.clusters:
                self.clusters[cluster_name] = {
                    'queries': [],
                    'strategy_performance': {},
                    'intent': intent,
                    'depth': depth,
                    'scope': scope
                }
            
            self.clusters[cluster_name]['queries'].append(query)
            self._save_clusters()
            
            return cluster_name, {
                'name': cluster_name,
                'intent': intent,
                'depth': depth,
                'scope': scope,
                'query_count': len(self.clusters[cluster_name]['queries']),
                'feature_embedding': self._get_feature_embedding(intent, depth, scope)
            }
    
    def _get_feature_embedding(self, intent: str, depth: str, scope: str) -> list:
        """Get multi-hot encoding for query dimensions"""
        embedding = []
        
        # Intent one-hot (6 dimensions)
        intent_vec = [0.0] * len(INTENT_TYPES)
        if intent in INTENT_TYPES:
            intent_vec[INTENT_TYPES.index(intent)] = 1.0
        embedding.extend(intent_vec)
        
        # Depth one-hot (3 dimensions)
        depth_vec = [0.0] * len(DEPTH_LEVELS)
        if depth in DEPTH_LEVELS:
            depth_vec[DEPTH_LEVELS.index(depth)] = 1.0
        embedding.extend(depth_vec)
        
        # Scope one-hot (2 dimensions)
        scope_vec = [0.0] * len(SCOPE_TYPES)
        if scope in SCOPE_TYPES:
            scope_vec[SCOPE_TYPES.index(scope)] = 1.0
        embedding.extend(scope_vec)
        
        return embedding  # Total: 11 dimensions
    
    def record_strategy_performance(self, cluster_name: str, strategy: str, reward: float):
        """Record strategy performance for cluster"""
        if cluster_name not in self.clusters:
            return
        
        perf = self.clusters[cluster_name]['strategy_performance']
        
        if strategy not in perf:
            perf[strategy] = {'total': 0, 'wins': 0, 'reward_sum': 0.0}
        
        perf[strategy]['total'] += 1
        perf[strategy]['reward_sum'] += reward
        if reward > 0:
            perf[strategy]['wins'] += 1
        
        self._save_clusters()
    
    def get_best_strategy_for_cluster(self, cluster_name: str) -> Optional[str]:
        """Get best performing strategy for cluster"""
        if cluster_name not in self.clusters:
            return None
        
        perf = self.clusters[cluster_name]['strategy_performance']
        
        if not perf:
            return None
        
        best_strategy = None
        best_win_rate = 0.0
        
        for strategy, stats in perf.items():
            if stats['total'] >= 5:
                win_rate = stats['wins'] / stats['total']
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_strategy = strategy
        
        return best_strategy if best_win_rate > 0.5 else None
    
    def get_cluster_stats(self) -> Dict:
        """Get statistics for all clusters"""
        stats = {}
        
        for cluster_name, cluster_data in self.clusters.items():
            strategy_perf = {}
            for strategy, perf in cluster_data['strategy_performance'].items():
                if perf['total'] > 0:
                    strategy_perf[strategy] = {
                        'uses': perf['total'],
                        'win_rate': round(perf['wins'] / perf['total'], 3),
                        'avg_reward': round(perf['reward_sum'] / perf['total'], 3)
                    }
            
            stats[cluster_name] = {
                'intent': cluster_data.get('intent', 'unknown'),
                'depth': cluster_data.get('depth', 'unknown'),
                'scope': cluster_data.get('scope', 'unknown'),
                'query_count': len(cluster_data['queries']),
                'example_queries': cluster_data['queries'][-3:] if cluster_data['queries'] else [],
                'strategy_performance': strategy_perf,
                'best_strategy': self.get_best_strategy_for_cluster(cluster_name)
            }
        
        return stats