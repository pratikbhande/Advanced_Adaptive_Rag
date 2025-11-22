# clustering.py
"""LLM-based semantic query clustering"""

import json
import os
import numpy as np
from typing import Dict, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer

class LLMQueryClusterer:
    """Intelligent query clustering using LLM"""
    
    def __init__(self, user_id: str, openai_api_key: str):
        self.user_id = user_id
        self.data_dir = "./rl_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.clusters_file = f"{self.data_dir}/query_clusters_{user_id}.json"
        self.clusters = self._load_clusters()
        
        # LLM for clustering
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=openai_api_key
        )
        
        # Sentence transformer for embeddings
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Clustering prompt template
        from prompt_template.clustering_prompt import CLUSTERING_PROMPT
        self.prompt_template = PromptTemplate.from_template(CLUSTERING_PROMPT)
    
    def _load_clusters(self) -> Dict:
        """Load existing clusters"""
        if os.path.exists(self.clusters_file):
            with open(self.clusters_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_clusters(self):
        """Save clusters"""
        with open(self.clusters_file, 'w') as f:
            json.dump(self.clusters, f, indent=2)
    
    def _get_clusters_summary(self) -> str:
        """Get summary of existing clusters for LLM"""
        if not self.clusters:
            return "No existing clusters yet."
        
        summary_parts = []
        for cluster_name, data in list(self.clusters.items())[:15]:
            example_queries = data['queries'][:3]
            query_count = len(data['queries'])
            summary_parts.append(
                f"• {cluster_name} ({query_count} queries): {', '.join(example_queries)}"
            )
        
        return "\n".join(summary_parts)
    
    def assign_cluster(self, query: str) -> Tuple[str, bool, Dict]:
        """
        Assign query to cluster using LLM
        Returns: (cluster_name, is_new_cluster, cluster_info)
        """
        existing_summary = self._get_clusters_summary()
        
        formatted_prompt = self.prompt_template.format(
            query=query,
            existing_clusters=existing_summary
        )
        
        try:
            response = self.llm.invoke(formatted_prompt)
            content = response.content.strip()
            
            # Parse response - expecting "GROUP: cluster_name"
            cluster_name = None
            for line in content.split('\n'):
                if line.startswith('GROUP:'):
                    cluster_name = line.replace('GROUP:', '').strip()
                    break
            
            if not cluster_name:
                cluster_name = f"cluster_{len(self.clusters)}"
            
            # Clean cluster name
            cluster_name = cluster_name.lower().replace(' ', '_')
            
            is_new = cluster_name not in self.clusters
            
            # Create/update cluster
            if is_new:
                self.clusters[cluster_name] = {
                    'queries': [query],
                    'strategy_performance': {},
                    'embedding': self.embedder.encode(query).tolist()
                }
            else:
                if query not in self.clusters[cluster_name]['queries']:
                    self.clusters[cluster_name]['queries'].append(query)
                    # Update cluster embedding (moving average)
                    old_emb = self.clusters[cluster_name]['embedding']
                    new_emb = self.embedder.encode(query).tolist()
                    n = len(self.clusters[cluster_name]['queries'])
                    self.clusters[cluster_name]['embedding'] = [
                        (old_emb[i] * (n-1) + new_emb[i]) / n 
                        for i in range(len(old_emb))
                    ]
            
            self._save_clusters()
            
            cluster_info = {
                'name': cluster_name,
                'query_count': len(self.clusters[cluster_name]['queries']),
                'embedding': self.clusters[cluster_name]['embedding']
            }
            
            return cluster_name, is_new, cluster_info
            
        except Exception as e:
            # Fallback to simple clustering
            return self._fallback_cluster(query)
    
    def _fallback_cluster(self, query: str) -> Tuple[str, bool, Dict]:
        """Fallback clustering using embeddings"""
        query_embedding = self.embedder.encode(query)
        
        if not self.clusters:
            new_cluster = "cluster_0"
            self.clusters[new_cluster] = {
                'queries': [query],
                'strategy_performance': {},
                'embedding': query_embedding.tolist()
            }
            self._save_clusters()
            return new_cluster, True, {
                'name': new_cluster,
                'query_count': 1,
                'embedding': query_embedding.tolist()
            }
        
        # Find most similar cluster
        best_cluster = None
        best_similarity = -1
        
        for cluster_name, data in self.clusters.items():
            cluster_emb = data['embedding']
            similarity = np.dot(query_embedding, cluster_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cluster_emb)
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster_name
        
        # If similarity too low, create new cluster
        if best_similarity < 0.6:
            new_cluster = f"cluster_{len(self.clusters)}"
            self.clusters[new_cluster] = {
                'queries': [query],
                'strategy_performance': {},
                'embedding': query_embedding.tolist()
            }
            self._save_clusters()
            return new_cluster, True, {
                'name': new_cluster,
                'query_count': 1,
                'embedding': query_embedding.tolist()
            }
        
        # Add to existing cluster
        self.clusters[best_cluster]['queries'].append(query)
        self._save_clusters()
        
        return best_cluster, False, {
            'name': best_cluster,
            'query_count': len(self.clusters[best_cluster]['queries']),
            'embedding': self.clusters[best_cluster]['embedding']
        }
    
    def record_strategy_performance(self, cluster_name: str, strategy: str, reward: float):
        """Record strategy performance for cluster"""
        if cluster_name not in self.clusters:
            return
        
        perf = self.clusters[cluster_name]['strategy_performance']
        
        if strategy not in perf:
            perf[strategy] = {'total': 0, 'reward_sum': 0.0}
        
        perf[strategy]['total'] += 1
        perf[strategy]['reward_sum'] += reward
        
        self._save_clusters()
    
    def get_best_strategy_for_cluster(self, cluster_name: str) -> Optional[str]:
        """Get best performing strategy for cluster"""
        if cluster_name not in self.clusters:
            return None
        
        perf = self.clusters[cluster_name]['strategy_performance']
        
        if not perf:
            return None
        
        best_strategy = None
        best_avg_reward = float('-inf')
        
        for strategy, stats in perf.items():
            if stats['total'] >= 3:  # Minimum samples
                avg_reward = stats['reward_sum'] / stats['total']
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_strategy = strategy
        
        return best_strategy if best_avg_reward > 0 else None
    
    def get_cluster_info(self, cluster_name: str) -> Dict:
        """Get detailed cluster information"""
        if cluster_name not in self.clusters:
            return {}
        
        cluster_data = self.clusters[cluster_name]
        
        strategy_stats = {}
        for strategy, stats in cluster_data['strategy_performance'].items():
            if stats['total'] > 0:
                strategy_stats[strategy] = {
                    'uses': stats['total'],
                    'avg_reward': round(stats['reward_sum'] / stats['total'], 3)
                }
        
        return {
            'name': cluster_name,
            'query_count': len(cluster_data['queries']),
            'example_queries': cluster_data['queries'][:3],
            'strategy_performance': strategy_stats,
            'best_strategy': self.get_best_strategy_for_cluster(cluster_name)
        }
    
    def get_all_clusters(self) -> Dict:
        """Get all clusters summary"""
        return {
            name: self.get_cluster_info(name)
            for name in self.clusters.keys()
        }