# testing_buddy.py
"""Enhanced Automated Testing Buddy with Aligned Persona Preferences"""

import time
import random
import numpy as np
from typing import List, Dict, Callable, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from config import *
from prompt_template import QUERY_GENERATION_PROMPT
from collections import defaultdict
from scipy import stats


class UserPersona:
    """Simulated user with specific preferences - ALIGNED with actual preferences"""
    
    def __init__(self, name: str, preferred_strategies: List[str], 
                 consistency: float = 0.85, expertise_level: str = "intermediate"):
        self.name = name
        self.preferred_strategies = set(preferred_strategies)
        self.consistency = consistency
        self.expertise_level = expertise_level
        self.query_history = []
        self.feedback_history = []
        
    def evaluate_response(self, strategy: str, answer: str, query_complexity: str) -> int:
        """
        Evaluate response - STRONGLY favor preferred strategies
        This makes testing results align with user expectations
        """
        # STRONG preference for selected strategies
        if strategy in self.preferred_strategies:
            # 85% chance of positive feedback for preferred strategies
            base_prob = 0.85
            
            # Bonus for complexity matching (makes it even better)
            complexity_bonus = {
                "simple": {"concise": 0.10},
                "moderate": {"detailed": 0.10, "structured": 0.10},
                "complex": {"analytical": 0.10, "detailed": 0.05}
            }
            
            if query_complexity in complexity_bonus:
                if strategy in complexity_bonus[query_complexity]:
                    base_prob = min(0.95, base_prob + complexity_bonus[query_complexity][strategy])
        else:
            # Non-preferred strategies get much lower probability
            base_prob = 0.35
            
            # Even if complexity matches, still penalize non-preferred
            complexity_match = {
                "simple": {"concise": 0.05},
                "moderate": {"detailed": 0.05, "structured": 0.05},
                "complex": {"analytical": 0.05, "detailed": 0.05}
            }
            
            if query_complexity in complexity_match and strategy in complexity_match[query_complexity]:
                base_prob = min(0.45, base_prob + complexity_match[query_complexity][strategy])
        
        # Apply consistency
        if random.random() > self.consistency:
            base_prob = 1 - base_prob  # Inconsistent feedback (rare)
        
        # Return feedback
        feedback = 1 if random.random() < base_prob else -1
        self.feedback_history.append({
            'strategy': strategy,
            'feedback': feedback,
            'complexity': query_complexity
        })
        
        return feedback
    
    def get_statistics(self) -> Dict:
        """Get persona statistics"""
        if not self.feedback_history:
            return {}
        
        total = len(self.feedback_history)
        positive = sum(1 for f in self.feedback_history if f['feedback'] > 0)
        
        strategy_stats = defaultdict(lambda: {'positive': 0, 'total': 0})
        for f in self.feedback_history:
            strategy_stats[f['strategy']]['total'] += 1
            if f['feedback'] > 0:
                strategy_stats[f['strategy']]['positive'] += 1
        
        return {
            'name': self.name,
            'total_interactions': total,
            'satisfaction_rate': positive / total,
            'preferred_strategies': list(self.preferred_strategies),
            'strategy_performance': {
                s: stats['positive'] / stats['total'] if stats['total'] > 0 else 0
                for s, stats in strategy_stats.items()
            }
        }


class BaselineStrategy:
    """Baseline strategies for comparison"""
    
    @staticmethod
    def random_strategy() -> str:
        """Random strategy selection"""
        return random.choice(STRATEGIES)
    
    @staticmethod
    def round_robin() -> str:
        """Round-robin strategy selection"""
        if not hasattr(BaselineStrategy, '_rr_counter'):
            BaselineStrategy._rr_counter = 0
        strategy = STRATEGIES[BaselineStrategy._rr_counter % len(STRATEGIES)]
        BaselineStrategy._rr_counter += 1
        return strategy
    
    @staticmethod
    def fixed_best(best_strategy: str = "detailed") -> str:
        """Always use the same strategy"""
        return best_strategy
    
    @staticmethod
    def complexity_based(complexity: str) -> str:
        """Strategy based only on complexity"""
        mapping = {
            "simple": "concise",
            "moderate": "structured",
            "complex": "analytical"
        }
        return mapping.get(complexity, "detailed")


class EvaluationMetrics:
    """Comprehensive evaluation metrics"""
    
    def __init__(self):
        self.interactions = []
        self.baseline_interactions = {}
        
    def add_interaction(self, strategy: str, reward: float, 
                       query: str, complexity: str, system: str = "rl"):
        """Record an interaction"""
        interaction = {
            'strategy': strategy,
            'reward': reward,
            'query': query,
            'complexity': complexity,
            'system': system,
            'timestamp': time.time()
        }
        
        if system == "rl":
            self.interactions.append(interaction)
        else:
            if system not in self.baseline_interactions:
                self.baseline_interactions[system] = []
            self.baseline_interactions[system].append(interaction)
    
    def calculate_cumulative_reward(self, system: str = "rl") -> List[float]:
        """Calculate cumulative reward over time"""
        if system == "rl":
            interactions = self.interactions
        else:
            interactions = self.baseline_interactions.get(system, [])
        
        if not interactions:
            return []
        
        cumulative = []
        total = 0
        for interaction in interactions:
            total += interaction['reward']
            cumulative.append(total)
        
        return cumulative
    
    def calculate_regret(self, optimal_reward: float = 1.0) -> List[float]:
        """Calculate regret compared to optimal strategy"""
        regret = []
        cumulative_regret = 0
        
        for interaction in self.interactions:
            instant_regret = optimal_reward - interaction['reward']
            cumulative_regret += instant_regret
            regret.append(cumulative_regret)
        
        return regret
    
    def calculate_success_rate(self, system: str = "rl", window: int = None) -> float:
        """Calculate success rate (positive feedback ratio)"""
        if system == "rl":
            interactions = self.interactions
        else:
            interactions = self.baseline_interactions.get(system, [])
        
        if not interactions:
            return 0.0
        
        if window:
            interactions = interactions[-window:]
        
        positive = sum(1 for i in interactions if i['reward'] > 0)
        return positive / len(interactions) if interactions else 0.0
    
    def compare_systems(self) -> Dict:
        """Compare RL system against baselines"""
        rl_rewards = [i['reward'] for i in self.interactions]
        
        comparisons = {}
        for system_name, baseline_data in self.baseline_interactions.items():
            baseline_rewards = [i['reward'] for i in baseline_data]
            
            if len(rl_rewards) == 0 or len(baseline_rewards) == 0:
                continue
            
            # Statistical test
            t_stat, p_value = stats.ttest_ind(rl_rewards, baseline_rewards)
            
            comparisons[system_name] = {
                'rl_mean': np.mean(rl_rewards),
                'baseline_mean': np.mean(baseline_rewards),
                'rl_std': np.std(rl_rewards),
                'baseline_std': np.std(baseline_rewards),
                'improvement': (np.mean(rl_rewards) - np.mean(baseline_rewards)) / abs(np.mean(baseline_rewards)) * 100 if np.mean(baseline_rewards) != 0 else 0,
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'better': np.mean(rl_rewards) > np.mean(baseline_rewards)
            }
        
        return comparisons
    
    def calculate_strategy_distribution(self, system: str = "rl") -> Dict[str, float]:
        """Calculate strategy usage distribution"""
        if system == "rl":
            interactions = self.interactions
        else:
            interactions = self.baseline_interactions.get(system, [])
        
        if not interactions:
            return {}
        
        strategy_counts = defaultdict(int)
        for interaction in interactions:
            strategy_counts[interaction['strategy']] += 1
        
        total = len(interactions)
        return {s: count / total for s, count in strategy_counts.items()}
    
    def get_summary(self) -> Dict:
        """Get comprehensive evaluation summary"""
        if not self.interactions:
            return {'error': 'No interactions recorded'}
        
        rl_rewards = [i['reward'] for i in self.interactions]
        
        return {
            'total_interactions': len(self.interactions),
            'rl_system': {
                'mean_reward': float(np.mean(rl_rewards)),
                'std_reward': float(np.std(rl_rewards)),
                'success_rate': self.calculate_success_rate('rl'),
                'final_cumulative_reward': self.calculate_cumulative_reward('rl')[-1] if self.interactions else 0,
                'strategy_distribution': self.calculate_strategy_distribution('rl')
            },
            'baselines': {
                name: {
                    'mean_reward': float(np.mean([i['reward'] for i in data])),
                    'success_rate': self.calculate_success_rate(name),
                    'strategy_distribution': self.calculate_strategy_distribution(name)
                }
                for name, data in self.baseline_interactions.items()
            },
            'comparisons': self.compare_systems(),
            'learning_curve': {
                'early_performance': self.calculate_success_rate('rl', window=10),
                'late_performance': self.calculate_success_rate('rl', window=10) if len(self.interactions) > 10 else 0,
            }
        }


class TestingBuddy:
    """Enhanced automated testing system with aligned preferences"""
    
    def __init__(self, rag_system, preferred_styles: List[str]):
        """Initialize Testing Buddy with user's preferred styles"""
        self.rag_system = rag_system
        self.preferred_styles = set(preferred_styles)
        
        # LLM for query generation
        self.query_generator = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.8,
            openai_api_key=self.rag_system.openai_api_key
        )
        
        # Create persona that MATCHES the selected preferences
        self.persona = UserPersona(
            f"User preferring {', '.join(preferred_styles)}", 
            preferred_styles,
            consistency=0.90,  # High consistency
            expertise_level="intermediate"
        )
        
        # Evaluation metrics
        self.metrics = EvaluationMetrics()
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'positive_feedback': 0,
            'negative_feedback': 0,
            'strategy_usage': {s: 0 for s in STRATEGIES},
            'total_time_ms': 0.0,
            'baseline_stats': {}
        }
    
    def generate_test_queries(self, text: str, num_queries: int = 20) -> List[str]:
        """Generate diverse test queries from text"""
        if len(text) > 3000:
            text = text[:3000] + "..."
        
        prompt = PromptTemplate.from_template(QUERY_GENERATION_PROMPT)
        formatted_prompt = prompt.format(text=text, num_queries=num_queries)
        
        try:
            response = self.query_generator.invoke(formatted_prompt)
            generated_text = response.content.strip()
            
            queries = []
            for line in generated_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    query = line.lstrip('0123456789.-) ').strip()
                    if query and len(query) > 10:
                        queries.append(query)
            
            if not queries:
                queries = [q.strip() for q in generated_text.split('\n') if len(q.strip()) > 10]
            
            return queries[:num_queries]
            
        except Exception as e:
            return self._generate_fallback_queries(text, num_queries)
    
    def _generate_fallback_queries(self, text: str, num_queries: int) -> List[str]:
        """Fallback query generation"""
        words = text.split()
        if len(words) < 10:
            return ["What is this about?"] * min(num_queries, 5)
        
        templates = [
            "What is {}?",
            "Explain {}",
            "How does {} work?",
            "What are the benefits of {}?",
            "Describe {}"
        ]
        
        topics = [w.strip('.,!?') for w in words if len(w) > 5 and w[0].isupper()]
        topics = list(set(topics))[:20]
        
        queries = []
        for i in range(min(num_queries, len(topics))):
            template = random.choice(templates)
            topic = topics[i % len(topics)]
            queries.append(template.format(topic))
        
        return queries
    
    def run_baseline_comparison(self, queries: List[str], progress_callback: Optional[Callable] = None) -> Dict:
        """Run baseline strategies for comparison"""
        baseline_results = {}
        
        baselines = {
            'random': BaselineStrategy.random_strategy,
            'round_robin': BaselineStrategy.round_robin,
            'fixed_best': lambda: BaselineStrategy.fixed_best('detailed')
        }
        
        for baseline_name, baseline_func in baselines.items():
            baseline_rewards = []
            
            for idx, query in enumerate(queries):
                try:
                    complexity = self.rag_system.analyze_query_complexity(query)
                    strategy = baseline_func()
                    
                    answer, metadata = self.rag_system.query(query)
                    
                    feedback = self.persona.evaluate_response(strategy, answer, complexity)
                    reward = POSITIVE_REWARD if feedback > 0 else NEGATIVE_REWARD
                    
                    self.metrics.add_interaction(strategy, reward, query, complexity, baseline_name)
                    baseline_rewards.append(reward)
                    
                    time.sleep(0.05)
                    
                except Exception as e:
                    continue
            
            baseline_results[baseline_name] = {
                'mean_reward': np.mean(baseline_rewards) if baseline_rewards else 0,
                'success_rate': sum(1 for r in baseline_rewards if r > 0) / len(baseline_rewards) if baseline_rewards else 0
            }
        
        return baseline_results
    
    def run_comprehensive_testing(
        self,
        text: str,
        num_queries: int = 30,
        include_baselines: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """Run comprehensive testing with aligned preferences"""
        
        if progress_callback:
            progress_callback(0, num_queries, "📝 Generating test queries...")
        
        queries = self.generate_test_queries(text, num_queries)
        
        if not queries:
            return {'error': 'Failed to generate queries'}
        
        if progress_callback:
            progress_callback(0, len(queries), f"👤 Testing with persona: {self.persona.name}")
        
        # Run RL system tests
        for idx, query in enumerate(queries):
            if progress_callback:
                progress_callback(idx + 1, len(queries), f"🧪 Testing ({idx + 1}/{len(queries)}): {query[:50]}...")
            
            try:
                complexity = self.rag_system.analyze_query_complexity(query)
                
                start_time = time.time()
                answer, metadata = self.rag_system.query(query)
                query_time = (time.time() - start_time) * 1000
                
                strategy = metadata['strategy']
                
                feedback = self.persona.evaluate_response(strategy, answer, complexity)
                self.rag_system.submit_feedback(feedback)
                
                reward = POSITIVE_REWARD if feedback > 0 else NEGATIVE_REWARD
                self.metrics.add_interaction(strategy, reward, query, complexity, 'rl')
                
                self.stats['total_queries'] += 1
                self.stats['total_time_ms'] += query_time
                self.stats['strategy_usage'][strategy] += 1
                
                if feedback > 0:
                    self.stats['positive_feedback'] += 1
                else:
                    self.stats['negative_feedback'] += 1
                
                time.sleep(0.1)
                
            except Exception as e:
                if progress_callback:
                    progress_callback(idx + 1, len(queries), f"⚠️ Error: {str(e)[:50]}")
                continue
        
        # Run baseline comparisons
        if include_baselines:
            if progress_callback:
                progress_callback(0, 1, "📊 Running baseline comparisons...")
            
            baseline_results = self.run_baseline_comparison(queries[:15], progress_callback)
            self.stats['baseline_stats'] = baseline_results
        
        results = self._compile_results()
        
        return results
    
    def _compile_results(self) -> Dict:
        """Compile comprehensive results"""
        evaluation_summary = self.metrics.get_summary()
        persona_stats = self.persona.get_statistics()
        
        results = {
            'total_queries': self.stats['total_queries'],
            'positive_feedback': self.stats['positive_feedback'],
            'negative_feedback': self.stats['negative_feedback'],
            'success_rate': (self.stats['positive_feedback'] / self.stats['total_queries'] * 100)
                           if self.stats['total_queries'] > 0 else 0,
            'strategy_usage': self.stats['strategy_usage'],
            'avg_time_ms': (self.stats['total_time_ms'] / self.stats['total_queries'])
                          if self.stats['total_queries'] > 0 else 0,
            'persona': persona_stats,
            'evaluation': evaluation_summary,
            'baseline_comparisons': self.stats.get('baseline_stats', {}),
            'learning_metrics': {
                'cumulative_reward': self.metrics.calculate_cumulative_reward('rl'),
                'regret': self.metrics.calculate_regret(),
                'early_success_rate': self.metrics.calculate_success_rate('rl', window=10),
                'late_success_rate': self.metrics.calculate_success_rate('rl', window=10) if self.stats['total_queries'] > 10 else 0
            }
        }
        
        return results
    
    def run_automated_testing(
        self,
        text: str,
        num_queries: int = 20,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """Quick testing without baselines"""
        return self.run_comprehensive_testing(text, num_queries, include_baselines=False, progress_callback=progress_callback)