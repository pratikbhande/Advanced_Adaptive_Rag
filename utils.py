# utils.py
"""Enhanced utility functions with evaluation capabilities"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import hashlib
from scipy import stats


# ============================================================================
#                    FILE OPERATIONS
# ============================================================================

def ensure_directory(path: str) -> None:
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)


def save_json(data: Dict[Any, Any], filepath: str) -> None:
    """Save data to JSON file with error handling"""
    try:
        ensure_directory(os.path.dirname(filepath))
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving JSON to {filepath}: {str(e)}")


def load_json(filepath: str, default: Dict = None) -> Dict:
    """Load data from JSON file"""
    if not os.path.exists(filepath):
        return default if default is not None else {}
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {filepath}: {str(e)}")
        return default if default is not None else {}


# ============================================================================
#                    DATETIME UTILITIES
# ============================================================================

def format_timestamp(dt: datetime = None) -> str:
    """Format datetime to ISO string"""
    if dt is None:
        dt = datetime.now()
    return dt.isoformat()


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO timestamp string to datetime"""
    try:
        return datetime.fromisoformat(timestamp_str)
    except:
        return datetime.now()


# ============================================================================
#                    SIMILARITY METRICS
# ============================================================================

def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def calculate_word_overlap(text1: str, text2: str) -> float:
    """Calculate word overlap similarity"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


# ============================================================================
#                    TEXT UTILITIES
# ============================================================================

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def generate_hash(text: str) -> str:
    """Generate hash for text"""
    return hashlib.md5(text.encode()).hexdigest()


# ============================================================================
#                    NUMERICAL UTILITIES
# ============================================================================

def format_number(num: float, decimals: int = 2) -> str:
    """Format number with specified decimals"""
    return f"{num:.{decimals}f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    if denominator == 0:
        return default
    return numerator / denominator


def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to [0, 1] range"""
    if not scores:
        return []
    
    scores_array = np.array(scores)
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)
    
    if max_score == min_score:
        return [0.5] * len(scores)
    
    return ((scores_array - min_score) / (max_score - min_score)).tolist()


# ============================================================================
#                    STATISTICAL UTILITIES
# ============================================================================

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate comprehensive statistics"""
    if not values:
        return {
            'count': 0,
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'var': 0.0,
            'q25': 0.0,
            'q75': 0.0
        }
    
    values = np.array(values)
    
    return {
        'count': len(values),
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'var': float(np.var(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75))
    }


def calculate_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate confidence interval"""
    if not values or len(values) < 2:
        return (0.0, 0.0)
    
    values = np.array(values)
    mean = np.mean(values)
    std_err = stats.sem(values)
    
    interval = std_err * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
    
    return (float(mean - interval), float(mean + interval))


def perform_ttest(
    group1: List[float],
    group2: List[float]
) -> Dict[str, float]:
    """Perform independent t-test"""
    if not group1 or not group2:
        return {
            't_statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'effect_size': 0.0
        }
    
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    # Calculate Cohen's d (effect size)
    pooled_std = np.sqrt(
        ((len(group1) - 1) * np.var(group1, ddof=1) +
         (len(group2) - 1) * np.var(group2, ddof=1)) /
        (len(group1) + len(group2) - 2)
    )
    
    if pooled_std == 0:
        effect_size = 0.0
    else:
        effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'effect_size': float(effect_size)
    }


def calculate_moving_average(
    values: List[float],
    window: int = 10
) -> List[float]:
    """Calculate moving average"""
    if not values or window <= 0:
        return values
    
    if len(values) < window:
        return values
    
    cumsum = np.cumsum(np.insert(values, 0, 0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    
    # Pad beginning with original values
    padded = values[:window-1] + ma.tolist()
    
    return padded


# ============================================================================
#                    EVALUATION METRICS
# ============================================================================

def calculate_regret(
    rewards: List[float],
    optimal_reward: float = 1.0
) -> List[float]:
    """Calculate cumulative regret"""
    if not rewards:
        return []
    
    regret = []
    cumulative = 0.0
    
    for reward in rewards:
        instant_regret = optimal_reward - reward
        cumulative += instant_regret
        regret.append(cumulative)
    
    return regret


def calculate_cumulative_reward(rewards: List[float]) -> List[float]:
    """Calculate cumulative reward over time"""
    if not rewards:
        return []
    
    cumulative = []
    total = 0.0
    
    for reward in rewards:
        total += reward
        cumulative.append(total)
    
    return cumulative


def calculate_success_rate(
    rewards: List[float],
    threshold: float = 0.0
) -> float:
    """Calculate success rate (rewards > threshold)"""
    if not rewards:
        return 0.0
    
    successes = sum(1 for r in rewards if r > threshold)
    return successes / len(rewards)


def calculate_windowed_success_rate(
    rewards: List[float],
    window: int = 10,
    threshold: float = 0.0
) -> List[float]:
    """Calculate success rate over sliding windows"""
    if not rewards or window <= 0:
        return []
    
    success_rates = []
    
    for i in range(len(rewards) - window + 1):
        window_rewards = rewards[i:i + window]
        rate = calculate_success_rate(window_rewards, threshold)
        success_rates.append(rate)
    
    return success_rates


def compare_performance(
    baseline_rewards: List[float],
    treatment_rewards: List[float]
) -> Dict[str, Any]:
    """Comprehensive performance comparison"""
    if not baseline_rewards or not treatment_rewards:
        return {'error': 'Insufficient data for comparison'}
    
    # Basic statistics
    baseline_stats = calculate_statistics(baseline_rewards)
    treatment_stats = calculate_statistics(treatment_rewards)
    
    # Statistical test
    test_results = perform_ttest(treatment_rewards, baseline_rewards)
    
    # Relative improvement
    if baseline_stats['mean'] != 0:
        relative_improvement = (
            (treatment_stats['mean'] - baseline_stats['mean']) /
            abs(baseline_stats['mean']) * 100
        )
    else:
        relative_improvement = 0.0
    
    # Confidence intervals
    baseline_ci = calculate_confidence_interval(baseline_rewards)
    treatment_ci = calculate_confidence_interval(treatment_rewards)
    
    return {
        'baseline': {
            'statistics': baseline_stats,
            'confidence_interval': baseline_ci
        },
        'treatment': {
            'statistics': treatment_stats,
            'confidence_interval': treatment_ci
        },
        'comparison': {
            'relative_improvement': relative_improvement,
            'absolute_difference': treatment_stats['mean'] - baseline_stats['mean'],
            't_test': test_results,
            'treatment_better': treatment_stats['mean'] > baseline_stats['mean']
        }
    }


# ============================================================================
#                    DISTRIBUTION ANALYSIS
# ============================================================================

def analyze_distribution(values: List[float]) -> Dict[str, Any]:
    """Analyze value distribution"""
    if not values:
        return {'error': 'No values provided'}
    
    values = np.array(values)
    
    # Basic stats
    basic_stats = calculate_statistics(values)
    
    # Normality test
    if len(values) >= 3:
        _, normality_p = stats.shapiro(values)
        is_normal = normality_p > 0.05
    else:
        normality_p = None
        is_normal = None
    
    # Skewness and kurtosis
    if len(values) >= 3:
        skewness = float(stats.skew(values))
        kurtosis = float(stats.kurtosis(values))
    else:
        skewness = 0.0
        kurtosis = 0.0
    
    return {
        'statistics': basic_stats,
        'normality': {
            'is_normal': is_normal,
            'p_value': normality_p
        },
        'shape': {
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    }


def calculate_percentile_ranks(values: List[float]) -> List[float]:
    """Calculate percentile rank for each value"""
    if not values:
        return []
    
    ranks = stats.rankdata(values, method='average')
    percentiles = (ranks - 1) / (len(values) - 1) * 100 if len(values) > 1 else [50.0] * len(values)
    
    return percentiles.tolist()


# ============================================================================
#                    TIME SERIES ANALYSIS
# ============================================================================

def detect_trend(
    values: List[float],
    method: str = 'linear'
) -> Dict[str, Any]:
    """Detect trend in time series data"""
    if not values or len(values) < 2:
        return {'trend': 'insufficient_data'}
    
    x = np.arange(len(values))
    y = np.array(values)
    
    # Linear regression
    slope, intercept = np.polyfit(x, y, 1)
    
    # Calculate R²
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Determine trend
    if abs(slope) < 0.01:
        trend = 'stable'
    elif slope > 0:
        trend = 'increasing'
    else:
        trend = 'decreasing'
    
    return {
        'trend': trend,
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_squared),
        'strength': 'strong' if abs(r_squared) > 0.7 else 'weak'
    }


def calculate_autocorrelation(
    values: List[float],
    max_lag: int = 10
) -> List[float]:
    """Calculate autocorrelation for different lags"""
    if not values or len(values) < 2:
        return []
    
    values = np.array(values)
    mean = np.mean(values)
    var = np.var(values)
    
    if var == 0:
        return [1.0] * min(max_lag, len(values))
    
    autocorr = []
    for lag in range(min(max_lag, len(values))):
        if lag == 0:
            autocorr.append(1.0)
        else:
            c = np.sum((values[:-lag] - mean) * (values[lag:] - mean)) / len(values)
            autocorr.append(c / var)
    
    return autocorr


# ============================================================================
#                    EXPORT UTILITIES
# ============================================================================

def create_evaluation_report(
    metrics: Dict[str, Any],
    filepath: str
) -> None:
    """Create and save comprehensive evaluation report"""
    report = {
        'generated_at': format_timestamp(),
        'metrics': metrics,
        'summary': _generate_summary(metrics)
    }
    
    save_json(report, filepath)


def _generate_summary(metrics: Dict[str, Any]) -> Dict[str, str]:
    """Generate human-readable summary of metrics"""
    summary = {}
    
    if 'total_interactions' in metrics:
        summary['total_interactions'] = f"{metrics['total_interactions']} interactions recorded"
    
    if 'success_rate' in metrics:
        rate = metrics['success_rate'] * 100
        summary['success_rate'] = f"{rate:.1f}% success rate"
    
    if 'learning_trend' in metrics:
        summary['learning'] = f"System showing {metrics['learning_trend']} trend"
    
    return summary


# ============================================================================
#                    HELPER FUNCTIONS
# ============================================================================

def clip_values(
    values: List[float],
    min_val: float = None,
    max_val: float = None
) -> List[float]:
    """Clip values to specified range"""
    if not values:
        return []
    
    clipped = values.copy()
    
    if min_val is not None:
        clipped = [max(v, min_val) for v in clipped]
    
    if max_val is not None:
        clipped = [min(v, max_val) for v in clipped]
    
    return clipped


def smooth_values(
    values: List[float],
    alpha: float = 0.3
) -> List[float]:
    """Exponential smoothing"""
    if not values:
        return []
    
    smoothed = [values[0]]
    
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    
    return smoothed