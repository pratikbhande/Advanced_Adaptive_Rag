# monitor.py
"""Enhanced performance monitoring"""

import time
import functools
from typing import Callable, Any, Dict
from collections import defaultdict
import psutil
import os

def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"⏱️  {func.__name__} executed in {execution_time:.3f}s")
        
        return result
    return wrapper

class PerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str):
        """End timing and record"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[f"{operation}_time"].append(duration)
            del self.start_times[operation]
            return duration
        return 0.0
    
    def record_metric(self, metric_name: str, value: float):
        """Record a metric value"""
        self.metrics[metric_name].append(value)
    
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        values = self.metrics.get(metric_name, [])
        
        if not values:
            return {
                'count': 0,
                'mean': 0.0,
                'min': 0.0,
                'max': 0.0,
                'last': 0.0
            }
        
        return {
            'count': len(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'last': values[-1]
        }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics"""
        return {
            metric: self.get_statistics(metric)
            for metric in self.metrics.keys()
        }
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.start_times.clear()

class ResourceMonitor:
    """Monitor system resources"""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
        except:
            return {
                'rss_mb': 0.0,
                'vms_mb': 0.0,
                'percent': 0.0
            }
    
    @staticmethod
    def get_cpu_usage() -> float:
        """Get CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get system information"""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'total_memory_mb': psutil.virtual_memory().total / 1024 / 1024,
                'available_memory_mb': psutil.virtual_memory().available / 1024 / 1024,
                'disk_usage_percent': psutil.disk_usage('/').percent
            }
        except:
            return {}