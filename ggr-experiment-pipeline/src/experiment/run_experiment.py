#!/usr/bin/env python3
"""
Simple LLM Query Experiment Script with vLLM
Uses predefined query templates to run experiments on datasets with GPU support
Includes comprehensive resource monitoring and KV cache tracking

*** OPTIMIZED FOR FAIR PERFORMANCE COMPARISON ***
- Monitoring runs in separate threads to avoid interference with inference timing
- Resource monitoring: 10 second intervals (reduced from 2s)
- vLLM metrics monitoring: 15 second intervals (reduced from 1s)
- Only saves comprehensive JSON stats file for focused analysis
- Uses modern prefix cache counters: hits/queries instead of deprecated hit_rate

Multi-GPU Usage Examples:
  # Use multiple GPUs to handle large models (recommended for 7B+ models)
  python run_experiment.py dataset.csv query_key --gpus "4,5,6,7"
  
  # Use with memory optimization for large sequence lengths
  python run_experiment.py dataset.csv query_key --gpus "6,7" --max-model-len 8192 --gpu-memory 0.95
  
  # Single GPU usage (backward compatible)
  python run_experiment.py dataset.csv query_key --gpu 0
  
Memory Optimization Tips:
  - Use --gpus with multiple GPUs for models requiring >16GB memory
  - Reduce --max-model-len to lower memory usage (e.g., 8192 instead of 32768)
  - Increase --gpu-memory to use more available GPU memory (up to 0.98)
  - Reduce --batch-size if running out of memory during inference
"""

import argparse
import pandas as pd
import json
import time
import os
import sys
import torch
import threading
import psutil
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import base64
from io import BytesIO

# Try to import prometheus_client for accessing vLLM metrics
try:
    import prometheus_client
    from prometheus_client import CollectorRegistry, REGISTRY
    prometheus_available = True
except ImportError:
    prometheus_available = False

# Try to import pynvml for enhanced GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    nvidia_available = True
except ImportError:
    nvidia_available = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check GPU availability and CUDA setup"""
    if (torch.cuda.is_available()):
        gpu_count = torch.cuda.device_count()
        logger.info(f"GPU Setup Detected:")
        logger.info(f"  - Available GPUs: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  - GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        logger.info(f"  - CUDA Version: {torch.version.cuda}")
        return True, gpu_count
    else:
        logger.warning("CUDA not available! vLLM will fall back to CPU (very slow)")
        return False, 0

def set_gpu_devices(gpu_ids: List[int]):
    """Set multiple GPU devices to use"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        
        # Validate all GPU IDs
        for gpu_id in gpu_ids:
            if gpu_id >= gpu_count:
                logger.error(f"GPU {gpu_id} not available. Available GPUs: 0-{gpu_count-1}")
                return False
        
        # Set CUDA_VISIBLE_DEVICES to include all specified GPUs
        gpu_str = ','.join(map(str, gpu_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
        
        logger.info(f"Set CUDA devices to GPUs: {gpu_ids}")
        for gpu_id in gpu_ids:
            gpu_name = torch.cuda.get_device_name(gpu_id)
            gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
            logger.info(f"  - GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        return True
    else:
        logger.error("No CUDA GPUs available")
        return False

def calculate_memory_requirements(model_path: str, max_seq_len: int = None) -> Dict[str, float]:
    """Estimate memory requirements for a model"""
    try:
        # Try to read config.json to get model parameters
        config_path = os.path.join(model_path, 'config.json') if os.path.isdir(model_path) else None
        
        if (config_path and os.path.exists(config_path)):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract key parameters
            hidden_size = config.get('hidden_size', 4096)
            num_layers = config.get('num_hidden_layers', config.get('num_layers', 32))
            num_attention_heads = config.get('num_attention_heads', 32)
            vocab_size = config.get('vocab_size', 151936)  # Qwen default
            max_position_embeddings = config.get('max_position_embeddings', 32768)
            
            if max_seq_len is None:
                max_seq_len = max_position_embeddings
            
            logger.info(f"Model config - Hidden size: {hidden_size}, Layers: {num_layers}, "    
                       f"Attention heads: {num_attention_heads}, Max seq len: {max_seq_len}")
            
            # Rough estimates (in GB)
            # Model weights: approximately 2 bytes per parameter for FP16
            approx_params = hidden_size * hidden_size * num_layers * 8  # Very rough estimate
            model_memory_gb = (approx_params * 2) / 1e9
            
            # KV cache: 2 * layers * hidden_size * max_seq_len * batch_size * 2 bytes (for key + value)
            # Assuming batch_size = 1 for base calculation
            kv_cache_per_token = 2 * num_layers * hidden_size * 2 / 1e9  # GB per token
            kv_cache_gb = kv_cache_per_token * max_seq_len
            
            return {
                'model_memory_gb': model_memory_gb,
                'kv_cache_gb': kv_cache_gb,
                'total_estimated_gb': model_memory_gb + kv_cache_gb,
                'max_seq_len': max_seq_len,
                'kv_cache_per_token_gb': kv_cache_per_token
            }
            
    except Exception as e:
        logger.warning(f"Could not estimate memory requirements: {e}")
    
    # Return defaults if estimation fails
    return {
        'model_memory_gb': 7.0,  # Rough estimate for 7B model
        'kv_cache_gb': 16.0,     # As mentioned in error
        'total_estimated_gb': 23.0,
        'max_seq_len': max_seq_len or 32768,
        'kv_cache_per_token_gb': 0.0005
    }

try:
    from vllm import LLM, SamplingParams
    vllm_available = True
    logger.info("vLLM imported successfully")
    
    # Try to import modern vLLM metrics (optional)
    try:
        from vllm.engine.metrics import LoggingStatLogger, Stats
        from vllm.engine.llm_engine import LLMEngine
        modern_vllm_metrics = True
        logger.info("Modern vLLM metrics support available")
    except ImportError:
        modern_vllm_metrics = False
        logger.info("Using legacy vLLM metrics approach")
        
except ImportError:
    logger.error("vLLM not available. Install with: pip install vllm")
    vllm_available = False
    modern_vllm_metrics = False
    sys.exit(1)

# Try to import NVIDIA monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    nvidia_available = True
except ImportError:
    logger.warning("pynvml not available. GPU metrics will be limited.")
    nvidia_available = False

# System prompt template
SYSTEM_PROMPT = """You are a data analyst. Use the provided JSON data to answer the user query based on the specified fields. Respond with only the answer, no extra formatting. Answer the below query: {QUERY}

Given the following data: {fields}"""

# Query templates organized by type (keeping existing ones)
QUERY_TEMPLATES = {
    # LLM Aggregation Queries
    "agg_movies_sentiment": {
        "type": "aggregation",
        "prompt": "Given the following fields of a movie description and a user review, assign a sentiment score for the review out of 5. Answer with ONLY a single integer between 1 (bad) and 5 (good).",
        "datasets": ["movies"]
    },
    "agg_products_sentiment": {
        "type": "aggregation", 
        "prompt": "Given the following fields of a product description and a user review, assign a sentiment score for the review out of 5. Answer with ONLY a single integer between 1 (bad) and 5 (good).",
        "datasets": ["products"]
    },
    
    # Multi-LLM Invocation Queries
    "multi_movies_sentiment": {
        "type": "multi_invocation",
        "prompt": "Given the following review, answer whether the sentiment associated is 'POSITIVE' or 'NEGATIVE'. Answer in all caps with ONLY 'POSITIVE' or 'NEGATIVE':",
        "datasets": ["movies"]
    },
    "multi_products_sentiment": {
        "type": "multi_invocation",
        "prompt": "Given the following review, answer whether the sentiment associated is 'POSITIVE' or 'NEGATIVE'. Answer in all caps with ONLY 'POSITIVE' or 'NEGATIVE':",
        "datasets": ["products"]
    },
    
    # LLM Filter Queries
    "filter_movies_kids": {
        "type": "filter",
        "prompt": "Given the following fields, answer in one word, 'Yes' or 'No', whether the movie would be suitable for kids. Answer with ONLY 'Yes' or 'No'.",
        "datasets": ["movies"]
    },
    "filter_products_sentiment": {
        "type": "filter",
        "prompt": "Given the following fields determine if the review speaks positively ('POSITIVE'), negatively ('NEGATIVE'), or neutral ('NEUTRAL') about the product. Answer only 'POSITIVE', 'NEGATIVE', or 'NEUTRAL', nothing else.",
        "datasets": ["products"]
    },
    "filter_bird_statistics": {
        "type": "filter",
        "prompt": "Given the following fields related to posts in an online codebase community, answer whether the post is related to statistics. Answer with only 'YES' or 'NO'.",
        "datasets": ["bird"]
    },
    "filter_pdmx_individual": {
        "type": "filter",
        "prompt": "Based on following fields, answer 'YES' or 'NO' if any of the song information references a specific individual. Answer only 'YES' or 'NO', nothing else.",
        "datasets": ["pdmx"]
    },
    "filter_beer_european": {
        "type": "filter",
        "prompt": "Based on the beer descriptions, does this beer have European origin? Answer 'YES' if it does or 'NO' if it doesn't.",
        "datasets": ["beer"]
    },
    
    # LLM Projection Queries
    "proj_movies_summary": {
        "type": "projection",
        "prompt": "Given information including movie descriptions and critic reviews, summarize the good qualities in this movie that led to a favorable rating.",
        "datasets": ["movies"]
    },
    "proj_products_consistency": {
        "type": "projection",
        "prompt": "Given the following fields related to amazon products, summarize the product, then answer whether the product description is consistent with the quality expressed in the review.",
        "datasets": ["products"]
    },
    "proj_bird_comment": {
        "type": "projection", 
        "prompt": "Given the following fields related to posts in an online codebase community, summarize how the comment Text related to the post body.",
        "datasets": ["bird"]
    },
    "proj_pdmx_music": {
        "type": "projection",
        "prompt": "Given the following fields, provide an overview on the music type, and analyze the given scores. Give exactly 50 words of summary.",
        "datasets": ["pdmx"]
    },
    "proj_beer_overview": {
        "type": "projection",
        "prompt": "Given the following fields, provide an high-level overview on the beer and review in a 20 words paragraph.",
        "datasets": ["beer"]
    },
    
    # RAG Queries
    "rag_fever": {
        "type": "rag",
        "prompt": "You are given 4 pieces of evidence as {evidence1}, {evidence2}, {evidence3}, and {evidence4}. You are also given a claim as {claim}. Answer SUPPORTS if the pieces of evidence support the given {claim}, REFUTES if the evidence refutes the given {claim}, or NOT ENOUGH INFO if there is not enough information to answer. Your answer should just be SUPPORTS, REFUTES, or NOT ENOUGH INFO and nothing else.",
        "datasets": ["fever"]
    },
    "rag_squad": {
        "type": "rag",
        "prompt": "Given a question and supporting contexts, answer the provided question.",
        "datasets": ["squad"]
    }
}


class ResourceMonitor:
    """Monitor system resources during inference with minimal performance impact"""
    
    def __init__(self, gpu_id: int = 0, sampling_interval: float = 5.0):
        self.gpu_id = gpu_id
        self.sampling_interval = max(sampling_interval, 5.0)  # Minimum 5 seconds to reduce impact
        self.monitoring = False
        self.metrics_log = []
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        self.monitoring = True
        self.metrics_log = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Started resource monitoring (interval: {self.sampling_interval}s)")
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10.0)
        logger.info("Stopped resource monitoring")
        
    def _monitor_loop(self):
        """Main monitoring loop running in background - optimized for minimal interference"""
        while self.monitoring:
            try:
                timestamp = time.time()
                # Use threading to collect metrics without blocking
                metrics = self._collect_metrics(timestamp)
                if metrics:
                    self.metrics_log.append(metrics)
            except Exception as e:
                logger.debug(f"Error collecting metrics: {e}")  # Use debug level to avoid spam
            
            time.sleep(self.sampling_interval)
    
    def _collect_metrics(self, timestamp: float) -> Dict[str, Any]:
        """Collect system and GPU metrics at a point in time"""
        metrics = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat()
        }
        
        # CPU and Memory metrics
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            metrics.update({
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / 1e9,
                'memory_available_gb': memory.available / 1e9,
                'memory_total_gb': memory.total / 1e9
            })
        except Exception as e:
            logger.warning(f"Error collecting CPU/Memory metrics: {e}")
        
        # GPU metrics via PyTorch CUDA
        if torch.cuda.is_available():
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated(self.gpu_id) / 1e9
                gpu_memory_reserved = torch.cuda.memory_reserved(self.gpu_id) / 1e9
                gpu_memory_total = torch.cuda.get_device_properties(self.gpu_id).total_memory / 1e9
                
                metrics.update({
                    'gpu_id': self.gpu_id,
                    'gpu_memory_allocated_gb': gpu_memory_allocated,
                    'gpu_memory_reserved_gb': gpu_memory_reserved, 
                    'gpu_memory_total_gb': gpu_memory_total,
                    'gpu_memory_utilization_percent': (gpu_memory_allocated / gpu_memory_total) * 100
                })
            except Exception as e:
                logger.warning(f"Error collecting GPU memory metrics: {e}")
        
        # Enhanced GPU metrics via pynvml
        if nvidia_available:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
                
                # GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics.update({
                    'gpu_utilization_percent': utilization.gpu,
                    'gpu_memory_controller_utilization_percent': utilization.memory
                })
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                metrics['gpu_temperature_celsius'] = temp
                
                # Power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    metrics['gpu_power_watts'] = power
                except:
                    pass  # Not all GPUs support power monitoring
                    
                # Clock speeds
                try:
                    graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                    metrics.update({
                        'gpu_graphics_clock_mhz': graphics_clock,
                        'gpu_memory_clock_mhz': memory_clock
                    })
                except:
                    pass
                    
            except Exception as e:
                logger.warning(f"Error collecting pynvml GPU metrics: {e}")
        
        return metrics
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics from collected metrics"""
        if not self.metrics_log:
            return {}
            
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(self.metrics_log)
        
        summary = {
            'monitoring_duration_seconds': len(df) * self.sampling_interval,
            'total_samples': len(df),
            'sampling_interval_seconds': self.sampling_interval
        }
        
        # CPU/Memory summary
        if 'cpu_percent' in df.columns:
            summary.update({
                'cpu_utilization_mean': df['cpu_percent'].mean(),
                'cpu_utilization_max': df['cpu_percent'].max(),
                'cpu_utilization_min': df['cpu_percent'].min(),
                'cpu_utilization_std': df['cpu_percent'].std(),
                'memory_utilization_mean': df['memory_percent'].mean(),
                'memory_utilization_max': df['memory_percent'].max(),
                'memory_used_gb_mean': df['memory_used_gb'].mean(),
                'memory_used_gb_max': df['memory_used_gb'].max()
            })
        
        # GPU summary
        if 'gpu_memory_utilization_percent' in df.columns:
            summary.update({
                'gpu_memory_utilization_mean': df['gpu_memory_utilization_percent'].mean(),
                'gpu_memory_utilization_max': df['gpu_memory_utilization_percent'].max(),
                'gpu_memory_allocated_gb_mean': df['gpu_memory_allocated_gb'].mean(),
                'gpu_memory_allocated_gb_max': df['gpu_memory_allocated_gb'].max()
            })
            
        if 'gpu_utilization_percent' in df.columns:
            summary.update({
                'gpu_compute_utilization_mean': df['gpu_utilization_percent'].mean(),
                'gpu_compute_utilization_max': df['gpu_utilization_percent'].max(),
                'gpu_compute_utilization_min': df['gpu_utilization_percent'].min(),
                'gpu_compute_utilization_std': df['gpu_utilization_percent'].std()
            })
            
        if 'gpu_temperature_celsius' in df.columns:
            summary.update({
                'gpu_temperature_mean': df['gpu_temperature_celsius'].mean(),
                'gpu_temperature_max': df['gpu_temperature_celsius'].max()
            })
            
        if 'gpu_power_watts' in df.columns:
            summary.update({
                'gpu_power_mean': df['gpu_power_watts'].mean(),
                'gpu_power_max': df['gpu_power_watts'].max()
            })
        
        return summary


class VLLMMetricsCollector:
    """Enhanced vLLM metrics collector using official Prometheus metrics"""
    
    def __init__(self, llm_instance=None, collection_interval=10.0):
        self.llm = llm_instance
        self.stats_history = []
        self.collection_interval = max(collection_interval, 10.0)  # Minimum 10 seconds for fair comparison
        self.monitoring = False
        self.monitor_thread = None
        self.modern_logging_available = False
        self.last_stats = {}
        
        # Official vLLM Prometheus metrics names from documentation
        self.vllm_metric_names = {
            # System stats - Scheduler State
            'num_requests_running': 'vllm:num_requests_running',
            'num_requests_waiting': 'vllm:num_requests_waiting',
            'num_requests_swapped': 'vllm:num_requests_swapped',  # Deprecated but may exist
            
            # KV Cache Usage - MOST IMPORTANT METRICS
            'gpu_cache_usage_perc': 'vllm:gpu_cache_usage_perc',
            'cpu_cache_usage_perc': 'vllm:cpu_cache_usage_perc',  # Deprecated but may exist
            
            # Modern Prefix Cache Counters (V1 engine and newer)
            'gpu_prefix_cache_hits': 'vllm:gpu_prefix_cache_hits',
            'gpu_prefix_cache_queries': 'vllm:gpu_prefix_cache_queries',
            'cpu_prefix_cache_hits': 'vllm:cpu_prefix_cache_hits',
            'cpu_prefix_cache_queries': 'vllm:cpu_prefix_cache_queries',
            
            # Token counters
            'prompt_tokens_total': 'vllm:prompt_tokens_total',
            'generation_tokens_total': 'vllm:generation_tokens_total',
            'num_preemptions_total': 'vllm:num_preemptions_total',
            
            # Request success
            'request_success_total': 'vllm:request_success_total',
            
            # Speculative decoding (if available)
            'spec_decode_draft_acceptance_rate': 'vllm:spec_decode_draft_acceptance_rate',
            'spec_decode_efficiency': 'vllm:spec_decode_efficiency',
            'spec_decode_num_accepted_tokens_total': 'vllm:spec_decode_num_accepted_tokens_total',
            'spec_decode_num_draft_tokens_total': 'vllm:spec_decode_num_draft_tokens_total',
            'spec_decode_num_emitted_tokens_total': 'vllm:spec_decode_num_emitted_tokens_total'
        }
        
        # Histogram metrics for detailed analysis
        self.vllm_histogram_names = {
            'iteration_tokens_total': 'vllm:iteration_tokens_total',
            'time_to_first_token_seconds': 'vllm:time_to_first_token_seconds',
            'time_per_output_token_seconds': 'vllm:time_per_output_token_seconds',
            'e2e_request_latency_seconds': 'vllm:e2e_request_latency_seconds',
            'request_queue_time_seconds': 'vllm:request_queue_time_seconds',
            'request_inference_time_seconds': 'vllm:request_inference_time_seconds',
            'request_prefill_time_seconds': 'vllm:request_prefill_time_seconds',
            'request_decode_time_seconds': 'vllm:request_decode_time_seconds',
            'request_prompt_tokens': 'vllm:request_prompt_tokens',
            'request_generation_tokens': 'vllm:request_generation_tokens',
            'request_max_num_generation_tokens': 'vllm:request_max_num_generation_tokens'
        }
        
        # Cache for metrics
        self.metrics_cache = {
            'prometheus_metrics': {},
            'histogram_metrics': {},
            'collection_timestamps': []
        }
        
    def initialize_logging(self):
        """Initialize vLLM metrics collection"""
        if not self.llm:
            logger.warning("No LLM instance available for metrics initialization")
            return False
            
        try:
            # Check if we can access the engine and its metrics
            if hasattr(self.llm, 'llm_engine'):
                engine = self.llm.llm_engine
                logger.info("Found llm_engine attribute")
                
                # Try to access engine metrics or stat_loggers
                if hasattr(engine, 'metrics') or hasattr(engine, 'stat_loggers'):
                    self.modern_logging_available = True
                    logger.info("vLLM Prometheus metrics should be available")
                    return True
                else:
                    logger.info("Engine found but metrics access uncertain")
                    self.modern_logging_available = True  # Try anyway
                    return True
                    
            else:
                logger.warning("Could not access llm_engine from vLLM instance")
                return False
                
        except Exception as e:
            logger.warning(f"Failed to initialize vLLM metrics: {e}")
            return False
    
    def collect_internal_stats(self) -> Dict[str, Any]:
        """Access vLLM's internal Stats object directly using private _get_stats() method"""
        stats_data = {
            'stats_available': False,
            'collection_method': 'internal_stats',
            'collection_time': time.time(),
            'metrics_found': {},
            'debug_info': {}
        }
        
        try:
            if not self.llm or not hasattr(self.llm, 'llm_engine'):
                stats_data['error'] = 'No LLM engine available'
                return stats_data
            
            engine = self.llm.llm_engine
            stats_data['debug_info']['engine_type'] = type(engine).__name__
            
            # Method 1: Use the proven approach from your example
            if hasattr(engine, '_get_stats'):
                try:
                    # Access internal stats (not a public API, may change)
                    stats = engine._get_stats()
                    stats_data['debug_info']['stats_method'] = '_get_stats() - direct approach'
                    
                    if stats:
                        # Extract all available metrics from Stats object using proven approach
                        metrics = {}
                        
                        # KV Cache metrics - MOST IMPORTANT - using exact approach from your example
                        if hasattr(stats, 'gpu_cache_usage_sys'):
                            gpu_cache_usage = stats.gpu_cache_usage_sys
                            metrics['gpu_cache_usage_sys'] = gpu_cache_usage
                            metrics['vllm:gpu_cache_usage_perc'] = gpu_cache_usage  # Also provide in standard format
                            
                        # Prefix cache hit rate - using getattr with default as in your example
                        prefix_hit_rate = getattr(stats, 'gpu_prefix_cache_hit_rate', 0)
                        if prefix_hit_rate > 0:
                            metrics['gpu_prefix_cache_hit_rate'] = prefix_hit_rate
                            metrics['vllm:gpu_prefix_cache_hit_rate'] = prefix_hit_rate
                            
                        # CPU cache metrics (if available)
                        if hasattr(stats, 'cpu_cache_usage_sys'):
                            metrics['cpu_cache_usage_sys'] = stats.cpu_cache_usage_sys
                        cpu_prefix_hit_rate = getattr(stats, 'cpu_prefix_cache_hit_rate', 0)
                        if cpu_prefix_hit_rate > 0:
                            metrics['cpu_prefix_cache_hit_rate'] = cpu_prefix_hit_rate
                        
                        # Scheduler state
                        if hasattr(stats, 'num_running_sys'):
                            metrics['num_running_sys'] = stats.num_running_sys
                            metrics['vllm:num_requests_running'] = stats.num_running_sys
                        if hasattr(stats, 'num_waiting_sys'):
                            metrics['num_waiting_sys'] = stats.num_waiting_sys
                            metrics['vllm:num_requests_waiting'] = stats.num_waiting_sys
                        if hasattr(stats, 'num_swapped_sys'):
                            metrics['num_swapped_sys'] = stats.num_swapped_sys
                            metrics['vllm:num_requests_swapped'] = stats.num_swapped_sys
                        
                        # Token processing
                        if hasattr(stats, 'num_prompt_tokens_iter'):
                            metrics['prompt_tokens_total'] = stats.num_prompt_tokens_iter
                            metrics['vllm:prompt_tokens_total'] = stats.num_prompt_tokens_iter
                        if hasattr(stats, 'num_generation_tokens_iter'):
                            metrics['generation_tokens_total'] = stats.num_generation_tokens_iter
                            metrics['vllm:generation_tokens_total'] = stats.num_generation_tokens_iter
                        if hasattr(stats, 'num_tokens_iter'):
                            metrics['tokens_total'] = stats.num_tokens_iter
                        
                        # Performance metrics
                        if hasattr(stats, 'time_to_first_tokens_iter') and stats.time_to_first_tokens_iter:
                            avg_ttft = sum(stats.time_to_first_tokens_iter) / len(stats.time_to_first_tokens_iter)
                            metrics['avg_time_to_first_token_seconds'] = avg_ttft
                        if hasattr(stats, 'time_per_output_tokens_iter') and stats.time_per_output_tokens_iter:
                            avg_tpot = sum(stats.time_per_output_tokens_iter) / len(stats.time_per_output_tokens_iter)
                            metrics['avg_time_per_output_token_seconds'] = avg_tpot
                        
                        # Request metrics  
                        if hasattr(stats, 'num_preemption_iter'):
                            metrics['num_preemptions'] = stats.num_preemption_iter
                        if hasattr(stats, 'time_e2e_requests') and stats.time_e2e_requests:
                            metrics['avg_e2e_latency_seconds'] = sum(stats.time_e2e_requests) / len(stats.time_e2e_requests)
                        
                        stats_data['metrics_found'] = metrics
                        stats_data['stats_available'] = True
                        stats_data['debug_info']['metrics_extracted'] = len(metrics)
                        
                        logger.debug(f"✅ Extracted {len(metrics)} metrics from internal Stats object using proven approach")
                        if 'gpu_cache_usage_sys' in metrics:
                            logger.debug(f"🗄️ GPU Cache Usage: {metrics['gpu_cache_usage_sys']*100:.1f}%")
                        if 'gpu_prefix_cache_hit_rate' in metrics:
                            logger.debug(f"🎯 GPU Prefix Cache Hit Rate: {metrics['gpu_prefix_cache_hit_rate']*100:.1f}%")
                        
                        return stats_data
                    else:
                        stats_data['debug_info']['stats_result'] = 'None returned from _get_stats()'
                        
                except Exception as e:
                    stats_data['debug_info']['_get_stats_error'] = str(e)
                    logger.debug(f"Error calling _get_stats(): {e}")
            
            # Method 2: Try to access scheduler's cache manager directly
            if hasattr(engine, 'scheduler'):
                try:
                    schedulers = engine.scheduler if isinstance(engine.scheduler, list) else [engine.scheduler]
                    stats_data['debug_info']['scheduler_access'] = 'found_schedulers'
                    
                    metrics = {}
                    for i, scheduler in enumerate(schedulers):
                        if hasattr(scheduler, 'block_manager'):
                            block_manager = scheduler.block_manager
                            
                            # Get KV cache usage directly from block manager
                            if hasattr(block_manager, 'get_num_free_gpu_blocks'):
                                free_gpu = block_manager.get_num_free_gpu_blocks()
                                if hasattr(engine, 'cache_config') and engine.cache_config.num_gpu_blocks:
                                    total_gpu = engine.cache_config.num_gpu_blocks
                                    gpu_usage = 1.0 - (free_gpu / total_gpu)
                                    metrics['gpu_cache_usage_sys'] = gpu_usage
                                    logger.debug(f"📊 Direct GPU cache usage: {gpu_usage*100:.1f}%")
                            
                            # Get CPU cache usage if available
                            if hasattr(block_manager, 'get_num_free_cpu_blocks'):
                                free_cpu = block_manager.get_num_free_cpu_blocks()
                                if hasattr(engine, 'cache_config') and engine.cache_config.num_cpu_blocks:
                                    total_cpu = engine.cache_config.num_cpu_blocks
                                    cpu_usage = 1.0 - (free_cpu / total_cpu)
                                    metrics['cpu_cache_usage_sys'] = cpu_usage
                        
                        # Try to get prefix cache counters (modern approach)
                        if hasattr(scheduler, 'get_prefix_cache_stats'):
                            try:
                                # Try to get prefix cache statistics directly
                                cache_stats = scheduler.get_prefix_cache_stats()
                                if isinstance(cache_stats, dict):
                                    metrics.update(cache_stats)
                            except Exception as e:
                                stats_data['debug_info']['cache_stats_error'] = str(e)
                        
                        # Fallback: Try individual prefix cache hit rates (deprecated but may exist)
                        if hasattr(scheduler, 'get_prefix_cache_hit_rate'):
                            try:
                                # Try different ways to access Device enum
                                try:
                                    # Method 1: Try importing Device from various locations
                                    try:
                                        from vllm.utils import Device
                                        gpu_hit_rate = scheduler.get_prefix_cache_hit_rate(Device.GPU)
                                        metrics['gpu_prefix_cache_hit_rate'] = gpu_hit_rate
                                        cpu_hit_rate = scheduler.get_prefix_cache_hit_rate(Device.CPU)
                                        metrics['cpu_prefix_cache_hit_rate'] = cpu_hit_rate
                                    except ImportError:
                                        # Try alternative import path
                                        from vllm.engine.llm_engine import Device
                                        gpu_hit_rate = scheduler.get_prefix_cache_hit_rate(Device.GPU)
                                        metrics['gpu_prefix_cache_hit_rate'] = gpu_hit_rate
                                        cpu_hit_rate = scheduler.get_prefix_cache_hit_rate(Device.CPU)
                                        metrics['cpu_prefix_cache_hit_rate'] = cpu_hit_rate
                                except (ImportError, AttributeError):
                                    # Method 2: Try with string device names
                                    try:
                                        gpu_hit_rate = scheduler.get_prefix_cache_hit_rate("gpu")
                                        metrics['gpu_prefix_cache_hit_rate'] = gpu_hit_rate
                                        cpu_hit_rate = scheduler.get_prefix_cache_hit_rate("cpu")
                                        metrics['cpu_prefix_cache_hit_rate'] = cpu_hit_rate
                                    except Exception:
                                        # Method 3: Try without device parameter (aggregate)
                                        try:
                                            hit_rate = scheduler.get_prefix_cache_hit_rate()
                                            metrics['prefix_cache_hit_rate'] = hit_rate
                                        except Exception:
                                            pass
                            except Exception as e:
                                stats_data['debug_info']['cache_hit_rate_error'] = str(e)
                    
                    if metrics:
                        stats_data['metrics_found'] = metrics
                        stats_data['stats_available'] = True
                        stats_data['debug_info']['metrics_extracted'] = len(metrics)
                        stats_data['debug_info']['extraction_method'] = 'direct_scheduler_access'
                        logger.debug(f"✅ Extracted {len(metrics)} metrics via direct scheduler access")
                        return stats_data
                        
                except Exception as e:
                    stats_data['debug_info']['scheduler_access_error'] = str(e)
                    
            # Method 3: Try to access model execution stats
            if hasattr(engine, 'model_executor'):
                try:
                    stats_data['debug_info']['model_executor_access'] = 'attempting'
                    # This is more complex and engine-version dependent
                    # Could potentially access worker stats here
                except Exception as e:
                    stats_data['debug_info']['model_executor_error'] = str(e)
            
            stats_data['error'] = 'No accessible internal stats methods found'
            return stats_data
            
        except Exception as e:
            stats_data['error'] = f'Error accessing internal stats: {str(e)}'
            stats_data['debug_info']['exception'] = str(e)
            logger.debug(f"Error in collect_internal_stats: {e}")
            return stats_data

    def collect_prometheus_metrics(self) -> Dict[str, Any]:
        """Enhanced Prometheus metrics collection with better debugging and fallback methods"""
        if not prometheus_available:
            return {
                'stats_available': False,
                'collection_method': 'prometheus_unavailable',
                'error': 'prometheus_client not available'
            }
            
        metrics_data = {
            'stats_available': False,
            'collection_method': 'prometheus_registry',
            'collection_time': time.time(),
            'metrics_found': {},
            'histogram_data': {},
            'debug_info': {}
        }
        
        try:
            # First, try internal stats collection (preferred method for offline inference)
            internal_stats = self.collect_internal_stats()
            if internal_stats.get('stats_available', False):
                # Merge internal stats with prometheus structure
                metrics_data.update(internal_stats)
                metrics_data['collection_method'] = 'internal_stats_primary'
                logger.debug("🎯 Using internal stats as primary collection method")
                
                # Still try to get Prometheus metrics as supplementary data
                try:
                    prometheus_stats = self._collect_prometheus_registry_only()
                    if prometheus_stats.get('stats_available', False):
                        # Merge with internal stats (internal takes priority)
                        prometheus_metrics = prometheus_stats.get('metrics_found', {})
                        for key, value in prometheus_metrics.items():
                            if key not in metrics_data['metrics_found']:
                                metrics_data['metrics_found'][key] = value
                        metrics_data['debug_info']['prometheus_supplement'] = len(prometheus_metrics)
                except Exception as e:
                    metrics_data['debug_info']['prometheus_supplement_error'] = str(e)
                
                return metrics_data
            
            # Fallback to Prometheus-only collection
            return self._collect_prometheus_registry_only()
            
        except Exception as e:
            metrics_data['error'] = f'Error in metrics collection: {str(e)}'
            metrics_data['debug_info']['exception'] = str(e)
            return metrics_data
    
    def _collect_prometheus_registry_only(self) -> Dict[str, Any]:
        """Original Prometheus registry collection method"""
        metrics_data = {
            'stats_available': False,
            'collection_method': 'prometheus_registry_only',
            'collection_time': time.time(),
            'metrics_found': {},
            'histogram_data': {},
            'debug_info': {}
        }
        
        try:
            # Get all metric families from the global Prometheus registry
            metric_families = list(REGISTRY.collect())
            metrics_data['debug_info']['total_metric_families'] = len(metric_families)
            
            found_vllm_metrics = {}
            found_histograms = {}
            all_metric_names = []
            
            for family in metric_families:
                metric_name = family.name
                all_metric_names.append(metric_name)
                
                # Check if this is a vLLM metric (expanded patterns)
                is_vllm_metric = (
                    metric_name.startswith('vllm:') or
                    metric_name.startswith('vllm_') or
                    'cache' in metric_name.lower() or
                    'prefix' in metric_name.lower() or
                    ('request' in metric_name.lower() and any(term in metric_name.lower() for term in ['running', 'waiting', 'queue']))
                )
                
                if is_vllm_metric:
                    logger.debug(f"Processing potential vLLM metric: {metric_name} (type: {family.type})")
                    
                    # Handle gauge and counter metrics
                    if family.type in ['gauge', 'counter']:
                        for sample in family.samples:
                            # Store all samples, not just exact matches
                            sample_name = sample.name
                            if sample_name not in found_vllm_metrics:
                                found_vllm_metrics[sample_name] = sample.value
                                logger.debug(f"Found metric {sample_name}: {sample.value}")
                    
                    # Handle histogram metrics
                    elif family.type == 'histogram':
                        histogram_data = {'buckets': {}, 'count': 0, 'sum': 0}
                        
                        for sample in family.samples:
                            if sample.name == f"{metric_name}_bucket":
                                le_value = sample.labels.get('le', 'unknown')
                                histogram_data['buckets'][le_value] = sample.value
                            elif sample.name == f"{metric_name}_count":
                                histogram_data['count'] = sample.value
                            elif sample.name == f"{metric_name}_sum":
                                histogram_data['sum'] = sample.value
                        
                        if histogram_data['count'] > 0 or histogram_data['sum'] > 0:
                            found_histograms[metric_name] = histogram_data
                            logger.debug(f"Found histogram {metric_name}: count={histogram_data['count']}, sum={histogram_data['sum']}")
            
            # Store debug info
            metrics_data['debug_info'].update({
                'all_metric_names': all_metric_names[:20],  # First 20 for debugging
                'vllm_metric_candidates': [name for name in all_metric_names if any(term in name.lower() for term in ['vllm', 'cache', 'prefix'])],
                'total_metrics_scanned': len(all_metric_names)
            })
            
            # Alternative: Try direct engine access if available
            if hasattr(self, 'llm') and self.llm and hasattr(self.llm, 'llm_engine'):
                try:
                    engine = self.llm.llm_engine
                    
                    # Try to get stats from engine directly (for older vLLM versions)
                    if hasattr(engine, 'get_stats'):
                        engine_stats = engine.get_stats()
                        logger.debug(f"Got engine stats: {engine_stats}")
                        # Add engine stats to found metrics with 'engine:' prefix
                        for key, value in engine_stats.items():
                            found_vllm_metrics[f'engine:{key}'] = value
                    
                    # Try to get model config info
                    if hasattr(engine, 'model_config'):
                        model_config = engine.model_config
                        found_vllm_metrics['engine:max_model_len'] = getattr(model_config, 'max_model_len', 0)
                        found_vllm_metrics['engine:enable_prefix_caching'] = getattr(model_config, 'enable_prefix_caching', False)
                    
                    logger.debug(f"Added {len([k for k in found_vllm_metrics if k.startswith('engine:')])} engine metrics")
                    
                except Exception as e:
                    logger.debug(f"Could not access engine stats: {e}")
                    metrics_data['debug_info']['engine_access_error'] = str(e)
            
            # Try accessing vLLM's internal stat logger if available
            try:
                # Look for vLLM stat loggers in the global registry
                for family in metric_families:
                    if 'stat' in family.name.lower() and family.samples:
                        for sample in family.samples:
                            if sample.name not in found_vllm_metrics:
                                found_vllm_metrics[f'stat:{sample.name}'] = sample.value
                                
                logger.debug(f"Added {len([k for k in found_vllm_metrics if k.startswith('stat:')])} stat logger metrics")
                
            except Exception as e:
                logger.debug(f"Could not access stat loggers: {e}")
                metrics_data['debug_info']['stat_logger_error'] = str(e)
            
            # Update metrics data with findings
            metrics_data['metrics_found'] = found_vllm_metrics
            metrics_data['histogram_data'] = found_histograms
            metrics_data['total_metrics_found'] = len(found_vllm_metrics)
            metrics_data['total_histograms_found'] = len(found_histograms)
            
            if found_vllm_metrics or found_histograms:
                metrics_data['stats_available'] = True
                logger.debug(f"✅ Prometheus registry collection: {len(found_vllm_metrics)} metrics, {len(found_histograms)} histograms")
            else:
                logger.debug("❌ No vLLM metrics found in Prometheus registry")
                
            # Log key findings for debugging
            key_cache_metrics = [k for k in found_vllm_metrics.keys() if 'cache' in k.lower()]
            if key_cache_metrics:
                logger.debug(f"🗄️  Found cache metrics: {key_cache_metrics}")
            
            key_request_metrics = [k for k in found_vllm_metrics.keys() if any(term in k.lower() for term in ['running', 'waiting', 'queue'])]
            if key_request_metrics:
                logger.debug(f"⏳ Found request metrics: {key_request_metrics}")
                
        except Exception as e:
            logger.debug(f"Error in Prometheus registry collection: {e}")
            metrics_data['stats_available'] = False
            metrics_data['error'] = str(e)
            metrics_data['debug_info']['exception'] = str(e)
            
        return metrics_data
    
    def simulate_metrics_for_testing(self, duration_seconds: int = 30, interval: float = 1.0):
        """Simulate vLLM metrics for testing purposes when real metrics aren't available"""
        import random
        import math
        
        logger.info(f"🧪 Simulating vLLM metrics for {duration_seconds} seconds (testing mode)")
        
        # Clear any existing history
        self.stats_history = []
        
        start_time = time.time()
        current_time = start_time
        
        # Simulate progressive cache usage and hit rates
        cache_queries = 0
        cache_hits = 0
        total_tokens = 0
        
        while current_time - start_time < duration_seconds:
            # Simulate realistic metrics progression
            elapsed = current_time - start_time
            progress = elapsed / duration_seconds
            
            # Simulate increasing cache queries and hits
            new_queries = random.randint(5, 20)
            cache_queries += new_queries
            
            # Hit rate improves over time (simulating prefix caching effectiveness)
            base_hit_rate = 0.1 + (progress * 0.7)  # 10% to 80% hit rate
            hit_rate_variance = random.uniform(-0.1, 0.1)
            actual_hit_rate = max(0.0, min(1.0, base_hit_rate + hit_rate_variance))
            
            new_hits = int(new_queries * actual_hit_rate)
            cache_hits += new_hits
            
            # Simulate token counts
            new_tokens = random.randint(100, 500)
            total_tokens += new_tokens
            
            # Simulate cache usage (gradual increase)
            cache_usage = min(0.9, 0.1 + (progress * 0.7) + random.uniform(-0.05, 0.05))
            
            # Simulate request queue
            running_requests = random.randint(0, 8)
            waiting_requests = random.randint(0, 5)
            
            # Create realistic metrics entry
            simulated_entry = {
                'timestamp': current_time,
                'collection_time': current_time,
                'stats_available': True,
                'collection_method': 'simulated_for_testing',
                'metrics_found': {
                    'vllm:gpu_cache_usage_perc': cache_usage,
                    'vllm:gpu_prefix_cache_hit_rate': cache_hits / max(cache_queries, 1),
                    'vllm:gpu_prefix_cache_hits': cache_hits,
                    'vllm:gpu_prefix_cache_queries': cache_queries,
                    'vllm:num_requests_running': running_requests,
                    'vllm:num_requests_waiting': waiting_requests,
                    'vllm:prompt_tokens_total': total_tokens * 0.6,  # Roughly 60% prompt tokens
                    'vllm:generation_tokens_total': total_tokens * 0.4,  # 40% generation tokens
                    'vllm:request_success_total': cache_queries,
                },
                'histogram_data': {
                    'vllm:time_to_first_token_seconds': {
                        'count': cache_queries,
                        'sum': cache_queries * random.uniform(0.1, 0.5),
                        'buckets': {}
                    },
                    'vllm:time_per_output_token_seconds': {
                        'count': total_tokens,
                        'sum': total_tokens * random.uniform(0.01, 0.05),
                        'buckets': {}
                    }
                },
                'total_metrics_found': 8,
                'total_histograms_found': 2
            }
            
            self.stats_history.append(simulated_entry)
            
            # Move to next time point
            current_time += interval
            time.sleep(0.01)  # Small sleep to avoid busy wait
        
        logger.info(f"✅ Generated {len(self.stats_history)} simulated metric data points")
        logger.info(f"📊 Final cache hit rate: {(cache_hits/cache_queries)*100:.1f}% ({cache_hits}/{cache_queries})")
        logger.info(f"🗄️ Final cache usage: {cache_usage*100:.1f}%")
        
        # Return the final stats for immediate analysis
        return self.get_comprehensive_stats()
    
    def enable_testing_mode(self):
        """Enable testing mode with simulated metrics"""
        logger.info("🧪 Enabling testing mode - will use simulated vLLM metrics")
        # Stop any existing monitoring
        if self.monitoring:
            self.stop_monitoring()
        
        # Generate some test data
        self.simulate_metrics_for_testing(duration_seconds=15, interval=0.5)
        logger.info(f"✅ Testing mode enabled with {len(self.stats_history)} data points")
    
    def start_monitoring(self):
        """Start background monitoring with minimal performance impact"""
        if self.monitoring:
            logger.info("vLLM metrics monitoring already active")
            return
            
        # Force initialization if not done
        if not self.modern_logging_available:
            self.initialize_logging()
            
        self.monitoring = True
        self.collection_interval = max(self.collection_interval, 10.0)  # Minimum 10 seconds to reduce impact
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Started vLLM metrics monitoring (interval: {self.collection_interval}s)")
        
    def stop_monitoring(self):
        """Stop background monitoring"""
        if not self.monitoring:
            return
            
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=15.0)
        logger.info("Stopped vLLM metrics monitoring")
        
    def _monitoring_loop(self):
        """Optimized monitoring loop with reduced interference"""
        consecutive_failures = 0
        max_failures = 5  # Reduce failure tolerance
        
        while self.monitoring:
            try:
                # Use a separate thread for metrics collection to avoid blocking
                stats = self.collect_prometheus_metrics()
                
                if stats.get('stats_available', False):
                    stats['timestamp'] = time.time()
                    self.stats_history.append(stats)
                    self.last_stats = stats.copy()
                    
                    # Update cache
                    self._update_metrics_cache(stats)
                    
                    # Reset failure counter on success
                    consecutive_failures = 0
                    
                    # Log key metrics less frequently
                    if len(self.stats_history) % 30 == 0:  # Log every 30 collections instead of 10
                        self._log_key_prometheus_metrics(stats)
                        
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        logger.debug(f"vLLM metrics collection failed {max_failures} times, reducing frequency")
                        self.collection_interval = min(self.collection_interval * 1.5, 30.0)  # Increase interval on failures
                        consecutive_failures = 0
                
                # Keep history manageable
                if len(self.stats_history) > 100:  # Reduce history size
                    self.stats_history = self.stats_history[-50:]
                    
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures == 1:
                    logger.debug(f"Error in vLLM monitoring loop: {e}")
                
            time.sleep(self.collection_interval)
    
    def _log_key_prometheus_metrics(self, stats):
        """Log key Prometheus metrics for debugging"""
        try:
            metrics_found = stats.get('metrics_found', {})
            
            # Enhanced metrics logging with categories
            cache_metrics_count = len([k for k in metrics_found.keys() if 'cache' in k.lower()])
            if cache_metrics_count > 0:
                logger.debug(f"📊 Found {cache_metrics_count} cache-related metrics")
                
                # KV Cache Usage
                if 'vllm:gpu_cache_usage_perc' in metrics_found:
                    cache_pct = metrics_found['vllm:gpu_cache_usage_perc'] * 100
                    logger.debug(f"   🗄️  GPU KV Cache Usage: {cache_pct:.1f}%")
                
                # Prefix Cache Hit Rate (may be deprecated)
                if 'vllm:gpu_prefix_cache_hit_rate' in metrics_found:
                    hit_rate = metrics_found['vllm:gpu_prefix_cache_hit_rate'] * 100
                    logger.debug(f"   🎯 GPU Prefix Cache Hit Rate: {hit_rate:.1f}%")
                
                # Cache hits and queries (for calculation)
                cache_hits = metrics_found.get('vllm:gpu_prefix_cache_hits', 0)
                cache_queries = metrics_found.get('vllm:gpu_prefix_cache_queries', 0)
                if cache_hits > 0 or cache_queries > 0:
                    logger.debug(f"   📈 Cache counters: {cache_hits} hits / {cache_queries} queries")
            
            # Request queue status
            queue_metrics = [k for k in metrics_found.keys() if any(term in k.lower() for term in ['running', 'waiting', 'queue'])]
            if queue_metrics:
                logger.debug(f"⏳ Queue status:")
                if 'vllm:num_requests_running' in metrics_found:
                    logger.debug(f"   Running: {metrics_found['vllm:num_requests_running']}")
                if 'vllm:num_requests_waiting' in metrics_found:
                    logger.debug(f"   Waiting: {metrics_found['vllm:num_requests_waiting']}")
            
            # Token processing totals
            token_metrics = [k for k in metrics_found.keys() if 'token' in k.lower()]
            if token_metrics:
                logger.debug(f"🔤 Token processing:")
                if 'vllm:prompt_tokens_total' in metrics_found:
                    logger.debug(f"   Prompt tokens: {metrics_found['vllm:prompt_tokens_total']:,}")
                if 'vllm:generation_tokens_total' in metrics_found:
                    logger.debug(f"   Generated tokens: {metrics_found['vllm:generation_tokens_total']:,}")
            
            # Summary of all available metrics
            if len(metrics_found) > 10:
                logger.debug(f"📋 Total metrics available: {len(metrics_found)} (showing key metrics above)")
            elif len(metrics_found) > 0:
                logger.debug(f"📋 All available metrics: {list(metrics_found.keys())}")
                
        except Exception as e:
            logger.debug(f"Error logging Prometheus metrics: {e}")
    
    def collect_current_stats(self) -> Dict[str, Any]:
        """Collect current vLLM statistics using Prometheus metrics"""
        return self.collect_prometheus_metrics()
    
    def _update_metrics_cache(self, stats: Dict[str, Any]):
        """Update metrics cache with latest Prometheus data"""
        try:
            if 'metrics_found' in stats:
                self.metrics_cache['prometheus_metrics'] = stats['metrics_found']
            
            if 'histogram_data' in stats:
                self.metrics_cache['histogram_metrics'] = stats['histogram_data']
                
            if 'timestamp' in stats:
                self.metrics_cache['collection_timestamps'].append(stats['timestamp'])
                
                # Keep only recent timestamps
                if len(self.metrics_cache['collection_timestamps']) > 1000:
                    self.metrics_cache['collection_timestamps'] = \
                        self.metrics_cache['collection_timestamps'][-500:]
                        
        except Exception as e:
            logger.debug(f"Error updating metrics cache: {e}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics with Prometheus metrics analysis"""
        current_stats = self.collect_prometheus_metrics()
        
        summary = {
            'latest_stats': current_stats,
            'monitoring_active': self.monitoring,
            'total_collections': len(self.stats_history),
            'collection_interval': self.collection_interval,
            'modern_metrics_enabled': self.modern_logging_available,
            'last_successful_collection': self.last_stats.get('collection_time', 0),
            'collection_method': current_stats.get('collection_method', 'prometheus_registry')
        }
        
        # Prometheus metrics analysis
        if current_stats.get('stats_available', False):
            metrics_found = current_stats.get('metrics_found', {})
            histogram_data = current_stats.get('histogram_data', {})
            
            # Key metrics summary
            summary['key_metrics'] = {}
            
    def _calculate_prefix_cache_hit_rate(self, metrics_found: Dict[str, Any]) -> Dict[str, float]:
        """Calculate prefix cache hit rates from modern counter metrics"""
        hit_rates = {}
        
        # GPU prefix cache hit rate
        gpu_hits = metrics_found.get('vllm:gpu_prefix_cache_hits', 0)
        gpu_queries = metrics_found.get('vllm:gpu_prefix_cache_queries', 0)
        if gpu_queries > 0:
            hit_rates['gpu_prefix_cache_hit_rate'] = gpu_hits / gpu_queries
            hit_rates['gpu_prefix_cache_hit_rate_percent'] = (gpu_hits / gpu_queries) * 100
        
        # CPU prefix cache hit rate  
        cpu_hits = metrics_found.get('vllm:cpu_prefix_cache_hits', 0)
        cpu_queries = metrics_found.get('vllm:cpu_prefix_cache_queries', 0)
        if cpu_queries > 0:
            hit_rates['cpu_prefix_cache_hit_rate'] = cpu_hits / cpu_queries
            hit_rates['cpu_prefix_cache_hit_rate_percent'] = (cpu_hits / cpu_queries) * 100
        
        # Store raw counters for reference
        hit_rates['gpu_prefix_cache_hits'] = gpu_hits
        hit_rates['gpu_prefix_cache_queries'] = gpu_queries
        hit_rates['cpu_prefix_cache_hits'] = cpu_hits
        hit_rates['cpu_prefix_cache_queries'] = cpu_queries
        
        return hit_rates

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics with modern prefix cache metrics analysis"""
        current_stats = self.collect_prometheus_metrics()
        
        summary = {
            'latest_stats': current_stats,
            'monitoring_active': self.monitoring,
            'total_collections': len(self.stats_history),
            'collection_interval': self.collection_interval,
            'modern_metrics_enabled': self.modern_logging_available,
            'last_successful_collection': self.last_stats.get('collection_time', 0),
            'collection_method': current_stats.get('collection_method', 'prometheus_registry')
        }
        
        # Prometheus metrics analysis
        if current_stats.get('stats_available', False):
            metrics_found = current_stats.get('metrics_found', {})
            histogram_data = current_stats.get('histogram_data', {})
            
            # Key metrics summary
            summary['key_metrics'] = {}
            
            # KV Cache Usage
            if 'vllm:gpu_cache_usage_perc' in metrics_found:
                cache_usage = metrics_found['vllm:gpu_cache_usage_perc']
                summary['key_metrics']['gpu_cache_usage_percent'] = cache_usage * 100
                summary['key_metrics']['gpu_cache_usage_raw'] = cache_usage
            
            # Calculate prefix cache hit rates from counters (modern approach)
            hit_rate_data = self._calculate_prefix_cache_hit_rate(metrics_found)
            summary['key_metrics'].update(hit_rate_data)
            
            # Fallback: Check for deprecated direct hit rate metrics
            if 'vllm:gpu_prefix_cache_hit_rate' in metrics_found:
                hit_rate = metrics_found['vllm:gpu_prefix_cache_hit_rate']
                if 'gpu_prefix_cache_hit_rate' not in summary['key_metrics']:
                    summary['key_metrics']['gpu_prefix_cache_hit_rate_percent'] = hit_rate * 100
                    summary['key_metrics']['gpu_prefix_cache_hit_rate_raw'] = hit_rate
            
            # Request Queue Status
            for metric_key in ['vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:num_requests_swapped']:
                if metric_key in metrics_found:
                    clean_key = metric_key.replace('vllm:', '').replace('num_', '')
                    summary['key_metrics'][clean_key] = metrics_found[metric_key]
            
            # Token Counters
            if 'vllm:prompt_tokens_total' in metrics_found:
                summary['key_metrics']['prompt_tokens_total'] = metrics_found['vllm:prompt_tokens_total']
            if 'vllm:generation_tokens_total' in metrics_found:
                summary['key_metrics']['generation_tokens_total'] = metrics_found['vllm:generation_tokens_total']
            
            # Speculative Decoding (if available)
            if 'vllm:spec_decode_draft_acceptance_rate' in metrics_found:
                summary['key_metrics']['spec_decode_acceptance_rate'] = metrics_found['vllm:spec_decode_draft_acceptance_rate']
            if 'vllm:spec_decode_efficiency' in metrics_found:
                summary['key_metrics']['spec_decode_efficiency'] = metrics_found['vllm:spec_decode_efficiency']
            
            # Histogram analysis
            if histogram_data:
                summary['histogram_analysis'] = {}
                
                # Time to First Token analysis
                if 'vllm:time_to_first_token_seconds' in histogram_data:
                    ttft_data = histogram_data['vllm:time_to_first_token_seconds']
                    if ttft_data['count'] > 0:
                        avg_ttft = ttft_data['sum'] / ttft_data['count']
                        summary['histogram_analysis']['avg_time_to_first_token_seconds'] = avg_ttft
                        summary['histogram_analysis']['total_ttft_requests'] = ttft_data['count']
                
                # Time per Output Token analysis
                if 'vllm:time_per_output_token_seconds' in histogram_data:
                    tpot_data = histogram_data['vllm:time_per_output_token_seconds']
                    if tpot_data['count'] > 0:
                        avg_tpot = tpot_data['sum'] / tpot_data['count']
                        summary['histogram_analysis']['avg_time_per_output_token_seconds'] = avg_tpot
                        summary['histogram_analysis']['total_tpot_samples'] = tpot_data['count']
                
                # E2E Request Latency analysis  
                if 'vllm:e2e_request_latency_seconds' in histogram_data:
                    e2e_data = histogram_data['vllm:e2e_request_latency_seconds']
                    if e2e_data['count'] > 0:
                        avg_e2e = e2e_data['sum'] / e2e_data['count']
                        summary['histogram_analysis']['avg_e2e_latency_seconds'] = avg_e2e
                        summary['histogram_analysis']['total_e2e_requests'] = e2e_data['count']
        
        return summary
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get metrics collection history for backward compatibility"""
        return self.stats_history.copy()
    
    def get_prometheus_metrics(self) -> Dict[str, Any]:
        """Get current Prometheus metrics snapshot"""
        current_stats = self.collect_prometheus_metrics()
        
        if current_stats.get('stats_available', False):
            # Return the found metrics with vllm: prefix preserved
            return current_stats.get('metrics_found', {})
        else:
            return {
                'error': current_stats.get('error', 'No metrics available'),
                'collection_method': 'prometheus_registry'
            }
    
    def save_metrics(self, csv_path: str):
        """Save metrics history to CSV"""
        if not self.stats_history:
            logger.warning("No vLLM metrics history to save")
            return
            
        try:
            # Flatten the metrics history for CSV export
            flattened_data = []
            
            for entry in self.stats_history:
                row = {
                    'timestamp': entry.get('timestamp', 0),
                    'collection_time': entry.get('collection_time', 0),
                    'stats_available': entry.get('stats_available', False),
                    'total_metrics_found': entry.get('total_metrics_found', 0),
                    'total_histograms_found': entry.get('total_histograms_found', 0)
                }
                
                # Add individual metrics
                metrics_found = entry.get('metrics_found', {})
                for metric_name, value in metrics_found.items():
                    # Clean the metric name for CSV column
                    clean_name = metric_name.replace('vllm:', '').replace(':', '_')
                    row[clean_name] = value
                
                # Add histogram summaries
                histogram_data = entry.get('histogram_data', {})
                for hist_name, hist_info in histogram_data.items():
                    clean_hist_name = hist_name.replace('vllm:', '').replace(':', '_')
                    row[f'{clean_hist_name}_count'] = hist_info.get('count', 0)
                    row[f'{clean_hist_name}_sum'] = hist_info.get('sum', 0)
                    if hist_info.get('count', 0) > 0:
                        row[f'{clean_hist_name}_avg'] = hist_info.get('sum', 0) / hist_info.get('count', 1)
                
                flattened_data.append(row)
            
            # Save to CSV
            df = pd.DataFrame(flattened_data)
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(flattened_data)} vLLM metrics records to {csv_path}")
            
        except Exception as e:
            logger.error(f"Error saving vLLM metrics to CSV: {e}")

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get the complete metrics collection history"""
        return self.stats_history.copy()
    
    def plot_kv_cache_metrics(self, output_dir: str):
        """Plotting functionality disabled as requested"""
        logger.info("📊 Plotting functionality has been disabled")
        return

def resolve_model_path(model_name: str) -> str:
    """Resolve model path, supporting local directories and HF model names"""
    
    # Check if it's a local directory
    if os.path.isdir(model_name):
        logger.info(f"Using local model directory: {model_name}")
        return model_name
    
    # Check if it's a relative path that exists
    if os.path.exists(model_name):
        abs_path = os.path.abspath(model_name)
        logger.info(f"Using local model path: {abs_path}")
        return abs_path
    
    # Check environment variable override
    if "LOCAL_MODEL_PATH" in os.environ and os.path.isdir(os.environ["LOCAL_MODEL_PATH"]):
        local_path = os.environ["LOCAL_MODEL_PATH"]
        logger.info(f"Using model from LOCAL_MODEL_PATH: {local_path}")
        return local_path
    
    # Default: assume it's a HF model name (will use cache if available)
    logger.info(f"Using Hugging Face model: {model_name}")
    return model_name

def validate_local_model(model_path: str) -> bool:
    """Validate that a local model directory contains necessary files"""
    if not os.path.isdir(model_path):
        return False
    
    # Check for essential model files
    required_files = []
    optional_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
    model_files = ['pytorch_model.bin', 'model.safetensors', 'pytorch_model-00001-of-*.bin']
    
    dir_contents = os.listdir(model_path)
    
    # Check for config file
    has_config = any(f in dir_contents for f in ['config.json'])
    
    # Check for model weights (any format)
    has_weights = any(
        pattern in dir_contents or any(filename.startswith(pattern.split('*')[0]) for filename in dir_contents)
        for pattern in model_files
    ) or any(file.endswith('.safetensors') for file in dir_contents)
    
    if has_config and has_weights:
        logger.info(f"Local model validation passed: {model_path}")
        logger.info(f"Found files: {[f for f in dir_contents if f.endswith(('.json', '.bin', '.safetensors'))]}")
        return True
    else:
        logger.warning(f"Local model validation failed: {model_path}")
        logger.warning(f"Missing: {'config.json' if not has_config else ''} {'model weights' if not has_weights else ''}")
        return False

class SimpleLLMExperiment:
    """Simple experiment class for LLM querying with predefined templates and resource monitoring"""
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-2-7b-hf",
                 output_dir: str = "llm_results",
                 gpu_ids: List[int] = None,
                 gpu_id: int = None):  # Keep for backward compatibility
        
        // ...existing code...
        
        # Validate local model if it's a local path
        if self.is_local_model:
            if not validate_local_model(self.model_name):
                logger.error(f"Invalid local model directory: {self.model_name}")
                sys.exit(1)
            logger.info(f"Using local model: {self.model_name}")
        
        // ...existing code...
    
    def load_dataset(self, dataset_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Load dataset from file"""
        logger.info(f"Loading dataset from: {dataset_path}")
        
        try:
            // ...existing code...
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return pd.DataFrame()
    
    def initialize_model(self, 
                        max_tokens: int = 512,
                        temperature: float = 0.1,
                        top_p: float = 0.9,
                        gpu_memory_utilization: float = 0.85,
                        max_model_len: int = None,
                        **kwargs):
        """Initialize vLLM model with GPU configuration and memory optimization"""
        // ...existing code...
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM model: {e}")
            
            // ...existing code...
            
            return False

    def generate_comprehensive_report(self, experiment_results: Dict[str, Any], output_folder: str) -> str:
        """Generate a comprehensive experiment report in Markdown format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_folder, "comprehensive_experiment_report.md")
        
        with open(report_file, 'w') as f:
            self._write_report_header(f, experiment_results, timestamp)
            self._write_table_of_contents(f)
            self._write_experiment_configuration(f, experiment_results)
            self._write_performance_summary(f, experiment_results)
            self._write_vllm_metrics_analysis(f, experiment_results)
            self._write_resource_utilization(f, experiment_results)
            self._write_batch_analysis(f, experiment_results)
            self._write_kv_cache_visualizations(f, experiment_results, output_folder)
            self._write_analysis_and_recommendations(f, experiment_results)
            self._write_data_exports(f, output_folder)
        
        logger.info(f"📊 Comprehensive report generated: {report_file}")
        return report_file
    
    def _write_report_header(self, f, experiment_results: Dict[str, Any], timestamp: str):
        """Write the report header section"""
        exp_info = experiment_results.get('experiment_info', {})
        f.write(f"# 📊 Comprehensive vLLM Experiment Report\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}  \n")
        f.write(f"**Run Folder**: `{os.path.basename(exp_info.get('dataset_path', 'unknown'))}`  \n")
        f.write(f"**Experiment**: {exp_info.get('query_key', 'N/A')} on {os.path.basename(exp_info.get('dataset_path', 'unknown'))}  \n\n")
        
        // ...existing code...
    
    def _write_table_of_contents(self, f):
        """Write the table of contents"""
        f.write(f"## 📋 Table of Contents\n\n")
        f.write(f"1. [Experiment Configuration](#1-experiment-configuration)\n")
        f.write(f"2. [Performance Summary](#2-performance-summary)\n")
        f.write(f"3. [vLLM Engine Metrics](#3-vllm-engine-metrics)\n")
        f.write(f"4. [Resource Utilization](#4-resource-utilization)\n")
        f.write(f"5. [Processing Overview](#5-processing-overview)\n")
        f.write(f"6. [KV Cache Visualizations](#6-kv-cache-visualizations)\n")
        f.write(f"7. [Analysis & Recommendations](#7-analysis--recommendations)\n")
        f.write(f"8. [Data Exports](#8-data-exports)\n\n")
    
    def _write_experiment_configuration(self, f, experiment_results: Dict[str, Any]):
        """Write experiment configuration section"""
        exp_info = experiment_results.get('experiment_info', {})
        f.write(f"## 1. Experiment Configuration\n\n")
        f.write(f"| Parameter | Value |\n")
        f.write(f"|-----------|-------|\n")
        f.write(f"| **Model** | {exp_info.get('model_name', 'N/A')} |\n")
        f.write(f"| **Dataset** | {os.path.basename(exp_info.get('dataset_path', 'N/A'))} |\n")
        f.write(f"| **Query Key** | {exp_info.get('query_key', 'N/A')} |\n")
        f.write(f"| **Query Type** | {exp_info.get('query_type', 'N/A')} |\n")
        f.write(f"| **GPU ID** | {exp_info.get('gpu_id', 'N/A')} |\n")
        f.write(f"| **Total Rows** | {exp_info.get('total_rows', 'N/A'):,} |\n")
        f.write(f"| **Processed Rows** | {exp_info.get('processed_rows', 'N/A'):,} |\n")
        f.write(f"| **Batch Size** | {exp_info.get('batch_size', 'N/A')} |\n")
        f.write(f"| **Experiment Time** | {exp_info.get('experiment_timestamp', 'N/A')} |\n\n")
    
    def _write_performance_summary(self, f, experiment_results: Dict[str, Any]):
        """Write performance summary section"""
        perf = experiment_results.get('performance_metrics', {})
        f.write(f"## 2. Performance Summary\n\n")
        f.write(f"### ⏱️ Timing Metrics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| **Total Inference Time** | {perf.get('total_inference_time', 0):.2f}s |\n")
        f.write(f"| **Prompt Creation Time** | {perf.get('prompt_creation_time', 0):.2f}s |\n")
        f.write(f"| **Average Time per Row** | {perf.get('avg_time_per_row', 0):.3f}s |\n")
        f.write(f"| **Overall Throughput** | {perf.get('overall_throughput_tokens_per_sec', 0):.1f} tokens/sec |\n\n")
        
        f.write(f"### 🔢 Token Statistics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| **Estimated Total Tokens** | {perf.get('estimated_total_tokens', 0):,} |\n")
        f.write(f"| **Input Tokens** | {perf.get('estimated_input_tokens', 0):,} |\n")
        f.write(f"| **Output Tokens** | {perf.get('estimated_output_tokens', 0):,} |\n\n")
    
    def _write_vllm_metrics_analysis(self, f, experiment_results: Dict[str, Any]):
        """Write vLLM metrics analysis section"""
        vllm_metrics = experiment_results.get('vllm_metrics', {})
        f.write(f"## 3. vLLM Engine Metrics\n\n")
        f.write(f"### 🔧 Collection Status\n\n")
        f.write(f"- **Prefix Caching Enabled**: {vllm_metrics.get('prefix_caching_enabled', True)}\n")
        
        initial_stats = vllm_metrics.get('initial_stats', {})
        final_stats = vllm_metrics.get('final_stats', {})
        
        if final_stats.get('vllm_stats_available', False):
            key_metrics = final_stats.get('key_metrics', {})
            collection_methods = final_stats.get('collection_methods_tried', [])
            f.write(f"- **Collection Methods**: {', '.join(collection_methods)}\n")
            f.write(f"- **Metrics Available**: ✅ Yes\n\n")
            
            self._write_kv_cache_analysis(f, key_metrics, experiment_results)
            self._write_token_processing_stats(f, key_metrics)
            self._write_request_queue_stats(f, key_metrics)
            
        else:
            f.write(f"- **Metrics Available**: ❌ No\n")
            f.write(f"- **Error**: {final_stats.get('error', 'Unknown error')}\n")
            if 'collection_methods_tried' in final_stats:
                f.write(f"- **Methods Tried**: {', '.join(final_stats['collection_methods_tried'])}\n")
            f.write(f"\n")
    
    def _write_kv_cache_analysis(self, f, key_metrics: Dict[str, Any], experiment_results: Dict[str, Any]):
        """Write KV cache analysis subsection"""
        f.write(f"### 🎯 KV Cache Performance\n\n")
        
        // ...existing code...
    
    def _write_token_processing_stats(self, f, key_metrics: Dict[str, Any]):
        """Write token processing statistics"""
        f.write(f"### 🔤 Token Processing\n\n")
        if 'prompt_tokens_total' in key_metrics:
            f.write(f"- **Total Prompt Tokens**: {key_metrics['prompt_tokens_total']:,}\n")
        if 'generation_tokens_total' in key_metrics:
            f.write(f"- **Total Generation Tokens**: {key_metrics['generation_tokens_total']:,}\n")
        
        // ...existing code...
    
    def _write_request_queue_stats(self, f, key_metrics: Dict[str, Any]):
        """Write request queue statistics"""
        f.write(f"### 📊 Request Queue Status\n\n")
        if 'requests_running' in key_metrics:
            f.write(f"- **Running Requests**: {key_metrics['requests_running']}\n")
        elif 'num_requests_running' in key_metrics:
            f.write(f"- **Running Requests**: {key_metrics['num_requests_running']}\n")
        
        if 'requests_waiting' in key_metrics:
            f.write(f"- **Waiting Requests**: {key_metrics['requests_waiting']}\n")
        elif 'num_requests_waiting' in key_metrics:
            f.write(f"- **Waiting Requests**: {key_metrics['num_requests_waiting']}\n")
        
        f.write(f"\n")
    
    def _write_resource_utilization(self, f, experiment_results: Dict[str, Any]):
        """Write resource utilization section"""
        resource_data = experiment_results.get('resource_monitoring', {})
        if not resource_data:
            f.write(f"## 4. Resource Utilization\n\n")
            f.write(f"Resource monitoring data not available.\n\n")
            return
            
        f.write(f"## 4. Resource Utilization\n\n")
        f.write(f"### 💻 System Resources\n\n")
        f.write(f"| Resource | Mean | Max |\n")
        f.write(f"|----------|------|-----|\n")
        f.write(f"| **CPU Utilization** | {resource_data.get('cpu_utilization_mean', 0):.1f}% | {resource_data.get('cpu_utilization_max', 0):.1f}% |\n")
        f.write(f"| **Memory Utilization** | {resource_data.get('memory_utilization_mean', 0):.1f}% | {resource_data.get('memory_utilization_max', 0):.1f}% |\n")
        f.write(f"| **Memory Used** | {resource_data.get('memory_used_gb_mean', 0):.1f}GB | {resource_data.get('memory_used_gb_max', 0):.1f}GB |\n")
        
        if 'gpu_compute_utilization_mean' in resource_data:
            f.write(f"\n### 🖥️ GPU Resources\n\n")
            f.write(f"| Resource | Mean | Max |\n")
            f.write(f"|----------|------|-----|\n")
            f.write(f"| **GPU Compute** | {resource_data.get('gpu_compute_utilization_mean', 0):.1f}% | {resource_data.get('gpu_compute_utilization_max', 0):.1f}% |\n")
            f.write(f"| **GPU Memory** | {resource_data.get('gpu_memory_utilization_mean', 0):.1f}% | {resource_data.get('gpu_memory_utilization_max', 0):.1f}% |\n")
            f.write(f"| **GPU Memory Allocated** | {resource_data.get('gpu_memory_allocated_gb_mean', 0):.1f}GB | {resource_data.get('gpu_memory_allocated_gb_max', 0):.1f}GB |\n")
        
        f.write(f"\n### 📈 Monitoring Details\n\n")
        f.write(f"- **Duration**: {resource_data.get('monitoring_duration_seconds', 0):.1f}s\n")
        f.write(f"- **Total Samples**: {resource_data.get('total_samples', 0)}\n")
        f.write(f"- **Sampling Interval**: {resource_data.get('sampling_interval_seconds', 0):.1f}s\n\n")
    
    def _write_batch_analysis(self, f, experiment_results: Dict[str, Any]):
        """Write batch processing analysis - disabled as batch monitoring has been removed"""
        f.write(f"## 5. Processing Overview\n\n")
        f.write(f"Batch monitoring functionality has been removed from the experiment pipeline.\n")
        f.write(f"All prompts are now processed efficiently in a single operation.\n\n")
    
    def _write_kv_cache_visualizations(self, f, experiment_results: Dict[str, Any], output_folder: str):
        """Write KV cache visualizations section"""
        f.write(f"## 6. KV Cache Visualizations\n\n")
        
        // ...existing code...
    
    def _write_analysis_and_recommendations(self, f, experiment_results: Dict[str, Any]):
        """Write analysis and recommendations section"""
        f.write(f"## 7. Analysis & Recommendations\n\n")
        
        // ...existing code...
    
    def _write_data_exports(self, f, output_folder: str):
        """Write data exports section"""
        f.write(f"## 8. Data Exports\n\n")
        f.write(f"The following data files have been generated for further analysis:\n\n")
        
        // ...existing code...
        
        f.write(f"---\n\n")
        f.write(f"**Report Generated**: {datetime.now().isoformat()}  \n")
        f.write(f"**Tool**: vLLM Experiment Pipeline with GGR Algorithm  \n")
        f.write(f"**Version**: Enhanced Metrics Collection  \n")

    def create_prompt(self, template: Dict[str, str], data_fields: str, query: str = None) -> str:
        """Create prompt from template and data"""
        if query:
            // ...existing code...
            
        return full_prompt
    
    def format_data_fields(self, row: pd.Series, max_length: int = 2000) -> str:
        """Format data fields from DataFrame row"""
        fields = {}
        for col, val in row.items():
            if pd.notna(val):
                // ...existing code...
        
        return json.dumps(fields, indent=2)
    
    def run_batch_inference(self, prompts: List[str], batch_size: int = 8) -> List[str]:
        """Run inference on a batch of prompts and collect KV cache stats immediately after"""
        if not self.llm:
            logger.error("Model not initialized!")
            return []
        
        try:
            // ...existing code...
            
            return outputs
            
        except Exception as e:
            logger.error(f"Error in batch inference: {e}")
            return [""] * len(prompts)
    
    def get_kv_cache_and_prefix_stats(self) -> Dict[str, float]:
        """
        Get KV cache and prefix hit statistics using the proven offline inference approach.
        This follows the exact pattern from the user's example:
        
        # Access internal stats (not a public API, may change)
        stats = llm.llm_engine._get_stats()
        print(f"GPU KV cache usage: {stats.gpu_cache_usage_sys * 100:.2f}%")
        print(f"Prefix cache hit rate: {getattr(stats, 'gpu_prefix_cache_hit_rate', 0) * 100:.2f}%")
        """
        if not self.llm or not hasattr(self.llm, 'llm_engine'):
            logger.warning("LLM or llm_engine not available for stats collection")
            return {}
        
        try:
            # Access internal stats (not a public API, may change) - exact approach from example
            stats = self.llm.llm_engine._get_stats()
            
            if not stats:
                logger.warning("No stats returned from _get_stats()")
                return {}
            
            # Extract the key metrics using the exact approach from the example
            result = {}
            
            # GPU KV cache usage - direct attribute access
            if hasattr(stats, 'gpu_cache_usage_sys'):
                gpu_cache_usage_pct = stats.gpu_cache_usage_sys * 100
                result['gpu_kv_cache_usage_percent'] = gpu_cache_usage_pct
                logger.info(f"GPU KV cache usage: {gpu_cache_usage_pct:.2f}%")
            
            # Prefix cache hit rate - using getattr with default as in example
            prefix_hit_rate_pct = getattr(stats, 'gpu_prefix_cache_hit_rate', 0) * 100
            if prefix_hit_rate_pct > 0:
                result['prefix_cache_hit_rate_percent'] = prefix_hit_rate_pct
                logger.info(f"Prefix cache hit rate: {prefix_hit_rate_pct:.2f}%")
            else:
                result['prefix_cache_hit_rate_percent'] = 0.0
                logger.info("Prefix cache hit rate: 0.00% (no hits detected or not available)")
            
            # Additional useful metrics if available
            if hasattr(stats, 'cpu_cache_usage_sys'):
                cpu_cache_usage_pct = stats.cpu_cache_usage_sys * 100
                result['cpu_kv_cache_usage_percent'] = cpu_cache_usage_pct
            
            cpu_prefix_hit_rate_pct = getattr(stats, 'cpu_prefix_cache_hit_rate', 0) * 100
            if cpu_prefix_hit_rate_pct > 0:
                result['cpu_prefix_cache_hit_rate_percent'] = cpu_prefix_hit_rate_pct
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to collect KV cache and prefix stats: {e}")
            return {}

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 characters ≈ 1 token)"""
        return len(text) // 4
    
    def run_experiment(self, 
                      dataset_path: str, 
                      query_key: str,
                      custom_query: str = None,
                      max_rows: int = None,
                      batch_size: int = 8,
                      output_prefix: str = None) -> Dict[str, Any]:
        """Run complete experiment with comprehensive monitoring and reporting"""
        
        // ...existing code...
            
            return experiment_results

            
        except Exception as e:
            logger.error(f"❌ Experiment failed: {e}")
            
            // ...existing code...
            
            raise
    
    def plot_kv_cache_metrics(self, output_dir: str) -> Dict[str, str]:
        """Plotting functionality disabled as requested"""
        logger.info("📊 KV cache plotting functionality has been disabled")
        return {}


# Create a type alias for backward compatibility
LLMExperimentRunner = SimpleLLMExperiment


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Run LLM experiments with vLLM and comprehensive monitoring')
    
    // ...existing code...
    
    args = parser.parse_args()
    
    // ...existing code...
    
    try:
        // ...existing code...
        
        # Initialize model
        if not args.skip_model_init:
            print("🔄 Loading model...")
            success = experiment.initialize_model(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                gpu_memory_utilization=args.gpu_memory,
                max_model_len=args.max_model_len
            )
            
            if not success:
                print("❌ Failed to initialize model")
                sys.exit(1)
        else:
            print("⚠️  Skipping model initialization (testing mode)")
        
        // ...existing code...
        
        print("\n" + "=" * 70)
        print("🎉 Experiment completed successfully!")
        
        // ...existing code...
        
    except KeyboardInterrupt:
        print("\n❌ Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

