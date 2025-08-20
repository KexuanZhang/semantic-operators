#!/usr/bin/env python3
"""
Simple LLM Query Experiment Script with vLLM
Uses predefined query templates to run experiments on datasets with GPU support
Includes comprehensive resource monitoring and KV cache tracking

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
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

# Try to import prometheus_client for accessing vLLM metrics
try:
    import prometheus_client
    from prometheus_client import CollectorRegistry, REGISTRY
    prometheus_available = True
except ImportError:
    logger.warning("prometheus_client not available. vLLM metrics will be limited.")
    prometheus_available = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check GPU availability and CUDA setup"""
    if torch.cuda.is_available():
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
    """Monitor system resources during inference"""
    
    def __init__(self, gpu_id: int = 0, sampling_interval: float = 2.0):
        self.gpu_id = gpu_id
        self.sampling_interval = sampling_interval
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
            self.monitor_thread.join(timeout=5.0)
        logger.info("Stopped resource monitoring")
        
    def _monitor_loop(self):
        """Main monitoring loop running in background"""
        while self.monitoring:
            try:
                timestamp = time.time()
                metrics = self._collect_metrics(timestamp)
                if metrics:
                    self.metrics_log.append(metrics)
            except Exception as e:
                logger.warning(f"Error collecting metrics: {e}")
            
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
    
    def __init__(self, llm_instance=None):
        self.llm = llm_instance
        self.stats_history = []
        self.collection_interval = 1.0
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
            
            # Prefix Cache Hit Rates - KEY PERFORMANCE INDICATORS
            'gpu_prefix_cache_hit_rate': 'vllm:gpu_prefix_cache_hit_rate',  # Deprecated but may exist
            'cpu_prefix_cache_hit_rate': 'vllm:cpu_prefix_cache_hit_rate',  # Deprecated but may exist
            
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
    
    def collect_prometheus_metrics(self) -> Dict[str, Any]:
        """Collect current Prometheus metrics from the global registry"""
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
            'histogram_data': {}
        }
        
        try:
            # Get all metric families from the global Prometheus registry
            metric_families = list(REGISTRY.collect())
            
            found_vllm_metrics = {}
            found_histograms = {}
            
            for family in metric_families:
                metric_name = family.name
                
                # Check if this is a vLLM metric we're interested in
                if metric_name.startswith('vllm:'):
                    
                    # Handle gauge and counter metrics
                    if family.type in ['gauge', 'counter']:
                        for sample in family.samples:
                            # Extract the metric value (usually the first sample)
                            if sample.name == metric_name:
                                found_vllm_metrics[metric_name] = sample.value
                                logger.debug(f"Found metric {metric_name}: {sample.value}")
                                break
                    
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
            
            # Update metrics data
            if found_vllm_metrics or found_histograms:
                metrics_data.update({
                    'stats_available': True,
                    'metrics_found': found_vllm_metrics,
                    'histogram_data': found_histograms,
                    'total_metrics_found': len(found_vllm_metrics),
                    'total_histograms_found': len(found_histograms)
                })
                
                logger.debug(f"Collected {len(found_vllm_metrics)} metrics and {len(found_histograms)} histograms")
            else:
                metrics_data['error'] = 'No vLLM metrics found in Prometheus registry'
                logger.debug("No vLLM metrics found in Prometheus registry")
                
        except Exception as e:
            metrics_data['error'] = f'Error collecting Prometheus metrics: {str(e)}'
            logger.debug(f"Error collecting Prometheus metrics: {e}")
        
        return metrics_data
    
    def start_monitoring(self):
        """Start background monitoring"""
        if self.monitoring:
            logger.info("vLLM metrics monitoring already active")
            return
            
        # Force initialization if not done
        if not self.modern_logging_available:
            self.initialize_logging()
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started vLLM Prometheus metrics monitoring thread")
        
    def stop_monitoring(self):
        """Stop background monitoring"""
        if not self.monitoring:
            return
            
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=3.0)
        logger.info("Stopped vLLM metrics monitoring")
        
    def _monitoring_loop(self):
        """Enhanced monitoring loop collecting Prometheus metrics"""
        consecutive_failures = 0
        max_failures = 10
        
        while self.monitoring:
            try:
                stats = self.collect_prometheus_metrics()
                
                if stats.get('stats_available', False):
                    stats['timestamp'] = time.time()
                    self.stats_history.append(stats)
                    self.last_stats = stats.copy()
                    
                    # Update cache
                    self._update_metrics_cache(stats)
                    
                    # Reset failure counter on success
                    consecutive_failures = 0
                    
                    # Log key metrics periodically
                    if len(self.stats_history) % 10 == 0:
                        self._log_key_prometheus_metrics(stats)
                        
                else:
                    consecutive_failures += 1
                    if consecutive_failures == 1:
                        logger.debug("vLLM Prometheus metrics collection failed, will retry")
                    elif consecutive_failures >= max_failures:
                        logger.warning(f"vLLM Prometheus metrics collection failed {max_failures} times")
                        consecutive_failures = 0  # Reset to avoid spam
                
                # Keep history manageable
                if len(self.stats_history) > 1000:
                    self.stats_history = self.stats_history[-500:]
                    
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures == 1:
                    logger.debug(f"Error in vLLM Prometheus monitoring loop: {e}")
                
            time.sleep(self.collection_interval)
    
    def _log_key_prometheus_metrics(self, stats):
        """Log key Prometheus metrics for debugging"""
        try:
            metrics_found = stats.get('metrics_found', {})
            
            # KV Cache Usage
            if 'vllm:gpu_cache_usage_perc' in metrics_found:
                cache_pct = metrics_found['vllm:gpu_cache_usage_perc'] * 100
                logger.debug(f"GPU KV Cache Usage: {cache_pct:.1f}%")
            
            # Prefix Cache Hit Rate (may be deprecated)
            if 'vllm:gpu_prefix_cache_hit_rate' in metrics_found:
                hit_rate = metrics_found['vllm:gpu_prefix_cache_hit_rate'] * 100
                logger.debug(f"GPU Prefix Cache Hit Rate: {hit_rate:.1f}%")
            
            # Request queue status
            if 'vllm:num_requests_running' in metrics_found:
                logger.debug(f"Running requests: {metrics_found['vllm:num_requests_running']}")
            if 'vllm:num_requests_waiting' in metrics_found:
                logger.debug(f"Waiting requests: {metrics_found['vllm:num_requests_waiting']}")
            
            # Token processing
            if 'vllm:prompt_tokens_total' in metrics_found:
                logger.debug(f"Total prompt tokens: {metrics_found['vllm:prompt_tokens_total']}")
            if 'vllm:generation_tokens_total' in metrics_found:
                logger.debug(f"Total generation tokens: {metrics_found['vllm:generation_tokens_total']}")
                
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
            
            # KV Cache Usage
            if 'vllm:gpu_cache_usage_perc' in metrics_found:
                cache_usage = metrics_found['vllm:gpu_cache_usage_perc']
                summary['key_metrics']['gpu_cache_usage_percent'] = cache_usage * 100
                summary['key_metrics']['gpu_cache_usage_raw'] = cache_usage
            
            # Prefix Cache Hit Rate (if available - may be deprecated)
            if 'vllm:gpu_prefix_cache_hit_rate' in metrics_found:
                hit_rate = metrics_found['vllm:gpu_prefix_cache_hit_rate']
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
        
        # Handle GPU specification (backward compatible)
        if gpu_ids is None:
            if gpu_id is not None:
                self.gpu_ids = [gpu_id]
                self.gpu_id = gpu_id  # Primary GPU for monitoring
            else:
                self.gpu_ids = [0]  # Default to GPU 0
                self.gpu_id = 0
        else:
            self.gpu_ids = gpu_ids
            self.gpu_id = gpu_ids[0]  # Use first GPU as primary for monitoring
        
        # Resolve model path (local or HuggingFace)
        self.model_name = resolve_model_path(model_name)
        self.is_local_model = os.path.exists(self.model_name)
        
        self.output_dir = output_dir
        self.llm = None
        self.sampling_params = None
        self.resource_monitor = ResourceMonitor(gpu_id=self.gpu_id)
        self.vllm_metrics_collector = VLLMMetricsCollector(llm_instance=None)  # Will be updated after model init
        
        # Validate local model if it's a local path
        if self.is_local_model:
            if not validate_local_model(self.model_name):
                logger.error(f"Invalid local model directory: {self.model_name}")
                sys.exit(1)
            logger.info(f"Using local model: {self.model_name}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check and set GPUs
        gpu_available, gpu_count = check_gpu_availability()
        if gpu_available:
            # Validate all requested GPUs are available
            for gpu_id in self.gpu_ids:
                if gpu_id >= gpu_count:
                    logger.error(f"GPU {gpu_id} not available. Available GPUs: 0-{gpu_count-1}")
                    sys.exit(1)
            
            if set_gpu_devices(self.gpu_ids):
                if len(self.gpu_ids) == 1:
                    logger.info(f"Using single GPU {self.gpu_ids[0]} for inference")
                else:
                    logger.info(f"Using multi-GPU setup: {self.gpu_ids} for inference")
            else:
                logger.error(f"Failed to set GPUs {self.gpu_ids}")
                sys.exit(1)
        else:
            logger.warning("No GPU available, will use CPU (very slow)")
    
    def load_dataset(self, dataset_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Load dataset from file"""
        logger.info(f"Loading dataset from: {dataset_path}")
        
        try:
            # Support multiple file formats
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.json'):
                df = pd.read_json(dataset_path)
            elif dataset_path.endswith('.jsonl'):
                df = pd.read_json(dataset_path, lines=True)
            elif dataset_path.endswith('.parquet'):
                df = pd.read_parquet(dataset_path)
            else:
                logger.error(f"Unsupported file format: {dataset_path}")
                return pd.DataFrame()
                
            logger.info(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
            logger.info(f"Columns: {list(df.columns)}")
            
            if max_rows and len(df) > max_rows:
                df = df.head(max_rows)
                logger.info(f"Limited to {max_rows} rows")
                
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
        logger.info(f"Initializing vLLM model: {self.model_name}")
        logger.info(f"Model type: {'Local' if self.is_local_model else 'HuggingFace'}")
        logger.info(f"Target GPUs: {self.gpu_ids}")
        
        # Check if user wants to force legacy engine for better stats compatibility
        use_legacy_engine = os.getenv('VLLM_USE_V1', '1') == '0'
        if use_legacy_engine:
            logger.info("🔧 Using legacy vLLM engine (VLLM_USE_V1=0) for better stats compatibility")
        else:
            logger.info("🚀 Using modern vLLM V1 engine (default)")
            logger.info("💡 Tip: Set VLLM_USE_V1=0 environment variable to use legacy engine with direct stats access")
        
        # Calculate memory requirements if local model
        if self.is_local_model:
            memory_info = calculate_memory_requirements(self.model_name, max_model_len)
            logger.info(f"Memory estimates - Model: {memory_info['model_memory_gb']:.1f}GB, "
                       f"KV Cache: {memory_info['kv_cache_gb']:.1f}GB, "
                       f"Total: {memory_info['total_estimated_gb']:.1f}GB")
            
            # Calculate total available GPU memory
            if torch.cuda.is_available():
                total_gpu_memory = sum(
                    torch.cuda.get_device_properties(gpu_id).total_memory / 1e9 
                    for gpu_id in self.gpu_ids
                )
                logger.info(f"Total available GPU memory: {total_gpu_memory:.1f}GB across {len(self.gpu_ids)} GPUs")
                
                # Auto-adjust max_model_len if not specified and memory is tight
                if max_model_len is None and memory_info['total_estimated_gb'] > total_gpu_memory * gpu_memory_utilization:
                    # Calculate a safer max_model_len based on available memory
                    available_memory = total_gpu_memory * gpu_memory_utilization - memory_info['model_memory_gb']
                    safe_max_len = int(available_memory / memory_info['kv_cache_per_token_gb'])
                    safe_max_len = min(safe_max_len, memory_info['max_seq_len'])
                    max_model_len = safe_max_len
                    logger.info(f"Auto-adjusting max_model_len to {max_model_len} based on available memory")
        
        try:
            # vLLM configuration for GPU inference with prefix caching enabled
            llm_config = {
                'model': self.model_name,
                'gpu_memory_utilization': gpu_memory_utilization,
                'max_num_seqs': 16,
                'enable_prefix_caching': True,  # Enable for KV cache reuse
                'disable_log_stats': False,     # Enable statistics logging for modern metrics
                'seed': 42,
                **kwargs
            }
            
            # Force legacy engine mode if requested for better stats compatibility
            if use_legacy_engine:
                # Try to force legacy engine by setting environment variable
                os.environ['VLLM_USE_V1'] = '0'
                logger.info("🔧 Forcing vLLM legacy engine mode for better stats access")
            
            # Multi-GPU configuration
            if len(self.gpu_ids) > 1:
                llm_config['tensor_parallel_size'] = len(self.gpu_ids)
                logger.info(f"Using tensor parallelism across {len(self.gpu_ids)} GPUs")
            
            # Set max_model_len if specified
            if max_model_len is not None:
                llm_config['max_model_len'] = max_model_len
                logger.info(f"Setting max_model_len to {max_model_len}")
            
            # Add local model specific configurations if needed
            if self.is_local_model:
                # For local models, we might need to be more explicit about tokenizer
                tokenizer_path = self.model_name
                if os.path.exists(os.path.join(self.model_name, 'tokenizer.json')):
                    llm_config['tokenizer'] = tokenizer_path
                    logger.info(f"Using tokenizer from: {tokenizer_path}")
            
            logger.info(f"vLLM Configuration: {llm_config}")
            
            self.llm = LLM(**llm_config)
            
            # Sampling parameters
            self.sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=42
            )
            
            # Log GPU memory usage after model loading
            if torch.cuda.is_available():
                for i, gpu_id in enumerate(self.gpu_ids):
                    allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
                    reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
                    total = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
                    utilization = (allocated / total) * 100
                    logger.info(f"GPU {gpu_id} Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, "
                               f"{total:.1f}GB total ({utilization:.1f}% util)")
            
            logger.info("vLLM model initialized successfully!")
            
            # Update and initialize the vLLM metrics collector
            self.vllm_metrics_collector.llm = self.llm
            metrics_init_success = self.vllm_metrics_collector.initialize_logging()
            if metrics_init_success:
                logger.info("Modern vLLM metrics collection enabled")
            else:
                logger.info("Using legacy vLLM metrics collection")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM model: {e}")
            
            # Log memory usage on all GPUs for debugging
            if torch.cuda.is_available():
                for gpu_id in self.gpu_ids:
                    try:
                        allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
                        total = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
                        logger.error(f"GPU {gpu_id} Memory: {allocated:.1f}GB allocated, {total:.1f}GB total")
                    except:
                        pass
            
            # Provide helpful error messages and suggestions
            if self.is_local_model:
                logger.error(f"Local model loading failed. Please check:")
                logger.error(f"1. Model files exist in: {self.model_name}")
                logger.error(f"2. Model format is compatible with vLLM")
                logger.error(f"3. All required files (config.json, model weights) are present")
            
            # Memory-related suggestions
            if "KV cache" in str(e) or "memory" in str(e).lower():
                logger.error("Memory optimization suggestions:")
                logger.error("1. Try using more GPUs: --gpus 4,5,6,7")
                logger.error("2. Increase gpu_memory_utilization: --gpu-memory 0.95")
                logger.error("3. Reduce max_model_len: --max-model-len 8192")
                logger.error("4. Reduce batch size: --batch-size 4")
                
                if torch.cuda.is_available():
                    total_memory = sum(
                        torch.cuda.get_device_properties(gpu_id).total_memory / 1e9 
                        for gpu_id in self.gpu_ids
                    )
                    logger.error(f"Total GPU memory available: {total_memory:.1f}GB across {len(self.gpu_ids)} GPUs")
            
            return False
    
    def get_vllm_stats(self) -> Dict[str, Any]:
        """Get vLLM internal statistics using multiple collection methods for version compatibility"""
        if not self.vllm_metrics_collector:
            logger.warning("vLLM metrics collector not available")
            return {'vllm_stats_available': False, 'error': 'Metrics collector not initialized'}
        
        # Try multiple methods to get vLLM stats for different versions
        stats = {'vllm_stats_available': False, 'collection_methods_tried': []}
        key_metrics = {}
        
        # Method 1: Try legacy _get_stats() for vLLM < 0.8 or VLLM_USE_V1=0
        try:
            if hasattr(self.llm, 'llm_engine') and hasattr(self.llm.llm_engine, '_get_stats'):
                legacy_stats = self.llm.llm_engine._get_stats()
                stats['collection_methods_tried'].append('legacy_get_stats')
                
                if legacy_stats:
                    logger.info("🔍 Legacy vLLM stats found:")
                    # Extract legacy format metrics
                    if hasattr(legacy_stats, 'gpu_cache_usage_perc'):
                        cache_usage = legacy_stats.gpu_cache_usage_perc
                        key_metrics['gpu_cache_usage_perc'] = cache_usage
                        key_metrics['gpu_cache_usage_percent'] = cache_usage * 100
                        logger.info(f"  📊 GPU KV Cache Usage: {cache_usage * 100:.1f}%")
                        stats['vllm_stats_available'] = True
                    
                    if hasattr(legacy_stats, 'gpu_prefix_cache_hit_rate'):
                        hit_rate = legacy_stats.gpu_prefix_cache_hit_rate
                        key_metrics['gpu_prefix_cache_hit_rate'] = hit_rate
                        key_metrics['gpu_prefix_cache_hit_rate_percent'] = hit_rate * 100
                        logger.info(f"  🎯 Prefix Cache Hit Rate: {hit_rate * 100:.1f}%")
                        stats['vllm_stats_available'] = True
                    
                    # Log all available attributes
                    available_attrs = [attr for attr in dir(legacy_stats) if not attr.startswith('_')]
                    logger.info(f"  📋 Available legacy stats attributes: {available_attrs}")
                    
                    # Other legacy metrics
                    for attr in ['num_requests_running', 'num_requests_waiting', 'prompt_tokens', 'generation_tokens']:
                        if hasattr(legacy_stats, attr):
                            value = getattr(legacy_stats, attr)
                            key_metrics[attr] = value
                            logger.info(f"  📊 {attr}: {value}")
                            stats['vllm_stats_available'] = True
                    
                    if hasattr(legacy_stats, 'gpu_cache_usage_sys'):
                        key_metrics['gpu_cache_usage_sys'] = legacy_stats.gpu_cache_usage_sys
                        logger.info(f"  💾 GPU Cache Usage (sys): {legacy_stats.gpu_cache_usage_sys}")
                        stats['vllm_stats_available'] = True
                            
        except Exception as e:
            logger.debug(f"Legacy stats collection failed: {e}")
            stats['collection_methods_tried'].append(f'legacy_get_stats_failed: {e}')
        
        # Method 2: Try Prometheus metrics collection for vLLM 0.8+
        try:
            prometheus_stats = self.vllm_metrics_collector.collect_prometheus_metrics()
            stats['collection_methods_tried'].append('prometheus_registry')
            
            if prometheus_stats.get('stats_available', False):
                metrics_found = prometheus_stats.get('metrics_found', {})
                histogram_data = prometheus_stats.get('histogram_data', {})
                
                logger.info(f"🔍 Prometheus metrics discovery:")
                logger.info(f"  📊 Total metrics found: {len(metrics_found)}")
                logger.info(f"  📈 Histogram metrics found: {len(histogram_data)}")
                
                if metrics_found:
                    logger.info(f"  📋 All available metrics:")
                    for metric_name, value in sorted(metrics_found.items()):
                        logger.info(f"    • {metric_name}: {value}")
                
                # Log all cache-related metrics for debugging
                all_cache_metrics = {k: v for k, v in metrics_found.items() 
                                   if any(term in k.lower() for term in ['cache', 'hit', 'prefix', 'query'])}
                if all_cache_metrics:
                    logger.info(f"  🗂️ Cache-related metrics found:")
                    for metric, value in all_cache_metrics.items():
                        logger.info(f"    • {metric}: {value}")
                else:
                    logger.info("  ⚠️ No cache-related metrics found")
                    
                # Try multiple possible metric names for cache hits/queries (vLLM version variations)
                cache_hit_variants = [
                    'vllm:gpu_prefix_cache_hits', 'vllm_gpu_prefix_cache_hits_total', 
                    'vllm_gpu_prefix_cache_hits', 'gpu_prefix_cache_hits',
                    'vllm:cache_hits', 'vllm_cache_hits_total',
                    'vllm:kv_cache_hits', 'vllm_kv_cache_hits_total',
                    'vllm:prefix_cache_hits', 'vllm_prefix_cache_hits_total'
                ]
                cache_query_variants = [
                    'vllm:gpu_prefix_cache_queries', 'vllm_gpu_prefix_cache_queries_total',
                    'vllm_gpu_prefix_cache_queries', 'gpu_prefix_cache_queries',
                    'vllm:cache_queries', 'vllm_cache_queries_total',
                    'vllm:kv_cache_queries', 'vllm_kv_cache_queries_total',
                    'vllm:prefix_cache_queries', 'vllm_prefix_cache_queries_total'
                ]
                
                gpu_cache_hits = 0
                gpu_cache_queries = 0
                cpu_cache_hits = 0 
                cpu_cache_queries = 0
                
                # Find GPU cache hits with extended search and reporting
                for variant in cache_hit_variants:
                    if variant in metrics_found:
                        gpu_cache_hits = metrics_found[variant]
                        logger.info(f"  ✅ Found GPU cache hits: {variant} = {gpu_cache_hits}")
                        break
                else:
                    # Try partial matches
                    for metric_name, value in metrics_found.items():
                        if 'gpu' in metric_name and any(term in metric_name.lower() for term in ['hit', 'cache']):
                            gpu_cache_hits = value
                            logger.info(f"  🔍 Partial match GPU cache hits: {metric_name} = {gpu_cache_hits}")
                            break
                    else:
                        logger.info("  ❌ No GPU cache hits metric found")
                
                # Find GPU cache queries with extended search and reporting
                for variant in cache_query_variants:
                    if variant in metrics_found:
                        gpu_cache_queries = metrics_found[variant]
                        logger.info(f"  ✅ Found GPU cache queries: {variant} = {gpu_cache_queries}")
                        break
                else:
                    # Try partial matches
                    for metric_name, value in metrics_found.items():
                        if ('gpu' in metric_name and 
                            any(term in metric_name.lower() for term in ['quer', 'request', 'cache']) and 
                            'hit' not in metric_name):
                            gpu_cache_queries = value
                            logger.info(f"  🔍 Partial match GPU cache queries: {metric_name} = {gpu_cache_queries}")
                            break
                    else:
                        logger.info("  ❌ No GPU cache queries metric found")
                
                # Calculate hit rates with proper validation and detailed reporting
                hit_rate_calculated = False
                
                if (gpu_cache_queries > 1 and gpu_cache_hits <= gpu_cache_queries and 
                    isinstance(gpu_cache_hits, (int, float)) and isinstance(gpu_cache_queries, (int, float))):
                    
                    # These look like real counters
                    gpu_hit_rate = gpu_cache_hits / gpu_cache_queries
                    key_metrics['gpu_prefix_cache_hit_rate'] = gpu_hit_rate
                    key_metrics['gpu_prefix_cache_hit_rate_percent'] = gpu_hit_rate * 100
                    key_metrics['gpu_prefix_cache_hits'] = int(gpu_cache_hits)
                    key_metrics['gpu_prefix_cache_queries'] = int(gpu_cache_queries)
                    logger.info(f"  🎯 Calculated GPU Prefix Cache Hit Rate: {gpu_hit_rate * 100:.2f}% ({int(gpu_cache_hits)}/{int(gpu_cache_queries)})")
                    hit_rate_calculated = True
                    stats['vllm_stats_available'] = True
                    
                elif gpu_cache_queries > 0 and gpu_cache_hits > 0:
                    # Values are suspicious - likely percentages/ratios rather than counters
                    logger.warning(f"  ⚠️ Suspicious cache values (likely not counters): hits={gpu_cache_hits}, queries={gpu_cache_queries}")
                    logger.warning(f"     These appear to be ratios/percentages, not actual hit/query counts")
                    logger.warning(f"     Skipping calculation to avoid misleading 100% hit rates")
                
                # KV Cache Usage
                if 'vllm:gpu_cache_usage_perc' in metrics_found:
                    cache_usage = metrics_found['vllm:gpu_cache_usage_perc']
                    key_metrics['gpu_cache_usage_perc'] = cache_usage
                    key_metrics['gpu_cache_usage_percent'] = cache_usage * 100
                    logger.info(f"  🔥 GPU KV Cache Usage: {cache_usage * 100:.1f}%")
                    stats['vllm_stats_available'] = True
                
                # Also try direct hit rate metric (may exist in some versions)
                direct_hit_rate_found = False
                for direct_metric_name in ['vllm:gpu_prefix_cache_hit_rate', 'gpu_prefix_cache_hit_rate', 
                                         'vllm:prefix_cache_hit_rate', 'vllm_gpu_prefix_cache_hit_rate']:
                    if direct_metric_name in metrics_found:
                        hit_rate_value = metrics_found[direct_metric_name]
                        logger.info(f"  🔍 Found direct hit rate metric: {direct_metric_name} = {hit_rate_value}")
                        
                        # Validate hit rate value
                        if 0 <= hit_rate_value <= 1:
                            # Value is a ratio (0-1)
                            key_metrics['gpu_prefix_cache_hit_rate'] = hit_rate_value
                            key_metrics['gpu_prefix_cache_hit_rate_percent'] = hit_rate_value * 100
                            logger.info(f"  🎯 Direct GPU Prefix Cache Hit Rate: {hit_rate_value * 100:.2f}%")
                            direct_hit_rate_found = True
                            stats['vllm_stats_available'] = True
                        elif 0 <= hit_rate_value <= 100:
                            # Value is already a percentage (0-100)
                            key_metrics['gpu_prefix_cache_hit_rate'] = hit_rate_value / 100
                            key_metrics['gpu_prefix_cache_hit_rate_percent'] = hit_rate_value
                            logger.info(f"  🎯 Direct GPU Prefix Cache Hit Rate: {hit_rate_value:.2f}%")
                            direct_hit_rate_found = True
                            stats['vllm_stats_available'] = True
                        else:
                            logger.warning(f"  ⚠️ Invalid hit rate value: {hit_rate_value} (expected 0-1 or 0-100)")
                        break
                
                # Request Queue Status
                for metric_key in ['vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:num_requests_swapped']:
                    if metric_key in metrics_found:
                        clean_key = metric_key.replace('vllm:', '').replace('num_', '')
                        key_metrics[clean_key] = metrics_found[metric_key]
                        logger.info(f"  📊 {clean_key}: {metrics_found[metric_key]}")
                        stats['vllm_stats_available'] = True
                
                # Token Processing Counters
                if 'vllm:prompt_tokens_total' in metrics_found:
                    key_metrics['prompt_tokens_total'] = metrics_found['vllm:prompt_tokens_total']
                    logger.info(f"  📝 Total Prompt Tokens: {metrics_found['vllm:prompt_tokens_total']}")
                    stats['vllm_stats_available'] = True
                
                if 'vllm:generation_tokens_total' in metrics_found:
                    key_metrics['generation_tokens_total'] = metrics_found['vllm:generation_tokens_total']
                    logger.info(f"  🔤 Total Generation Tokens: {metrics_found['vllm:generation_tokens_total']}")
                    stats['vllm_stats_available'] = True
                
                # Performance Histograms Analysis
                if histogram_data:
                    logger.info(f"  📈 Performance histograms available:")
                    histogram_summary = {}
                    
                    for hist_name, hist_info in histogram_data.items():
                        count = hist_info.get('count', 0)
                        sum_val = hist_info.get('sum', 0)
                        logger.info(f"    • {hist_name}: {count} samples, {sum_val:.3f} total")
                        
                        if count > 0:
                            avg_val = sum_val / count
                            if 'time_to_first_token' in hist_name:
                                histogram_summary['avg_time_to_first_token_seconds'] = avg_val
                                logger.info(f"      → Avg TTFT: {avg_val:.3f}s")
                            elif 'time_per_output_token' in hist_name:
                                histogram_summary['avg_time_per_output_token_seconds'] = avg_val
                                logger.info(f"      → Avg TPOT: {avg_val:.4f}s")
                            elif 'e2e_request_latency' in hist_name:
                                histogram_summary['avg_e2e_latency_seconds'] = avg_val
                                logger.info(f"      → Avg E2E Latency: {avg_val:.3f}s")
                    
                    key_metrics['histogram_summary'] = histogram_summary
                    
        except Exception as e:
            logger.debug(f"Prometheus metrics collection failed: {e}")
            stats['collection_methods_tried'].append(f'prometheus_failed: {e}')
        
        # ... rest of the methods (3, 4) remain the same but add similar detailed logging
        
        # Final summary of what was found
        stats['key_metrics'] = key_metrics
        stats['prefix_caching_enabled'] = True
        
        # Detailed summary logging
        logger.info(f"\n📊 vLLM METRICS COLLECTION SUMMARY:")
        logger.info(f"  Collection methods tried: {', '.join(stats['collection_methods_tried'])}")
        logger.info(f"  Metrics available: {stats['vllm_stats_available']}")
        logger.info(f"  Total key metrics found: {len(key_metrics)}")
        
        if key_metrics:
            logger.info(f"  📋 Key metrics collected:")
            for metric, value in key_metrics.items():
                if isinstance(value, dict):
                    logger.info(f"    • {metric}: {len(value)} items")
                else:
                    logger.info(f"    • {metric}: {value}")
        else:
            logger.info(f"  ❌ No key metrics collected")
            
        return stats

    def generate_performance_report(self, experiment_results: Dict[str, Any]):
        """Generate comprehensive performance report in Markdown format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_key = experiment_results['experiment_info']['query_key']
        gpu_info = f"gpu{experiment_results['experiment_info']['gpu_id']}"
        
        # Use the same timestamped folder structure
        run_folder = os.path.join(self.output_dir, f"{query_key}_{gpu_info}_{timestamp}")
        os.makedirs(run_folder, exist_ok=True)
        
        report_file = os.path.join(run_folder, "performance_report.md")
        
        with open(report_file, 'w') as f:
            f.write(f"# vLLM Inference Performance Report\n\n")
            f.write(f"**Run Folder**: `{os.path.basename(run_folder)}`  \n")
            f.write(f"**Generated**: {datetime.now().isoformat()}  \n\n")
            
            # Experiment configuration
            f.write(f"## Experiment Configuration\n\n")
            exp_info = experiment_results['experiment_info']
            f.write(f"- **Query Template**: {exp_info['query_key']}\n")
            f.write(f"- **Query Type**: {exp_info['query_type']}\n")
            f.write(f"- **Model**: {exp_info['model_name']}\n")
            f.write(f"- **GPU Configuration**: GPU {exp_info['gpu_id']}\n")
            f.write(f"- **Dataset**: {os.path.basename(exp_info['dataset_path'])}\n")
            f.write(f"- **Data Rows Processed**: {exp_info['processed_rows']} / {exp_info['total_rows']}\n")
            f.write(f"- **Batch Size**: {exp_info['batch_size']}\n")
            f.write(f"- **Inference Timestamp**: {exp_info['experiment_timestamp']}\n\n")
            
            # Performance summary
            f.write(f"## Inference Performance Summary\n\n")
            perf = experiment_results['performance_metrics']
            f.write(f"- **Total Inference Time**: {perf['total_inference_time']:.2f} seconds\n")
            f.write(f"- **Prompt Creation Time**: {perf['prompt_creation_time']:.2f} seconds\n")
            f.write(f"- **Average Time per Query**: {perf['avg_time_per_row']:.3f} seconds\n")
            f.write(f"- **Overall Throughput**: {perf['overall_throughput_tokens_per_sec']:.1f} tokens/second\n")
            f.write(f"- **Token Processing Summary**:\n")
            f.write(f"  - Input Tokens (estimated): {perf['estimated_input_tokens']:,}\n")
            f.write(f"  - Output Tokens (estimated): {perf['estimated_output_tokens']:,}\n")
            f.write(f"  - Total Tokens Processed: {perf['estimated_total_tokens']:,}\n")
            f.write(f"- **Batch Processing**: {perf['successful_batches']} successful, {perf['failed_batches']} failed\n\n")
            
            # vLLM Engine Metrics with detailed reporting
            f.write(f"## vLLM Engine Metrics\n\n")
            vllm_metrics = experiment_results['vllm_metrics']
            
            initial_stats = vllm_metrics['initial_stats']
            final_stats = vllm_metrics['final_stats']
            
            f.write(f"- **Prefix Caching**: Enabled (for KV cache optimization)\n")
            f.write(f"- **Metrics Collection**: {final_stats.get('vllm_stats_available', False)}\n")
            
            if final_stats.get('vllm_stats_available', False):
                key_metrics = final_stats.get('key_metrics', {})
                collection_methods = final_stats.get('collection_methods_tried', [])
                
                f.write(f"- **Collection Methods Used**: {', '.join(collection_methods)}\n")
                f.write(f"- **Total Metrics Retrieved**: {len(key_metrics)}\n\n")
                
                f.write(f"### Retrieved Metrics Summary\n\n")
                
                # Report what was actually found, categorized
                cache_metrics = {}
                performance_metrics = {}
                request_metrics = {}
                token_metrics = {}
                
                for metric, value in key_metrics.items():
                    if 'cache' in metric.lower():
                        cache_metrics[metric] = value
                    elif any(term in metric.lower() for term in ['time', 'latency', 'throughput']):
                        performance_metrics[metric] = value
                    elif 'request' in metric.lower():
                        request_metrics[metric] = value
                    elif 'token' in metric.lower():
                        token_metrics[metric] = value
                    else:
                        # Uncategorized metrics
                        performance_metrics[metric] = value
                
                # Cache Performance Analysis
                if cache_metrics:
                    f.write(f"#### Cache Performance Metrics\n\n")
                    
                    # GPU Cache Usage
                    if 'gpu_cache_usage_percent' in cache_metrics:
                        f.write(f"- **GPU KV Cache Usage**: {cache_metrics['gpu_cache_usage_percent']:.1f}%\n")
                    elif 'gpu_cache_usage_sys' in cache_metrics:
                        f.write(f"- **GPU KV Cache Usage**: {cache_metrics['gpu_cache_usage_sys']}\n")
                    
                    # Prefix Cache Hit Rate Analysis
                    hit_rate_found = False
                    if 'gpu_prefix_cache_hit_rate_percent' in cache_metrics:
                        hit_rate = cache_metrics['gpu_prefix_cache_hit_rate_percent']
                        hits = cache_metrics.get('gpu_prefix_cache_hits', 0)
                        queries = cache_metrics.get('gpu_prefix_cache_queries', 0)
                        
                        # Only report if we have valid data
                        if queries > 10 or hit_rate < 99:
                            hit_rate_found = True
                            
                            if 'gpu_prefix_cache_hits' in cache_metrics and 'gpu_prefix_cache_queries' in cache_metrics:
                                f.write(f"- **Prefix Cache Hit Rate**: {hit_rate:.2f}% ({hits:,} hits / {queries:,} queries)\n")
                            else:
                                f.write(f"- **Prefix Cache Hit Rate**: {hit_rate:.2f}%\n")
                            
                            # Analysis based on hit rate (generic, not GGR-specific)
                            if hit_rate > 70:
                                f.write(f"  - ✅ **EXCELLENT** - Very high cache reuse indicates efficient request patterns\n")
                                f.write(f"  - The input data or query ordering is enabling substantial prefix reuse\n")
                            elif hit_rate > 40:
                                f.write(f"  - ✅ **GOOD** - High cache reuse showing significant efficiency gains\n")
                                f.write(f"  - Clear evidence of beneficial prefix caching from the request patterns\n")
                            elif hit_rate > 15:
                                f.write(f"  - ⚠️ **MODERATE** - Some prefix reuse detected, moderate efficiency gain\n")
                                f.write(f"  - There may be opportunities to improve request ordering for better caching\n")
                            elif hit_rate > 5:
                                f.write(f"  - ⚠️ **LOW** - Minimal prefix reuse, limited caching benefits\n")
                                f.write(f"  - Input data may have low similarity or suboptimal ordering for caching\n")
                            else:
                                f.write(f"  - ❌ **VERY LOW** - Almost no prefix reuse detected\n")
                                f.write(f"  - Consider data characteristics or alternative optimization strategies\n")
                        else:
                            f.write(f"- **Prefix Cache Hit Rate**: Insufficient data ({queries} queries processed)\n")
                            f.write(f"  - Need more inference requests to generate meaningful cache statistics\n")
                    
                    # CPU cache (if available)
                    if 'cpu_prefix_cache_hit_rate_percent' in cache_metrics:
                        cpu_hit_rate = cache_metrics['cpu_prefix_cache_hit_rate_percent']
                        if 'cpu_prefix_cache_hits' in cache_metrics and 'cpu_prefix_cache_queries' in cache_metrics:
                            cpu_hits = cache_metrics['cpu_prefix_cache_hits']
                            cpu_queries = cache_metrics['cpu_prefix_cache_queries']
                            f.write(f"- **CPU Prefix Cache Hit Rate**: {cpu_hit_rate:.1f}% ({cpu_hits:,} hits / {cpu_queries:,} queries)\n")
                        else:
                            f.write(f"- **CPU Prefix Cache Hit Rate**: {cpu_hit_rate:.1f}%\n")
                        hit_rate_found = True
                    
                    if not hit_rate_found:
                        f.write(f"- **Prefix Cache Hit Rate**: Not available\n")
                        f.write(f"  - Possible reasons:\n")
                        f.write(f"    • Insufficient inference requests processed\n")
                        f.write(f"    • vLLM version compatibility (try VLLM_USE_V1=0)\n")
                        f.write(f"    • Metrics collection timing (cache stats accumulate over time)\n")
                    
                    f.write(f"\n")
                else:
                    f.write(f"#### Cache Metrics\n\n")
                    f.write(f"- **Cache Metrics**: None retrieved from vLLM engine\n")
                    f.write(f"- This may indicate metrics collection limitations or insufficient processing time\n\n")
                
                # Request Processing Metrics
                if request_metrics or any(k in key_metrics for k in ['requests_running', 'requests_waiting', 'num_requests_running', 'num_requests_waiting']):
                    f.write(f"#### Request Processing Status\n\n")
                    
                    for key in ['requests_running', 'num_requests_running']:
                        if key in key_metrics:
                            f.write(f"- **Running Requests**: {key_metrics[key]}\n")
                            break
                    
                    for key in ['requests_waiting', 'num_requests_waiting']:
                        if key in key_metrics:
                            f.write(f"- **Waiting Requests**: {key_metrics[key]}\n")
                            break
                    
                    f.write(f"\n")
                
                # Token Processing Metrics
                if token_metrics or any(k in key_metrics for k in ['prompt_tokens_total', 'generation_tokens_total', 'prompt_tokens', 'generation_tokens']):
                    f.write(f"#### Token Processing Statistics\n\n")
                    
                    for key in ['prompt_tokens_total', 'prompt_tokens']:
                        if key in key_metrics:
                            f.write(f"- **Total Prompt Tokens**: {key_metrics[key]:,}\n")
                            break
                    
                    for key in ['generation_tokens_total', 'generation_tokens']:
                        if key in key_metrics:
                            f.write(f"- **Total Generation Tokens**: {key_metrics[key]:,}\n")
                            break
                    
                    f.write(f"\n")
                
                # Performance Latency Metrics
                histogram_summary = key_metrics.get('histogram_summary', {})
                if histogram_summary:
                    f.write(f"#### Performance Latency Analysis\n\n")
                    if 'avg_time_to_first_token_seconds' in histogram_summary:
                        f.write(f"- **Average Time to First Token (TTFT)**: {histogram_summary['avg_time_to_first_token_seconds']:.3f}s\n")
                    if 'avg_time_per_output_token_seconds' in histogram_summary:
                        f.write(f"- **Average Time per Output Token (TPOT)**: {histogram_summary['avg_time_per_output_token_seconds']:.4f}s\n")
                    if 'avg_e2e_latency_seconds' in histogram_summary:
                        f.write(f"- **Average End-to-End Latency**: {histogram_summary['avg_e2e_latency_seconds']:.3f}s\n")
                    f.write(f"\n")
                
                # Raw metrics dump for debugging
                if key_metrics:
                    f.write(f"#### Complete Retrieved Metrics\n\n")
                    f.write(f"```\n")
                    for metric, value in sorted(key_metrics.items()):
                        if isinstance(value, dict):
                            f.write(f"{metric}: {len(value)} items\n")
                            for sub_key, sub_value in value.items():
                                f.write(f"  {sub_key}: {sub_value}\n")
                        else:
                            f.write(f"{metric}: {value}\n")
                    f.write(f"```\n\n")
                    
            else:
                f.write(f"- **vLLM Metrics**: Not available in this run\n")
                if 'error' in initial_stats:
                    f.write(f"  - Error: {initial_stats['error']}\n")
                f.write(f"  - This may be due to vLLM version compatibility or prometheus_client availability\n")
                f.write(f"  - Consider using VLLM_USE_V1=0 for legacy metrics access\n\n")
            
            # Resource utilization (unchanged)
            f.write(f"## System Resource Utilization\n\n")
            resource = experiment_results['resource_monitoring']
            
            if resource:
                f.write(f"### Monitoring Overview\n")
                f.write(f"- **Monitoring Duration**: {resource.get('monitoring_duration_seconds', 0):.1f} seconds\n")
                f.write(f"- **Samples Collected**: {resource.get('total_samples', 0)}\n")
                f.write(f"- **Sampling Interval**: {resource.get('sampling_interval_seconds', 0)}s\n\n")
                
                f.write(f"### CPU and Memory Usage\n")
                f.write(f"- **CPU Utilization**: {resource.get('cpu_utilization_mean', 0):.1f}% avg, {resource.get('cpu_utilization_max', 0):.1f}% peak\n")
                f.write(f"- **Memory Utilization**: {resource.get('memory_utilization_mean', 0):.1f}% avg, {resource.get('memory_utilization_max', 0):.1f}% peak\n")
                f.write(f"- **Memory Usage**: {resource.get('memory_used_gb_mean', 0):.1f} GB avg, {resource.get('memory_used_gb_max', 0):.1f} GB peak\n\n")
                
                f.write(f"### GPU Resource Usage\n")
                if 'gpu_compute_utilization_mean' in resource:
                    f.write(f"- **GPU Compute Utilization**: {resource['gpu_compute_utilization_mean']:.1f}% avg, {resource['gpu_compute_utilization_max']:.1f}% peak\n")
                    f.write(f"- **GPU Memory Utilization**: {resource.get('gpu_memory_utilization_mean', 0):.1f}% avg, {resource.get('gpu_memory_utilization_max', 0):.1f}% peak\n")
                    f.write(f"- **GPU Memory Allocated**: {resource.get('gpu_memory_allocated_gb_mean', 0):.1f} GB avg, {resource.get('gpu_memory_allocated_gb_max', 0):.1f} GB peak\n")
                    
                    if 'gpu_temperature_mean' in resource:
                        f.write(f"- **GPU Temperature**: {resource['gpu_temperature_mean']:.1f}°C avg, {resource['gpu_temperature_max']:.1f}°C peak\n")
                    
                    if 'gpu_power_mean' in resource:
                        f.write(f"- **GPU Power Draw**: {resource['gpu_power_mean']:.1f}W avg, {resource['gpu_power_max']:.1f}W peak\n")
                else:
                    f.write(f"- **GPU Monitoring**: Limited (basic PyTorch CUDA metrics only)\n")
                    f.write(f"- **GPU Memory Utilization**: {resource.get('gpu_memory_utilization_mean', 0):.1f}% avg\n")
            else:
                f.write(f"- **Resource Monitoring**: No data collected\n")
            
            f.write(f"\n")
            
            # Analysis notes (updated to be more general)
            f.write(f"## Analysis Notes\n\n")
            f.write(f"### Experimental Setup\n")
            f.write(f"- This report analyzes vLLM inference performance on the provided dataset\n")
            f.write(f"- Prefix caching was enabled to optimize KV cache reuse across similar requests\n")
            f.write(f"- Resource monitoring tracked system utilization during inference\n")
            f.write(f"- Token estimates are based on character count (≈4 chars per token)\n\n")
            
            f.write(f"### Data Processing Characteristics\n")
            f.write(f"- **Dataset**: {os.path.basename(exp_info['dataset_path'])}\n")
            f.write(f"- **Query Pattern**: {exp_info['query_type']} queries using template '{exp_info['query_key']}'\n")
            f.write(f"- **Processing Method**: Batch inference with size {exp_info['batch_size']}\n")
            
            # Cache performance interpretation (generic)
            if final_stats.get('vllm_stats_available', False):
                key_metrics = final_stats.get('key_metrics', {})
                if 'gpu_prefix_cache_hit_rate_percent' in key_metrics:
                    hit_rate = key_metrics['gpu_prefix_cache_hit_rate_percent']
                    f.write(f"- **Cache Efficiency**: {hit_rate:.1f}% prefix cache hit rate indicates ")
                    if hit_rate > 50:
                        f.write(f"high similarity between requests, enabling substantial computational reuse\n")
                    elif hit_rate > 20:
                        f.write(f"moderate similarity between requests with some computational reuse\n")
                    else:
                        f.write(f"low similarity between requests with limited computational reuse\n")
            
            f.write(f"\n### Performance Optimization Notes\n")
            f.write(f"- Higher prefix cache hit rates indicate more efficient processing due to request similarity\n")
            f.write(f"- Request ordering can significantly impact cache effectiveness\n")
            f.write(f"- Batch size affects both throughput and memory utilization\n")
            f.write(f"- GPU utilization patterns reflect the computational intensity of the workload\n")
            
            f.write(f"\n---\n")
            f.write(f"**Run ID**: `{os.path.basename(run_folder)}`  \n")
            f.write(f"**Generated**: {datetime.now().isoformat()}  \n")
            f.write(f"**vLLM Performance Analysis Report**\n")
        
        logger.info(f"Performance report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Simple LLM Query Experiment with Predefined Templates and Performance Monitoring")
    
    # Required arguments
    parser.add_argument("dataset_path", help="Path to the dataset file (CSV/JSON/JSONL/Parquet)")
    parser.add_argument("query_key", help="Query template key to use", 
                       choices=list(QUERY_TEMPLATES.keys()))
    
    # Model configuration - Updated to support local paths and multi-GPU
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf",
                       help="vLLM model name (HuggingFace) or local path to model directory")
    parser.add_argument("--gpu", type=int, default=0,
                       help="Single GPU device to use (0, 1, 2, 3, etc.) - for backward compatibility")
    parser.add_argument("--gpus", type=str, 
                       help="Multiple GPU devices to use (comma-separated, e.g., '4,5,6,7' or '6,7')")
    parser.add_argument("--gpu-memory", type=float, default=0.85,
                       help="GPU memory utilization fraction (0.0-1.0)")
    
    # Data configuration
    parser.add_argument("--max-rows", type=int, help="Maximum number of rows to process")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    
    # LLM configuration
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--max-model-len", type=int, 
                       help="Maximum model sequence length (reduces memory usage)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    
    # Output configuration
    parser.add_argument("--output-dir", default="llm_results", help="Output directory")
    
    # Monitoring configuration
    parser.add_argument("--monitor-interval", type=float, default=2.0, 
                       help="Resource monitoring interval in seconds")
    
    # Model validation
    parser.add_argument("--validate-model", action="store_true", 
                       help="Validate model path/name before running experiment")
    
    # Utility arguments
    parser.add_argument("--list-queries", action="store_true", help="List available query templates")
    parser.add_argument("--check-gpu", action="store_true", help="Check GPU availability and exit")
    
    args = parser.parse_args()
    
    if args.check_gpu:
        gpu_available, gpu_count = check_gpu_availability()
        if gpu_available:
            logger.info(f"GPU check completed. {gpu_count} GPU(s) available.")
        else:
            logger.info("No GPUs available.")
        return
    
    if args.list_queries:
        print("Available Query Templates:")
        print("=" * 80)
        for key, info in QUERY_TEMPLATES.items():
            print(f"🔹 {key}")
            print(f"   Type: {info['type']}")
            print(f"   Datasets: {', '.join(info['datasets'])}")
            print(f"   Prompt: {info['prompt'][:100]}...")
            print()
        return
    
    # Validate model if requested
    if args.validate_model:
        resolved_path = resolve_model_path(args.model)
        if os.path.exists(resolved_path):
            if validate_local_model(resolved_path):
                logger.info(f"✅ Model validation passed: {resolved_path}")
            else:
                logger.error(f"❌ Model validation failed: {resolved_path}")
                return
        else:
            logger.info(f"🌐 HuggingFace model (will be downloaded if needed): {args.model}")
        return
    
    # Parse GPU configuration
    gpu_ids = None
    gpu_id = None
    
    if args.gpus:
        # Parse comma-separated GPU list
        try:
            gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
            logger.info(f"Using multi-GPU setup: {gpu_ids}")
        except ValueError:
            logger.error(f"Invalid GPU list format: {args.gpus}. Use comma-separated integers like '4,5,6,7'")
            return
    else:
        # Use single GPU (backward compatibility)
        gpu_id = args.gpu
        logger.info(f"Using single GPU: {gpu_id}")
    
    # Initialize experiment with GPU configuration
    experiment = SimpleLLMExperiment(
        model_name=args.model,
        output_dir=args.output_dir,
        gpu_ids=gpu_ids,
        gpu_id=gpu_id
    )
    
    # Configure resource monitoring
    experiment.resource_monitor.sampling_interval = args.monitor_interval
    
    # Initialize model with custom parameters
    model_kwargs = {
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'gpu_memory_utilization': args.gpu_memory
    }
    
    # Add max_model_len if specified
    if args.max_model_len:
        model_kwargs['max_model_len'] = args.max_model_len
        logger.info(f"Setting max_model_len to {args.max_model_len}")
    
    if not experiment.initialize_model(**model_kwargs):
        logger.error("Failed to initialize model")
        return
    
    # Run experiment
    experiment.run_experiment(
        dataset_path=args.dataset_path,
        query_key=args.query_key,
        max_rows=args.max_rows,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()