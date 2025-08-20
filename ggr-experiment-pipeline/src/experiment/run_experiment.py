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
    """Enhanced vLLM metrics collector with better error handling and fallback methods"""
    
    def __init__(self, llm_instance=None):
        self.llm = llm_instance
        self.logger = None
        self.stats_history = []
        self.collection_interval = 1.0
        self.monitoring = False
        self.monitor_thread = None
        self.modern_logging_available = False
        self.last_stats = {}
        
        # Enhanced metrics cache
        self.metrics_cache = {
            'gpu_cache_usage_perc': [],
            'gpu_cache_usage_sys': [],
            'cpu_cache_usage_sys': [],
            'gpu_prefix_cache_hit_rate': [],
            'cpu_prefix_cache_hit_rate': [],
            'num_requests_running': [],
            'num_requests_waiting': [],
            'num_requests_swapped': [],
            'prompt_tokens_total': 0,
            'generation_tokens_total': 0,
            'iteration_tokens_total': [],
            'time_to_first_token_seconds': [],
            'time_per_output_token_seconds': [],
            'e2e_request_latency_seconds': [],
            'request_queue_time_seconds': [],
            'request_inference_time_seconds': [],
            'request_prompt_tokens': [],
            'request_generation_tokens': [],
            'request_success_total': 0,
            'num_preemptions_total': 0,
            'num_batched_tokens': []
        }
        
    def initialize_logging(self):
        """Initialize vLLM logging with multiple fallback approaches"""
        if not self.llm:
            logger.warning("No LLM instance available for metrics initialization")
            return False
            
        try:
            # Try to enable stats logging via disable_log_stats parameter
            if hasattr(self.llm, 'llm_engine'):
                engine = self.llm.llm_engine
                logger.info("Found llm_engine attribute")
                
                # Try to access engine config and enable stats
                if hasattr(engine, 'log_stats') and not engine.log_stats:
                    try:
                        engine.log_stats = True
                        logger.info("Enabled log_stats on engine")
                    except Exception as e:
                        logger.debug(f"Could not enable log_stats: {e}")
                
                # Try to manually add LoggingStatLogger if modern metrics available
                if modern_vllm_metrics:
                    try:
                        from vllm.engine.metrics import LoggingStatLogger
                        if not hasattr(engine, 'stat_loggers') or not engine.stat_loggers:
                            engine.stat_loggers = {}
                        
                        if 'logging' not in engine.stat_loggers:
                            # Create logger with basic configuration
                            stat_logger = LoggingStatLogger(labels={})
                            engine.stat_loggers['logging'] = stat_logger
                            logger.info("Added LoggingStatLogger to engine")
                            self.logger = stat_logger
                            self.modern_logging_available = True
                        
                    except Exception as e:
                        logger.debug(f"Could not add LoggingStatLogger: {e}")
                
                self.modern_logging_available = True
                logger.info("vLLM metrics initialization successful")
                return True
                
            else:
                logger.warning("Could not access llm_engine from vLLM instance")
                return False
                
        except Exception as e:
            logger.warning(f"Failed to initialize vLLM logging: {e}")
            return False
    
    def start_monitoring(self):
        """Start background monitoring with better error handling"""
        if self.monitoring:
            logger.info("vLLM metrics monitoring already active")
            return
            
        # Force initialization if not done
        if not self.modern_logging_available:
            self.initialize_logging()
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started vLLM metrics monitoring thread")
        
    def stop_monitoring(self):
        """Stop background monitoring"""
        if not self.monitoring:
            return
            
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=3.0)
        logger.info("Stopped vLLM metrics monitoring")
        
    def _monitoring_loop(self):
        """Enhanced monitoring loop with better error handling"""
        consecutive_failures = 0
        max_failures = 10
        
        while self.monitoring:
            try:
                stats = self.collect_current_stats()
                
                if stats and stats.get('stats_available', False):
                    stats['timestamp'] = time.time()
                    self.stats_history.append(stats)
                    self.last_stats = stats.copy()
                    
                    # Update metrics cache
                    self._update_metrics_cache(stats)
                    
                    # Reset failure counter on success
                    consecutive_failures = 0
                    
                    # Log key metrics periodically
                    if len(self.stats_history) % 10 == 0:
                        self._log_key_metrics(stats)
                        
                else:
                    consecutive_failures += 1
                    if consecutive_failures == 1:
                        logger.debug("vLLM stats collection failed, will retry")
                    elif consecutive_failures >= max_failures:
                        logger.warning(f"vLLM stats collection failed {max_failures} times, continuing with limited metrics")
                        consecutive_failures = 0  # Reset to avoid spam
                
                # Keep history manageable
                if len(self.stats_history) > 1000:
                    self.stats_history = self.stats_history[-500:]
                    
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures == 1:
                    logger.debug(f"Error in vLLM monitoring loop: {e}")
                
            time.sleep(self.collection_interval)
    
    def _log_key_metrics(self, stats):
        """Log key metrics for debugging"""
        try:
            if stats.get('gpu_cache_usage_sys') is not None:
                cache_pct = stats['gpu_cache_usage_sys'] * 100
                logger.debug(f"KV Cache Usage: {cache_pct:.1f}%")
            
            if stats.get('gpu_prefix_cache_hit_rate') is not None:
                hit_rate = stats['gpu_prefix_cache_hit_rate'] * 100
                logger.debug(f"Prefix Cache Hit Rate: {hit_rate:.1f}%")
                
            if stats.get('num_requests_running') is not None:
                logger.debug(f"Running requests: {stats['num_requests_running']}")
                
        except Exception as e:
            logger.debug(f"Error logging key metrics: {e}")
    
    def collect_current_stats(self) -> Dict[str, Any]:
        """Enhanced stats collection with multiple fallback methods"""
        stats = {
            'stats_available': False,
            'modern_metrics': self.modern_logging_available,
            'collection_time': time.time(),
            'collection_method': 'unknown'
        }
        
        if not self.llm:
            stats['error'] = 'No LLM instance available'
            return stats
            
        try:
            # Method 1: Try modern _get_stats method
            if hasattr(self.llm, 'llm_engine'):
                engine = self.llm.llm_engine
                
                if hasattr(engine, '_get_stats'):
                    try:
                        engine_stats = engine._get_stats()
                        if engine_stats:
                            stats.update({
                                'stats_available': True,
                                'collection_method': 'llm_engine._get_stats',
                                **engine_stats
                            })
                            
                            # Convert to prometheus format
                            prometheus_stats = self._map_to_prometheus_metrics(engine_stats)
                            stats.update(prometheus_stats)
                            
                            logger.debug(f"Collected stats via _get_stats: {len(engine_stats)} fields")
                            return stats
                    except Exception as e:
                        logger.debug(f"_get_stats method failed: {e}")
                
                # Method 2: Try to access scheduler stats
                if hasattr(engine, 'scheduler'):
                    try:
                        scheduler_stats = self._get_scheduler_stats(engine)
                        if scheduler_stats:
                            stats.update({
                                'stats_available': True,
                                'collection_method': 'scheduler_stats',
                                **scheduler_stats
                            })
                            logger.debug(f"Collected scheduler stats: {len(scheduler_stats)} fields")
                            return stats
                    except Exception as e:
                        logger.debug(f"Scheduler stats failed: {e}")
                
                # Method 3: Try to access cache manager stats  
                if hasattr(engine, 'cache_config') or hasattr(engine, 'cache'):
                    try:
                        cache_stats = self._get_cache_stats(engine)
                        if cache_stats:
                            stats.update({
                                'stats_available': True,
                                'collection_method': 'cache_stats',
                                **cache_stats
                            })
                            logger.debug(f"Collected cache stats: {len(cache_stats)} fields")
                            return stats
                    except Exception as e:
                        logger.debug(f"Cache stats failed: {e}")
            
            # Method 4: Try alternative engine access paths
            if hasattr(self.llm, '_engine'):
                try:
                    alt_engine = self.llm._engine
                    if hasattr(alt_engine, '_get_stats'):
                        engine_stats = alt_engine._get_stats()
                        if engine_stats:
                            stats.update({
                                'stats_available': True,
                                'collection_method': '_engine._get_stats',
                                **engine_stats
                            })
                            prometheus_stats = self._map_to_prometheus_metrics(engine_stats)
                            stats.update(prometheus_stats)
                            return stats
                except Exception as e:
                    logger.debug(f"Alternative engine access failed: {e}")
            
            # Method 5: Fallback - basic GPU memory stats
            if torch.cuda.is_available():
                try:
                    gpu_stats = self._get_basic_gpu_stats()
                    stats.update({
                        'stats_available': True,
                        'collection_method': 'basic_gpu_fallback',
                        **gpu_stats
                    })
                    logger.debug("Using basic GPU stats fallback")
                    return stats
                except Exception as e:
                    logger.debug(f"Basic GPU stats failed: {e}")
                    
            stats['error'] = 'All collection methods failed'
            
        except Exception as e:
            stats['error'] = f'Exception in stats collection: {str(e)}'
            logger.debug(f"Stats collection exception: {e}")
        
        return stats
    
    def _get_scheduler_stats(self, engine) -> Dict[str, Any]:
        """Try to get stats from scheduler"""
        stats = {}
        
        try:
            scheduler = engine.scheduler
            
            # Try to get request counts
            if hasattr(scheduler, 'running'):
                stats['num_requests_running'] = len(scheduler.running)
            if hasattr(scheduler, 'waiting'):
                stats['num_requests_waiting'] = len(scheduler.waiting)
            if hasattr(scheduler, 'swapped'):
                stats['num_requests_swapped'] = len(scheduler.swapped)
                
            logger.debug(f"Scheduler stats: {stats}")
            
        except Exception as e:
            logger.debug(f"Error getting scheduler stats: {e}")
            
        return stats
    
    def _get_cache_stats(self, engine) -> Dict[str, Any]:
        """Try to get cache-related stats"""
        stats = {}
        
        try:
            # Look for cache manager
            if hasattr(engine, 'cache_config'):
                cache_config = engine.cache_config
                
                # Try to get cache usage information
                if hasattr(cache_config, 'cache_size'):
                    stats['cache_size'] = cache_config.cache_size
                    
                if hasattr(cache_config, 'block_size'):
                    stats['cache_block_size'] = cache_config.block_size
            
            # Look for GPU cache manager
            if hasattr(engine, 'gpu_cache'):
                gpu_cache = engine.gpu_cache
                if hasattr(gpu_cache, 'get_num_free_blocks'):
                    try:
                        free_blocks = gpu_cache.get_num_free_blocks()
                        if hasattr(gpu_cache, 'get_num_total_blocks'):
                            total_blocks = gpu_cache.get_num_total_blocks()
                            used_blocks = total_blocks - free_blocks
                            usage_ratio = used_blocks / total_blocks if total_blocks > 0 else 0
                            
                            stats.update({
                                'gpu_cache_free_blocks': free_blocks,
                                'gpu_cache_total_blocks': total_blocks,
                                'gpu_cache_used_blocks': used_blocks,
                                'gpu_cache_usage_sys': usage_ratio
                            })
                    except Exception as e:
                        logger.debug(f"Error getting cache block info: {e}")
            
            logger.debug(f"Cache stats: {stats}")
            
        except Exception as e:
            logger.debug(f"Error getting cache stats: {e}")
            
        return stats
    
    def _get_basic_gpu_stats(self) -> Dict[str, Any]:
        """Fallback method using basic GPU memory stats"""
        stats = {}
        
        try:
            if torch.cuda.is_available():
                gpu_id = 0  # Use first GPU
                allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
                reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
                total = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
                
                # Estimate cache usage (very rough)
                cache_usage_estimate = reserved / total if total > 0 else 0
                
                stats.update({
                    'gpu_memory_allocated_gb': allocated,
                    'gpu_memory_reserved_gb': reserved,
                    'gpu_memory_total_gb': total,
                    'gpu_cache_usage_sys': cache_usage_estimate,  # Rough estimate
                    'num_requests_running': 1,  # Assume something is running if we're collecting stats
                    'collection_note': 'Estimated from GPU memory usage'
                })
                
        except Exception as e:
            logger.debug(f"Error getting basic GPU stats: {e}")
            
        return stats
    
    def _map_to_prometheus_metrics(self, engine_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced mapping to Prometheus-style metrics"""
        prometheus_metrics = {}
        
        try:
            # GPU Cache usage mapping
            for cache_key in ['gpu_cache_usage', 'gpu_cache_usage_sys', 'gpu_kv_cache_usage']:
                if cache_key in engine_stats:
                    cache_usage = engine_stats[cache_key]
                    if isinstance(cache_usage, (int, float)):
                        # Ensure it's in 0-1 range
                        if cache_usage > 1:
                            prometheus_metrics['gpu_cache_usage_perc'] = cache_usage / 100.0
                        else:
                            prometheus_metrics['gpu_cache_usage_perc'] = cache_usage
                        break
            
            # CPU Cache usage
            if 'cpu_cache_usage_sys' in engine_stats:
                prometheus_metrics['cpu_cache_usage_perc'] = engine_stats['cpu_cache_usage_sys']
            
            # Prefix cache hit rates
            for hit_rate_key in ['gpu_prefix_cache_hit_rate', 'prefix_cache_hit_rate']:
                if hit_rate_key in engine_stats:
                    prometheus_metrics['gpu_prefix_cache_hit_rate'] = engine_stats[hit_rate_key]
                    break
                    
            if 'cpu_prefix_cache_hit_rate' in engine_stats:
                prometheus_metrics['cpu_prefix_cache_hit_rate'] = engine_stats['cpu_prefix_cache_hit_rate']
            
            # Request queue metrics with multiple possible keys
            request_mappings = [
                (['num_requests_running', 'running_requests'], 'num_requests_running'),
                (['num_requests_waiting', 'waiting_requests', 'pending_requests'], 'num_requests_waiting'),
                (['num_requests_swapped', 'swapped_requests'], 'num_requests_swapped')
            ]
            
            for source_keys, target_key in request_mappings.items():
                for source_key in source_keys:
                    if source_key in engine_stats:
                        prometheus_metrics[target_key] = engine_stats[source_key]
                        break
            
            # Token metrics
            token_mappings = {
                'num_batched_tokens': 'iteration_tokens_total',
                'prompt_tokens': 'prompt_tokens_total',
                'generation_tokens': 'generation_tokens_total'
            }
            
            for source_key, target_key in token_mappings.items():
                if source_key in engine_stats:
                    prometheus_metrics[target_key] = engine_stats[source_key]
            
            # Preemption count
            if 'num_preemptions' in engine_stats:
                prometheus_metrics['num_preemptions_total'] = engine_stats['num_preemptions']
            
        except Exception as e:
            logger.debug(f"Error mapping to Prometheus metrics: {e}")
        
        return prometheus_metrics
    
    def _update_metrics_cache(self, stats: Dict[str, Any]):
        """Enhanced metrics cache update"""
        try:
            # Cache usage metrics
            for cache_key in ['gpu_cache_usage_perc', 'gpu_cache_usage_sys']:
                if cache_key in stats:
                    value = stats[cache_key]
                    if isinstance(value, (int, float)) and 0 <= value <= 1:
                        self.metrics_cache[cache_key].append(value)
            
            # CPU cache usage
            if 'cpu_cache_usage_sys' in stats:
                value = stats['cpu_cache_usage_sys']
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    self.metrics_cache['cpu_cache_usage_sys'].append(value)
            
            # Prefix cache hit rates
            for hit_rate_key in ['gpu_prefix_cache_hit_rate', 'cpu_prefix_cache_hit_rate']:
                if hit_rate_key in stats:
                    value = stats[hit_rate_key]
                    if isinstance(value, (int, float)) and 0 <= value <= 1:
                        self.metrics_cache[hit_rate_key].append(value)
            
            # Request queue metrics
            for metric in ['num_requests_running', 'num_requests_waiting', 'num_requests_swapped']:
                if metric in stats:
                    value = stats[metric]
                    if isinstance(value, (int, float)) and value >= 0:
                        self.metrics_cache[metric].append(value)
            
            # Token counters
            if 'prompt_tokens_total' in stats:
                self.metrics_cache['prompt_tokens_total'] += stats.get('prompt_tokens_total', 0)
            if 'generation_tokens_total' in stats:
                self.metrics_cache['generation_tokens_total'] += stats.get('generation_tokens_total', 0)
                
            # Batched tokens (iteration)
            if 'iteration_tokens_total' in stats or 'num_batched_tokens' in stats:
                tokens = stats.get('iteration_tokens_total', stats.get('num_batched_tokens', 0))
                if isinstance(tokens, (int, float)) and tokens > 0:
                    self.metrics_cache['num_batched_tokens'].append(tokens)
            
            # Preemption counter
            if 'num_preemptions_total' in stats:
                self.metrics_cache['num_preemptions_total'] += stats.get('num_preemptions_total', 0)
                
        except Exception as e:
            logger.debug(f"Error updating metrics cache: {e}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Enhanced comprehensive statistics"""
        current_stats = self.collect_current_stats()
        
        summary = {
            'latest_stats': current_stats,
            'monitoring_active': self.monitoring,
            'total_collections': len(self.stats_history),
            'collection_interval': self.collection_interval,
            'modern_metrics_enabled': self.modern_logging_available,
            'last_successful_collection': self.last_stats.get('collection_time', 0),
            'collection_method': current_stats.get('collection_method', 'unknown')
        }
        
        # Enhanced Prometheus-style metrics summary
        summary['prometheus_metrics'] = {}
        
        # GPU Cache metrics
        if self.metrics_cache['gpu_cache_usage_perc']:
            values = self.metrics_cache['gpu_cache_usage_perc']
            summary['prometheus_metrics']['gpu_cache_usage_perc'] = {
                'values_count': len(values),
                'current': values[-1] if values else 0,
                'mean': sum(values) / len(values),
                'max': max(values),
                'min': min(values)
            }
        
        # Prefix cache hit rate
        if self.metrics_cache['gpu_prefix_cache_hit_rate']:
            values = self.metrics_cache['gpu_prefix_cache_hit_rate']
            summary['prometheus_metrics']['gpu_prefix_cache_hit_rate'] = {
                'values_count': len(values),
                'current': values[-1] if values else 0,
                'mean': sum(values) / len(values),
                'max': max(values),
                'min': min(values)
            }
        
        # Request queue metrics
        for metric in ['num_requests_running', 'num_requests_waiting', 'num_requests_swapped']:
            if self.metrics_cache[metric]:
                values = self.metrics_cache[metric]
                summary['prometheus_metrics'][metric] = {
                    'values_count': len(values),
                    'current': values[-1] if values else 0,
                    'mean': sum(values) / len(values),
                    'max': max(values)
                }
        
        # Token processing stats
        summary['prometheus_metrics']['token_counters'] = {
            'prompt_tokens_total': self.metrics_cache['prompt_tokens_total'],
            'generation_tokens_total': self.metrics_cache['generation_tokens_total'],
            'batched_tokens_samples': len(self.metrics_cache['num_batched_tokens'])
        }
        
        # Add historical analysis if we have sufficient data
        if len(self.stats_history) > 5:
            try:
                history_df = pd.DataFrame(self.stats_history)
                
                # KV Cache utilization analysis
                cache_columns = ['gpu_cache_usage_sys', 'gpu_cache_usage_perc']
                for col in cache_columns:
                    if col in history_df.columns:
                        cache_data = history_df[col].dropna()
                        if len(cache_data) > 0:
                            summary.update({
                                f'historical_{col}_mean': cache_data.mean(),
                                f'historical_{col}_max': cache_data.max(),
                                f'historical_{col}_min': cache_data.min(),
                                f'historical_{col}_std': cache_data.std(),
                                f'historical_{col}_samples': len(cache_data)
                            })
                
                # Prefix cache hit rate analysis
                if 'gpu_prefix_cache_hit_rate' in history_df.columns:
                    hit_rate_data = history_df['gpu_prefix_cache_hit_rate'].dropna()
                    if len(hit_rate_data) > 0:
                        summary.update({
                            'historical_prefix_hit_rate_mean': hit_rate_data.mean(),
                            'historical_prefix_hit_rate_max': hit_rate_data.max(),
                            'historical_prefix_hit_rate_min': hit_rate_data.min(),
                            'historical_prefix_hit_rate_samples': len(hit_rate_data)
                        })
                
            except Exception as e:
                logger.debug(f"Error computing historical analysis: {e}")
        
        return summary
    
    def get_prometheus_metrics(self) -> Dict[str, Any]:
        """Get current Prometheus-style metrics snapshot with enhanced data"""
        metrics = {
            'vllm:monitoring_active': self.monitoring,
            'vllm:total_collections': len(self.stats_history),
            'vllm:modern_metrics_enabled': self.modern_logging_available
        }
        
        # Add current values
        if self.metrics_cache['gpu_cache_usage_perc']:
            metrics['vllm:gpu_cache_usage_perc'] = self.metrics_cache['gpu_cache_usage_perc'][-1]
        
        if self.metrics_cache['gpu_prefix_cache_hit_rate']:
            metrics['vllm:gpu_prefix_cache_hit_rate'] = self.metrics_cache['gpu_prefix_cache_hit_rate'][-1]
        
        for metric in ['num_requests_running', 'num_requests_waiting', 'num_requests_swapped']:
            if self.metrics_cache[metric]:
                metrics[f'vllm:{metric}'] = self.metrics_cache[metric][-1]
        
        metrics.update({
            'vllm:prompt_tokens_total': self.metrics_cache['prompt_tokens_total'],
            'vllm:generation_tokens_total': self.metrics_cache['generation_tokens_total'],
            'vllm:num_preemptions_total': self.metrics_cache['num_preemptions_total']
        })
        
        # Add latest stats if available
        if self.last_stats:
            metrics['vllm:last_collection_method'] = self.last_stats.get('collection_method', 'unknown')
            metrics['vllm:last_collection_time'] = self.last_stats.get('collection_time', 0)
        
        return metrics