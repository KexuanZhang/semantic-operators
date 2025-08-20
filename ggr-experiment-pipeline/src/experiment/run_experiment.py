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
except ImportError:
    logger.error("vLLM not available. Install with: pip install vllm")
    vllm_available = False
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
                'seed': 42,
                **kwargs
            }
            
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
        """Try to extract vLLM internal statistics"""
        vllm_stats = {}
        
        try:
            # Try to access internal stats (may not work in all vLLM versions)
            if hasattr(self.llm, 'llm_engine') and hasattr(self.llm.llm_engine, '_get_stats'):
                stats = self.llm.llm_engine._get_stats()
                
                # Extract relevant metrics
                vllm_stats.update({
                    'vllm_stats_available': True,
                    'gpu_cache_usage_sys': getattr(stats, 'gpu_cache_usage_sys', None),
                    'gpu_prefix_cache_hit_rate': getattr(stats, 'gpu_prefix_cache_hit_rate', None),
                    'prompt_tokens': getattr(stats, 'prompt_tokens', None),
                    'generation_tokens': getattr(stats, 'generation_tokens', None),
                    'num_requests_running': getattr(stats, 'num_requests_running', None),
                    'num_requests_waiting': getattr(stats, 'num_requests_waiting', None)
                })
                
                logger.info(f"vLLM Stats - KV Cache Usage: {vllm_stats.get('gpu_cache_usage_sys', 'N/A')}")
                logger.info(f"vLLM Stats - Prefix Cache Hit Rate: {vllm_stats.get('gpu_prefix_cache_hit_rate', 'N/A')}")
                
            else:
                vllm_stats['vllm_stats_available'] = False
                logger.warning("vLLM internal stats not accessible in this version")
                
        except Exception as e:
            vllm_stats['vllm_stats_available'] = False
            vllm_stats['vllm_stats_error'] = str(e)
            logger.warning(f"Could not access vLLM internal stats: {e}")
            
        return vllm_stats
    
    def format_row_data(self, row: pd.Series, query_key: str) -> Dict[str, Any]:
        """Format row data based on query type"""
        row_dict = row.to_dict()
        
        # Handle special cases for RAG queries
        if query_key == "rag_fever":
            required_fields = ['evidence1', 'evidence2', 'evidence3', 'evidence4', 'claim']
            return {field: str(row_dict.get(field, '')) for field in required_fields}
        
        elif query_key == "rag_squad":
            return {
                'question': str(row_dict.get('question', '')),
                'context': str(row_dict.get('context', ''))
            }
        
        # For other queries, return all fields formatted
        formatted_dict = {}
        for k, v in row_dict.items():
            if pd.isna(v):
                formatted_dict[k] = "N/A"
            else:
                formatted_dict[k] = str(v)
        return formatted_dict
    
    def create_prompt(self, row: pd.Series, query_key: str) -> str:
        """Create prompt for a specific query and row"""
        if query_key not in QUERY_TEMPLATES:
            raise ValueError(f"Unknown query key: {query_key}")
        
        query_info = QUERY_TEMPLATES[query_key]
        user_prompt = query_info["prompt"]
        
        # Format row data
        row_data = self.format_row_data(row, query_key)
        
        # Handle RAG queries differently
        if query_info["type"] == "rag":
            if query_key == "rag_fever":
                formatted_prompt = user_prompt.format(**row_data)
                system_context = SYSTEM_PROMPT.format(
                    QUERY=formatted_prompt,
                    fields=json.dumps(row_data, indent=2)
                )
            elif query_key == "rag_squad":
                formatted_prompt = f"{user_prompt}\n\nQuestion: {row_data['question']}\nContext: {row_data['context']}"
                system_context = SYSTEM_PROMPT.format(
                    QUERY=formatted_prompt,
                    fields=json.dumps(row_data, indent=2)
                )
        else:
            # Standard formatting for other query types
            system_context = SYSTEM_PROMPT.format(
                QUERY=user_prompt,
                fields=json.dumps(row_data, indent=2)
            )
        
        return system_context
    
    def run_experiment(self,
                      dataset_path: str,
                      query_key: str,
                      max_rows: Optional[int] = None,
                      batch_size: int = 8) -> Dict[str, Any]:
        """Run the LLM experiment with predefined query template and comprehensive monitoring"""
        
        if query_key not in QUERY_TEMPLATES:
            logger.error(f"Invalid query key: {query_key}")
            logger.info(f"Available queries: {list(QUERY_TEMPLATES.keys())}")
            return {}
        
        # Load dataset
        df = self.load_dataset(dataset_path, max_rows)
        if df.empty:
            logger.error("Failed to load dataset or dataset is empty")
            return {}
        
        # Initialize model if not done
        if self.llm is None:
            if not self.initialize_model():
                return {}
        
        query_info = QUERY_TEMPLATES[query_key]
        logger.info(f"Running experiment with query: {query_key}")
        logger.info(f"Query type: {query_info['type']}")
        logger.info(f"Query prompt: {query_info['prompt'][:100]}...")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Using GPU: {self.gpu_id}")
        logger.info(f"Batch size: {batch_size}")
        
        # Create prompts for all rows
        prompts = []
        prompt_creation_start = time.perf_counter()
        
        for idx, row in df.iterrows():
            try:
                prompt = self.create_prompt(row, query_key)
                prompts.append(prompt)
            except Exception as e:
                logger.warning(f"Failed to create prompt for row {idx}: {e}")
                prompts.append("")
        
        prompt_creation_time = time.perf_counter() - prompt_creation_start
        logger.info(f"Created {len(prompts)} prompts in {prompt_creation_time:.2f}s")
        
        # Calculate total input tokens (rough estimate)
        total_input_chars = sum(len(p) for p in prompts if p)
        estimated_input_tokens = total_input_chars // 4  # Rough estimate: 4 chars per token
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        # Get initial vLLM stats
        initial_vllm_stats = self.get_vllm_stats()
        
        # Run inference with detailed timing
        results = []
        batch_metrics = []
        inference_start_time = time.perf_counter()
        
        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = [p for p in prompts[batch_start:batch_end] if p]
            
            if not batch_prompts:
                continue
                
            batch_idx = batch_start // batch_size + 1
            total_batches = (len(prompts) - 1) // batch_size + 1
            logger.info(f"Processing batch {batch_idx}/{total_batches} (GPU {self.gpu_id})")
            
            try:
                batch_start_time = time.perf_counter()
                outputs = self.llm.generate(batch_prompts, self.sampling_params)
                batch_end_time = time.perf_counter()
                batch_duration = batch_end_time - batch_start_time
                
                # Count tokens in this batch
                batch_input_chars = sum(len(p) for p in batch_prompts)
                batch_output_chars = sum(len(output.outputs[0].text) for output in outputs if output.outputs)
                batch_input_tokens = batch_input_chars // 4
                batch_output_tokens = batch_output_chars // 4
                batch_total_tokens = batch_input_tokens + batch_output_tokens
                
                batch_throughput = batch_total_tokens / batch_duration if batch_duration > 0 else 0
                
                # Store batch metrics
                batch_metrics.append({
                    'batch_idx': batch_idx,
                    'batch_size': len(batch_prompts),
                    'batch_duration': batch_duration,
                    'batch_input_tokens': batch_input_tokens,
                    'batch_output_tokens': batch_output_tokens,
                    'batch_total_tokens': batch_total_tokens,
                    'batch_throughput_tokens_per_sec': batch_throughput
                })
                
                logger.info(f"Batch {batch_idx} completed: {batch_duration:.2f}s, "
                           f"{batch_throughput:.1f} tokens/sec")
                
                for i, output in enumerate(outputs):
                    row_idx = batch_start + i
                    generated_text = output.outputs[0].text.strip() if output.outputs else ""
                    
                    results.append({
                        'row_index': row_idx,
                        'query_key': query_key,
                        'query_type': query_info['type'],
                        'prompt': batch_prompts[i][:500] + "..." if len(batch_prompts[i]) > 500 else batch_prompts[i],
                        'response': generated_text,
                        'batch_idx': batch_idx,
                        'batch_duration': batch_duration,
                        'gpu_id': self.gpu_id,
                        'timestamp': datetime.now().isoformat()
                    })
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                # Still log the failed batch
                batch_metrics.append({
                    'batch_idx': batch_idx,
                    'batch_size': len(batch_prompts),
                    'batch_duration': 0,
                    'batch_input_tokens': 0,
                    'batch_output_tokens': 0,
                    'batch_total_tokens': 0,
                    'batch_throughput_tokens_per_sec': 0,
                    'error': str(e)
                })
        
        inference_end_time = time.perf_counter()
        total_inference_time = inference_end_time - inference_start_time
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        
        # Get final vLLM stats
        final_vllm_stats = self.get_vllm_stats()
        
        # Calculate summary metrics
        total_output_chars = sum(len(r['response']) for r in results)
        total_output_tokens = total_output_chars // 4
        total_tokens = estimated_input_tokens + total_output_tokens
        overall_throughput = total_tokens / total_inference_time if total_inference_time > 0 else 0
        
        # Get resource monitoring summary
        resource_summary = self.resource_monitor.get_summary_stats()
        
        # Compile comprehensive experiment results
        experiment_results = {
            'experiment_info': {
                'dataset_path': dataset_path,
                'query_key': query_key,
                'query_type': query_info['type'],
                'query_prompt': query_info['prompt'],
                'model_name': self.model_name,
                'gpu_id': self.gpu_id,
                'total_rows': len(df),
                'processed_rows': len(results),
                'batch_size': batch_size,
                'experiment_timestamp': datetime.now().isoformat()
            },
            'performance_metrics': {
                'prompt_creation_time': prompt_creation_time,
                'total_inference_time': total_inference_time,
                'total_experiment_time': prompt_creation_time + total_inference_time,
                'avg_time_per_row': total_inference_time / len(results) if results else 0,
                'estimated_input_tokens': estimated_input_tokens,
                'estimated_output_tokens': total_output_tokens,
                'estimated_total_tokens': total_tokens,
                'overall_throughput_tokens_per_sec': overall_throughput,
                'successful_batches': len([b for b in batch_metrics if 'error' not in b]),
                'failed_batches': len([b for b in batch_metrics if 'error' in b])
            },
            'vllm_metrics': {
                'initial_stats': initial_vllm_stats,
                'final_stats': final_vllm_stats,
                'prefix_caching_enabled': True
            },
            'resource_monitoring': resource_summary,
            'batch_metrics': batch_metrics,
            'results': results
        }
        
        # Save results and generate report
        self.save_results(experiment_results)
        self.generate_performance_report(experiment_results)
        
        # Log summary
        logger.info(f"Experiment completed!")
        logger.info(f"Processed {len(results)} rows in {total_inference_time:.2f} seconds")
        logger.info(f"Overall throughput: {overall_throughput:.1f} tokens/second")
        logger.info(f"Used GPU: {self.gpu_id}")
        
        return experiment_results
    
    def save_results(self, experiment_results: Dict[str, Any]):
        """Save experiment results to timestamped folder"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_key = experiment_results['experiment_info']['query_key']
        gpu_info = f"gpu{experiment_results['experiment_info']['gpu_id']}"
        
        # Create timestamped folder for this run
        run_folder = os.path.join(self.output_dir, f"{query_key}_{gpu_info}_{timestamp}")
        os.makedirs(run_folder, exist_ok=True)
        
        # Save detailed results as JSON
        json_file = os.path.join(run_folder, "experiment_results.json")
        with open(json_file, 'w') as f:
            json.dump(experiment_results, f, indent=2, default=str)
        logger.info(f"Detailed results saved to: {json_file}")
        
        # Save results as CSV
        results_df = pd.DataFrame(experiment_results['results'])
        csv_file = os.path.join(run_folder, "query_results.csv")
        results_df.to_csv(csv_file, index=False)
        logger.info(f"Results CSV saved to: {csv_file}")
        
        # Save batch metrics as CSV
        batch_df = pd.DataFrame(experiment_results['batch_metrics'])
        batch_csv = os.path.join(run_folder, "batch_metrics.csv")
        batch_df.to_csv(batch_csv, index=False)
        logger.info(f"Batch metrics saved to: {batch_csv}")
        
        # Save resource monitoring data if available
        if self.resource_monitor.metrics_log:
            resource_df = pd.DataFrame(self.resource_monitor.metrics_log)
            resource_csv = os.path.join(run_folder, "resource_metrics.csv")
            resource_df.to_csv(resource_csv, index=False)
            logger.info(f"Resource metrics saved to: {resource_csv}")
        
        # Save a summary file with key metrics
        summary_file = os.path.join(run_folder, "experiment_summary.txt")
        with open(summary_file, 'w') as f:
            exp_info = experiment_results['experiment_info']
            perf = experiment_results['performance_metrics']
            
            f.write(f"LLM Query Experiment Summary\n")
            f.write(f"=" * 50 + "\n\n")
            
            f.write(f"Experiment Details:\n")
            f.write(f"- Query Key: {exp_info['query_key']}\n")
            f.write(f"- Query Type: {exp_info['query_type']}\n")
            f.write(f"- Model: {exp_info['model_name']}\n")
            f.write(f"- GPU ID: {exp_info['gpu_id']}\n")
            f.write(f"- Dataset: {exp_info['dataset_path']}\n")
            f.write(f"- Total Rows: {exp_info['total_rows']}\n")
            f.write(f"- Processed Rows: {exp_info['processed_rows']}\n")
            f.write(f"- Batch Size: {exp_info['batch_size']}\n")
            f.write(f"- Timestamp: {exp_info['experiment_timestamp']}\n\n")
            
            f.write(f"Performance Metrics:\n")
            f.write(f"- Total Inference Time: {perf['total_inference_time']:.2f} seconds\n")
            f.write(f"- Average Time per Row: {perf['avg_time_per_row']:.3f} seconds\n")
            f.write(f"- Overall Throughput: {perf['overall_throughput_tokens_per_sec']:.1f} tokens/second\n")
            f.write(f"- Total Tokens Processed: {perf['estimated_total_tokens']:,}\n")
            f.write(f"- Successful Batches: {perf['successful_batches']}\n")
            f.write(f"- Failed Batches: {perf['failed_batches']}\n\n")
            
            # vLLM stats if available
            vllm_stats = experiment_results['vllm_metrics']['final_stats']
            if vllm_stats.get('vllm_stats_available', False):
                f.write(f"vLLM Statistics:\n")
                f.write(f"- KV Cache Usage: {vllm_stats.get('gpu_cache_usage_sys', 'N/A')}\n")
                if vllm_stats.get('gpu_prefix_cache_hit_rate') is not None:
                    hit_rate = vllm_stats['gpu_prefix_cache_hit_rate'] * 100
                    f.write(f"- Prefix Cache Hit Rate: {hit_rate:.1f}%\n")
                f.write(f"\n")
            
            f.write(f"Files in this run:\n")
            f.write(f"- experiment_results.json: Complete experiment data\n")
            f.write(f"- query_results.csv: Query responses and metadata\n")
            f.write(f"- batch_metrics.csv: Per-batch performance data\n")
            f.write(f"- resource_metrics.csv: System resource monitoring\n")
            f.write(f"- performance_report.md: Detailed analysis report\n")
            f.write(f"- experiment_summary.txt: This summary file\n")
        
        logger.info(f"Experiment summary saved to: {summary_file}")
        logger.info(f"All results saved to folder: {run_folder}")

    def generate_performance_report(self, experiment_results: Dict[str, Any]):
        """Generate comprehensive performance report in Markdown format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_key = experiment_results['experiment_info']['query_key']
        gpu_info = f"gpu{experiment_results['experiment_info']['gpu_id']}"
        
        # Use the same timestamped folder structure
        run_folder = os.path.join(self.output_dir, f"{query_key}_{gpu_info}_{timestamp}")
        os.makedirs(run_folder, exist_ok=True)  # Ensure folder exists
        
        report_file = os.path.join(run_folder, "performance_report.md")
        
        with open(report_file, 'w') as f:
            f.write(f"# LLM Query Experiment Performance Report\n\n")
            f.write(f"**Run Folder**: `{os.path.basename(run_folder)}`  \n")
            f.write(f"**Generated**: {datetime.now().isoformat()}  \n\n")
            
            # Experiment configuration
            f.write(f"## Experiment Configuration\n\n")
            exp_info = experiment_results['experiment_info']
            f.write(f"- **Query Key**: {exp_info['query_key']}\n")
            f.write(f"- **Query Type**: {exp_info['query_type']}\n")
            f.write(f"- **Model**: {exp_info['model_name']}\n")
            f.write(f"- **GPU ID**: {exp_info['gpu_id']}\n")
            f.write(f"- **Dataset**: {os.path.basename(exp_info['dataset_path'])}\n")
            f.write(f"- **Total Rows**: {exp_info['total_rows']}\n")
            f.write(f"- **Processed Rows**: {exp_info['processed_rows']}\n")
            f.write(f"- **Batch Size**: {exp_info['batch_size']}\n")
            f.write(f"- **Experiment Time**: {exp_info['experiment_timestamp']}\n\n")
            
            # Performance summary
            f.write(f"## Performance Summary\n\n")
            perf = experiment_results['performance_metrics']
            f.write(f"- **Total Inference Time**: {perf['total_inference_time']:.2f} seconds\n")
            f.write(f"- **Prompt Creation Time**: {perf['prompt_creation_time']:.2f} seconds\n")
            f.write(f"- **Average Time per Row**: {perf['avg_time_per_row']:.3f} seconds\n")
            f.write(f"- **Overall Throughput**: {perf['overall_throughput_tokens_per_sec']:.1f} tokens/second\n")
            f.write(f"- **Estimated Total Tokens**: {perf['estimated_total_tokens']:,}\n")
            f.write(f"  - Input Tokens: {perf['estimated_input_tokens']:,}\n")
            f.write(f"  - Output Tokens: {perf['estimated_output_tokens']:,}\n")
            f.write(f"- **Successful Batches**: {perf['successful_batches']}\n")
            f.write(f"- **Failed Batches**: {perf['failed_batches']}\n\n")
            
            # vLLM metrics
            f.write(f"## vLLM Engine Metrics\n\n")
            vllm_metrics = experiment_results['vllm_metrics']
            f.write(f"- **Prefix Caching Enabled**: {vllm_metrics['prefix_caching_enabled']}\n")
            
            initial_stats = vllm_metrics['initial_stats']
            final_stats = vllm_metrics['final_stats']
            
            if initial_stats.get('vllm_stats_available', False):
                f.write(f"- **Initial KV Cache Usage**: {initial_stats.get('gpu_cache_usage_sys', 'N/A')}\n")
                f.write(f"- **Final KV Cache Usage**: {final_stats.get('gpu_cache_usage_sys', 'N/A')}\n")
                f.write(f"- **Prefix Cache Hit Rate**: {final_stats.get('gpu_prefix_cache_hit_rate', 'N/A')}\n")
                
                if final_stats.get('gpu_prefix_cache_hit_rate') is not None:
                    hit_rate = final_stats['gpu_prefix_cache_hit_rate'] * 100
                    f.write(f"  - **Hit Rate Percentage**: {hit_rate:.1f}%\n")
            else:
                f.write(f"- **vLLM Internal Stats**: Not available in this version\n")
                if 'vllm_stats_error' in initial_stats:
                    f.write(f"  - Error: {initial_stats['vllm_stats_error']}\n")
            f.write(f"\n")
            
            # Resource utilization
            f.write(f"## Resource Utilization Summary\n\n")
            resource = experiment_results['resource_monitoring']
            
            if resource:
                f.write(f"### Monitoring Overview\n")
                f.write(f"- **Monitoring Duration**: {resource.get('monitoring_duration_seconds', 0):.1f} seconds\n")
                f.write(f"- **Total Samples**: {resource.get('total_samples', 0)}\n")
                f.write(f"- **Sampling Interval**: {resource.get('sampling_interval_seconds', 0)} seconds\n\n")
                
                f.write(f"### CPU and Memory\n")
                f.write(f"- **CPU Utilization**: {resource.get('cpu_utilization_mean', 0):.1f}% (avg), {resource.get('cpu_utilization_max', 0):.1f}% (max)\n")
                f.write(f"- **Memory Utilization**: {resource.get('memory_utilization_mean', 0):.1f}% (avg), {resource.get('memory_utilization_max', 0):.1f}% (max)\n")
                f.write(f"- **Memory Usage**: {resource.get('memory_used_gb_mean', 0):.1f} GB (avg), {resource.get('memory_used_gb_max', 0):.1f} GB (max)\n\n")
                
                f.write(f"### GPU Utilization\n")
                if 'gpu_compute_utilization_mean' in resource:
                    f.write(f"- **GPU Compute Utilization**: {resource['gpu_compute_utilization_mean']:.1f}% (avg), {resource['gpu_compute_utilization_max']:.1f}% (max)\n")
                    f.write(f"- **GPU Memory Utilization**: {resource.get('gpu_memory_utilization_mean', 0):.1f}% (avg), {resource.get('gpu_memory_utilization_max', 0):.1f}% (max)\n")
                    f.write(f"- **GPU Memory Allocated**: {resource.get('gpu_memory_allocated_gb_mean', 0):.1f} GB (avg), {resource.get('gpu_memory_allocated_gb_max', 0):.1f} GB (max)\n")
                    
                    if 'gpu_temperature_mean' in resource:
                        f.write(f"- **GPU Temperature**: {resource['gpu_temperature_mean']:.1f}°C (avg), {resource['gpu_temperature_max']:.1f}°C (max)\n")
                    
                    if 'gpu_power_mean' in resource:
                        f.write(f"- **GPU Power Consumption**: {resource['gpu_power_mean']:.1f}W (avg), {resource['gpu_power_max']:.1f}W (max)\n")
                else:
                    f.write(f"- **GPU Metrics**: Limited (basic PyTorch CUDA metrics only)\n")
                    f.write(f"- **GPU Memory Utilization**: {resource.get('gpu_memory_utilization_mean', 0):.1f}% (avg)\n")
            else:
                f.write(f"- **Resource Monitoring**: No data collected\n")
            
            f.write(f"\n")
            
            # Batch performance analysis
            f.write(f"## Batch Performance Analysis\n\n")
            batch_metrics = experiment_results['batch_metrics']
            
            if batch_metrics:
                batch_df = pd.DataFrame(batch_metrics)
                f.write(f"- **Total Batches**: {len(batch_metrics)}\n")
                f.write(f"- **Average Batch Duration**: {batch_df['batch_duration'].mean():.3f} seconds\n")
                f.write(f"- **Average Batch Throughput**: {batch_df['batch_throughput_tokens_per_sec'].mean():.1f} tokens/second\n")
                f.write(f"- **Max Batch Throughput**: {batch_df['batch_throughput_tokens_per_sec'].max():.1f} tokens/second\n")
                f.write(f"- **Average Batch Size**: {batch_df['batch_size'].mean():.1f}\n\n")
                
                # Top performing batches
                top_batches = batch_df.nlargest(3, 'batch_throughput_tokens_per_sec')[['batch_idx', 'batch_throughput_tokens_per_sec', 'batch_duration']]
                f.write(f"### Top 3 Performing Batches\n")
                for _, batch in top_batches.iterrows():
                    f.write(f"- Batch {int(batch['batch_idx'])}: {batch['batch_throughput_tokens_per_sec']:.1f} tokens/sec ({batch['batch_duration']:.3f}s)\n")
                f.write(f"\n")
            
            # File structure
            f.write(f"## Generated Files\n\n")
            f.write(f"This experiment run generated the following files in `{os.path.basename(run_folder)}/`:\n\n")
            f.write(f"- **`experiment_results.json`**: Complete experiment data and metadata\n")
            f.write(f"- **`query_results.csv`**: Query responses and processing details\n")
            f.write(f"- **`batch_metrics.csv`**: Per-batch performance metrics\n")
            f.write(f"- **`resource_metrics.csv`**: System resource monitoring data\n")
            f.write(f"- **`performance_report.md`**: This detailed analysis report\n")
            f.write(f"- **`experiment_summary.txt`**: Quick summary of key metrics\n\n")
            
            # System information
            f.write(f"## System Information\n\n")
            if torch.cuda.is_available():
                gpu_id = experiment_results['experiment_info']['gpu_id']
                gpu_name = torch.cuda.get_device_name(gpu_id)
                gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
                f.write(f"- **GPU**: {gpu_name}\n")
                f.write(f"- **GPU Memory**: {gpu_memory:.1f} GB\n")
                f.write(f"- **CUDA Version**: {torch.version.cuda}\n")
                
            f.write(f"- **PyTorch Version**: {torch.__version__}\n")
            f.write(f"- **Python Version**: {sys.version.split()[0]}\n")
            f.write(f"- **CPU Count**: {psutil.cpu_count()}\n")
            f.write(f"- **Total System Memory**: {psutil.virtual_memory().total / 1e9:.1f} GB\n\n")
            
            # Analysis notes
            f.write(f"## Analysis Notes\n\n")
            f.write(f"- This experiment used vLLM with prefix caching enabled to optimize KV cache reuse\n")
            f.write(f"- Resource monitoring was performed every {resource.get('sampling_interval_seconds', 2)} seconds during inference\n")
            f.write(f"- Token counts are estimated based on character length (4 chars ≈ 1 token)\n")
            f.write(f"- Throughput includes both input and output tokens\n")
            
            if final_stats.get('gpu_prefix_cache_hit_rate') is not None:
                hit_rate = final_stats['gpu_prefix_cache_hit_rate'] * 100
                if hit_rate > 50:
                    f.write(f"- **High prefix cache hit rate ({hit_rate:.1f}%) indicates effective reuse of computations**\n")
                elif hit_rate > 20:
                    f.write(f"- Moderate prefix cache hit rate ({hit_rate:.1f}%) suggests some reuse benefits\n")
                else:
                    f.write(f"- Low prefix cache hit rate ({hit_rate:.1f}%) indicates limited reuse opportunities\n")
            
            f.write(f"\n---\n")
            f.write(f"**Run ID**: `{os.path.basename(run_folder)}`  \n")
            f.write(f"**Report generated**: {datetime.now().isoformat()}  \n")
        
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