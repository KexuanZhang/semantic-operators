#!/usr/bin/env python3
"""
vLLM Experiment Runner with KV Cache Monitoring

This module implements a comprehensive experiment runner that:
1. Accepts user inputs: dataset path, model path, query type, and GPU devices
2. Serves a vLLM model on specified GPUs
3. Queries each row of data with the specified query type
4. Logs KV cache usage and performance metrics

Features:
- Multi-GPU support with configurable GPU selection
- Comprehensive KV cache monitoring via Prometheus metrics
- Support for multiple query types with predefined templates
- Real-time metrics logging and collection
- Automatic server lifecycle management
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import threading
import signal
import atexit
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import pandas as pd
import requests
from prometheus_client.parser import text_string_to_metric_families

# System prompt template
SYSTEM_PROMPT = """You are a data analyst. Use the provided JSON data to answer the user query based on the specified fields. Respond with only the answer, no extra formatting. Answer the below query: {QUERY}

Given the following data: {fields}"""

# Query templates organized by type
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


class VLLMExperimentRunner:
    """Main experiment runner for vLLM with comprehensive KV cache monitoring"""
    
    def __init__(self, dataset_path: str, model_path: str, query_type: str, 
                 gpus: str, host: str = "127.0.0.1", port: int = 8000):
        """
        Initialize the experiment runner
        
        Args:
            dataset_path: Path to the dataset file
            model_path: Path to the LLM model
            query_type: Type of query to run (from QUERY_TEMPLATES)
            gpus: GPU devices to use (e.g., "6,7")
            host: Server host address
            port: Server port
        """
        self.dataset_path = Path(dataset_path)
        self.model_path = model_path
        self.query_type = query_type
        self.gpus = gpus
        self.host = host
        self.port = port
        
        # Validate inputs
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        if query_type not in QUERY_TEMPLATES:
            raise ValueError(f"Unknown query type: {query_type}. Available: {list(QUERY_TEMPLATES.keys())}")
        
        # Set up GPU environment
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        
        # Initialize state
        self.server_process = None
        self.server_url = f"http://{host}:{port}"
        self.metrics_url = f"{self.server_url}/metrics"
        self.metrics_data = []
        self.inference_results = []
        self.stop_monitoring = False
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = Path(f"results/experiment_{timestamp}")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.result_dir / "experiment.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Log experiment configuration
        self.logger.info("="*50)
        self.logger.info("vLLM Experiment Configuration")
        self.logger.info("="*50)
        self.logger.info(f"Dataset: {self.dataset_path}")
        self.logger.info(f"Model: {self.model_path}")
        self.logger.info(f"Query Type: {self.query_type}")
        self.logger.info(f"GPUs: {self.gpus}")
        self.logger.info(f"Server URL: {self.server_url}")
        self.logger.info(f"Results Directory: {self.result_dir}")
        self.logger.info("="*50)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_monitoring = True
        self.stop_server()
    
    def verify_gpu_setup(self) -> bool:
        """Verify GPU setup"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                self.logger.warning("CUDA is not available")
                return False
            
            gpu_list = [int(g.strip()) for g in self.gpus.split(',')]
            num_available = torch.cuda.device_count()
            
            for gpu_id in gpu_list:
                if gpu_id >= num_available:
                    self.logger.error(f"GPU {gpu_id} not available. Found {num_available} GPUs.")
                    return False
                
                gpu_name = torch.cuda.get_device_name(gpu_id)
                gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                self.logger.info(f"GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"GPU verification failed: {e}")
            return False
    
    def start_server(self) -> bool:
        """Start vLLM server"""
        self.logger.info("Starting vLLM server...")
        
        if not self.verify_gpu_setup():
            return False
        
        gpu_count = len(self.gpus.split(','))
        
        # Build server command
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--tensor-parallel-size", str(gpu_count),
            "--gpu-memory-utilization", "0.85",
            "--enable-prefix-caching",
            "--disable-log-stats",  # We'll collect metrics manually
            "--max-model-len", "8192",
            "--dtype", "auto"
        ]
        
        self.logger.info(f"Server command: {' '.join(cmd)}")
        
        try:
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = self.gpus
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            # Wait for server to start
            self.logger.info("Waiting for server to start...")
            start_time = time.time()
            
            while time.time() - start_time < 300:  # 5 minute timeout
                try:
                    response = requests.get(f"{self.server_url}/health", timeout=5)
                    if response.status_code == 200:
                        self.logger.info("Server started successfully!")
                        
                        # Log initial metrics
                        initial_metrics = self.get_metrics()
                        gpu_cache_usage = initial_metrics.get('vllm:gpu_cache_usage_perc', 0)
                        self.logger.info(f"Initial GPU cache usage: {gpu_cache_usage:.1f}%")
                        
                        return True
                except requests.exceptions.RequestException:
                    pass
                
                # Check if process is still running
                if self.server_process.poll() is not None:
                    stdout, stderr = self.server_process.communicate()
                    self.logger.error("Server process terminated unexpectedly")
                    self.logger.error(f"STDERR: {stderr[-1000:]}")
                    return False
                
                time.sleep(5)
            
            self.logger.error("Server startup timeout")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            return False
    
    def stop_server(self):
        """Stop vLLM server"""
        if self.server_process:
            self.logger.info("Stopping vLLM server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.logger.warning("Server didn't terminate gracefully, killing...")
                self.server_process.kill()
                self.server_process.wait()
            self.server_process = None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics from vLLM server"""
        try:
            response = requests.get(self.metrics_url, timeout=5)
            response.raise_for_status()
            
            # Log the raw metrics text for debugging
            self.logger.debug("RAW METRICS RESPONSE:")
            self.logger.debug(response.text[:2000] + "..." if len(response.text) > 2000 else response.text)
            
            metrics = {}
            
            for family in text_string_to_metric_families(response.text):
                for sample in family.samples:
                    # Create metric key with labels if present
                    if sample.labels:
                        label_str = ",".join([f"{k}={v}" for k, v in sample.labels.items()])
                        key = f"{sample.name}[{label_str}]"
                    else:
                        key = sample.name
                    
                    metrics[key] = sample.value
            
            self.logger.info(f"Collected {len(metrics)} metrics from Prometheus endpoint")
            return metrics
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error getting metrics: {e}")
            return {}
    
    def monitor_metrics(self):
        """Monitor metrics in background thread"""
        self.logger.info("Starting metrics monitoring...")
        
        sample_count = 0
        while not self.stop_monitoring:
            try:
                metrics = self.get_metrics()
                if metrics:
                    # Add timestamp
                    metrics['_timestamp'] = time.time()
                    self.metrics_data.append(metrics)
                    sample_count += 1
                    
                    # Log detailed metrics for first few samples and periodically
                    if sample_count <= 3 or sample_count % 10 == 0:
                        self.log_key_metrics(metrics)
                    else:
                        # Just log basic info
                        kv_usage = metrics.get('vllm:gpu_cache_usage_perc', 
                                             metrics.get('vllm_gpu_cache_usage_perc', 0))
                        running_requests = metrics.get('vllm:num_requests_running', 0)
                        if kv_usage > 0 or running_requests > 0:
                            self.logger.info(f"Sample {sample_count}: KV Cache: {kv_usage:.1f}%, Running: {running_requests}")
                else:
                    self.logger.warning("No metrics received from server")
                
                time.sleep(10)  # Sample every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics monitoring error: {e}")
                time.sleep(10)
        
        self.logger.info(f"Metrics monitoring stopped. Collected {len(self.metrics_data)} samples.")
    
    def log_key_metrics(self, metrics: Dict[str, Any]):
        """Log important metrics"""
        # Log ALL metrics for debugging
        self.logger.info("="*60)
        self.logger.info("ALL AVAILABLE METRICS:")
        self.logger.info("="*60)
        
        # Sort metrics by name for easier reading
        sorted_metrics = sorted(metrics.items())
        
        for key, value in sorted_metrics:
            if key != '_timestamp':  # Skip our internal timestamp
                self.logger.info(f"{key}: {value}")
        
        self.logger.info("="*60)
        
        # Also specifically look for cache-related metrics
        cache_related = {}
        kv_related = {}
        prefix_related = {}
        request_related = {}
        
        for key, value in metrics.items():
            key_lower = key.lower()
            if 'cache' in key_lower:
                cache_related[key] = value
            if 'kv' in key_lower:
                kv_related[key] = value
            if 'prefix' in key_lower:
                prefix_related[key] = value
            if 'request' in key_lower or 'running' in key_lower or 'waiting' in key_lower:
                request_related[key] = value
        
        if cache_related:
            self.logger.info("CACHE-RELATED METRICS:")
            for key, value in cache_related.items():
                self.logger.info(f"  {key}: {value}")
        else:
            self.logger.warning("NO CACHE-RELATED METRICS FOUND!")
            
        if kv_related:
            self.logger.info("KV-RELATED METRICS:")
            for key, value in kv_related.items():
                self.logger.info(f"  {key}: {value}")
        else:
            self.logger.warning("NO KV-RELATED METRICS FOUND!")
            
        if prefix_related:
            self.logger.info("PREFIX-RELATED METRICS:")
            for key, value in prefix_related.items():
                self.logger.info(f"  {key}: {value}")
        else:
            self.logger.warning("NO PREFIX-RELATED METRICS FOUND!")
            
        if request_related:
            self.logger.info("REQUEST-RELATED METRICS:")
            for key, value in request_related.items():
                self.logger.info(f"  {key}: {value}")
        else:
            self.logger.warning("NO REQUEST-RELATED METRICS FOUND!")
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset from file"""
        self.logger.info(f"Loading dataset: {self.dataset_path}")
        
        try:
            if self.dataset_path.suffix.lower() == '.csv':
                df = pd.read_csv(self.dataset_path)
                data = df.to_dict('records')
            elif self.dataset_path.suffix.lower() == '.json':
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("JSON must contain a list of records")
            else:
                raise ValueError(f"Unsupported file format: {self.dataset_path.suffix}")
            
            self.logger.info(f"Loaded {len(data)} records from dataset")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
    
    def build_query_prompt(self, data_row: Dict[str, Any]) -> str:
        """Build query prompt for a data row"""
        template = QUERY_TEMPLATES[self.query_type]
        query_prompt = template["prompt"]
        
        # Convert data row to formatted fields string
        # Use consistent formatting to enable prefix caching
        fields_str = json.dumps(data_row, indent=2, sort_keys=True)
        
        # Use system prompt template
        full_prompt = SYSTEM_PROMPT.format(
            QUERY=query_prompt,
            fields=fields_str
        )
        
        return full_prompt
    
    def query_model(self, prompt: str, row_index: int) -> Dict[str, Any]:
        """Query the model with a prompt"""
        # Use consistent parameters to enable prefix caching
        payload = {
            "model": self.model_path,
            "messages": [
                {"role": "system", "content": QUERY_TEMPLATES[self.query_type]["prompt"]},
                {"role": "user", "content": f"Given the following data: {json.dumps(self.current_data_row, indent=2, sort_keys=True)}"}
            ],
            "max_tokens": 512,
            "temperature": 0.0,  # Use 0.0 for consistent results
            "seed": 42  # Add seed for consistency
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            inference_time = time.time() - start_time
            result_data = response.json()
            
            return {
                'row_index': row_index,
                'success': True,
                'response': result_data['choices'][0]['message']['content'].strip(),
                'inference_time': inference_time,
                'prompt_tokens': result_data.get('usage', {}).get('prompt_tokens', 0),
                'completion_tokens': result_data.get('usage', {}).get('completion_tokens', 0),
                'timestamp': time.time()
            }
            
        except Exception as e:
            inference_time = time.time() - start_time
            self.logger.error(f"Query failed for row {row_index}: {e}")
            
            return {
                'row_index': row_index,
                'success': False,
                'error': str(e),
                'inference_time': inference_time,
                'timestamp': time.time()
            }
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment"""
        self.logger.info("Starting vLLM experiment...")
        
        # Load dataset
        try:
            data = self.load_dataset()
        except Exception as e:
            return {"status": "failed", "error": f"Dataset loading failed: {e}"}
        
        # Start server
        if not self.start_server():
            return {"status": "failed", "error": "Failed to start vLLM server"}
        
        # Start metrics monitoring
        self.stop_monitoring = False
        metrics_thread = threading.Thread(target=self.monitor_metrics, daemon=True)
        metrics_thread.start()
        
        # Wait a bit for initial metrics
        time.sleep(5)
        
        # Run inference on each row
        experiment_start_time = time.time()
        total_inference_time = 0
        successful_queries = 0
        
        self.logger.info(f"Processing {len(data)} rows with query type: {self.query_type}")
        
        # For testing cache hits, let's duplicate some rows if dataset is small
        if len(data) < 5:
            self.logger.info("Small dataset detected, duplicating rows to test caching")
            # Duplicate the first row multiple times
            original_data = data.copy()
            data = [original_data[0]] * 5 + original_data
            self.logger.info(f"Extended dataset to {len(data)} rows for cache testing")
        
        for i, row in enumerate(data):
            self.logger.info(f"Processing row {i+1}/{len(data)}")
            
            # Store current row for the modified query method
            self.current_data_row = row
            
            # Build query prompt
            prompt = self.build_query_prompt(row)
            
            # Query model
            result = self.query_model(prompt, i)
            self.inference_results.append(result)
            
            if result['success']:
                total_inference_time += result['inference_time']
                successful_queries += 1
            
            # Smaller delay to see caching effects better
            time.sleep(0.5)
        
        experiment_end_time = time.time()
        total_experiment_time = experiment_end_time - experiment_start_time
        
        # Wait a bit more to collect final metrics
        time.sleep(2)
        
        # Stop monitoring
        self.stop_monitoring = True
        metrics_thread.join(timeout=10)
        
        # Calculate summary
        summary = self.calculate_summary(
            total_experiment_time, total_inference_time, 
            successful_queries, len(data)
        )
        
        # Save results
        self.save_results(summary)
        
        # Stop server
        self.stop_server()
        
        self.logger.info("Experiment completed successfully!")
        return summary
    
    def calculate_summary(self, total_time: float, inference_time: float, 
                         successful: int, total: int) -> Dict[str, Any]:
        """Calculate experiment summary statistics"""
        metrics_summary = self.extract_metrics_summary()
        
        summary = {
            "status": "completed",
            "experiment_config": {
                "dataset_path": str(self.dataset_path),
                "model_path": self.model_path,
                "query_type": self.query_type,
                "gpus": self.gpus,
                "total_rows": total
            },
            "timing_stats": {
                "total_experiment_time": total_time,
                "total_inference_time": inference_time,
                "average_inference_time": inference_time / successful if successful > 0 else 0,
                "throughput_requests_per_second": successful / total_time if total_time > 0 else 0
            },
            "success_stats": {
                "successful_queries": successful,
                "failed_queries": total - successful,
                "success_rate": successful / total if total > 0 else 0
            },
            "kv_cache_metrics": metrics_summary,
            "metrics_collected": len(self.metrics_data),
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
    
    def extract_metrics_summary(self) -> Dict[str, Any]:
        """Extract key metrics summary"""
        if not self.metrics_data:
            self.logger.warning("No metrics data collected!")
            return {}
        
        self.logger.info(f"Analyzing metrics from {len(self.metrics_data)} samples...")
        
        # Get all unique metric names across all samples
        all_metric_names = set()
        for metrics in self.metrics_data:
            all_metric_names.update(metrics.keys())
        
        self.logger.info(f"Found {len(all_metric_names)} unique metric names:")
        for name in sorted(all_metric_names):
            if name != '_timestamp':
                self.logger.info(f"  - {name}")
        
        # Try to find cache metrics across all samples
        all_cache_queries = []
        all_cache_hits = []
        all_kv_usage = []
        
        # Extended list of possible metric names
        possible_cache_query_keys = [
            'vllm:cache_query_total',
            'vllm_cache_query_total',
            'cache_query_total',
            'vllm:prefix_cache_query_total',
            'vllm_prefix_cache_query_total',
            'prefix_cache_query_total',
            'vllm:cache_queries_total',
            'vllm_cache_queries_total'
        ]
        
        possible_cache_hit_keys = [
            'vllm:cache_query_hit_total', 
            'vllm_cache_query_hit_total',
            'cache_query_hit_total',
            'vllm:prefix_cache_hit_total',
            'vllm_prefix_cache_hit_total',
            'prefix_cache_hit_total',
            'vllm:cache_hits_total',
            'vllm_cache_hits_total'
        ]
        
        possible_cache_usage_keys = [
            'vllm:gpu_cache_usage_perc',
            'vllm_gpu_cache_usage_perc', 
            'gpu_cache_usage_perc',
            'vllm:kv_cache_usage_perc',
            'vllm_kv_cache_usage_perc',
            'kv_cache_usage_perc',
            'vllm:cache_usage_perc',
            'vllm_cache_usage_perc'
        ]
        
        # Look for any metrics containing these keywords
        cache_query_metrics = []
        cache_hit_metrics = []
        cache_usage_metrics = []
        
        for metrics in self.metrics_data:
            for key in metrics.keys():
                key_lower = key.lower()
                if any(keyword in key_lower for keyword in ['cache', 'prefix']) and 'query' in key_lower and 'hit' not in key_lower:
                    cache_query_metrics.append((key, metrics[key]))
                elif any(keyword in key_lower for keyword in ['cache', 'prefix']) and 'hit' in key_lower:
                    cache_hit_metrics.append((key, metrics[key]))
                elif 'usage' in key_lower and any(keyword in key_lower for keyword in ['cache', 'kv']):
                    cache_usage_metrics.append((key, metrics[key]))
        
        self.logger.info("FOUND CACHE QUERY METRICS:")
        for key, value in cache_query_metrics:
            self.logger.info(f"  {key}: {value}")
            
        self.logger.info("FOUND CACHE HIT METRICS:")
        for key, value in cache_hit_metrics:
            self.logger.info(f"  {key}: {value}")
            
        self.logger.info("FOUND CACHE USAGE METRICS:")
        for key, value in cache_usage_metrics:
            self.logger.info(f"  {key}: {value}")
        
        # Try standard extraction
        for metrics in self.metrics_data:
            # Try different metric names for cache queries
            found_query = False
            for cache_query_key in possible_cache_query_keys:
                if cache_query_key in metrics:
                    all_cache_queries.append(metrics[cache_query_key])
                    found_query = True
                    break
            if not found_query:
                all_cache_queries.append(0)
                
            # Try different metric names for cache hits    
            found_hit = False
            for cache_hit_key in possible_cache_hit_keys:
                if cache_hit_key in metrics:
                    all_cache_hits.append(metrics[cache_hit_key])
                    found_hit = True
                    break
            if not found_hit:
                all_cache_hits.append(0)
                
            # Try different metric names for cache usage
            found_usage = False
            for kv_usage_key in possible_cache_usage_keys:
                if kv_usage_key in metrics:
                    all_kv_usage.append(metrics[kv_usage_key])
                    found_usage = True
                    break
            if not found_usage:
                all_kv_usage.append(0)
        
        # Calculate deltas and stats
        total_cache_queries = max(all_cache_queries) - min(all_cache_queries) if all_cache_queries else 0
        total_cache_hits = max(all_cache_hits) - min(all_cache_hits) if all_cache_hits else 0
        max_kv_usage = max(all_kv_usage) if all_kv_usage else 0
        avg_kv_usage = sum(all_kv_usage) / len(all_kv_usage) if all_kv_usage else 0
        
        # Check for any running requests
        max_running_requests = 0
        total_requests = 0
        for metrics in self.metrics_data:
            max_running_requests = max(max_running_requests, metrics.get('vllm:num_requests_running', 0))
            total_requests = max(total_requests, metrics.get('vllm:num_requests_total', 0))
        
        # Summary of findings
        self.logger.info("METRICS ANALYSIS SUMMARY:")
        self.logger.info(f"  Cache Queries Range: {min(all_cache_queries) if all_cache_queries else 0} - {max(all_cache_queries) if all_cache_queries else 0}")
        self.logger.info(f"  Cache Hits Range: {min(all_cache_hits) if all_cache_hits else 0} - {max(all_cache_hits) if all_cache_hits else 0}")
        self.logger.info(f"  KV Usage Range: {min(all_kv_usage) if all_kv_usage else 0:.1f}% - {max(all_kv_usage) if all_kv_usage else 0:.1f}%")
        self.logger.info(f"  Total Cache Queries: {total_cache_queries}")
        self.logger.info(f"  Total Cache Hits: {total_cache_hits}")
        self.logger.info(f"  Max KV Usage: {max_kv_usage:.1f}%")
        self.logger.info(f"  Average KV Usage: {avg_kv_usage:.1f}%")
        
        return {
            "max_kv_cache_usage_percent": max_kv_usage,
            "average_kv_cache_usage_percent": avg_kv_usage,
            "total_cache_queries": total_cache_queries,
            "total_cache_hits": total_cache_hits,
            "cache_hit_rate_percent": (total_cache_hits / total_cache_queries * 100) if total_cache_queries > 0 else 0,
            "max_concurrent_requests": max_running_requests,
            "total_requests_processed": total_requests,
            "metrics_samples_collected": len(self.metrics_data),
            "unique_metric_names_found": len(all_metric_names),
            "raw_cache_query_values": all_cache_queries,
            "raw_cache_hit_values": all_cache_hits,
            "raw_kv_usage_values": all_kv_usage
        }
    
    def save_results(self, summary: Dict[str, Any]):
        """Save experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary_file = self.result_dir / f"experiment_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        self.logger.info(f"Summary saved: {summary_file}")
        
        # Save detailed results
        results_file = self.result_dir / f"inference_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.inference_results, f, indent=2, default=str)
        self.logger.info(f"Results saved: {results_file}")
        
        # Save metrics data
        if self.metrics_data:
            metrics_file = self.result_dir / f"metrics_data_{timestamp}.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_data, f, indent=2, default=str)
            self.logger.info(f"Metrics saved: {metrics_file}")
            
            # Save a human-readable metrics summary
            metrics_summary_file = self.result_dir / f"metrics_summary_{timestamp}.txt"
            with open(metrics_summary_file, 'w') as f:
                f.write("VLLM METRICS SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                # Get all unique metric names
                all_metric_names = set()
                for metrics in self.metrics_data:
                    all_metric_names.update(metrics.keys())
                
                f.write(f"Total samples collected: {len(self.metrics_data)}\n")
                f.write(f"Unique metrics found: {len(all_metric_names)}\n\n")
                
                f.write("ALL METRIC NAMES:\n")
                for name in sorted(all_metric_names):
                    if name != '_timestamp':
                        f.write(f"  - {name}\n")
                
                f.write("\nFIRST SAMPLE VALUES:\n")
                if self.metrics_data:
                    first_sample = self.metrics_data[0]
                    for key, value in sorted(first_sample.items()):
                        if key != '_timestamp':
                            f.write(f"  {key}: {value}\n")
                
                f.write("\nLAST SAMPLE VALUES:\n")
                if self.metrics_data:
                    last_sample = self.metrics_data[-1]
                    for key, value in sorted(last_sample.items()):
                        if key != '_timestamp':
                            f.write(f"  {key}: {value}\n")
            
            self.logger.info(f"Metrics summary saved: {metrics_summary_file}")
        else:
            self.logger.warning("No metrics data to save!")
        
        # Save CSV for easy analysis
        if self.inference_results:
            csv_file = self.result_dir / f"inference_results_{timestamp}.csv"
            df = pd.DataFrame(self.inference_results)
            df.to_csv(csv_file, index=False)
            self.logger.info(f"CSV saved: {csv_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="vLLM Experiment Runner with KV Cache Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""s
Available Query Types:
{chr(10).join([f"  {k}: {v['prompt'][:80]}..." for k, v in QUERY_TEMPLATES.items()])}

Examples:
  # Basic usage
  python experiment.py dataset.csv /path/to/model filter_movies_kids --gpus "6,7"
  
  # Single GPU
  python experiment.py dataset.json /path/to/model agg_products_sentiment --gpus "7"
  
  # Custom server settings
  python experiment.py data.csv model_path proj_movies_summary --gpus "6,7" --port 8001
        """
    )
    
    parser.add_argument("dataset_path", help="Path to dataset file (CSV or JSON)")
    parser.add_argument("model_path", help="Path to LLM model")
    parser.add_argument("query_type", help="Query type", choices=list(QUERY_TEMPLATES.keys()))
    parser.add_argument("--gpus", required=True, help="GPU devices to use (e.g., '6,7' or '7')")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    
    args = parser.parse_args()
    
    try:
        # Initialize and run experiment
        runner = VLLMExperimentRunner(
            dataset_path=args.dataset_path,
            model_path=args.model_path,
            query_type=args.query_type,
            gpus=args.gpus,
            host=args.host,
            port=args.port
        )
        
        # Run experiment
        summary = runner.run_experiment()
        
        # Print summary
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        print(f"Status: {summary['status']}")
        print(f"Total Queries: {summary['experiment_config']['total_rows']}")
        print(f"Successful: {summary['success_stats']['successful_queries']}")
        print(f"Success Rate: {summary['success_stats']['success_rate']:.1%}")
        print(f"Total Time: {summary['timing_stats']['total_experiment_time']:.2f}s")
        print(f"Avg Inference Time: {summary['timing_stats']['average_inference_time']:.3f}s")
        print(f"Throughput: {summary['timing_stats']['throughput_requests_per_second']:.2f} req/s")
        
        if summary.get('kv_cache_metrics'):
            kv_metrics = summary['kv_cache_metrics']
            print(f"Max KV Cache Usage: {kv_metrics['max_kv_cache_usage_percent']:.1f}%")
            print(f"Average KV Cache Usage: {kv_metrics['average_kv_cache_usage_percent']:.1f}%")
            print(f"Cache Hit Rate: {kv_metrics['cache_hit_rate_percent']:.1f}%")
            print(f"Total Cache Queries: {kv_metrics['total_cache_queries']}")
            print(f"Total Cache Hits: {kv_metrics['total_cache_hits']}")
            print(f"Unique Metrics Found: {kv_metrics['unique_metric_names_found']}")
            print(f"Metrics Samples Collected: {kv_metrics['metrics_samples_collected']}")
        
        print("="*50)
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()