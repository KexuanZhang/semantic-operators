#!/usr/bin/env python3
"""
vLLM Server Experiment Script

This script hosts a vLLM server with a language model and processes datasets
for sentiment analysis experiments. It collects comprehensive metrics including
KV cache usage, inference times, and throughput statistics.

Usage:
    python server_exp.py --model MODEL_NAME --dataset DATASET_PATH [options]

Example:
    python server_exp.py --model microsoft/DialoGPT-medium --dataset data/reviews.csv
"""

import os
import sys
import json
import time
import argparse
import subprocess
import threading
import requests
import pandas as pd
import signal
import atexit
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from prometheus_client.parser import text_string_to_metric_families
from dataclasses import dataclass
import logging


@dataclass
class ExperimentConfig:
    """Configuration for the experiment"""
    model_name: str
    dataset_path: str
    host: str = "localhost"
    port: int = 8000
    result_dir: str = "results"
    max_tokens: int = 10
    temperature: float = 0.0
    timeout: int = 30
    metrics_interval: float = 1.0
    server_startup_timeout: int = 300
    
    # GPU Configuration
    tensor_parallel_size: int = 1  # Number of GPUs for tensor parallelism
    gpu_memory_utilization: float = 0.9  # GPU memory utilization (0.0-1.0)
    dtype: str = "auto"  # Data type: auto, half, float16, bfloat16, float32
    max_model_len: Optional[int] = None  # Maximum sequence length
    quantization: Optional[str] = None  # awq, gptq, squeezellm, fp8
    enforce_eager: bool = False  # Disable CUDA graph (for debugging)
    enable_chunked_prefill: bool = True  # Enable chunked prefill for better memory usage
    max_num_seqs: int = 256  # Maximum number of sequences in a batch
    cuda_visible_devices: Optional[str] = None  # Specific GPU selection


class VLLMServerExperiment:
    """Main class for running vLLM server sentiment analysis experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.server_url = f"http://{config.host}:{config.port}"
        self.metrics_url = f"{self.server_url}/metrics"
        
        # Create result directory
        self.result_dir = Path(config.result_dir)
        self.result_dir.mkdir(exist_ok=True)
        
        # Initialize state
        self.server_process = None
        self.metrics_data = []
        self.inference_results = []
        self.stop_monitoring = False
        self.experiment_start_time = None
        
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
        """Verify GPU setup before starting the server"""
        try:
            import torch
            
            # Check if CUDA is available
            if not torch.cuda.is_available():
                self.logger.warning("CUDA is not available. Server will run on CPU.")
                return True  # Allow CPU fallback
            
            # Check number of available GPUs
            num_gpus = torch.cuda.device_count()
            self.logger.info(f"Found {num_gpus} GPU(s)")
            
            # Verify tensor parallel size doesn't exceed available GPUs
            if self.config.tensor_parallel_size > num_gpus:
                self.logger.error(
                    f"Tensor parallel size ({self.config.tensor_parallel_size}) "
                    f"exceeds available GPUs ({num_gpus})"
                )
                return False
            
            # Print GPU information
            for i in range(min(num_gpus, self.config.tensor_parallel_size)):
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    self.logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
                except Exception as e:
                    self.logger.warning(f"Could not get info for GPU {i}: {e}")
            
            # Check GPU memory utilization setting
            if not 0.1 <= self.config.gpu_memory_utilization <= 1.0:
                self.logger.warning(
                    f"GPU memory utilization ({self.config.gpu_memory_utilization}) "
                    f"should be between 0.1 and 1.0"
                )
            
            return True
            
        except ImportError:
            self.logger.warning("PyTorch not available for GPU verification")
            return True  # Continue anyway
        except Exception as e:
            self.logger.error(f"Error verifying GPU setup: {e}")
            return False

    def start_server(self) -> bool:
        """Start the vLLM server with metrics and GPU configuration enabled"""
        self.logger.info(f"Starting vLLM server with model: {self.config.model_name}")
        
        # Verify GPU setup first
        if not self.verify_gpu_setup():
            self.logger.error("GPU verification failed")
            return False
        
        # Construct server command with GPU configuration
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model_name,
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--disable-log-stats",  # We'll collect metrics manually
            "--enable-prefix-caching",  # Enable prefix caching for better KV cache metrics
            
            # GPU Configuration
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--dtype", self.config.dtype,
            "--max-num-seqs", str(self.config.max_num_seqs),
        ]
        
        # Add optional parameters
        if self.config.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.config.max_model_len)])
        
        if self.config.quantization:
            cmd.extend(["--quantization", self.config.quantization])
        
        if self.config.enforce_eager:
            cmd.append("--enforce-eager")
        
        if self.config.enable_chunked_prefill:
            cmd.append("--enable-chunked-prefill")
        
        self.logger.info(f"Starting server with command: {' '.join(cmd)}")
        
        try:
            # Start server process
            env = dict(os.environ)
            
            # Set CUDA visibility if needed
            if self.config.cuda_visible_devices:
                env['CUDA_VISIBLE_DEVICES'] = self.config.cuda_visible_devices
            
            self.server_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True, 
                bufsize=1,
                env=env
            )
            
            # Wait for server to be ready
            self.logger.info("Waiting for server to start...")
            start_time = time.time()
            
            while time.time() - start_time < self.config.server_startup_timeout:
                try:
                    # Check if server is responding
                    response = requests.get(f"{self.server_url}/health", timeout=5)
                    if response.status_code == 200:
                        self.logger.info("Server started successfully!")
                        
                        # Log initial GPU metrics if available
                        initial_metrics = self.get_metrics()
                        gpu_cache_usage = initial_metrics.get('vllm:gpu_cache_usage_perc', 0)
                        if gpu_cache_usage >= 0:
                            self.logger.info(f"Initial GPU cache usage: {gpu_cache_usage:.1f}%")
                        
                        return True
                except requests.exceptions.RequestException:
                    pass
                
                # Check if process is still running
                if self.server_process.poll() is not None:
                    stdout, stderr = self.server_process.communicate()
                    self.logger.error(f"Server process terminated unexpectedly")
                    self.logger.error(f"STDOUT: {stdout[-1000:]}")  # Last 1000 chars
                    self.logger.error(f"STDERR: {stderr[-1000:]}")  # Last 1000 chars
                    return False
                
                time.sleep(2)
            
            self.logger.error("Server failed to start within timeout period")
            return False
            
        except Exception as e:
            self.logger.error(f"Error starting server: {e}")
            return False
    
    def stop_server(self):
        """Stop the vLLM server"""
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
        """Get current metrics from the server with comprehensive vLLM metrics collection"""
        try:
            response = requests.get(self.metrics_url, timeout=5)
            response.raise_for_status()
            
            metrics = {}
            histograms = {}
            counters = {}
            gauges = {}
            
            for family in text_string_to_metric_families(response.text):
                metric_type = family.type
                
                for sample in family.samples:
                    # Create metric key with labels
                    key = sample.name
                    label_info = {}
                    
                    if sample.labels:
                        label_str = ",".join([f"{k}={v}" for k, v in sample.labels.items()])
                        key_with_labels = f"{sample.name}[{label_str}]"
                        label_info = sample.labels
                    else:
                        key_with_labels = sample.name
                    
                    # Store raw metric
                    metrics[key_with_labels] = sample.value
                    
                    # Categorize metrics by type for analysis
                    metric_info = {
                        'value': sample.value,
                        'labels': label_info,
                        'type': metric_type
                    }
                    
                    if metric_type == 'histogram':
                        if key not in histograms:
                            histograms[key] = {}
                        histograms[key][key_with_labels] = metric_info
                    elif metric_type == 'counter':
                        counters[key_with_labels] = metric_info
                    elif metric_type == 'gauge':
                        gauges[key_with_labels] = metric_info
            
            # Add categorized metrics for analysis
            metrics['_meta'] = {
                'histograms': histograms,
                'counters': counters,
                'gauges': gauges,
                'collection_timestamp': time.time()
            }
            
            return metrics
            
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Error querying metrics: {e}")
            return {}
    
    def monitor_metrics(self):
        """Monitor metrics in a separate thread"""
        self.logger.info("Starting metrics monitoring...")
        
        while not self.stop_monitoring:
            timestamp = time.time()
            metrics = self.get_metrics()
            
            if metrics:
                metrics['timestamp'] = timestamp
                metrics['relative_time'] = timestamp - self.experiment_start_time if self.experiment_start_time else 0
                self.metrics_data.append(metrics)
                
                # Log key performance and cache metrics
                self._log_key_metrics(metrics)
            
            time.sleep(self.config.metrics_interval)
    
    def _log_key_metrics(self, metrics: Dict[str, Any]):
        """Log important vLLM metrics for monitoring"""
        # GPU Cache Usage
        kv_usage = metrics.get('vllm:gpu_cache_usage_perc', 0)
        
        # Prefix Cache Metrics (new counters approach)
        cache_queries = metrics.get('vllm:cache_query_total', 0)
        cache_hits = metrics.get('vllm:cache_query_hit_total', 0)
        
        # Request counts
        running_requests = metrics.get('vllm:num_requests_running', 0)
        waiting_requests = metrics.get('vllm:num_requests_waiting', 0)
        
        # Token generation metrics
        prompt_tokens = metrics.get('vllm:prompt_tokens_total', 0)
        generation_tokens = metrics.get('vllm:generation_tokens_total', 0)
        
        # Success/failure counts
        success_requests = metrics.get('vllm:request_success_total', 0)
        
        # Log comprehensive metrics
        if any([kv_usage > 0, cache_queries > 0, running_requests > 0, waiting_requests > 0]):
            self.logger.debug(
                f"Metrics - KV Cache: {kv_usage:.1f}% | "
                f"Cache Queries: {cache_queries} (Hits: {cache_hits}) | "
                f"Requests - Running: {running_requests}, Waiting: {waiting_requests} | "
                f"Tokens - Prompt: {prompt_tokens}, Generated: {generation_tokens}"
            )
    
    def _extract_key_metrics_summary(self) -> Dict[str, Any]:
        """Extract summary of key metrics from collected data"""
        if not self.metrics_data:
            return {}
        
        last_metrics = self.metrics_data[-1]
        first_metrics = self.metrics_data[0] if len(self.metrics_data) > 1 else last_metrics
        
        # Calculate deltas for counters
        prompt_tokens_delta = last_metrics.get('vllm:prompt_tokens_total', 0) - first_metrics.get('vllm:prompt_tokens_total', 0)
        generation_tokens_delta = last_metrics.get('vllm:generation_tokens_total', 0) - first_metrics.get('vllm:generation_tokens_total', 0)
        cache_queries_delta = last_metrics.get('vllm:cache_query_total', 0) - first_metrics.get('vllm:cache_query_total', 0)
        cache_hits_delta = last_metrics.get('vllm:cache_query_hit_total', 0) - first_metrics.get('vllm:cache_query_hit_total', 0)
        
        # Calculate hit rate
        cache_hit_rate = (cache_hits_delta / cache_queries_delta * 100) if cache_queries_delta > 0 else 0
        
        # Peak usage metrics
        max_kv_usage = max([m.get('vllm:gpu_cache_usage_perc', 0) for m in self.metrics_data])
        max_running_requests = max([m.get('vllm:num_requests_running', 0) for m in self.metrics_data])
        max_waiting_requests = max([m.get('vllm:num_requests_waiting', 0) for m in self.metrics_data])
        
        # Histogram summaries for key latency metrics
        ttft_histogram = self._extract_histogram_summary('vllm:time_to_first_token_seconds', last_metrics)
        tpot_histogram = self._extract_histogram_summary('vllm:time_per_output_token_seconds', last_metrics)
        e2e_histogram = self._extract_histogram_summary('vllm:e2e_request_latency_seconds', last_metrics)
        
        return {
            'cache_metrics': {
                'max_kv_cache_usage_percent': max_kv_usage,
                'total_cache_queries': cache_queries_delta,
                'total_cache_hits': cache_hits_delta,
                'cache_hit_rate_percent': cache_hit_rate,
            },
            'token_metrics': {
                'total_prompt_tokens': prompt_tokens_delta,
                'total_generation_tokens': generation_tokens_delta,
                'total_tokens': prompt_tokens_delta + generation_tokens_delta,
            },
            'request_metrics': {
                'max_running_requests': max_running_requests,
                'max_waiting_requests': max_waiting_requests,
                'final_successful_requests': last_metrics.get('vllm:request_success_total', 0),
            },
            'latency_histograms': {
                'time_to_first_token': ttft_histogram,
                'time_per_output_token': tpot_histogram,
                'end_to_end_latency': e2e_histogram,
            }
        }
    
    def _extract_histogram_summary(self, metric_prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract histogram summary statistics"""
        histogram_data = {}
        total_count = 0
        
        # Find histogram buckets and count
        for key, value in metrics.items():
            if key.startswith(metric_prefix):
                if key.endswith('_count'):
                    total_count = value
                elif '_bucket' in key:
                    # Extract bucket threshold
                    bucket_part = key.split('[')[1].split(',')[0] if '[' in key else ''
                    if 'le=' in bucket_part:
                        threshold = bucket_part.split('le=')[1].strip('"')
                        histogram_data[f'bucket_{threshold}'] = value
        
        return {
            'total_count': total_count,
            'buckets': histogram_data
        }
    
    def query_llm(self, text: str, row_index: int) -> Dict[str, Any]:
        """Query the LLM with a sentiment analysis prompt"""
        prompt = f"Analyze the sentiment of the following text and respond with only 'positive' or 'negative':\n\n{text}"
        
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            inference_time = time.time() - start_time
            
            result_data = response.json()
            raw_answer = result_data['choices'][0]['message']['content'].strip().lower()
            
            # Parse sentiment
            if 'positive' in raw_answer:
                sentiment = 'positive'
            elif 'negative' in raw_answer:
                sentiment = 'negative'
            else:
                sentiment = 'unknown'
            
            return {
                'row_index': row_index,
                'text': text[:100] + "..." if len(text) > 100 else text,
                'raw_response': raw_answer,
                'parsed_sentiment': sentiment,
                'inference_time': inference_time,
                'timestamp': time.time(),
                'success': True,
                'usage': result_data.get('usage', {}),
                'model': result_data.get('model', self.config.model_name)
            }
            
        except Exception as e:
            inference_time = time.time() - start_time
            self.logger.error(f"Error querying LLM for row {row_index}: {e}")
            return {
                'row_index': row_index,
                'text': text[:100] + "..." if len(text) > 100 else text,
                'raw_response': None,
                'parsed_sentiment': 'error',
                'inference_time': inference_time,
                'timestamp': time.time(),
                'success': False,
                'error': str(e),
                'usage': {},
                'model': self.config.model_name
            }
    
    def load_dataset(self, dataset_path: str) -> List[str]:
        """Load dataset from various file formats"""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        self.logger.info(f"Loading dataset from: {dataset_path}")
        
        try:
            if dataset_path.suffix.lower() == '.csv':
                df = pd.read_csv(dataset_path)
                # Find text columns
                text_columns = df.select_dtypes(include=['object']).columns
                if len(text_columns) == 0:
                    raise ValueError("No text columns found in CSV file")
                
                # Use first text column or look for common column names
                text_col = None
                for preferred_name in ['text', 'review', 'comment', 'content', 'message']:
                    if preferred_name in df.columns:
                        text_col = preferred_name
                        break
                
                if text_col is None:
                    text_col = text_columns[0]
                
                texts = df[text_col].astype(str).tolist()
                self.logger.info(f"Loaded {len(texts)} texts from column '{text_col}'")
                return texts
                
            elif dataset_path.suffix.lower() == '.json':
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    if isinstance(data[0], str):
                        return data
                    elif isinstance(data[0], dict):
                        # Try to find text field
                        for key in ['text', 'review', 'comment', 'content', 'message']:
                            if key in data[0]:
                                texts = [item[key] for item in data]
                                self.logger.info(f"Loaded {len(texts)} texts from key '{key}'")
                                return texts
                        # Fallback to first string field
                        for key, value in data[0].items():
                            if isinstance(value, str):
                                texts = [item[key] for item in data]
                                self.logger.info(f"Loaded {len(texts)} texts from key '{key}'")
                                return texts
                
                raise ValueError("Could not parse JSON dataset structure")
                
            elif dataset_path.suffix.lower() == '.txt':
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                self.logger.info(f"Loaded {len(lines)} lines from text file")
                return lines
                
            else:
                raise ValueError(f"Unsupported file format: {dataset_path.suffix}")
                
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment"""
        self.logger.info("Starting vLLM server experiment")
        
        # Load dataset
        try:
            texts = self.load_dataset(self.config.dataset_path)
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            return {"status": "failed", "error": f"Dataset loading failed: {e}"}
        
        # Start server
        if not self.start_server():
            return {"status": "failed", "error": "Failed to start vLLM server"}
        
        # Start metrics monitoring
        self.stop_monitoring = False
        metrics_thread = threading.Thread(target=self.monitor_metrics, daemon=True)
        metrics_thread.start()
        
        # Run inference experiment
        self.experiment_start_time = time.time()
        total_inference_time = 0
        successful_inferences = 0
        
        self.logger.info(f"Processing {len(texts)} texts...")
        
        for i, text in enumerate(texts):
            self.logger.info(f"Processing row {i+1}/{len(texts)}")
            
            result = self.query_llm(text, i)
            self.inference_results.append(result)
            
            if result['success']:
                total_inference_time += result['inference_time']
                successful_inferences += 1
            
            # Small delay between requests to avoid overwhelming the server
            time.sleep(0.1)
        
        experiment_end_time = time.time()
        total_experiment_time = experiment_end_time - self.experiment_start_time
        
        # Stop monitoring
        self.stop_monitoring = True
        metrics_thread.join(timeout=5)
        
        # Calculate summary statistics
        key_metrics_summary = self._extract_key_metrics_summary()
        
        summary = {
            "status": "completed",
            "experiment_config": {
                "model_name": self.config.model_name,
                "dataset_path": str(self.config.dataset_path),
                "dataset_size": len(texts),
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            },
            "timing_stats": {
                "total_experiment_time": total_experiment_time,
                "total_inference_time": total_inference_time,
                "average_inference_time": total_inference_time / successful_inferences if successful_inferences > 0 else 0,
                "throughput_requests_per_second": successful_inferences / total_experiment_time if total_experiment_time > 0 else 0,
            },
            "success_stats": {
                "total_requests": len(texts),
                "successful_requests": successful_inferences,
                "failed_requests": len(texts) - successful_inferences,
                "success_rate": successful_inferences / len(texts) if len(texts) > 0 else 0,
            },
            "sentiment_distribution": self._calculate_sentiment_distribution(),
            "vllm_metrics_summary": key_metrics_summary,
            "metrics_collected": len(self.metrics_data),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save results
        self._save_results(summary)
        
        # Stop server
        self.stop_server()
        
        self.logger.info("Experiment completed successfully")
        return summary
    
    def _calculate_sentiment_distribution(self) -> Dict[str, int]:
        """Calculate distribution of sentiment predictions"""
        distribution = {"positive": 0, "negative": 0, "unknown": 0, "error": 0}
        for result in self.inference_results:
            sentiment = result.get('parsed_sentiment', 'error')
            distribution[sentiment] = distribution.get(sentiment, 0) + 1
        return distribution
    
    def _save_results(self, summary: Dict[str, Any]):
        """Save experiment results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary_file = self.result_dir / f"experiment_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        self.logger.info(f"Saved experiment summary to: {summary_file}")
        
        # Save detailed inference results
        results_file = self.result_dir / f"inference_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.inference_results, f, indent=2, default=str)
        self.logger.info(f"Saved inference results to: {results_file}")
        
        # Save metrics data
        if self.metrics_data:
            metrics_file = self.result_dir / f"metrics_data_{timestamp}.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_data, f, indent=2, default=str)
            self.logger.info(f"Saved metrics data to: {metrics_file}")
        
        # Save CSV for easy analysis
        if self.inference_results:
            csv_file = self.result_dir / f"inference_results_{timestamp}.csv"
            df = pd.DataFrame(self.inference_results)
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Saved inference results CSV to: {csv_file}")


def main():
    """Main function to parse arguments and run experiment"""
    parser = argparse.ArgumentParser(
        description="vLLM Server Sentiment Analysis Experiment with GPU Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic GPU usage (single GPU)
    python server_exp.py --model microsoft/DialoGPT-medium --dataset data/reviews.csv

    # Multi-GPU setup with tensor parallelism
    python server_exp.py --model meta-llama/Llama-2-7b-chat-hf --dataset data/reviews.csv \\
                        --tensor-parallel-size 2 --gpu-memory-utilization 0.85

    # Large model with quantization for memory efficiency
    python server_exp.py --model meta-llama/Llama-2-13b-chat-hf --dataset data/reviews.csv \\
                        --quantization awq --gpu-memory-utilization 0.95

    # Specific GPU selection
    CUDA_VISIBLE_DEVICES=0,1 python server_exp.py --model meta-llama/Llama-2-7b-chat-hf \\
                        --dataset data/reviews.csv --tensor-parallel-size 2

    # CPU fallback (for testing without GPU)
    CUDA_VISIBLE_DEVICES="" python server_exp.py --model microsoft/DialoGPT-medium \\
                        --dataset data/reviews.csv
        """
    )
    
    # Basic arguments
    parser.add_argument("--model", required=True, help="Name or path of the model to load")
    parser.add_argument("--dataset", required=True, help="Path to the dataset file (CSV, JSON, or TXT)")
    parser.add_argument("--host", default="localhost", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--result-dir", default="results", help="Directory to save results")
    parser.add_argument("--max-tokens", type=int, default=10, help="Maximum tokens for each response")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout for each inference request")
    parser.add_argument("--metrics-interval", type=float, default=1.0, help="Metrics collection interval")
    parser.add_argument("--server-timeout", type=int, default=300, help="Server startup timeout")
    
    # GPU Configuration Arguments
    parser.add_argument("--tensor-parallel-size", type=int, default=1, 
                       help="Number of GPUs for tensor parallelism (default: 1)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                       help="GPU memory utilization ratio (0.1-1.0, default: 0.9)")
    parser.add_argument("--dtype", default="auto", 
                       choices=["auto", "half", "float16", "bfloat16", "float32"],
                       help="Data type for model weights (default: auto)")
    parser.add_argument("--max-model-len", type=int, help="Maximum sequence length")
    parser.add_argument("--quantization", choices=["awq", "gptq", "squeezellm", "fp8"],
                       help="Quantization method for memory efficiency")
    parser.add_argument("--max-num-seqs", type=int, default=256,
                       help="Maximum number of sequences in a batch (default: 256)")
    parser.add_argument("--enforce-eager", action="store_true",
                       help="Disable CUDA graph for debugging (may reduce performance)")
    parser.add_argument("--disable-chunked-prefill", action="store_true",
                       help="Disable chunked prefill")
    parser.add_argument("--cuda-visible-devices", help="Comma-separated list of GPU IDs to use (e.g., 0,1,2)")
    
    args = parser.parse_args()
    
    # Create configuration with GPU settings
    config = ExperimentConfig(
        model_name=args.model,
        dataset_path=args.dataset,
        host=args.host,
        port=args.port,
        result_dir=args.result_dir,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        metrics_interval=args.metrics_interval,
        server_startup_timeout=args.server_timeout,
        
        # GPU Configuration
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        quantization=args.quantization,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        cuda_visible_devices=args.cuda_visible_devices,
    )
    
    # Run experiment
    experiment = VLLMServerExperiment(config)
    
    try:
        summary = experiment.run_experiment()
        
        if summary["status"] == "completed":
            print("\n" + "="*80)
            print("EXPERIMENT COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"Model: {summary['experiment_config']['model_name']}")
            print(f"Dataset: {summary['experiment_config']['dataset_path']}")
            print(f"Total requests: {summary['success_stats']['total_requests']}")
            print(f"Successful requests: {summary['success_stats']['successful_requests']}")
            print(f"Success rate: {summary['success_stats']['success_rate']:.2%}")
            print(f"Total experiment time: {summary['timing_stats']['total_experiment_time']:.2f}s")
            print(f"Total inference time: {summary['timing_stats']['total_inference_time']:.2f}s")
            print(f"Average inference time: {summary['timing_stats']['average_inference_time']:.3f}s")
            print(f"Throughput: {summary['timing_stats']['throughput_requests_per_second']:.2f} req/s")
            print(f"Sentiment distribution: {summary['sentiment_distribution']}")
            print(f"Metrics collected: {summary['metrics_collected']} data points")
            
            # Print comprehensive vLLM metrics summary
            if summary['vllm_metrics_summary']:
                vllm_metrics = summary['vllm_metrics_summary']
                print("\nVLLM Metrics Summary:")
                print("-" * 40)
                
                # Cache metrics
                if 'cache_metrics' in vllm_metrics:
                    cache = vllm_metrics['cache_metrics']
                    print(f"Max KV Cache Usage: {cache['max_kv_cache_usage_percent']:.1f}%")
                    print(f"Total Cache Queries: {cache['total_cache_queries']}")
                    print(f"Total Cache Hits: {cache['total_cache_hits']}")
                    print(f"Cache Hit Rate: {cache['cache_hit_rate_percent']:.1f}%")
                
                # Token metrics
                if 'token_metrics' in vllm_metrics:
                    tokens = vllm_metrics['token_metrics']
                    print(f"Total Prompt Tokens: {tokens['total_prompt_tokens']}")
                    print(f"Total Generation Tokens: {tokens['total_generation_tokens']}")
                    print(f"Total Tokens: {tokens['total_tokens']}")
                
                # Request metrics
                if 'request_metrics' in vllm_metrics:
                    requests = vllm_metrics['request_metrics']
                    print(f"Max Running Requests: {requests['max_running_requests']}")
                    print(f"Max Waiting Requests: {requests['max_waiting_requests']}")
                
                # Latency histograms
                if 'latency_histograms' in vllm_metrics:
                    histograms = vllm_metrics['latency_histograms']
                    for hist_name, hist_data in histograms.items():
                        if hist_data and hist_data.get('total_count', 0) > 0:
                            print(f"{hist_name.replace('_', ' ').title()} samples: {hist_data['total_count']}")
            
            print(f"Results saved to: {config.result_dir}")
        else:
            print(f"\nExperiment failed: {summary.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nExperiment failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()