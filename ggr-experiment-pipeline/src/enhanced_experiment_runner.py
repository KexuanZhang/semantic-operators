"""
Enhanced GGR Experiment Runner with vLLM Inference
Extends the original experiment pipeline to include vLLM inference and resource monitoring
"""
import pandas as pd
import numpy as np
import os
import time
import json
import threading
from typing import List, Tuple, Optional, Dict, Any
import logging
from datetime import datetime

# Import original components
from .data_preprocessing import (
    load_dataset, 
    discover_functional_dependencies, 
    preprocess_data,
    parse_functional_dependencies,
    validate_functional_dependencies
)
from .ggr_algorithm import ggr

# Resource monitoring imports
try:
    import psutil
    import pynvml
    monitoring_available = True
except ImportError:
    monitoring_available = False

# vLLM imports
try:
    from vllm import LLM, SamplingParams
    vllm_available = True
except ImportError:
    vllm_available = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedGGRExperiment:
    """
    Enhanced GGR experiment class that includes vLLM inference and resource monitoring
    """
    
    def __init__(self, 
                 dataset_path: str,
                 output_dir: str = "results",
                 experiment_name: Optional[str] = None,
                 model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize enhanced experiment
        
        Args:
            dataset_path: Path to the dataset CSV file
            output_dir: Directory to save results
            experiment_name: Name for this experiment
            model_name: vLLM model name for inference
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.experiment_name = experiment_name or f"enhanced_ggr_exp_{int(time.time())}"
        self.model_name = model_name
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {}
        self.resource_monitor = ResourceMonitor() if monitoring_available else None
        self.vllm_metrics = VLLMMetricsCollector()
        
    def run_complete_experiment(self,
                               functional_dependencies: Optional[List[Tuple[str, str]]] = None,
                               columns_of_interest: Optional[List[str]] = None,
                               max_depth: int = 100,
                               discover_fds: bool = True,
                               fd_confidence: float = 0.95,
                               handle_missing: str = 'drop',
                               run_inference: bool = True,
                               max_inference_samples: int = 50,
                               sampling_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run the complete enhanced experiment including GGR and vLLM inference
        
        Args:
            functional_dependencies: Pre-defined FDs
            columns_of_interest: Specific columns to analyze
            max_depth: Maximum recursion depth for GGR
            discover_fds: Whether to auto-discover FDs
            fd_confidence: Confidence threshold for FD discovery
            handle_missing: How to handle missing values
            run_inference: Whether to run vLLM inference
            max_inference_samples: Maximum samples for inference
            sampling_params: vLLM sampling parameters
            
        Returns:
            Complete experiment results
        """
        logger.info(f"Starting enhanced GGR experiment: {self.experiment_name}")
        start_time = time.time()
        
        try:
            # Phase 1: Original GGR workflow
            logger.info("Phase 1: Running GGR algorithm...")
            ggr_results = self._run_ggr_phase(
                functional_dependencies, columns_of_interest, max_depth,
                discover_fds, fd_confidence, handle_missing
            )
            self.results.update(ggr_results)
            
            # Phase 2: vLLM Inference (if enabled and available)
            if run_inference and vllm_available:
                logger.info("Phase 2: Running vLLM inference...")
                inference_results = self._run_inference_phase(
                    max_inference_samples, sampling_params
                )
                self.results.update(inference_results)
            else:
                logger.info("Phase 2: Skipping vLLM inference (disabled or not available)")
                self.results['inference_skipped'] = True
            
            # Phase 3: Analysis and comparison
            logger.info("Phase 3: Analyzing results...")
            analysis_results = self._run_analysis_phase()
            self.results.update(analysis_results)
            
            # Save all results
            total_time = time.time() - start_time
            self.results['total_experiment_time'] = total_time
            self._save_complete_results()
            
            logger.info(f"Enhanced experiment completed in {total_time:.2f} seconds")
            return self.results
            
        except Exception as e:
            logger.error(f"Enhanced experiment failed: {e}")
            self.results['error'] = str(e)
            self._save_complete_results()
            raise
    
    def _run_ggr_phase(self, functional_dependencies, columns_of_interest, 
                       max_depth, discover_fds, fd_confidence, handle_missing):
        """Run the GGR algorithm phase"""
        # Load and preprocess data
        df = load_dataset(self.dataset_path)
        processed_df = preprocess_data(df, columns_of_interest, handle_missing)
        
        # Handle functional dependencies
        if functional_dependencies is None and discover_fds:
            functional_dependencies = discover_functional_dependencies(
                processed_df, max_lhs_size=2, min_confidence=fd_confidence
            )
        elif functional_dependencies is None:
            functional_dependencies = []
        
        functional_dependencies = validate_functional_dependencies(processed_df, functional_dependencies)
        
        # Apply GGR algorithm
        ggr_start_time = time.time()
        total_hits, reordered_table = ggr(processed_df, functional_dependencies, max_depth=max_depth)
        ggr_end_time = time.time()
        
        # Save reordered dataset
        if reordered_table:
            reordered_df = pd.DataFrame(reordered_table, columns=processed_df.columns.tolist())
            reordered_path = os.path.join(self.output_dir, f"{self.experiment_name}_reordered_table.csv")
            reordered_df.to_csv(reordered_path, index=False)
        
        return {
            'dataset_info': {'path': self.dataset_path, 'shape': df.shape},
            'processed_shape': processed_df.shape,
            'functional_dependencies': functional_dependencies,
            'ggr_total_hits': total_hits,
            'ggr_execution_time': ggr_end_time - ggr_start_time,
            'reordered_table_size': len(reordered_table) if reordered_table else 0,
            'reordered_dataset_path': reordered_path if reordered_table else None
        }
    
    def _run_inference_phase(self, max_samples, sampling_params):
        """Run the vLLM inference phase"""
        if not vllm_available:
            return {'inference_error': 'vLLM not available'}
        
        # Load reordered dataset
        reordered_path = self.results.get('reordered_dataset_path')
        if not reordered_path or not os.path.exists(reordered_path):
            return {'inference_error': 'Reordered dataset not available'}
        
        reordered_df = pd.read_csv(reordered_path)
        if max_samples:
            reordered_df = reordered_df.head(max_samples)
        
        # Create prompts
        prompts = self._create_inference_prompts(reordered_df)
        
        # Initialize vLLM
        default_sampling = {
            'temperature': 0.7,
            'top_p': 0.9,
            'max_tokens': 100,
            'seed': 42
        }
        if sampling_params:
            default_sampling.update(sampling_params)
        
        try:
            llm = LLM(
                model=self.model_name,
                gpu_memory_utilization=0.8,
                enable_prefix_caching=True,
                seed=default_sampling['seed']
            )
            sampling_params_obj = SamplingParams(**default_sampling)
            
            # Start resource monitoring
            if self.resource_monitor:
                self.resource_monitor.start()
            
            # Run inference with timing
            inference_start_time = time.time()
            outputs = llm.generate(prompts, sampling_params_obj)
            inference_end_time = time.time()
            
            # Stop monitoring
            if self.resource_monitor:
                self.resource_monitor.stop()
            
            # Collect results
            inference_time = inference_end_time - inference_start_time
            throughput = len(prompts) / inference_time
            
            # Try to collect vLLM stats
            self.vllm_metrics.collect_engine_stats(llm, {
                'total_prompts': len(prompts),
                'inference_time': inference_time,
                'throughput': throughput
            })
            
            return {
                'inference_completed': True,
                'inference_samples': len(prompts),
                'inference_time': inference_time,
                'inference_throughput': throughput,
                'model_name': self.model_name,
                'sampling_params': default_sampling,
                'resource_metrics_available': self.resource_monitor is not None
            }
            
        except Exception as e:
            return {'inference_error': str(e)}
    
    def _create_inference_prompts(self, df: pd.DataFrame) -> List[str]:
        """Create inference prompts from dataset"""
        prompts = []
        for _, row in df.iterrows():
            if 'review_content' in df.columns and 'movie_title' in df.columns:
                prompt = f"Analyze the following movie review and determine the sentiment:\\n\\nMovie: {row.get('movie_title', 'Unknown')}\\nReview: {row.get('review_content', 'No review')}\n\\nSentiment:"
            else:
                prompt = f"Analyze the following data and provide insights:\\n{dict(row)}\\n\\nInsights:"
            prompts.append(prompt)
        return prompts
    
    def _run_analysis_phase(self):
        """Run analysis and comparison phase"""
        analysis = {
            'experiment_timestamp': datetime.now().isoformat(),
            'experiment_name': self.experiment_name,
            'ggr_effectiveness': {
                'total_hits': self.results.get('ggr_total_hits', 0),
                'processing_time': self.results.get('ggr_execution_time', 0),
                'reordered_samples': self.results.get('reordered_table_size', 0)
            }
        }
        
        # Inference analysis
        if self.results.get('inference_completed'):
            analysis['inference_performance'] = {
                'throughput': self.results.get('inference_throughput', 0),
                'samples_processed': self.results.get('inference_samples', 0),
                'total_time': self.results.get('inference_time', 0)
            }
            
            # Resource usage analysis
            if self.resource_monitor:
                metrics = self.resource_monitor.get_metrics()
                if metrics:
                    df_resources = pd.DataFrame(metrics)
                    analysis['resource_usage'] = {
                        'avg_cpu_percent': df_resources['cpu_percent'].mean() if 'cpu_percent' in df_resources else 0,
                        'peak_cpu_percent': df_resources['cpu_percent'].max() if 'cpu_percent' in df_resources else 0,
                        'avg_memory_percent': df_resources['memory_percent'].mean() if 'memory_percent' in df_resources else 0,
                        'monitoring_samples': len(metrics)
                    }
        
        return analysis
    
    def _save_complete_results(self):
        """Save all experiment results"""
        # Save main results as JSON
        results_file = os.path.join(self.output_dir, f"{self.experiment_name}_complete_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save resource metrics if available
        if self.resource_monitor:
            resource_file = os.path.join(self.output_dir, f"{self.experiment_name}_resource_metrics.csv")
            self.resource_monitor.save_metrics(resource_file)
        
        # Save vLLM metrics if available
        vllm_file = os.path.join(self.output_dir, f"{self.experiment_name}_vllm_metrics.json")
        self.vllm_metrics.save_metrics(vllm_file)
        
        logger.info(f"Complete results saved to {results_file}")


class ResourceMonitor:
    """Monitor system resources during experiments"""
    
    def __init__(self, interval: int = 5):
        self.interval = interval
        self.monitoring = False
        self.metrics = []
        self.thread = None
        
        if monitoring_available:
            try:
                pynvml.nvmlInit()
                self.gpu_available = True
            except:
                self.gpu_available = False
        else:
            self.gpu_available = False
    
    def start(self):
        if not self.monitoring:
            self.monitoring = True
            self.thread = threading.Thread(target=self._monitor)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self):
        if self.monitoring:
            self.monitoring = False
            if self.thread:
                self.thread.join(timeout=self.interval + 2)
    
    def _monitor(self):
        while self.monitoring:
            try:
                metric = {'timestamp': time.time()}
                
                if monitoring_available:
                    metric['cpu_percent'] = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    metric['memory_percent'] = memory.percent
                    metric['memory_used_gb'] = memory.used / (1024**3)
                
                if self.gpu_available:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metric['gpu_util_percent'] = util.gpu
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    metric['gpu_memory_used_gb'] = memory_info.used / (1024**3)
                
                self.metrics.append(metric)
                time.sleep(self.interval)
            except:
                time.sleep(self.interval)
    
    def get_metrics(self):
        return self.metrics.copy()
    
    def save_metrics(self, filepath):
        df = pd.DataFrame(self.metrics)
        df.to_csv(filepath, index=False)


class VLLMMetricsCollector:
    """Collect vLLM internal metrics"""
    
    def __init__(self):
        self.metrics = {'runs': []}
    
    def collect_engine_stats(self, llm, run_info):
        try:
            stats = None
            if hasattr(llm, 'llm_engine'):
                if hasattr(llm.llm_engine, '_get_stats'):
                    stats = llm.llm_engine._get_stats()
            
            metric_data = {
                'run_info': run_info,
                'timestamp': time.time(),
                'stats': self._extract_stats(stats) if stats else {}
            }
            self.metrics['runs'].append(metric_data)
        except Exception as e:
            logger.warning(f"Could not collect vLLM stats: {e}")
    
    def _extract_stats(self, stats):
        extracted = {}
        for attr in ['gpu_cache_usage_sys', 'gpu_prefix_cache_hit_rate', 'prompt_tokens', 'generation_tokens']:
            if hasattr(stats, attr):
                extracted[attr] = getattr(stats, attr)
        return extracted
    
    def get_metrics(self):
        return self.metrics.copy()
    
    def save_metrics(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)


def run_enhanced_experiment(dataset_path: str,
                           model_name: str = "microsoft/DialoGPT-medium",
                           functional_dependencies: Optional[str] = None,
                           columns: Optional[str] = None,
                           output_dir: str = "enhanced_results",
                           experiment_name: Optional[str] = None,
                           max_depth: int = 100,
                           run_inference: bool = True,
                           max_inference_samples: int = 50) -> Dict[str, Any]:
    """
    Run enhanced GGR experiment with vLLM inference
    
    Args:
        dataset_path: Path to dataset CSV
        model_name: vLLM model name
        functional_dependencies: FDs as string
        columns: Columns of interest as string
        output_dir: Output directory
        experiment_name: Experiment name
        max_depth: GGR max depth
        run_inference: Whether to run inference
        max_inference_samples: Max samples for inference
        
    Returns:
        Complete experiment results
    """
    from .data_preprocessing import parse_functional_dependencies
    
    # Parse arguments
    fds = parse_functional_dependencies(functional_dependencies) if functional_dependencies else None
    cols = [col.strip() for col in columns.split(',')] if columns else None
    
    # Create and run experiment
    experiment = EnhancedGGRExperiment(dataset_path, output_dir, experiment_name, model_name)
    return experiment.run_complete_experiment(
        functional_dependencies=fds,
        columns_of_interest=cols,
        max_depth=max_depth,
        run_inference=run_inference,
        max_inference_samples=max_inference_samples
    )
