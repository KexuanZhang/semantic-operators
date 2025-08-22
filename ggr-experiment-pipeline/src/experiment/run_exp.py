import os
import pandas as pd
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from vllm import LLM, SamplingParams
import torch
import torch.distributed as dist
from tqdm import tqdm


class SentimentAnalysisExperiment:
    """
    A class to perform sentiment analysis experiments using vLLM.
    
    This class handles:
    - Loading local LLM models using vLLM
    - Processing datasets for sentiment analysis
    - Saving results and inference times
    """
    
    def __init__(self, model_path: str, results_dir: str = "results", gpu_devices: Optional[List[int]] = None):
        """
        Initialize the experiment with model path and results directory.
        
        Args:
            model_path: Path to the local LLM model
            results_dir: Directory to save experiment results
            gpu_devices: List of GPU device IDs to use (e.g., [0, 1] or [6, 7])
        """
        self.model_path = model_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.gpu_devices = gpu_devices
        
        # Set environment variable for cached outputs
        os.environ["VLLM_USE_CACHED_OUTPUTS"] = "True"
        
        # Force vLLM to use v0 to access cache metrics via internal APIs
        os.environ["VLLM_USE_V1"] = "0"
        
        # Set GPU devices if specified
        if gpu_devices and torch.cuda.is_available():
            # Set CUDA_VISIBLE_DEVICES to the specified GPUs
            gpu_ids_str = ",".join(str(gpu_id) for gpu_id in gpu_devices)
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
            print(f"Using GPU devices: {gpu_devices} (CUDA_VISIBLE_DEVICES={gpu_ids_str})")
            
            # Set the primary device to the first GPU in the list
            torch.cuda.set_device(0)  # After setting CUDA_VISIBLE_DEVICES, use index 0
        elif torch.cuda.is_available():
            print("Using default GPU device")
        else:
            print("CUDA not available, using CPU")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize model and sampling parameters
        self.llm = None
        self.sampling_params = SamplingParams(
            temperature=0.7,  # Slightly higher for more natural responses
            top_p=0.9,
            max_tokens=100,   # Allow longer responses
            # Remove stop tokens to get full response
        )
        
        # For tracking vLLM internal metrics
        self.cache_metrics = []
        
    def load_model(self):
        """Load the vLLM model."""
        try:
            print(f"Loading model from: {self.model_path}")
            self.llm = LLM(
                model=self.model_path,
                trust_remote_code=True,
                gpu_memory_utilization=0.95,  # Use 95% of GPU memory
                max_model_len=6256,  # Set to estimated maximum based on available memory
                # Enable additional memory optimizations
                enforce_eager=True,  # Use eager execution to save memory
                disable_custom_all_reduce=True,  # Disable custom all-reduce to save memory
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Try with even more conservative settings if first attempt fails
            print("Retrying with more conservative memory settings...")
            try:
                self.llm = LLM(
                    model=self.model_path,
                    trust_remote_code=True,
                    gpu_memory_utilization=0.90,  # Reduce to 90%
                    max_model_len=4000,  # Further reduce max length
                    enforce_eager=True,
                    disable_custom_all_reduce=True,
                )
                print("Model loaded successfully with conservative settings!")
            except Exception as e2:
                print(f"Failed with conservative settings too: {e2}")
                raise
    
    def create_sentiment_prompt(self, text: str) -> str:
        """
        Create a prompt for sentiment analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Analyze the sentiment of the following text and respond with only 'POSITIVE' or 'NEGATIVE':

Text: {text}

Sentiment:"""
        return prompt
    
    def get_cache_stats(self):
        """
        Get cache statistics from vLLM internal Stats object.
        This accesses internal APIs which may change in future versions.
        """
        try:
            # Access internal engine stats - this is version-dependent
            if hasattr(self.llm, 'llm_engine'):
                engine = self.llm.llm_engine
                
                # For vLLM v1 engines (0.10.0+), try to access stats through different paths
                stats = None
                
                # Try different paths to access stats object
                if hasattr(engine, 'engine_core') and hasattr(engine.engine_core, 'stats'):
                    stats = engine.engine_core.stats
                elif hasattr(engine, '_get_stats'):
                    # Older versions
                    stats = engine._get_stats()
                elif hasattr(engine, 'scheduler') and hasattr(engine.scheduler, 'stats'):
                    stats = engine.scheduler.stats
                
                if stats:
                    cache_metrics = {}
                    
                    # Extract key metrics
                    if hasattr(stats, 'gpu_cache_usage_sys'):
                        cache_metrics['gpu_cache_usage_perc'] = stats.gpu_cache_usage_sys
                    
                    if hasattr(stats, 'gpu_prefix_cache_hit_rate'):
                        cache_metrics['gpu_prefix_cache_hit_rate'] = stats.gpu_prefix_cache_hit_rate
                    
                    # Try alternative attribute names
                    for attr in dir(stats):
                        if 'cache' in attr.lower() and 'hit' in attr.lower():
                            cache_metrics[attr] = getattr(stats, attr)
                        elif 'cache' in attr.lower() and 'usage' in attr.lower():
                            cache_metrics[attr] = getattr(stats, attr)
                    
                    return cache_metrics
                    
        except Exception as e:
            print(f"Warning: Could not access cache stats: {e}")
            
        return {}
        
    def analyze_sentiment(self, text: str) -> str:
        """
        Analyze sentiment of a single text using the LLM.
        
        Args:
            text: Text to analyze
            
        Returns:
            Raw LLM response
        """
        if self.llm is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        prompt = self.create_sentiment_prompt(text)
        
        try:
            # Generate response
            output = self.llm.generate([prompt], self.sampling_params)
            response = output[0].outputs[0].text.strip()
            
            # Return raw response without any processing
            return response
                
        except Exception as e:
            print(f"Error during inference: {e}")
            return f"Error: {str(e)}"
    
    def _analyze_cache_metrics(self) -> Dict[str, Any]:
        """
        Analyze collected cache metrics to provide summary statistics.
        
        Returns:
            Dictionary containing cache performance summary
        """
        if not self.cache_metrics:
            return {'note': 'No cache metrics collected'}
        
        summary = {
            'total_inferences': len(self.cache_metrics),
            'cache_hit_rates': [],
            'cache_usage_percentages': [],
            'detailed_metrics': []
        }
        
        for i, metric in enumerate(self.cache_metrics):
            inference_summary = {
                'inference_index': i,
                'timestamp': metric['timestamp'],
                'before_stats': metric['before'],
                'after_stats': metric['after']
            }
            
            # Extract cache hit rates if available
            if 'gpu_prefix_cache_hit_rate' in metric.get('after', {}):
                hit_rate = metric['after']['gpu_prefix_cache_hit_rate']
                summary['cache_hit_rates'].append(hit_rate)
                inference_summary['cache_hit_rate'] = hit_rate
            
            # Extract cache usage if available
            if 'gpu_cache_usage_perc' in metric.get('after', {}):
                usage = metric['after']['gpu_cache_usage_perc']
                summary['cache_usage_percentages'].append(usage)
                inference_summary['cache_usage_perc'] = usage
            
            summary['detailed_metrics'].append(inference_summary)
        
        # Calculate averages
        if summary['cache_hit_rates']:
            summary['average_cache_hit_rate'] = sum(summary['cache_hit_rates']) / len(summary['cache_hit_rates'])
            summary['max_cache_hit_rate'] = max(summary['cache_hit_rates'])
            summary['min_cache_hit_rate'] = min(summary['cache_hit_rates'])
        
        if summary['cache_usage_percentages']:
            summary['average_cache_usage'] = sum(summary['cache_usage_percentages']) / len(summary['cache_usage_percentages'])
            summary['max_cache_usage'] = max(summary['cache_usage_percentages'])
            summary['min_cache_usage'] = min(summary['cache_usage_percentages'])
        
        return summary

    def process_dataset(self, dataset_path: str, text_column: str,
                       batch_size: int = 1) -> Dict[str, Any]:
        """
        Process an entire dataset for sentiment analysis.
        
        Args:
            dataset_path: Path to the dataset CSV file
            text_column: Name of the column containing text to analyze
            batch_size: Number of samples to process at once
            
        Returns:
            Dictionary containing results and metadata
        """
        print(f"Loading dataset from: {dataset_path}")
        
        # Load dataset
        try:
            df = pd.read_csv(dataset_path)
            print(f"Loaded dataset with {len(df)} rows")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
        
        # Validate text column
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataset. Available columns: {list(df.columns)}")
        
        # Initialize results tracking
        start_time = time.time()
        
        # Get initial cache stats
        initial_cache_stats = self.get_cache_stats()
        
        print(f"Starting sentiment analysis for {len(df)} samples...")
        
        # Create summary - count unique responses
        response_counts = {}
        
        # Process each row
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            text = str(row[text_column])
            
            # Skip empty texts
            if not text or text.lower() == 'nan':
                sentiment = "POSITIVE"  # Default for empty texts
            else:
                # Analyze sentiment
                sentiment = self.analyze_sentiment(text)
            
            # Count responses for summary
            response_counts[sentiment] = response_counts.get(sentiment, 0) + 1
        
        # Calculate total time
        end_time = time.time()
        total_inference_time = end_time - start_time
        
        # Get final cache stats
        final_cache_stats = self.get_cache_stats()
        
        # Store overall cache metrics
        overall_cache_metrics = {
            'initial_stats': initial_cache_stats,
            'final_stats': final_cache_stats,
            'experiment_start': datetime.fromtimestamp(start_time).isoformat(),
            'experiment_end': datetime.fromtimestamp(end_time).isoformat()
        }
        
        # Create summary - count unique responses
        response_counts = {}
        for r in results:
            response = r['llm_response']  # Use raw LLM response
            response_counts[response] = response_counts.get(response, 0) + 1
        
        experiment_metadata = {
            'model_path': self.model_path,
            'dataset_path': dataset_path,
            'text_column': text_column,
            'total_samples': len(df),
            'processed_samples': len(df),
            'response_counts': response_counts,
            'total_inference_time_seconds': total_inference_time,
            'average_inference_time_seconds': total_inference_time / len(df) if len(df) > 0 else 0,
            'experiment_timestamp': datetime.now().isoformat(),
            'gpu_devices': self.gpu_devices,
            'sampling_params': {
                'temperature': self.sampling_params.temperature,
                'top_p': self.sampling_params.top_p,
                'max_tokens': self.sampling_params.max_tokens,
            },
            'cache_metrics_summary': self._analyze_cache_metrics(),
            'overall_cache_metrics': overall_cache_metrics
        }
        
        return {
            'results': [],  # No per-inference results saved
            'metadata': experiment_metadata
        }
    
    def save_results(self, experiment_data: Dict[str, Any], experiment_name: Optional[str] = None):
        """
        Save experiment results to files.
        
        Args:
            experiment_data: Results and metadata from process_dataset
            experiment_name: Optional name for the experiment files
        """
        if experiment_name is None:
            experiment_name = f"sentiment_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save detailed results
        results_file = self.results_dir / f"{experiment_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        # Save results as CSV for easy analysis
        if experiment_data['results']:
            results_df = pd.DataFrame(experiment_data['results'])
            csv_file = self.results_dir / f"{experiment_name}_results.csv"
            results_df.to_csv(csv_file, index=False)
        else:
            csv_file = self.results_dir / f"{experiment_name}_no_individual_results.txt"
            with open(csv_file, 'w') as f:
                f.write("No per-inference results saved. See metadata for overall statistics.")
        
        # Save summary metadata
        metadata_file = self.results_dir / f"{experiment_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(experiment_data['metadata'], f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"- Detailed results: {results_file}")
        print(f"- CSV results: {csv_file}")
        print(f"- Metadata: {metadata_file}")
        
        # Print summary
        metadata = experiment_data['metadata']
        print(f"\nExperiment Summary:")
        print(f"- Total samples: {metadata['total_samples']}")
        print(f"- Response counts: {metadata['response_counts']}")
        print(f"- Total inference time: {metadata['total_inference_time_seconds']:.2f} seconds")
        print(f"- Average inference time per sample: {metadata['average_inference_time_seconds']:.4f} seconds")
        
        # Print cache metrics summary if available
        cache_summary = metadata.get('cache_metrics_summary', {})
        overall_cache = metadata.get('overall_cache_metrics', {})
        
        if overall_cache.get('initial_stats') and overall_cache.get('final_stats'):
            print(f"\nOverall Cache Performance:")
            initial = overall_cache['initial_stats']
            final = overall_cache['final_stats']
            
            if 'gpu_prefix_cache_hit_rate' in final:
                print(f"- Final cache hit rate: {final['gpu_prefix_cache_hit_rate']:.2%}")
            
            if 'gpu_cache_usage_perc' in final:
                print(f"- Final GPU cache usage: {final['gpu_cache_usage_perc']:.2%}")
        
        if 'average_cache_hit_rate' in cache_summary:
            print(f"\nPrefix Cache Performance:")
            print(f"- Average cache hit rate: {cache_summary['average_cache_hit_rate']:.2%}")
            print(f"- Max cache hit rate: {cache_summary['max_cache_hit_rate']:.2%}")
            print(f"- Min cache hit rate: {cache_summary['min_cache_hit_rate']:.2%}")
        
        if 'average_cache_usage' in cache_summary:
            print(f"- Average GPU cache usage: {cache_summary['average_cache_usage']:.2%}")
            print(f"- Max GPU cache usage: {cache_summary['max_cache_usage']:.2%}")
            print(f"- Min GPU cache usage: {cache_summary['min_cache_usage']:.2%}")
    
    def cleanup(self):
        """Clean up resources."""
        if dist.is_initialized():
            dist.destroy_process_group()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """Main function to run the sentiment analysis experiment."""
    parser = argparse.ArgumentParser(description="Run sentiment analysis experiment with vLLM")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the local LLM model directory"
    )
    
    parser.add_argument(
        "--dataset-path", 
        type=str,
        required=True,
        help="Path to the dataset CSV file"
    )
    
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of the column containing text to analyze (default: 'text')"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results (default: 'results')"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name for the experiment files (default: auto-generated)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)"
    )
    
    parser.add_argument(
        "--gpus",
        type=str,
        help="Comma-separated list of GPU device IDs to use for inference (e.g., '6,7' or '0,1,2,3'). If not specified, will use default GPU."
    )
    
    args = parser.parse_args()
    
    # Parse GPU devices
    gpu_devices = None
    if args.gpus:
        try:
            gpu_devices = [int(gpu_id.strip()) for gpu_id in args.gpus.split(',')]
            print(f"Parsed GPU devices: {gpu_devices}")
        except ValueError:
            print(f"Error: Invalid GPU device format '{args.gpus}'. Use comma-separated integers like '6,7'")
            return 1
    
    # Validate input paths
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        return 1
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path does not exist: {args.dataset_path}")
        return 1
    
    try:
        # Initialize experiment
        experiment = SentimentAnalysisExperiment(
            model_path=args.model_path,
            results_dir=args.results_dir,
            gpu_devices=gpu_devices
        )
        
        # Load model
        experiment.load_model()
        
        # Process dataset
        experiment_data = experiment.process_dataset(
            dataset_path=args.dataset_path,
            text_column=args.text_column,
            batch_size=args.batch_size
        )
        
        # Save results
        experiment.save_results(experiment_data, args.experiment_name)
        
        # Cleanup
        experiment.cleanup()
        
        print("\nExperiment completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())