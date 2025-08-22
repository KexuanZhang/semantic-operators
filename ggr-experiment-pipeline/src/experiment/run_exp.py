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
        
        # Initialize results
        results = []
        start_time = time.time()
        
        print(f"Starting sentiment analysis for {len(df)} samples...")
        
        # Process each row
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            text = str(row[text_column])
            
            # Skip empty texts
            if not text or text.lower() == 'nan':
                sentiment = "POSITIVE"  # Default for empty texts
                row_inference_time = 0.0
            else:
                # Analyze sentiment
                row_start_time = time.time()
                sentiment = self.analyze_sentiment(text)
                row_end_time = time.time()
                row_inference_time = row_end_time - row_start_time
            
            # Store result - just save the raw LLM response
            result = {
                'index': idx,
                'text': text,
                'llm_response': sentiment,  # Raw LLM response
                'inference_time': row_inference_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add other columns from original dataset
            for col in df.columns:
                if col != text_column:
                    result[f'original_{col}'] = row[col]
            
            results.append(result)
        
        # Calculate total time
        end_time = time.time()
        total_inference_time = end_time - start_time
        
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
            'processed_samples': len(results),
            'response_counts': response_counts,
            'total_inference_time_seconds': total_inference_time,
            'average_inference_time_seconds': total_inference_time / len(results) if results else 0,
            'experiment_timestamp': datetime.now().isoformat(),
            'gpu_devices': self.gpu_devices,
            'sampling_params': {
                'temperature': self.sampling_params.temperature,
                'top_p': self.sampling_params.top_p,
                'max_tokens': self.sampling_params.max_tokens,
            }
        }
        
        return {
            'results': results,
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