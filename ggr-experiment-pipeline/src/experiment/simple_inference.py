#!/usr/bin/env python3
"""
Simple LLM Inference Script
Takes dataset and query arguments, feeds them to the LLM, and outputs only inference time in JSON format.

Usage:
    python simple_inference.py dataset.csv query_key [--custom-query "Your query"] [--max-rows N] [--output output.json]

Examples:
    python simple_inference.py movie_data.csv movie_analysis
    python simple_inference.py data.csv custom --custom-query "Analyze this data:" --max-rows 100 --output results.json
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

# Import vLLM
try:
    from vllm import LLM, SamplingParams
    import torch
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install: pip install vllm torch")
    sys.exit(1)

# Set up basic logging
logging.basicConfig(level=logging.WARNING)  # Only show warnings and errors
logger = logging.getLogger(__name__)

# Simple query templates - minimal set
QUERY_TEMPLATES = {
    'movie_analysis': {
        'prompt': """Based on the following movie data, provide a brief analysis:

Data: {data_fields}

Analysis:""",
        'type': 'analysis'
    },
    'summarize': {
        'prompt': """Summarize the following data in 2-3 sentences:

Data: {data_fields}

Summary:""",
        'type': 'summarization'
    },
    'extract': {
        'prompt': """Extract key information from the following data:

Data: {data_fields}

Key Information:""",
        'type': 'extraction'
    },
    'custom': {
        'prompt': '{custom_query}\n\nData: {data_fields}\n\nResponse:',
        'type': 'custom'
    }
}

class SimpleLLMInference:
    """Minimal LLM inference class focused only on timing measurement"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        self.model_name = model_name
        self.llm = None
        self.sampling_params = None
    
    def initialize_model(self, 
                        max_tokens: int = 256,
                        temperature: float = 0.1,
                        top_p: float = 0.9) -> bool:
        """Initialize the LLM model"""
        try:
            # Basic vLLM configuration
            self.llm = LLM(
                model=self.model_name,
                gpu_memory_utilization=0.85,
                enable_prefix_caching=True,
                seed=42
            )
            
            self.sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=42
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            return False
    
    def load_dataset(self, dataset_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Load dataset from file"""
        try:
            # Support multiple formats
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.json'):
                df = pd.read_json(dataset_path)
            elif dataset_path.endswith('.jsonl'):
                df = pd.read_json(dataset_path, lines=True)
            elif dataset_path.endswith('.parquet'):
                df = pd.read_parquet(dataset_path)
            else:
                raise ValueError(f"Unsupported file format: {dataset_path}")
            
            if max_rows and len(df) > max_rows:
                df = df.head(max_rows)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return pd.DataFrame()
    
    def format_data_fields(self, row: pd.Series, max_length: int = 1500) -> str:
        """Format data fields from DataFrame row"""
        fields = {}
        for col, val in row.items():
            if pd.notna(val):
                str_val = str(val)
                if len(str_val) > max_length:
                    str_val = str_val[:max_length] + "..."
                fields[col] = str_val
        
        return json.dumps(fields, indent=2)
    
    def create_prompt(self, template: Dict[str, str], data_fields: str, custom_query: str = None) -> str:
        """Create prompt from template and data"""
        if custom_query and template.get('type') == 'custom':
            return template['prompt'].format(custom_query=custom_query, data_fields=data_fields)
        else:
            return template['prompt'].format(data_fields=data_fields)
    
    def run_inference(self, prompts: List[str]) -> tuple:
        """Run inference and return results with timing"""
        if not self.llm:
            raise ValueError("Model not initialized")
        
        start_time = time.time()
        results = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        inference_time = time.time() - start_time
        
        # Extract generated text
        outputs = []
        for result in results:
            if result.outputs:
                generated_text = result.outputs[0].text.strip()
                outputs.append(generated_text)
            else:
                outputs.append("")
        
        return outputs, inference_time
    
    def run_simple_experiment(self, 
                             dataset_path: str, 
                             query_key: str,
                             custom_query: str = None,
                             max_rows: int = None) -> Dict[str, Any]:
        """Run the simple inference experiment"""
        
        # Load dataset
        df = self.load_dataset(dataset_path, max_rows=max_rows)
        if df.empty:
            raise ValueError("Failed to load dataset or dataset is empty")
        
        # Get query template
        if query_key not in QUERY_TEMPLATES:
            if custom_query:
                template = {"prompt": custom_query + "\n\nData: {data_fields}\n\nResponse:", "type": "custom"}
            else:
                raise ValueError(f"Query key '{query_key}' not found and no custom query provided")
        else:
            template = QUERY_TEMPLATES[query_key]
        
        # Create prompts
        prompts = []
        for idx, row in df.iterrows():
            data_fields = self.format_data_fields(row)
            prompt = self.create_prompt(template, data_fields, custom_query)
            prompts.append(prompt)
        
        # Run inference
        results, inference_time = self.run_inference(prompts)
        
        # Return minimal results focused on timing
        return {
            'inference_time_seconds': inference_time,
            'dataset_path': dataset_path,
            'query_key': query_key,
            'custom_query': custom_query,
            'total_rows_processed': len(df),
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'avg_time_per_row': inference_time / len(df) if len(df) > 0 else 0
        }

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Simple LLM Inference with Timing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_inference.py movie_data.csv movie_analysis
  python simple_inference.py data.csv custom --custom-query "Analyze this:" --max-rows 50
  python simple_inference.py data.csv summarize --output results.json
        """
    )
    
    parser.add_argument('dataset', help='Path to dataset file (CSV, JSON, JSONL, Parquet)')
    parser.add_argument('query_key', help='Query template key or "custom" for custom query', 
                       choices=list(QUERY_TEMPLATES.keys()))
    parser.add_argument('--custom-query', help='Custom query text (required if query_key is "custom")')
    parser.add_argument('--max-rows', type=int, help='Maximum number of rows to process')
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf', 
                       help='Model name or path (default: meta-llama/Llama-2-7b-hf)')
    parser.add_argument('--output', help='Output JSON file path (default: print to stdout)')
    parser.add_argument('--max-tokens', type=int, default=256, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.1, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p sampling parameter')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.query_key == 'custom' and not args.custom_query:
        parser.error("--custom-query is required when query_key is 'custom'")
    
    if not os.path.exists(args.dataset):
        parser.error(f"Dataset file not found: {args.dataset}")
    
    try:
        # Initialize inference engine
        inference = SimpleLLMInference(model_name=args.model)
        
        # Initialize model
        print(f"Initializing model: {args.model}...", file=sys.stderr)
        if not inference.initialize_model(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        ):
            print("Failed to initialize model", file=sys.stderr)
            sys.exit(1)
        
        print(f"Loading dataset: {args.dataset}...", file=sys.stderr)
        print(f"Running inference with query: {args.query_key}...", file=sys.stderr)
        
        # Run experiment
        results = inference.run_simple_experiment(
            dataset_path=args.dataset,
            query_key=args.query_key,
            custom_query=args.custom_query,
            max_rows=args.max_rows
        )
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}", file=sys.stderr)
        else:
            print(json.dumps(results, indent=2))
            
        print(f"Completed in {results['inference_time_seconds']:.2f} seconds", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
