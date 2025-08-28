#!/usr/bin/env python3
"""
Enhanced LLM Query Experiment Script with Local vLLM Stats Logging
Uses local modified vLLM with detailed stats logging for offline inference experiments

*** ENHANCED WITH DETAILED STATS LOGGING ***
- Uses local modified vLLM with KV cache and prefix hit rate logging
- Comprehensive resource monitoring during inference
- Detailed performance metrics collection
- JSON export of all stats for analysis
"""

import argparse
import pandas as pd
import json
import time
import os
import sys
import torch
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import local vLLM setup - use a more flexible approach to find the module
try:
    # First try the direct import (if script is run from the correct directory)
    logger.info("Trying direct import of use_local_vllm...")
    try:
        from use_local_vllm import initialize_experiment_with_local_vllm, create_enhanced_llm
    except ImportError:
        # If that fails, try to find the module by adding potential paths
        logger.info("Direct import failed, trying to locate module...")
        
        # Try to find the module in parent directories
        current_dir = os.path.dirname(os.path.abspath(__file__))
        potential_paths = [
            # Try current directory
            current_dir,
            # Try one level up
            os.path.dirname(current_dir),
            # Try two levels up
            os.path.dirname(os.path.dirname(current_dir)),
            # Try hardcoded paths for different environments
            '/home/data/so/semantic-operators/ggr-experiment-pipeline',
            '/Users/zhang/Desktop/huawei/so1/semantic-operators/ggr-experiment-pipeline'
        ]
        
        module_found = False
        for path in potential_paths:
            if os.path.exists(os.path.join(path, 'use_local_vllm.py')):
                logger.info(f"Found use_local_vllm.py in: {path}")
                sys.path.insert(0, path)
                from use_local_vllm import initialize_experiment_with_local_vllm, create_enhanced_llm
                module_found = True
                break
        
        if not module_found:
            raise ImportError("Could not locate use_local_vllm.py in any of the expected paths")

# Initialize local vLLM
logger.info("Initializing local vLLM with stats logging...")
initialize_experiment_with_local_vllm()

# Now import vLLM components
from vllm import SamplingParams
from vllm.offline_llm_with_stats import OfflineLLMWithStats

# Query templates (same as your original)
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

def prepare_prompt(query_template: str, row: pd.Series) -> str:
    """Prepare a prompt based on the query template and data row."""
    template_info = QUERY_TEMPLATES[query_template]
    prompt_template = template_info["prompt"]
    
    # Create field data from the row
    fields = {}
    for column in row.index:
        if pd.notna(row[column]):
            fields[column] = str(row[column])
    
    # Add diagnostic logging for empty fields
    if not fields:
        logger.warning(f"No valid fields found in row: {row.to_dict()}")
    
    # Format the prompt
    try:
        # Try to format with named placeholders first
        formatted_prompt = prompt_template.format(**fields)
    except KeyError as e:
        # Log the missing keys
        logger.warning(f"KeyError in prompt formatting: {e}. Available keys: {list(fields.keys())}")
        # If named placeholders fail, append fields as JSON
        fields_json = json.dumps(fields, indent=2)
        formatted_prompt = f"{prompt_template}\n\nData fields:\n{fields_json}"
    
    # Ensure we have actual data in the prompt
    if len(formatted_prompt) <= len(prompt_template) + 10:
        logger.warning("Prompt seems too short, might be missing data. Adding all fields as JSON.")
        fields_json = json.dumps(fields, indent=2)
        formatted_prompt = f"{prompt_template}\n\nData fields:\n{fields_json}"
    
    logger.debug(f"Prepared prompt (length: {len(formatted_prompt)}): {formatted_prompt[:100]}...")
    
    return formatted_prompt

def run_enhanced_experiment(csv_file: str, query_key: str, model_path: str = "Qwen/Qwen1.5-14B",
                          batch_size: int = 10, max_rows: Optional[int] = None, 
                          output_dir: str = "results", gpu_ids: Optional[str] = None, **llm_kwargs):
    """
    Run experiment with enhanced stats logging using local vLLM.
    
    Args:
        csv_file: Path to dataset CSV
        query_key: Query template key
        model_path: Model path or name
        batch_size: Batch size for inference
        max_rows: Maximum number of rows to process
        output_dir: Output directory for results
        gpu_ids: Comma-separated list of GPU IDs to use
        **llm_kwargs: Additional LLM arguments
    """
    # Validate query key
    if query_key not in QUERY_TEMPLATES:
        raise ValueError(f"Unknown query key: {query_key}. Available: {list(QUERY_TEMPLATES.keys())}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset from {csv_file}")
    df = pd.read_csv(csv_file)
    
    if max_rows:
        df = df.head(max_rows)
    
    logger.info(f"Dataset loaded: {len(df)} rows")
    
    # Set GPU devices if specified
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        logger.info(f"Set CUDA_VISIBLE_DEVICES={gpu_ids}")
        
        # Parse GPU IDs and set tensor_parallel_size if not explicitly specified
        gpu_id_list = [int(gpu.strip()) for gpu in gpu_ids.split(',')]
        if 'tensor_parallel_size' not in llm_kwargs:
            tensor_parallel_size = len(gpu_id_list)
            llm_kwargs['tensor_parallel_size'] = tensor_parallel_size
            logger.info(f"Set tensor_parallel_size={tensor_parallel_size} based on number of GPUs")
    
    # Initialize enhanced LLM with stats logging
    logger.info(f"Initializing enhanced LLM with model: {model_path}")
    
    # Set default LLM parameters
    llm_kwargs.setdefault('tensor_parallel_size', 1)
    llm_kwargs.setdefault('max_model_len', 2048)
    llm_kwargs.setdefault('gpu_memory_utilization', 0.9)
    llm_kwargs.setdefault('enable_prefix_caching', True)
    llm_kwargs.setdefault('log_stats_interval', max(1, batch_size // 2))  # Log every few batches
    
    llm = create_enhanced_llm(model_path, **llm_kwargs)
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1,  # Low temperature for consistent results
        top_p=0.9,
        max_tokens=200,
        stop=["</s>", "<|endoftext|>"]
    )
    
    # Process data in batches
    results = []
    all_stats = []
    start_time = time.time()
    
    logger.info(f"Starting inference with batch size {batch_size}")
    
    for batch_idx in range(0, len(df), batch_size):
        batch_end = min(batch_idx + batch_size, len(df))
        batch_df = df.iloc[batch_idx:batch_end]
        
        logger.info(f"Processing batch {batch_idx//batch_size + 1}/{(len(df)-1)//batch_size + 1} "
                   f"(rows {batch_idx+1}-{batch_end})")
        
        # Prepare prompts for this batch
        batch_prompts = []
        for idx, row in batch_df.iterrows():
            prompt = prepare_prompt(query_key, row)
            if not prompt or len(prompt.strip()) < 10:
                logger.warning(f"Row {idx}: Generated an empty or very short prompt")
                # Try to create a backup prompt with raw data
                fields_json = json.dumps(row.to_dict(), indent=2)
                prompt = f"{QUERY_TEMPLATES[query_key]['prompt']}\n\nRAW DATA:\n{fields_json}"
                logger.info(f"Created backup prompt with raw data for row {idx}")
            
            # Log prompt length for diagnostics
            logger.debug(f"Row {idx}: Prompt length = {len(prompt)} characters")
            batch_prompts.append(prompt)
        
        # Run inference with detailed stats logging
        batch_start_time = time.time()
        try:
            outputs = llm.generate(
                batch_prompts, 
                sampling_params, 
                log_detailed_stats=True
            )
            batch_end_time = time.time()
        except Exception as e:
            logger.error(f"Error during batch generation: {e}")
            logger.error(f"First prompt in batch: {batch_prompts[0][:200]}...")
            logger.error(f"Batch size: {len(batch_prompts)}")
            logger.error("This could be due to CUDA errors, model loading issues, or prompt formatting problems")
            raise
        
        # Process results
        for i, output in enumerate(outputs):
            row_idx = batch_idx + i
            generated_text = output.outputs[0].text
            
            # Print each LLM response with current stats
            print(f"\n{'='*50}")
            print(f"Response {row_idx+1}/{len(df)}:")
            print(f"{'='*50}")
            print(f"{generated_text}")
            print(f"{'='*50}")
            
            # Get and print current stats
            current_batch_stats = llm.get_current_stats()
            print(f"Current Stats:")
            if 'engine_kv_cache_usage' in current_batch_stats:
                print(f"• KV Cache Usage: {current_batch_stats['engine_kv_cache_usage']:.2%}")
            if 'engine_prefix_cache_stats_hits' in current_batch_stats and 'engine_prefix_cache_stats_queries' in current_batch_stats:
                hits = current_batch_stats['engine_prefix_cache_stats_hits']
                queries = current_batch_stats['engine_prefix_cache_stats_queries']
                hit_rate = (hits / queries * 100) if queries > 0 else 0
                print(f"• Prefix Cache Hit Rate: {hit_rate:.2f}% ({hits}/{queries})")
            print(f"• Tokens - Prompt: {len(output.prompt_token_ids)}, Generated: {len(output.outputs[0].token_ids)}")
            print(f"{'='*50}\n")
            
            result = {
                'row_id': row_idx,
                'query_type': query_key,
                'prompt': batch_prompts[i][:200] + "..." if len(batch_prompts[i]) > 200 else batch_prompts[i],
                'generated_text': generated_text,
                'finish_reason': str(output.outputs[0].finish_reason),
                'prompt_tokens': len(output.prompt_token_ids),
                'generated_tokens': len(output.outputs[0].token_ids),
                'batch_idx': batch_idx // batch_size,
                'inference_time': batch_end_time - batch_start_time
            }
            results.append(result)
        
        # Collect current stats
        current_stats = llm.get_current_stats()
        current_stats.update({
            'batch_idx': batch_idx // batch_size,
            'timestamp': datetime.now().isoformat(),
            'processed_rows': batch_end,
            'total_rows': len(df)
        })
        all_stats.append(current_stats)
        
        logger.info(f"Batch {batch_idx//batch_size + 1} completed in {batch_end_time - batch_start_time:.2f}s")
    
    total_time = time.time() - start_time
    
    # Get final comprehensive stats
    final_stats = llm.get_current_stats()
    final_stats.update({
        'total_inference_time': total_time,
        'total_rows_processed': len(df),
        'average_time_per_row': total_time / len(df),
        'query_type': query_key,
        'model_path': model_path,
        'batch_size': batch_size,
        'completion_time': datetime.now().isoformat()
    })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save only LLM responses to CSV
    llm_responses = [result['generated_text'] for result in results]
    results_df = pd.DataFrame({'llm_response': llm_responses})
    results_file = os.path.join(output_dir, f"{query_key}_results_{timestamp}.csv")
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to: {results_file} (contains only LLM response column)")
    
    # Save detailed stats
    stats_file = os.path.join(output_dir, f"{query_key}_detailed_stats_{timestamp}.json")
    detailed_stats = {
        'final_stats': final_stats,
        'batch_stats': all_stats,
        'experiment_config': {
            'csv_file': csv_file,
            'query_key': query_key,
            'model_path': model_path,
            'batch_size': batch_size,
            'max_rows': max_rows,
            'llm_kwargs': llm_kwargs
        }
    }
    
    with open(stats_file, 'w') as f:
        json.dump(detailed_stats, f, indent=2)
    logger.info(f"Detailed stats saved to: {stats_file}")
    
    # Log summary
    logger.info("="*60)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("="*60)
    logger.info(f"Query Type: {query_key}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Rows Processed: {len(df)}")
    logger.info(f"Total Time: {total_time:.2f}s")
    logger.info(f"Avg Time per Row: {total_time/len(df):.3f}s")
    logger.info(f"Results: {results_file}")
    logger.info(f"Stats: {stats_file}")
    
    # Print key stats
    if 'engine_kv_cache_usage' in final_stats:
        logger.info(f"Final KV Cache Usage: {final_stats['engine_kv_cache_usage']:.2%}")
    
    if 'engine_prefix_cache_stats_hits' in final_stats and 'engine_prefix_cache_stats_queries' in final_stats:
        hits = final_stats['engine_prefix_cache_stats_hits']
        queries = final_stats['engine_prefix_cache_stats_queries']
        if queries > 0:
            hit_rate = hits / queries * 100
            logger.info(f"Prefix Cache Hit Rate: {hit_rate:.2f}% ({hits}/{queries})")
    
    logger.info("="*60)
    
    return results_df, detailed_stats

def main():
    """Main function to run enhanced experiments."""
    parser = argparse.ArgumentParser(description='Enhanced LLM Query Experiment with Stats Logging')
    parser.add_argument('csv_file', help='Path to the CSV dataset file')
    parser.add_argument('query_key', choices=list(QUERY_TEMPLATES.keys()), 
                       help='Query template key')
    parser.add_argument('--model', default='Qwen/Qwen1.5-14B', 
                       help='Model path or name (default: Qwen/Qwen1.5-14B)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for inference (default: 10)')
    parser.add_argument('--max-rows', type=int, default=None,
                       help='Maximum number of rows to process (default: all)')
    parser.add_argument('--output-dir', default='results_enhanced',
                       help='Output directory for results (default: results_enhanced)')
    parser.add_argument('--max-model-len', type=int, default=2048,
                       help='Maximum model length (default: 2048)')
    parser.add_argument('--gpu-memory', type=float, default=0.7,
                       help='GPU memory utilization (default: 0.7)')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                       help='Tensor parallel size (default: 1)')
    parser.add_argument('--gpus', type=str, default=None,
                       help='Comma-separated list of GPU IDs to use (e.g., "6,7")')
    parser.add_argument('--log-stats', action='store_true',
                       help='Enable detailed stats logging (default)')
    parser.add_argument('--disable-log-stats', action='store_true',
                       help='Disable detailed stats logging')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='INFO', help='Set the logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Set logging level
    logging_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(logging_level)
    logger.info(f"Set logging level to {args.log_level}")
    
    # Check if files exist
    if not os.path.exists(args.csv_file):
        logger.error(f"CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    try:
        # Run enhanced experiment
        results_df, detailed_stats = run_enhanced_experiment(
            csv_file=args.csv_file,
            query_key=args.query_key,
            model_path=args.model,
            batch_size=args.batch_size,
            max_rows=args.max_rows,
            output_dir=args.output_dir,
            gpu_ids=args.gpus,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory,
            tensor_parallel_size=args.tensor_parallel_size,
            log_stats=not args.disable_log_stats,
            disable_log_stats=args.disable_log_stats
        )
        
        logger.info("✅ Enhanced experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
