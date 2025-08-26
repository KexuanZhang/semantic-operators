import argparse
import json
import os
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Vector
import pandas as pd

# Prompt template
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


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON or CSV file"""
    if dataset_path.endswith('.json'):
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    elif dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
        return df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}")


def format_prompt(prompt_template: str, data_row: Dict[str, Any]) -> str:
    """Format the prompt with data from a row"""
    if prompt_template in QUERY_TEMPLATES:
        query_info = QUERY_TEMPLATES[prompt_template]
        query = query_info["prompt"]
        
        # Convert data row to formatted fields string
        fields_str = json.dumps(data_row, indent=2)
        
        # Use system prompt template
        formatted_prompt = SYSTEM_PROMPT.format(
            QUERY=query,
            fields=fields_str
        )
        return formatted_prompt
    else:
        # Custom prompt - format with data row fields
        try:
            return prompt_template.format(**data_row)
        except KeyError as e:
            print(f"Warning: Key {e} not found in data row, using raw prompt")
            return prompt_template


def run_inference(llm: LLM, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
    """Run inference on the prompts and return generated texts"""
    outputs = llm.generate(prompts, sampling_params)
    results = []
    
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        results.append(generated_text)
    
    return results


def print_metrics(llm: LLM):
    """Print all metrics from the LLM"""
    print("\n" + "="*80)
    print("METRICS")
    print("="*80)
    
    for metric in llm.get_metrics():
        if isinstance(metric, Gauge):
            print(f"{metric.name} (gauge) = {metric.value}")
        elif isinstance(metric, Counter):
            print(f"{metric.name} (counter) = {metric.value}")
        elif isinstance(metric, Vector):
            print(f"{metric.name} (vector) = {metric.values}")
        elif isinstance(metric, Histogram):
            print(f"{metric.name} (histogram)")
            print(f"    sum = {metric.sum}")
            print(f"    count = {metric.count}")
            for bucket_le, value in metric.buckets.items():
                print(f"    {bucket_le} = {value}")


def main():
    parser = argparse.ArgumentParser(description="Run LLM inference experiment with metrics")
    parser.add_argument("--prompt", type=str, required=True, 
                       help="Prompt template name from QUERY_TEMPLATES or custom prompt string")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to dataset file (JSON or CSV)")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model or model name")
    parser.add_argument("--gpu_devices", type=str, default="0",
                       help="Comma-separated GPU device IDs (e.g., '0,1,2')")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="Maximum tokens to generate")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Set GPU devices
    if args.gpu_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
        print(f"Using GPU devices: {args.gpu_devices}")
    
    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    dataset = load_dataset(args.dataset_path)
    print(f"Loaded {len(dataset)} rows")
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    # Initialize LLM
    print(f"Initializing LLM with model: {args.model_path}")
    llm_kwargs = {
        "model": args.model_path,
        "disable_log_stats": False,
    }
    
    if args.batch_size:
        llm_kwargs["max_num_batched_tokens"] = args.batch_size
    
    llm = LLM(**llm_kwargs)
    
    # Format prompts for each row
    print("Formatting prompts...")
    prompts = []
    for i, row in enumerate(dataset):
        formatted_prompt = format_prompt(args.prompt, row)
        prompts.append(formatted_prompt)
        if i < 3:  # Show first 3 prompts as examples
            print(f"\nExample prompt {i+1}:")
            print("-" * 50)
            print(formatted_prompt[:200] + "..." if len(formatted_prompt) > 200 else formatted_prompt)
    
    print(f"\nRunning inference on {len(prompts)} prompts...")
    
    # Run inference
    results = run_inference(llm, prompts, sampling_params)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    for i, (prompt, result) in enumerate(zip(prompts[:5], results[:5])):  # Show first 5 results
        print(f"\nResult {i+1}:")
        print("-" * 50)
        print(f"Prompt: {prompt[:100]}...")
        print(f"Generated: {result}")
    
    if len(results) > 5:
        print(f"\n... and {len(results) - 5} more results")
    
    # Print metrics
    print_metrics(llm)
    
    # Save results to file
    output_file = "experiment_results.json"
    output_data = {
        "experiment_config": {
            "prompt": args.prompt,
            "dataset_path": args.dataset_path,
            "model_path": args.model_path,
            "gpu_devices": args.gpu_devices,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "num_samples": len(results)
        },
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()