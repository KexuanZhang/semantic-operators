#!/usr/bin/env python3
"""
LLM Inference Script with KVTuner Integration

This script runs LLM inference on a dataset with KVTuner quantization:
1. Loads a dataset from a CSV file
2. Initializes an LLM model with KVTuner quantized cache
3. Runs inference on each row using a specified prompt template
4. Saves the results and statistics

Usage:
    python llm_inference.py --dataset path/to/dataset.csv --model model_name --kvtuner_scheme pertoken [--gpu_ids "0,1"]
"""

import os
import sys
import pandas as pd
import argparse
import time
import json
import torch
import datetime
import yaml
from tqdm import tqdm
from pathlib import Path

# Add KVTuner to Python path
kvtuner_path = "/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner"
sys.path.insert(0, kvtuner_path)

try:
    from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig, FlexibleVanillaQuantizedCache
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure KVTuner is properly installed and accessible")
    sys.exit(1)

def setup_directories(timestamp):
    """Create result directory with timestamp"""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference_results")
    result_dir = os.path.join(base_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def get_model_basename(model_path):
    """Extract model basename for config file lookup"""
    if os.path.exists(model_path):
        # Local path - get directory name
        return os.path.basename(os.path.normpath(model_path))
    else:
        # HuggingFace model name - replace slashes
        return model_path.replace("/", "_")

def load_kvtuner_config(model_name, scheme, kvtuner_dir):
    """Load KVTuner configuration from calibration presets"""
    model_basename = get_model_basename(model_name)
    
    # Look for exact match first
    config_filename = f"{model_basename}_{scheme}_KVTuner4_0.yaml"
    config_path = os.path.join(kvtuner_dir, "calibration_presets", config_filename)
    
    if os.path.exists(config_path):
        print(f"Loading KVTuner config: {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    # Try alternative naming patterns
    alternative_names = [
        f"{model_basename}_{scheme}_KVTuner6_0.yaml",
        f"{model_basename}_{scheme}_KVTuner4_1.yaml",
        f"{model_basename}_{scheme}_KVTuner6_1.yaml"
    ]
    
    for alt_name in alternative_names:
        alt_path = os.path.join(kvtuner_dir, "calibration_presets", alt_name)
        if os.path.exists(alt_path):
            print(f"Loading KVTuner config: {alt_path}")
            with open(alt_path, 'r') as f:
                return yaml.safe_load(f)
    
    print(f"Warning: No KVTuner config found for {model_basename} with {scheme} scheme")
    print(f"Searched for: {config_filename}")
    return None

def create_kvtuner_cache(model_name, scheme, kvtuner_dir):
    """Create KVTuner quantized cache"""
    
    # Load configuration
    per_layer_config = load_kvtuner_config(model_name, scheme, kvtuner_dir)
    
    # Set axis configuration based on scheme
    if scheme == "kivi":
        axis_key = 1  # Per-channel for keys in KiVi
        axis_value = 0  # Per-token for values in KiVi
        q_group_size = 32
        residual_length = 32
    else:  # pertoken
        axis_key = 0  # Per-token for keys
        axis_value = 0  # Per-token for values
        q_group_size = -1
        residual_length = 0
    
    # Create cache configuration
    cache_config = FlexibleQuantizedCacheConfig(
        device="cuda",
        per_layer_quant=per_layer_config is not None,
        per_layer_config=per_layer_config,
        asym=True,
        axis_key=axis_key,
        axis_value=axis_value,
        q_group_size=q_group_size,
        residual_length=residual_length,
        compute_dtype=torch.float16
    )
    
    # Create and return the KV cache
    return FlexibleVanillaQuantizedCache(cache_config=cache_config)

def initialize_llm(model_name, kvtuner_scheme, kvtuner_dir, tokenizer_name=None, 
                   max_model_len=None, gpu_ids=None):
    """Initialize the LLM with KVTuner quantized cache
    
    Args:
        model_name (str): HuggingFace model name or local path to model
        kvtuner_scheme (str): KVTuner quantization scheme ('pertoken' or 'kivi')
        kvtuner_dir (str): Path to KVTuner directory
        tokenizer_name (str, optional): Name or path of the tokenizer
        max_model_len (int, optional): Maximum model context length
        gpu_ids (str, optional): Comma-separated GPU IDs to use
    """
    # Set specific GPU devices if specified
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    print(f"Loading model: {model_name}")
    print(f"KVTuner scheme: {kvtuner_scheme}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer_path = tokenizer_name if tokenizer_name else model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=False,
        trust_remote_code=True
    )
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model and tokenizer loaded successfully")
    
    return model, tokenizer

def format_prompt_by_model(model_name, text):
    """Format prompt based on model type"""
    model_name_lower = model_name.lower() if model_name else ""
    
    # Qwen models
    if "qwen" in model_name_lower:
        return f"<|im_start|>system\nYou are a helpful assistant that provides clear, concise, and accurate answers.\n<|im_end|>\n<|im_start|>user\n{text}\n<|im_end|>\n<|im_start|>assistant\n"
    
    # LLaMA family models (LLaMA, Mistral, etc)
    elif any(name in model_name_lower for name in ["llama", "mistral", "vicuna"]):
        return f"<s>[INST] {text} [/INST]"
    
    # ChatGLM models
    elif "chatglm" in model_name_lower:
        return f"[Round 1]\n\n问：{text}\n\n答："
    
    # Gemma models
    elif "gemma" in model_name_lower:
        return f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
    
    # TinyLlama models
    elif "tinyllama" in model_name_lower and "chat" in model_name_lower:
        return f"<|system|>\nYou are a helpful assistant. Answer directly and concisely.\n<|user|>\n{text}\n<|assistant|>"
    
    # Mixtral models
    elif "mixtral" in model_name_lower:
        return f"<s>[INST] {text} [/INST]"
    
    # Yi models
    elif "yi" in model_name_lower:
        return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    
    # GPT models
    elif any(name in model_name_lower for name in ["gpt", "opt"]):
        return f"User: {text}\nAssistant:"
    
    # BERT/T5/Flan models
    elif any(name in model_name_lower for name in ["bert", "t5", "flan"]):
        return f"{text}"
    
    # Generic instruction format
    else:
        return f"### Instruction:\n{text}\n\n### Response:\n"

def clean_response_by_model(model_name, response):
    """Clean response based on model type"""
    model_name_lower = model_name.lower() if model_name else ""
    
    # Qwen models
    if "qwen" in model_name_lower and "<|im_end|>" in response:
        return response.split("<|im_end|>")[0].strip()
    
    # LLaMA family models
    elif any(name in model_name_lower for name in ["llama", "mistral", "vicuna"]):
        if "</s>" in response:
            return response.split("</s>")[0].strip()
    
    # ChatGLM models
    elif "chatglm" in model_name_lower and "assistant" in response:
        try:
            return response.split("assistant")[1].strip()
        except IndexError:
            pass
    
    # TinyLlama models
    elif "tinyllama" in model_name_lower:
        if "<|assistant|>" in response:
            return response.split("<|assistant|>")[-1].strip()
        elif "<|user|>" in response:
            return response.split("<|user|>")[0].strip()
    
    # Yi models
    elif "yi" in model_name_lower and "<|im_end|>" in response:
        return response.split("<|im_end|>")[0].strip()
    
    # Gemma models
    elif "gemma" in model_name_lower and "<end_of_turn>" in response:
        return response.split("<end_of_turn>")[0].strip()
    
    return response.strip()

def llm_inference_kvtuner(model, tokenizer, kv_cache, prompt, model_name=None, max_new_tokens=200):
    """Run inference with KVTuner quantized cache"""
    
    # Format prompt based on model type
    formatted_prompt = format_prompt_by_model(model_name, prompt)
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            past_key_values=kv_cache,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic generation
            temperature=0.1,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Extract generated text (skip prompt)
    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    # Clean response based on model type
    cleaned_response = clean_response_by_model(model_name, generated_text)
    
    # Calculate token counts
    prompt_tokens = inputs.input_ids.shape[1]
    response_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    
    return cleaned_response, prompt_tokens, response_tokens

def process_dataset(df, model, tokenizer, kvtuner_scheme, kvtuner_dir, prompt_template, 
                   columns_to_include, model_name=None, max_new_tokens=200):
    """Process each row in the dataset with KVTuner inference"""
    results = []
    start_time = time.time()
    total_tokens = 0
    total_prompt_tokens = 0
    total_response_tokens = 0
    
    # Check if required columns exist in the dataset
    available_columns = df.columns.tolist()
    used_columns = [col for col in columns_to_include if col in available_columns]
    
    if not used_columns:
        used_columns = [available_columns[0]]
        print(f"No specified columns found, using first column: {used_columns[0]}")
    
    # Check if prompt template is valid with available columns
    try:
        # Test formatting with dummy values
        test_values = {col: f"test_{col}" for col in used_columns}
        prompt_template.format(**test_values)
    except KeyError as e:
        # Create a simple fallback template using the first available column
        prompt_template = f"Analyze this: {{{used_columns[0]}}}"
        print(f"Invalid prompt template, using fallback: {prompt_template}")
    
    print(f"Processing {len(df)} rows with KVTuner ({kvtuner_scheme} scheme)")
    
    # Add progress bar for inference
    with tqdm(total=len(df), desc="Processing queries", unit="queries", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        for index, row in df.iterrows():
            # Create a fresh KV cache for each row to avoid cross-contamination
            kv_cache = create_kvtuner_cache(model_name, kvtuner_scheme, kvtuner_dir)
            
            row_data = {}
            
            # Extract values for specified columns
            input_values = {col: str(row[col]) for col in used_columns}
            
            # Create prompt using template and row values
            try:
                prompt = prompt_template.format(**input_values)
            except KeyError as e:
                # Use a simple fallback prompt with the first column
                prompt = f"Analyze this: {row[used_columns[0]]}"
                
            # Run inference with timing
            inference_start = time.time()
            try:
                response, prompt_tokens, response_tokens = llm_inference_kvtuner(
                    model, tokenizer, kv_cache, prompt, model_name, max_new_tokens
                )
            except Exception as e:
                print(f"Error processing row {index}: {str(e)}")
                response = f"Error: {str(e)}"
                prompt_tokens = 0
                response_tokens = 0
            
            inference_end = time.time()
            
            # Update token counts
            total_prompt_tokens += prompt_tokens
            total_response_tokens += response_tokens
            total_tokens += prompt_tokens + response_tokens
            
            # Store results
            row_data.update(input_values)
            row_data["llm_response"] = response
            row_data["inference_time"] = inference_end - inference_start
            row_data["prompt_tokens"] = prompt_tokens
            row_data["response_tokens"] = response_tokens
            row_data["total_tokens"] = prompt_tokens + response_tokens
            row_data["kvtuner_scheme"] = kvtuner_scheme
            
            results.append(row_data)
            
            # Update progress bar
            pbar.update(1)
    
    end_time = time.time()
    
    # Calculate stats
    stats = {
        "total_rows": len(df),
        "total_time": end_time - start_time,
        "avg_time_per_row": (end_time - start_time) / len(df),
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_response_tokens": total_response_tokens,
        "avg_tokens_per_row": total_tokens / len(df) if len(df) > 0 else 0,
        "avg_prompt_tokens_per_row": total_prompt_tokens / len(df) if len(df) > 0 else 0,
        "avg_response_tokens_per_row": total_response_tokens / len(df) if len(df) > 0 else 0,
        "kvtuner_scheme": kvtuner_scheme,
        "model_name": model_name
    }
    
    return results, stats

def save_results(results, stats, dataset, result_dir, filename_prefix):
    """Save inference results and stats to the specified directory"""
    # Save processed results as JSON
    results_path = os.path.join(result_dir, f"{filename_prefix}_kvtuner_inference_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save stats as JSON
    stats_path = os.path.join(result_dir, f"{filename_prefix}_kvtuner_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save summary as text
    summary_path = os.path.join(result_dir, f"{filename_prefix}_kvtuner_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("LLM Inference with KVTuner Summary\n")
        f.write("===================================\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {stats['model_name']}\n")
        f.write(f"KVTuner Scheme: {stats['kvtuner_scheme']}\n")
        f.write(f"Dataset Size: {stats['total_rows']} rows\n")
        f.write(f"Total Processing Time: {stats['total_time']:.2f} seconds\n")
        f.write(f"Average Time per Row: {stats['avg_time_per_row']:.4f} seconds\n")
        f.write(f"Total Tokens Processed: {stats['total_tokens']}\n")
        f.write(f"Average Tokens per Row: {stats['avg_tokens_per_row']:.2f}\n")
        f.write(f"Average Prompt Tokens: {stats['avg_prompt_tokens_per_row']:.2f}\n")
        f.write(f"Average Response Tokens: {stats['avg_response_tokens_per_row']:.2f}\n")
    
    # Save results as CSV for easier analysis
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(result_dir, f"{filename_prefix}_kvtuner_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    print(f"Results saved to: {result_dir}")
    print(f"  - JSON: {results_path}")
    print(f"  - CSV: {csv_path}")
    print(f"  - Stats: {stats_path}")
    print(f"  - Summary: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Run LLM inference with KVTuner quantization on a dataset.')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset CSV file.')
    
    # LLM configuration
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model name or path to local model directory.')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='Optional tokenizer name or path (useful for local models).')
    parser.add_argument('--max_model_len', type=int, default=None,
                        help='Maximum model context length.')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='Specific GPU IDs to use, comma-separated (e.g., "0,1" or "6,7").')
    
    # KVTuner configuration
    parser.add_argument('--kvtuner_scheme', type=str, default='pertoken', 
                        choices=['pertoken', 'kivi'],
                        help='KVTuner quantization scheme (default: pertoken).')
    parser.add_argument('--kvtuner_dir', type=str, 
                        default='/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner',
                        help='Path to KVTuner directory.')
    
    # Inference configuration
    parser.add_argument('--prompt_template', type=str, 
                        default='Analyze the following text: {text_content}. Is this a positive or negative text?',
                        help='Prompt template for LLM inference.')
    parser.add_argument('--include_columns', type=str, nargs='+', 
                        default=None, help='Columns to include in the prompt template.')
    parser.add_argument('--max_new_tokens', type=int, default=200,
                        help='Maximum new tokens to generate per response.')
    
    # Execution options
    parser.add_argument('--max_rows', type=int, default=None, 
                        help='Maximum number of rows to process (for testing).')
    parser.add_argument('--output_prefix', type=str, default='dataset',
                        help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Validate KVTuner directory
    if not os.path.exists(args.kvtuner_dir):
        print(f"Error: KVTuner directory not found: {args.kvtuner_dir}")
        sys.exit(1)
    
    # Create timestamped result directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = setup_directories(timestamp)
    
    # Start timing
    start_time = time.time()
    
    print("="*60)
    print("LLM Inference with KVTuner")
    print("="*60)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = pd.read_csv(args.dataset)
    
    # Get dataset filename without extension for output naming
    filename_prefix = args.output_prefix
    if filename_prefix == 'dataset':
        filename_prefix = os.path.splitext(os.path.basename(args.dataset))[0]
    
    # Apply max rows limit if specified
    if args.max_rows is not None:
        dataset = dataset.head(args.max_rows)
        print(f"Limited dataset to {args.max_rows} rows")
    
    print(f"Dataset loaded: {len(dataset)} rows, {len(dataset.columns)} columns")
    
    # If no columns specified, try to find content column by name patterns
    if args.include_columns is None:
        content_columns = [col for col in dataset.columns if any(name in col.lower() for name in 
                          ['content', 'text', 'review', 'description', 'comment', 'body'])]
        if content_columns:
            args.include_columns = [content_columns[0]]
            print(f"Auto-selected text column: {content_columns[0]}")
        else:
            args.include_columns = [dataset.columns[0]]
            print(f"Using first column as text input: {dataset.columns[0]}")
    
    # Initialize LLM with KVTuner
    model, tokenizer = initialize_llm(
        model_name=args.model,
        kvtuner_scheme=args.kvtuner_scheme,
        kvtuner_dir=args.kvtuner_dir,
        tokenizer_name=args.tokenizer,
        max_model_len=args.max_model_len,
        gpu_ids=args.gpu_ids
    )
    
    # Process dataset with KVTuner inference
    results, stats = process_dataset(
        dataset,
        model,
        tokenizer,
        args.kvtuner_scheme,
        args.kvtuner_dir,
        args.prompt_template,
        args.include_columns,
        args.model,
        args.max_new_tokens
    )
    
    # End timing and update stats
    end_time = time.time()
    total_time = end_time - start_time
    stats["total_experiment_time"] = total_time
    
    # Save results
    save_results(results, stats, dataset, result_dir, filename_prefix)
    
    # Print summary
    print("\n" + "="*60)
    print("INFERENCE COMPLETED")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"KVTuner Scheme: {args.kvtuner_scheme}")
    print(f"Processed: {stats['total_rows']} rows")
    print(f"Total Time: {stats['total_time']:.2f}s")
    print(f"Average Time per Row: {stats['avg_time_per_row']:.3f}s")
    print(f"Total Tokens: {stats['total_tokens']}")
    print(f"Results saved to: {result_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
