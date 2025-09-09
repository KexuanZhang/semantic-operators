#!/usr/bin/env python3
# filepath: /Users/zhang/Desktop/huawei/untitled folder 6/semantic-operators/old/llm_inference.py
"""
LLM Inference Script

This script runs LLM inference on a dataset:
1. Loads a dataset from a CSV file
2. Initializes an LLM model (local or from HuggingFace)
3. Runs inference on each row using a specified prompt template
4. Saves the results and statistics

Usage:
    python llm_inference.py --dataset path/to/dataset.csv --model model_name [--gpu_ids "0,1"] [--tp_size 2]
"""

import os
import pandas as pd
import argparse
import time
import json
import torch
import datetime
from vllm import LLM, SamplingParams
import torch.distributed as dist

def setup_directories(timestamp):
    """Create result directory with timestamp"""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference_results")
    result_dir = os.path.join(base_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def initialize_llm(model_name, tensor_parallel_size=None, tokenizer_name=None, 
                max_model_len=None, gpu_memory_utilization=None, gpu_ids=None):
    """Initialize the LLM with vLLM
    
    Args:
        model_name (str): HuggingFace model name or local path to model
        tensor_parallel_size (int, optional): Number of GPUs to use for tensor parallelism
        tokenizer_name (str, optional): Name or path of the tokenizer (useful for local models)
        max_model_len (int, optional): Maximum model context length
        gpu_memory_utilization (float, optional): Fraction of GPU memory to use (0.0-1.0)
        gpu_ids (str, optional): Comma-separated GPU IDs to use (e.g., "0,1" or "6,7")
    """
    # Set environment variable for cached outputs
    os.environ["VLLM_USE_CACHED_OUTPUTS"] = "True"
    
    # Set specific GPU devices if specified
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        print(f"Using specific GPU IDs: {gpu_ids}")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Check if model_name is a local path
    is_local = os.path.exists(model_name)
    model_source = "local path" if is_local else "Hugging Face Hub"
    print(f"Initializing LLM from {model_source}: {model_name}")
    
    # Default parameters
    llm_params = {
        "model": model_name,
        "trust_remote_code": True,
    }
    
    # Add optional parameters if provided
    if tensor_parallel_size is not None:
        llm_params["tensor_parallel_size"] = tensor_parallel_size
        print(f"Using tensor parallelism with {tensor_parallel_size} GPUs")
    
    if tokenizer_name is not None:
        llm_params["tokenizer"] = tokenizer_name
        print(f"Using custom tokenizer: {tokenizer_name}")
    
    if max_model_len is not None:
        llm_params["max_model_len"] = max_model_len
        print(f"Using custom context length: {max_model_len}")
    
    if gpu_memory_utilization is not None:
        llm_params["gpu_memory_utilization"] = gpu_memory_utilization
        print(f"Using GPU memory utilization: {gpu_memory_utilization}")
    
    try:
        llm = LLM(**llm_params)
        print(f"Model {model_name} loaded successfully")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Attempting to fallback to TinyLlama model...")
        llm_params["model"] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        # Remove tokenizer param for fallback model if it was set
        if "tokenizer" in llm_params:
            del llm_params["tokenizer"]
        llm = LLM(**llm_params)
        print("Fallback model loaded successfully")
    
    # Define sampling parameters based on model type
    if model_name:
        model_name_lower = model_name.lower()
        
        # For smaller models, use more conservative parameters
        if any(name in model_name_lower for name in ["tiny", "small", "base"]):
            sampling_params = SamplingParams(
                temperature=0.1,  # Lower temperature for more predictable outputs
                top_p=0.9,
                max_tokens=100
            )
        else:
            # For larger models, we can use slightly higher temperature
            sampling_params = SamplingParams(
                temperature=0.3,
                top_p=0.95,
                max_tokens=200
            )
    else:
        # Default sampling parameters
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=150
        )
    
    print(f"Initialized sampling parameters: temp={sampling_params.temperature}, top_p={sampling_params.top_p}, max_tokens={sampling_params.max_tokens}")
    
    return llm, sampling_params

def llm_inference(llm, sampling_params, prompt, model_name=None):
    """Run inference with the LLM and return the result
    
    Args:
        llm: The vLLM model instance
        sampling_params: SamplingParams for generation
        prompt: The text prompt to send to the model
        model_name: Name/path of the model (used for format detection)
    """
    # Format prompt based on model type (instruction formatting)
    if model_name:
        model_name_lower = model_name.lower()
        
        # Qwen models
        if "qwen" in model_name_lower:
            formatted_prompt = f"<|im_start|>system\nYou are a helpful assistant that provides clear, concise, and accurate answers.\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
        
        # LLaMA family models (LLaMA, Mistral, Vicuna, etc)
        elif any(name in model_name_lower for name in ["llama", "mistral", "vicuna"]):
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # ChatGLM models
        elif "chatglm" in model_name_lower:
            formatted_prompt = f"[gMASK]system\nYou are a helpful assistant that provides accurate answers.\n\nuser\n{prompt}\n\nassistant\n"
        
        # Gemma models
        elif "gemma" in model_name_lower:
            formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        # TinyLlama Chat models
        elif "tinyllama" in model_name_lower and "chat" in model_name_lower:
            formatted_prompt = f"<|system|>\nYou are a helpful assistant. Answer directly and concisely.\n<|user|>\n{prompt}\n<|assistant|>"
        
        # Mixtral models
        elif "mixtral" in model_name_lower:
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Yi models
        elif "yi" in model_name_lower:
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # GPT models (typically use straightforward prompting)
        elif any(name in model_name_lower for name in ["gpt", "opt"]):
            formatted_prompt = f"User: {prompt}\nAssistant:"
        
        # BERT/T5/Flan models (use direct prompting)
        elif any(name in model_name_lower for name in ["bert", "t5", "flan"]):
            formatted_prompt = f"{prompt}"
        
        # Generic instruction format for other models
        else:
            formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
            
        print(f"Using prompt format for model type: {model_name_lower}")
    else:
        # Fallback to direct prompt with explicit instruction
        formatted_prompt = f"Answer the following question directly and concisely: {prompt}\n"
        print("Using default prompt format (no model specified)")
        
    # Print the first part of the prompt (for debugging)
    print(f"Formatted prompt preview: {formatted_prompt[:50]}...")
    
    # Adjust sampling params for better, more controlled outputs
    adjusted_params = SamplingParams(
        temperature=0.1,  # Lower temperature for more deterministic responses
        top_p=0.9,
        max_tokens=200,   # Increased response length for more complete answers
        stop=["<|im_end|>", "</s>", "<|endoftext|>", "<|end|>", "<|user|>"]  # Stop tokens for various models
    )
    
    # Generate the response
    try:
        # Use the adjusted parameters instead of the passed sampling_params
        print(f"\n----- INFERENCE REQUEST -----")
        print(f"Model: {model_name if model_name else 'Unknown'}")
        print(f"Prompt length: {len(formatted_prompt)} chars")
        print(f"Prompt preview: {formatted_prompt[:100]}...\n")
        
        # Run the actual inference
        output = llm.generate(formatted_prompt, adjusted_params)
        
        if not output or len(output) == 0 or len(output[0].outputs) == 0:
            print("ERROR: Model returned empty output")
            return "Error: Model returned empty output. Please try again with different parameters."
        
        response = output[0].outputs[0].text
        print(f"----- RAW RESPONSE -----\n{response[:100]}...\n")
        
        # Clean up response based on model type
        if model_name:
            model_name_lower = model_name.lower()
            
            # Clean Qwen responses
            if "qwen" in model_name_lower and "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0].strip()
            
            # Clean Llama responses
            elif any(name in model_name_lower for name in ["llama", "mistral", "vicuna"]):
                # Sometimes responses include the instruction prefix
                if "[/INST]" in response:
                    parts = response.split("[/INST]")
                    if len(parts) > 1:
                        response = parts[1].strip()
            
            # Clean TinyLlama responses
            elif "tinyllama" in model_name_lower:
                if "<|assistant|>" in response:
                    response = response.replace("<|assistant|>", "").strip()
                elif "<|user|>" in response:
                    # If the model starts generating a new user turn, cut it off
                    response = response.split("<|user|>")[0].strip()
                    
            # Clean ChatGLM responses
            elif "chatglm" in model_name_lower and "assistant" in response:
                try:
                    response = response.split("assistant\n", 1)[1].strip()
                except IndexError:
                    pass
            
            # Check for empty response after cleaning
            if not response or response.isspace():
                print("WARNING: Response was empty after cleaning")
                return "Error: Model returned an empty response after formatting."
        
        print(f"----- CLEANED RESPONSE -----\n{response[:100]}...\n")
        return response.strip()
    except Exception as e:
        error_msg = f"Error during inference: {e}"
        print(error_msg)
        
        # Attempt to recover with a simpler prompt if there was an error
        try:
            print("Attempting recovery with simpler prompt...")
            simple_prompt = f"Answer briefly: {prompt}"
            simple_output = llm.generate(simple_prompt, adjusted_params)
            if len(simple_output) > 0 and len(simple_output[0].outputs) > 0:
                return f"[RECOVERED RESPONSE] {simple_output[0].outputs[0].text.strip()}"
            else:
                return f"Error generating response: {str(e)}"
        except:
            return f"Error generating response: {str(e)}"

def process_dataset(df, llm, sampling_params, prompt_template, columns_to_include, model_name=None):
    """Process each row in the dataset with LLM inference"""
    results = []
    start_time = time.time()
    total_tokens = 0
    
    # Check if required columns exist in the dataset
    available_columns = df.columns.tolist()
    used_columns = [col for col in columns_to_include if col in available_columns]
    
    if not used_columns:
        print(f"Warning: None of the specified columns {columns_to_include} found in dataset!")
        print(f"Available columns are: {available_columns}")
        print("Using first column as default input")
        used_columns = [available_columns[0]]
        
    print(f"Using columns for inference: {used_columns}")
    
    # Check if prompt template is valid with available columns
    try:
        # Test formatting with dummy values
        test_values = {col: f"test_{col}" for col in used_columns}
        prompt_template.format(**test_values)
    except KeyError as e:
        print(f"Error: Prompt template references column {e} which is not available in the dataset")
        print(f"Available columns: {available_columns}")
        print("Falling back to a simple template using available columns")
        # Create a simple fallback template using the first available column
        prompt_template = f"Analyze this: {{{used_columns[0]}}}"
        print(f"New template: '{prompt_template}'")
    
    for index, row in df.iterrows():
        row_data = {}
        
        # Extract values for specified columns
        input_values = {col: str(row[col]) for col in used_columns}
        
        # Create prompt using template and row values
        try:
            prompt = prompt_template.format(**input_values)
        except KeyError as e:
            print(f"Error formatting prompt at row {index}: {e}")
            # Use a simple fallback prompt with the first column
            prompt = f"Analyze this: {row[used_columns[0]]}"
            
        # Run inference
        inference_start = time.time()
        # Pass the model_name parameter rather than using args.model directly
        response = llm_inference(llm, sampling_params, prompt, model_name)
        inference_end = time.time()
        
        # Calculate tokens (approximate)
        prompt_tokens = len(prompt.split())
        response_tokens = len(response.split())
        total_tokens += prompt_tokens + response_tokens
        
        # Store results
        row_data.update(input_values)
        row_data["llm_response"] = response.strip()
        row_data["inference_time"] = inference_end - inference_start
        row_data["prompt_tokens"] = prompt_tokens
        row_data["response_tokens"] = response_tokens
        
        results.append(row_data)
        
        # Print progress every 10 rows
        if index % 10 == 0:
            print(f"Processed {index} rows. Latest inference time: {row_data['inference_time']:.2f}s")
    
    end_time = time.time()
    
    # Calculate stats
    stats = {
        "total_rows": len(df),
        "total_time": end_time - start_time,
        "avg_time_per_row": (end_time - start_time) / len(df),
        "total_tokens": total_tokens,
        "avg_tokens_per_row": total_tokens / len(df) if len(df) > 0 else 0
    }
    
    return results, stats

def save_results(results, stats, dataset, result_dir, filename_prefix):
    """Save inference results and stats to the specified directory"""
    # Save processed results as JSON
    results_path = os.path.join(result_dir, f"{filename_prefix}_inference_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save stats as JSON
    stats_path = os.path.join(result_dir, f"{filename_prefix}_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save summary as text
    summary_path = os.path.join(result_dir, f"{filename_prefix}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("LLM Inference Summary\n")
        f.write("====================\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset Size: {stats['total_rows']} rows\n")
        f.write(f"Total Processing Time: {stats['total_time']:.2f} seconds\n")
        f.write(f"Average Time per Row: {stats['avg_time_per_row']:.4f} seconds\n")
        f.write(f"Total Tokens Processed: {stats['total_tokens']}\n")
        f.write(f"Average Tokens per Row: {stats['avg_tokens_per_row']:.2f}\n")
        
    print(f"Results saved to {result_dir}")

def main():
    parser = argparse.ArgumentParser(description='Run LLM inference on a dataset.')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset CSV file.')
    
    # LLM configuration
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                        help='HuggingFace model name or path to local model directory.')
    parser.add_argument('--tp_size', type=int, default=None,
                        help='Tensor parallel size (number of GPUs to use).')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='Specific GPU IDs to use, comma-separated (e.g., "0,1" or "6,7").')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='Optional tokenizer name or path (useful for local models).')
    parser.add_argument('--max_model_len', type=int, default=None,
                        help='Maximum model context length.')
    parser.add_argument('--gpu_memory', type=float, default=None,
                        help='Fraction of GPU memory to use (0.0-1.0).')
    parser.add_argument('--prompt_template', type=str, 
                        default='Analyze the following text: {text_content}. Is this a positive or negative text?',
                        help='Prompt template for LLM inference.')
    parser.add_argument('--include_columns', type=str, nargs='+', 
                        default=None, help='Columns to include in the prompt template.')
    
    # Execution options
    parser.add_argument('--max_rows', type=int, default=None, 
                        help='Maximum number of rows to process (for testing).')
    parser.add_argument('--output_prefix', type=str, default='dataset',
                        help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Create timestamped result directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = setup_directories(timestamp)
    
    # Start timing
    start_time = time.time()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    dataset = pd.read_csv(args.dataset)
    
    # Get dataset filename without extension for output naming
    filename_prefix = args.output_prefix
    if filename_prefix == 'dataset':
        filename_prefix = os.path.splitext(os.path.basename(args.dataset))[0]
    
    # Apply max rows limit if specified
    if args.max_rows is not None:
        dataset = dataset.head(args.max_rows)
        print(f"Limited dataset to {args.max_rows} rows")
    
    # If no columns specified, use all columns
    if args.include_columns is None:
        # Try to find content column by name patterns
        content_columns = [col for col in dataset.columns if any(name in col.lower() for name in 
                          ['content', 'text', 'review', 'description', 'comment', 'body'])]
        if content_columns:
            args.include_columns = content_columns
            print(f"Auto-detected content columns: {content_columns}")
        else:
            args.include_columns = dataset.columns.tolist()
            print(f"Using all columns for inference")
    
    # Initialize LLM
    llm, sampling_params = initialize_llm(
        model_name=args.model, 
        tensor_parallel_size=args.tp_size,
        tokenizer_name=args.tokenizer,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory,
        gpu_ids=args.gpu_ids
    )
    
    # Process dataset with LLM inference
    print("Running LLM inference on dataset...")
    results, stats = process_dataset(
        dataset,
        llm,
        sampling_params,
        args.prompt_template,
        args.include_columns,
        args.model  # Pass the model name to properly format prompts
    )
    
    # End timing and update stats
    end_time = time.time()
    total_time = end_time - start_time
    stats["total_experiment_time"] = total_time
    
    print(f"Total experiment time: {total_time:.2f} seconds")
    
    # Save results
    save_results(results, stats, dataset, result_dir, filename_prefix)
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    
    print(f"All results saved to directory: {result_dir}")

if __name__ == "__main__":
    main()
