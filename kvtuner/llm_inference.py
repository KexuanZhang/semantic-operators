#!/usr/bin/env python3
"""
LLM Inference Script with Dual Cache Support

This script runs LLM inference on a dataset with configurable cache modes:
1. Loads a dataset from a CSV file
2. Initializes an LLM model with either KVTuner quantized cache or basic vLLM cache
3. Runs inference on each row using a specified prompt template
4. Saves the results and statistics

Cache Modes:
- kvtuner: Uses KVTuner's mixed precision quantized cache for memory efficiency
- basic: Uses vLLM's default cache implementation

Usage:
    # With KVTuner quantized cache (default)
    python llm_inference.py --dataset path/to/dataset.csv --model model_name --cache_mode kvtuner --kvtuner_scheme pertoken
    
    # With basic vLLM cache
    python llm_inference.py --dataset path/to/dataset.csv --model model_name --cache_mode basic
    
    # With custom GPU memory settings
    python llm_inference.py --dataset path/to/dataset.csv --model model_name --gpu_memory_utilization 0.6 --gpu_ids "0,1"
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

# Suppress vLLM's internal progress bars and verbose logging
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["VLLM_SHOW_PROGRESS_BARS"] = "0"
os.environ["VLLM_DISABLE_TQDM"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["VLLM_TRACE_FUNCTION"] = "0"

# Additional logging suppression
import logging
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# Disable tqdm globally by monkey-patching
import tqdm
tqdm.tqdm.__init__ = lambda self, *args, **kwargs: None
tqdm.tqdm.update = lambda self, *args, **kwargs: None
tqdm.tqdm.close = lambda self, *args, **kwargs: None
tqdm.tqdm.__enter__ = lambda self: self
tqdm.tqdm.__exit__ = lambda self, *args, **kwargs: None

# Add paths for vLLM and KVTuner
vllm_path = "/Users/zhang/Desktop/huawei/untitled folder 6/vllm"
kvtuner_path = "/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner"
semantic_operators_path = "/Users/zhang/Desktop/huawei/untitled folder 6/semantic-operators"
sys.path.insert(0, vllm_path)
sys.path.insert(0, kvtuner_path)

try:
    # Import vLLM components
    from vllm import LLM, SamplingParams
    print("vLLM imported successfully")
    
    # Test if KVTuner quantization is properly registered
    try:
        from vllm.model_executor.layers.quantization import QuantizationMethods
        if hasattr(QuantizationMethods, '__args__') and "kvtuner" in QuantizationMethods.__args__:
            print("KVTuner quantization method registered successfully")
        else:
            print("Warning: KVTuner quantization method not found in vLLM")
    except Exception as e:
        print(f"Warning: Could not verify KVTuner registration: {e}")
    
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure vLLM is properly installed and accessible")
    print("vLLM path:", vllm_path)
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

def initialize_llm_vllm(model_name, cache_mode='kvtuner', kvtuner_scheme='pertoken', 
                        kvtuner_dir=None, tokenizer_name=None, max_model_len=None, 
                        gpu_ids=None, gpu_memory_utilization=0.75):
    """Initialize the LLM using vLLM with configurable cache mode
    
    Args:
        model_name (str): HuggingFace model name or local path to model
        cache_mode (str): Cache mode - 'kvtuner' for quantized cache, 'basic' for default vLLM cache
        kvtuner_scheme (str): KVTuner quantization scheme ('pertoken' or 'kivi')
        kvtuner_dir (str): Path to KVTuner directory (required for kvtuner mode)
        tokenizer_name (str, optional): Name or path of the tokenizer
        max_model_len (int, optional): Maximum model context length
        gpu_ids (str, optional): Comma-separated GPU IDs to use
        gpu_memory_utilization (float): Fraction of GPU memory to use (0.1 to 1.0)
    """
    # Set specific GPU devices if specified
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        print(f"Using GPU devices: {gpu_ids}")
    
    # Clear GPU memory if torch is available
    try:
        torch.cuda.empty_cache()
        print("GPU memory cleared")
    except Exception as e:
        print(f"Could not clear GPU memory: {e}")
    
    print(f"Loading model with vLLM: {model_name}")
    print(f"Cache mode: {cache_mode}")
    if cache_mode == 'kvtuner':
        print(f"KVTuner scheme: {kvtuner_scheme}")
    
    # Initialize vLLM with base configuration
    llm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": 1,  # Single GPU by default
        "trust_remote_code": True,
        "dtype": "float16",
        "gpu_memory_utilization": gpu_memory_utilization,  # Configurable memory usage
        "enforce_eager": True,  # Disable CUDA graphs to save memory
        "disable_log_stats": True,  # Disable internal logging/stats
    }
    
    # Track temporary config file for cleanup
    temp_config_file = None
    
    # Configure cache mode
    if cache_mode == 'kvtuner':
        if not kvtuner_dir:
            raise ValueError("kvtuner_dir is required when using cache_mode='kvtuner'")
            
        # Get KVTuner config path
        model_basename = get_model_basename(model_name)
        config_filename = f"{model_basename}_{kvtuner_scheme}_KVTuner4_0.yaml"
        kvtuner_config_path = os.path.join(kvtuner_dir, "calibration_presets", config_filename)
        
        # Check if config exists, try alternatives if not
        if not os.path.exists(kvtuner_config_path):
            alternative_names = [
                f"{model_basename}_{kvtuner_scheme}_KVTuner6_0.yaml",
                f"{model_basename}_{kvtuner_scheme}_KVTuner4_1.yaml",
                f"{model_basename}_{kvtuner_scheme}_KVTuner6_1.yaml"
            ]
            
            for alt_name in alternative_names:
                alt_path = os.path.join(kvtuner_dir, "calibration_presets", alt_name)
                if os.path.exists(alt_path):
                    kvtuner_config_path = alt_path
                    break
            else:
                print(f"Warning: No KVTuner config found for {model_basename} with {kvtuner_scheme} scheme")
                print(f"Searched for: {config_filename}")
                print("Falling back to basic cache mode...")
                cache_mode = 'basic'
        
        # Add KVTuner configuration if available
        if cache_mode == 'kvtuner' and os.path.exists(kvtuner_config_path):
            print(f"Using KVTuner config: {kvtuner_config_path}")
            
            # KVTuner integration requires the config file to be in JSON format and 
            # named "kvtuner_config.yaml" in the model directory. However, vLLM's 
            # quantization loading system expects JSON, so we need to convert YAML to JSON.
            import shutil
            import json
            model_dir = model_name if os.path.isdir(model_name) else None
            
            if model_dir and os.access(model_dir, os.W_OK):
                # Load the YAML config and convert to JSON
                try:
                    with open(kvtuner_config_path, 'r') as f:
                        yaml_config = yaml.safe_load(f)
                    
                    # Create both YAML and JSON versions in the model directory
                    target_yaml_path = os.path.join(model_dir, "kvtuner_config.yaml")
                    target_json_path = os.path.join(model_dir, "kvtuner_config.json")
                    temp_config_file = target_yaml_path  # Track for cleanup message
                    
                    # Copy the original YAML file (needed by KVTuner implementation)
                    shutil.copy2(kvtuner_config_path, target_yaml_path)
                    print(f"Copied KVTuner YAML config to: {target_yaml_path}")
                    
                    # Also create a JSON version (for vLLM's quantization loader)
                    with open(target_json_path, 'w') as f:
                        json.dump(yaml_config, f, indent=2)
                    print(f"Created KVTuner JSON config at: {target_json_path}")
                    
                    # Verify files exist and show directory contents
                    print(f"Verifying files in model directory: {model_dir}")
                    yaml_exists = os.path.exists(target_yaml_path)
                    json_exists = os.path.exists(target_json_path)
                    print(f"YAML file exists: {yaml_exists}")
                    print(f"JSON file exists: {json_exists}")
                    
                    # List all config-like files in the directory
                    config_files = [f for f in os.listdir(model_dir) if f.endswith(('.json', '.yaml', '.yml'))]
                    print(f"All config files in model dir: {config_files}")
                    
                    llm_kwargs.update({
                        "quantization": "kvtuner"
                    })
                    
                except Exception as e:
                    print(f"Failed to process KVTuner config: {e}")
                    print("Trying symlink approach...")
                    try:
                        # Try creating symlinks instead
                        target_yaml_path = os.path.join(model_dir, "kvtuner_config.yaml")
                        if os.path.exists(target_yaml_path):
                            os.remove(target_yaml_path)
                        os.symlink(kvtuner_config_path, target_yaml_path)
                        print(f"Created symlink to KVTuner config: {target_yaml_path}")
                        temp_config_file = target_yaml_path
                        
                        # Also try to create JSON version
                        try:
                            with open(kvtuner_config_path, 'r') as f:
                                yaml_config = yaml.safe_load(f)
                            target_json_path = os.path.join(model_dir, "kvtuner_config.json")
                            with open(target_json_path, 'w') as f:
                                json.dump(yaml_config, f, indent=2)
                            print(f"Created KVTuner JSON config at: {target_json_path}")
                        except Exception as json_error:
                            print(f"Could not create JSON version: {json_error}")
                        
                        llm_kwargs.update({
                            "quantization": "kvtuner"
                        })
                    except Exception as e2:
                        print(f"Failed to create symlink: {e2}")
                        print("Falling back to basic cache mode...")
                        cache_mode = 'basic'
                        temp_config_file = None
            else:
                # For HuggingFace models or read-only directories, we can't modify the cache directory easily
                # Fall back to basic cache mode with a warning
                if not model_dir:
                    print("KVTuner requires local model directory to place config file.")
                    print("For HuggingFace models, download the model locally first.")
                else:
                    print(f"Cannot write to model directory: {model_dir}")
                    print("Check permissions or use a local model copy.")
                print("Falling back to basic cache mode...")
                cache_mode = 'basic'
        else:
            print("KVTuner config not found, using basic cache mode")
            cache_mode = 'basic'
    
    if cache_mode == 'basic':
        print("Using default vLLM cache (no quantization)")
        # No additional configuration needed for basic cache
    
    # Add optional parameters
    if max_model_len:
        llm_kwargs["max_model_len"] = max_model_len
    
    if tokenizer_name:
        llm_kwargs["tokenizer"] = tokenizer_name
    
    # Set tensor parallel size based on GPU count
    if gpu_ids:
        gpu_count = len(gpu_ids.split(','))
        if gpu_count > 1:
            llm_kwargs["tensor_parallel_size"] = gpu_count
            print(f"Using tensor parallelism with {gpu_count} GPUs")
    
    try:
        print("Initializing vLLM model...")
        print(f"Model kwargs:")
        for key, value in llm_kwargs.items():
            print(f"  {key}: {value}")
        
        # Create LLM instance
        llm = LLM(**llm_kwargs)
        print("✓ vLLM model loaded successfully")
        
        # Print model info (minimal)
        print(f"  Model: {model_name}")
        print(f"  Cache mode: {cache_mode}")
        if cache_mode == 'kvtuner':
            print(f"  KVTuner scheme: {kvtuner_scheme}")
        print(f"  GPU memory: {gpu_memory_utilization:.1%}")
        
        # Note: We leave the KVTuner config file in place as vLLM may need it during operation
        if temp_config_file:
            print(f"Note: KVTuner config file created at: {temp_config_file}")
                
        return llm
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error initializing vLLM model: {e}")
        
        # Print the full traceback for debugging
        import traceback
        print("Full error traceback:")
        traceback.print_exc()
        
        # Check for common KVTuner integration issues
        if "get_config_filenames() missing 1 required positional argument" in error_msg:
            print("\n" + "="*60)
            print("KVTUNER INTEGRATION ERROR DETECTED")
            print("="*60)
            print("This error indicates the KVTuner integration in vLLM has a method signature issue.")
            print("The get_config_filenames() method signature is inconsistent between base and implementation.")
            print("\nThis issue should be fixed in the current vLLM installation.")
            print("If you're seeing this error, the fix may not have been applied correctly.")
            print("\nTo work around this:")
            print("1. Use --cache_mode basic to bypass KVTuner")
            print("2. Or verify the vLLM quantization configs are properly fixed")
            print("="*60)
            
            if cache_mode == 'kvtuner':
                print("\nAttempting automatic fallback to basic cache mode...")
                try:
                    # Retry with basic cache mode
                    basic_kwargs = {k: v for k, v in llm_kwargs.items() 
                                  if k not in ['quantization', 'kvtuner_config_path', 
                                             'kvtuner_scheme', 'kvtuner_backend']}
                    llm = LLM(**basic_kwargs)
                    print("✓ Successfully initialized with basic cache mode")
                    return llm
                except Exception as fallback_error:
                    print(f"✗ Fallback to basic cache also failed: {fallback_error}")
        
        print("Model kwargs:")
        for key, value in llm_kwargs.items():
            print(f"  {key}: {value}")
        raise

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

def llm_inference_vllm(llm, prompt, model_name=None, max_new_tokens=200):
    """Run inference with vLLM and KVTuner quantization"""
    
    try:
        # Format prompt based on model type
        formatted_prompt = format_prompt_by_model(model_name, prompt)
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=max_new_tokens,
            skip_special_tokens=True
        )
        
        # Additional runtime suppression to prevent progress bars
        import sys
        import io
        import warnings
        from contextlib import redirect_stderr, redirect_stdout
        
        # Suppress warnings and additional verbose output
        warnings.filterwarnings("ignore")
        
        # Generate response using vLLM with maximum output suppression
        # Temporarily capture all stdout/stderr to prevent any progress bars
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            outputs = llm.generate([formatted_prompt], sampling_params, use_tqdm=False)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        # Extract response
        output = outputs[0]
        generated_text = output.outputs[0].text
        
        # Clean response based on model type
        cleaned_response = clean_response_by_model(model_name, generated_text)
        
        # Calculate token counts (approximate from vLLM output)
        prompt_tokens = len(output.prompt_token_ids) if hasattr(output, 'prompt_token_ids') else 0
        response_tokens = len(output.outputs[0].token_ids) if hasattr(output.outputs[0], 'token_ids') else len(generated_text.split())
        
        return cleaned_response, prompt_tokens, response_tokens
        
    except Exception as e:
        print(f"Error during inference: {e}")
        print(f"Prompt (first 100 chars): {prompt[:100]}...")
        return f"Error: {str(e)}", 0, 0

def process_dataset(df, llm, cache_mode, kvtuner_scheme, kvtuner_dir, prompt_template, 
                   columns_to_include, model_name=None, max_new_tokens=200):
    """Process each row in the dataset with configurable cache inference"""
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
    
    print(f"Processing {len(df)} rows with {cache_mode} cache")
    if cache_mode == 'kvtuner':
        print(f"KVTuner scheme: {kvtuner_scheme}")
    
    # Add simple progress tracking for inference
    with tqdm(total=len(df), desc="Processing", unit="rows", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
              disable=False, leave=False, miniters=1, mininterval=2.0) as pbar:
        
        for index, row in df.iterrows():
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
                response, prompt_tokens, response_tokens = llm_inference_vllm(
                    llm, prompt, model_name, max_new_tokens
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
            row_data["kvtuner_scheme"] = kvtuner_scheme if cache_mode == 'kvtuner' else 'none'
            row_data["cache_mode"] = cache_mode
            
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
        "kvtuner_scheme": kvtuner_scheme if cache_mode == 'kvtuner' else 'none',
        "cache_mode": cache_mode,
        "model_name": model_name
    }
    
    return results, stats

def save_results(results, stats, dataset, result_dir, filename_prefix, cache_mode):
    """Save inference results and stats to the specified directory"""
    # Use cache mode in filename
    cache_suffix = f"_{cache_mode}" if cache_mode == 'basic' else "_kvtuner"
    
    # Save processed results as JSON
    results_path = os.path.join(result_dir, f"{filename_prefix}{cache_suffix}_inference_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save stats as JSON
    stats_path = os.path.join(result_dir, f"{filename_prefix}{cache_suffix}_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save summary as text
    summary_path = os.path.join(result_dir, f"{filename_prefix}{cache_suffix}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"LLM Inference with {cache_mode.title()} Cache Summary\n")
        f.write("=" * (30 + len(cache_mode)) + "\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {stats['model_name']}\n")
        f.write(f"Cache Mode: {stats['cache_mode']}\n")
        if stats['cache_mode'] == 'kvtuner':
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
    csv_path = os.path.join(result_dir, f"{filename_prefix}{cache_suffix}_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    print(f"Results saved to: {result_dir}")
    print(f"  - JSON: {results_path}")
    print(f"  - CSV: {csv_path}")
    print(f"  - Stats: {stats_path}")
    print(f"  - Summary: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Run LLM inference with configurable cache modes (KVTuner quantized or basic vLLM cache) on a dataset.')
    
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
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.75,
                        help='Fraction of GPU memory to use (0.1 to 1.0, default: 0.75).')
    
    # Cache configuration
    parser.add_argument('--cache_mode', type=str, default='kvtuner',
                        choices=['kvtuner', 'basic'],
                        help='Cache mode: "kvtuner" for quantized cache, "basic" for default vLLM cache (default: kvtuner).')
    
    # KVTuner configuration
    parser.add_argument('--kvtuner_scheme', type=str, default='pertoken', 
                        choices=['pertoken', 'kivi'],
                        help='KVTuner quantization scheme (default: pertoken). Only used with --cache_mode kvtuner.')
    parser.add_argument('--kvtuner_dir', type=str, 
                        default='/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner',
                        help='Path to KVTuner directory. Only used with --cache_mode kvtuner.')
    
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
    
    # Validate cache mode configuration
    if args.cache_mode == 'kvtuner':
        # Validate KVTuner directory
        if not os.path.exists(args.kvtuner_dir):
            print(f"Error: KVTuner directory not found: {args.kvtuner_dir}")
            print("Either install KVTuner or use --cache_mode basic")
            sys.exit(1)
    
    # Validate GPU memory utilization
    if not (0.1 <= args.gpu_memory_utilization <= 1.0):
        print(f"Error: gpu_memory_utilization must be between 0.1 and 1.0, got {args.gpu_memory_utilization}")
        sys.exit(1)
    
    # Create timestamped result directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = setup_directories(timestamp)
    
    # Start timing
    start_time = time.time()
    
    print("="*60)
    print(f"LLM Inference with {args.cache_mode.title()} Cache")
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
    
    # Initialize LLM with configurable cache
    llm = initialize_llm_vllm(
        model_name=args.model,
        cache_mode=args.cache_mode,
        kvtuner_scheme=args.kvtuner_scheme,
        kvtuner_dir=args.kvtuner_dir if args.cache_mode == 'kvtuner' else None,
        tokenizer_name=args.tokenizer,
        max_model_len=args.max_model_len,
        gpu_ids=args.gpu_ids,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # Process dataset with configurable cache inference
    results, stats = process_dataset(
        dataset,
        llm,
        args.cache_mode,
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
    save_results(results, stats, dataset, result_dir, filename_prefix, args.cache_mode)
    
    # Print summary
    print("\n" + "="*60)
    print("INFERENCE COMPLETED")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Cache Mode: {args.cache_mode}")
    if args.cache_mode == 'kvtuner':
        print(f"KVTuner Scheme: {args.kvtuner_scheme}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization:.1%}")
    print(f"Processed: {stats['total_rows']} rows")
    print(f"Total Time: {stats['total_time']:.2f}s")
    print(f"Average Time per Row: {stats['avg_time_per_row']:.3f}s")
    print(f"Total Tokens: {stats['total_tokens']}")
    print(f"Results saved to: {result_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
