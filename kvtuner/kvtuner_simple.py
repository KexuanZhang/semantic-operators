#!/usr/bin/env python3
"""
Simple runner for KVTuner inference with Qwen2.5-3B-Instruct

This script provides a simplified interface for running KVTuner inference
with preset configurations and easy dataset processing.
"""

import os
import sys
import argparse
import json
import time
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add KVTuner to Python path
sys.path.append("/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner")

try:
    from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig, FlexibleVanillaQuantizedCache
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import yaml
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure KVTuner is properly installed and accessible")
    sys.exit(1)

class KVTunerRunner:
    """Simplified KVTuner inference runner"""
    
    def __init__(self, model_path, scheme="pertoken"):
        self.model_path = model_path
        self.scheme = scheme
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model from: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=False,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("Model and tokenizer loaded successfully")
        
    def create_kv_cache(self):
        """Create KV cache with preset configuration"""
        
        # Determine config path
        config_name = f"Qwen2.5-3B-Instruct_{self.scheme}_KVTuner4_0.yaml"
        config_path = f"/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner/calibration_presets/{config_name}"
        
        if not os.path.exists(config_path):
            print(f"Warning: Config file not found: {config_path}")
            print("Using default configuration...")
            
        # Set axis configuration based on scheme
        if self.scheme == "kivi":
            axis_key = 1  # Per-channel for keys
            axis_value = 0  # Per-token for values
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
            per_layer_quant=True,
            per_layer_config_path=config_path if os.path.exists(config_path) else None,
            asym=True,
            axis_key=axis_key,
            axis_value=axis_value,
            q_group_size=q_group_size,
            residual_length=residual_length
        )
        
        return FlexibleVanillaQuantizedCache(cache_config=cache_config)
    
    def format_prompt(self, text):
        """Format prompt for Qwen models"""
        return f"<|im_start|>system\nYou are a helpful assistant that provides clear, concise, and accurate answers.\n<|im_end|>\n<|im_start|>user\n{text}\n<|im_end|>\n<|im_start|>assistant\n"
    
    def run_inference(self, prompt, max_new_tokens=256):
        """Run inference on a single prompt"""
        
        # Create fresh KV cache
        kv_cache = self.create_kv_cache()
        
        # Format prompt
        formatted_prompt = self.format_prompt(prompt)
        
        # Tokenize input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        
        # Run inference
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                past_key_values=kv_cache,
                use_cache=True,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        
        # Decode output
        output_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Clean up output
        if "<|im_end|>" in output_text:
            output_text = output_text.split("<|im_end|>")[0].strip()
        
        return output_text.strip()
    
    def process_dataset(self, dataset_path, text_column="text", max_rows=None, 
                       prompt_template="Analyze the following text: {text}",
                       max_new_tokens=256):
        """Process entire dataset"""
        
        print(f"Loading dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        if max_rows:
            df = df.head(max_rows)
            print(f"Limited to {max_rows} rows")
        
        print(f"Dataset loaded: {len(df)} rows")
        print(f"Available columns: {list(df.columns)}")
        
        # Check text column
        if text_column not in df.columns:
            print(f"Column '{text_column}' not found. Available: {list(df.columns)}")
            text_column = df.columns[0]
            print(f"Using first column: '{text_column}'")
        
        results = []
        total_time = 0
        
        print(f"Starting inference with {self.scheme} scheme...")
        
        for idx, row in df.iterrows():
            text = str(row[text_column])
            
            # Create prompt
            try:
                if "{" in prompt_template and "}" in prompt_template:
                    prompt = prompt_template.format(**{text_column: text})
                else:
                    prompt = f"{prompt_template} {text}"
            except:
                prompt = f"Analyze the following text: {text}"
            
            # Run inference with timing
            start_time = time.time()
            try:
                completion = self.run_inference(prompt, max_new_tokens)
            except Exception as e:
                print(f"Error at row {idx}: {e}")
                completion = f"Error: {str(e)}"
            
            inference_time = time.time() - start_time
            total_time += inference_time
            
            result = {
                "id": idx,
                "input_text": text,
                "prompt": prompt,
                "completion": completion,
                "inference_time": inference_time
            }
            results.append(result)
            
            # Show progress
            if (idx + 1) % 5 == 0:
                avg_time = total_time / (idx + 1)
                print(f"Processed {idx+1}/{len(df)} samples. Avg time: {avg_time:.2f}s")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./results/kvtuner_simple_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        results_file = os.path.join(output_dir, "results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save as CSV
        csv_file = os.path.join(output_dir, "results.csv")
        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_file, index=False)
        
        # Print summary
        print(f"\nInference completed!")
        print(f"Processed: {len(results)} samples")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time: {total_time/len(results):.3f}s per sample")
        print(f"Results saved to: {output_dir}")
        
        return results

def run_simple_test(model_path, scheme="pertoken"):
    """Run a simple test with a few prompts"""
    
    runner = KVTunerRunner(model_path, scheme)
    runner.load_model()
    
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot learning to paint."
    ]
    
    print(f"Running simple test with {scheme} scheme...")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {prompt}")
        
        start_time = time.time()
        completion = runner.run_inference(prompt, max_new_tokens=128)
        inference_time = time.time() - start_time
        
        print(f"Completion: {completion}")
        print(f"Time: {inference_time:.3f}s")

def main():
    parser = argparse.ArgumentParser(description="Simple KVTuner inference runner")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to local Qwen2.5-3B-Instruct model")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset CSV file (optional)")
    parser.add_argument("--scheme", type=str, default="pertoken",
                        choices=["pertoken", "kivi"],
                        help="Quantization scheme")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column name for text input")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Maximum rows to process")
    parser.add_argument("--prompt_template", type=str, 
                        default="Analyze the following text: {text}",
                        help="Prompt template")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum new tokens to generate")
    parser.add_argument("--test", action="store_true",
                        help="Run simple test instead of dataset processing")
    
    args = parser.parse_args()
    
    # Verify model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        return
    
    try:
        if args.test:
            # Run simple test
            run_simple_test(args.model_path, args.scheme)
        elif args.dataset:
            # Process dataset
            if not os.path.exists(args.dataset):
                print(f"Error: Dataset file does not exist: {args.dataset}")
                return
            
            runner = KVTunerRunner(args.model_path, args.scheme)
            runner.load_model()
            runner.process_dataset(
                args.dataset, 
                args.text_column, 
                args.max_rows,
                args.prompt_template,
                args.max_new_tokens
            )
        else:
            print("Please provide either --dataset for dataset processing or --test for simple test")
            print("Use --help for more information")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
