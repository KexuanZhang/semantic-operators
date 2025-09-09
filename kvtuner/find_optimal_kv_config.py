#!/usr/bin/env python3
# Script to find optimal KV config for a given model
# Usage: python find_optimal_kv_config.py --model_name "meta-llama/Meta-Llama-3-8B" --scheme pertoken --output_dir ./configs

import os
import sys
import argparse
import torch
import yaml
import json
import logging
from datetime import datetime

# Add path to KVTuner
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "KVTuner"))

def parse_args():
    parser = argparse.ArgumentParser(description="Find optimal KV configuration for a model")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Path to local model or HF model name")
    parser.add_argument("--scheme", type=str, default="pertoken", 
                        choices=["pertoken", "kivi"],
                        help="Quantization scheme: pertoken (per-token for both K&V) or kivi (per-channel for K, per-token for V)")
    parser.add_argument("--output_dir", type=str, default="./configs", 
                        help="Directory to save KV configs")
    parser.add_argument("--target_bits", type=float, default=4.0, 
                        help="Target average bit width")
    parser.add_argument("--search_method", type=str, default="optuna", 
                        choices=["brute", "optuna"], 
                        help="Search method: brute force or optuna")
    parser.add_argument("--n_trials", type=int, default=30, 
                        help="Number of trials for optuna search")
    parser.add_argument("--sample_limit", type=int, default=20, 
                        help="Number of samples for evaluation")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug logging")
    return parser.parse_args()

def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("kv_config_search")

def select_search_method(args, logger):
    """Select the search method based on arguments"""
    try:
        if args.search_method == "brute":
            from search_brute_force import run_search
            logger.info("Using brute force search method")
            return run_search
        else:  # Default to optuna
            if args.scheme == "pertoken":
                from search_optuna_vanilla import run_search
                logger.info("Using optuna search with per-token scheme")
                return run_search
            else:  # kivi
                from search_optuna_adaptive import run_search
                logger.info("Using optuna search with KiVi scheme")
                return run_search
    except ImportError as e:
        logger.error(f"Failed to import search method: {e}")
        logger.error("Make sure you have the KVTuner repository accessible")
        sys.exit(1)

def get_model_name_for_filename(model_name):
    """Convert model name to a filename-friendly format"""
    # Remove path prefixes and replace slashes with underscores
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
    
    # Replace any characters that might be problematic in filenames
    return model_name.replace(' ', '_').replace('-', '_')

def main():
    args = parse_args()
    logger = setup_logging(args.debug)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select search method
    run_search = select_search_method(args, logger)
    
    # Set up search parameters
    search_params = {
        "model_name": args.model_name,
        "target_bits": args.target_bits,
        "sample_limit": args.sample_limit,
        "n_trials": args.n_trials if args.search_method == "optuna" else None,
    }
    
    if args.scheme == "kivi":
        search_params["axis_key"] = 1  # per-channel for keys
        search_params["axis_value"] = 0  # per-token for values
        search_params["q_group_size"] = 32
        search_params["residual_length"] = 32
    else:  # pertoken
        search_params["axis_key"] = 0  # per-token for keys
        search_params["axis_value"] = 0  # per-token for values
        search_params["q_group_size"] = -1
        search_params["residual_length"] = 0
    
    # Log search parameters
    logger.info(f"Starting search for model: {args.model_name}")
    logger.info(f"Scheme: {args.scheme}, Target bits: {args.target_bits}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Run search
        result = run_search(**search_params)
        
        if result is None or not isinstance(result, dict):
            logger.error("Search failed to return a valid configuration")
            return
        
        # Save configuration
        model_name = get_model_name_for_filename(args.model_name)
        config_filename = f"{model_name}_{args.scheme}_{int(args.target_bits)}_bits_{timestamp}.yaml"
        config_path = os.path.join(args.output_dir, config_filename)
        
        with open(config_path, 'w') as f:
            yaml.dump(result, f)
        
        logger.info(f"Configuration saved to: {config_path}")
        
        # Also save metadata
        metadata = {
            "model_name": args.model_name,
            "scheme": args.scheme,
            "target_bits": args.target_bits,
            "search_method": args.search_method,
            "timestamp": timestamp,
        }
        
        metadata_path = os.path.join(args.output_dir, f"{model_name}_{args.scheme}_{int(args.target_bits)}_bits_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Search complete! Metadata saved to: {metadata_path}")
        
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
