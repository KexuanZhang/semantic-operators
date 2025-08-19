#!/usr/bin/env python3
"""
Enhanced GGR Experiment Pipeline with vLLM Inference
Command-line interface for running complete GGR experiments with vLLM inference monitoring
"""
import argparse
import sys
import os
import logging
from typing import Optional

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.enhanced_experiment_runner import run_enhanced_experiment
    enhanced_available = True
except ImportError as e:
    print(f"Warning: Enhanced experiment runner not fully available: {e}")
    enhanced_available = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function for enhanced experiment CLI"""
    parser = argparse.ArgumentParser(
        description="Enhanced GGR Algorithm Experiment Pipeline with vLLM Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete experiment with GGR and vLLM inference
  python main_enhanced.py data/dataset.csv --model microsoft/DialoGPT-medium
  
  # GGR only (skip inference)
  python main_enhanced.py data/dataset.csv --no-inference
  
  # Custom model and parameters
  python main_enhanced.py data/dataset.csv --model meta-llama/Llama-2-7b-chat-hf --max-samples 100
  
  # With functional dependencies
  python main_enhanced.py data/dataset.csv --fds "movie_id->title,movie_id->genre" --model gpt2
        """
    )
    
    # Required arguments
    parser.add_argument('dataset_path', 
                       help='Path to the input CSV dataset')
    
    # Model configuration
    parser.add_argument('--model', '--model-name',
                       type=str,
                       default='microsoft/DialoGPT-medium',
                       help='vLLM model name for inference (default: microsoft/DialoGPT-medium)')
    
    # GGR configuration
    parser.add_argument('--fds', '--functional-dependencies',
                       type=str,
                       help='Functional dependencies as comma-separated pairs (e.g., "col1->col2,col3->col4")')
    
    parser.add_argument('--columns', '--columns-of-interest',
                       type=str,
                       help='Comma-separated list of columns to analyze')
    
    parser.add_argument('--max-depth',
                       type=int,
                       default=100,
                       help='Maximum recursion depth for GGR algorithm (default: 100)')
    
    parser.add_argument('--no-discover-fds',
                       action='store_true',
                       help='Disable automatic discovery of functional dependencies')
    
    # Inference configuration
    parser.add_argument('--no-inference',
                       action='store_true',
                       help='Skip vLLM inference (GGR only)')
    
    parser.add_argument('--max-samples',
                       type=int,
                       default=50,
                       help='Maximum samples for inference (default: 50, use 0 for all)')
    
    parser.add_argument('--temperature',
                       type=float,
                       default=0.7,
                       help='Sampling temperature for inference (default: 0.7)')
    
    parser.add_argument('--max-tokens',
                       type=int,
                       default=100,
                       help='Maximum tokens to generate per prompt (default: 100)')
    
    # Output configuration
    parser.add_argument('--output', '--output-dir',
                       type=str,
                       default='enhanced_results',
                       help='Output directory for results (default: enhanced_results)')
    
    parser.add_argument('--name', '--experiment-name',
                       type=str,
                       help='Name for this experiment (auto-generated if not specified)')
    
    # Other options
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not os.path.exists(args.dataset_path):
        logger.error(f"Dataset file not found: {args.dataset_path}")
        sys.exit(1)
    
    if not enhanced_available:
        logger.error("Enhanced experiment runner not available. Please ensure all dependencies are installed.")
        logger.error("Try: pip install -r requirements-vllm.txt")
        sys.exit(1)
    
    # Prepare sampling parameters
    sampling_params = {
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'top_p': 0.9,
        'seed': 42
    }
    
    try:
        logger.info("Starting enhanced GGR experiment pipeline...")
        
        # Run enhanced experiment
        results = run_enhanced_experiment(
            dataset_path=args.dataset_path,
            model_name=args.model,
            functional_dependencies=args.fds,
            columns=args.columns,
            output_dir=args.output,
            experiment_name=args.name,
            max_depth=args.max_depth,
            run_inference=not args.no_inference,
            max_inference_samples=args.max_samples if args.max_samples > 0 else None
        )
        
        # Print detailed summary
        print("\\n" + "="*70)
        print("ENHANCED GGR EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*70)
        
        # GGR Results
        print(f"Experiment Name: {results.get('experiment_name', 'Unknown')}")
        print(f"Dataset: {results.get('dataset_info', {}).get('path', 'Unknown')}")
        print(f"Dataset Shape: {results.get('dataset_info', {}).get('shape', 'Unknown')}")
        print(f"Processed Shape: {results.get('processed_shape', 'Unknown')}")
        
        print(f"\\n📊 GGR ALGORITHM RESULTS:")
        print(f"  Functional Dependencies: {len(results.get('functional_dependencies', []))}")
        print(f"  Total Hits: {results.get('ggr_total_hits', 0)}")
        print(f"  GGR Execution Time: {results.get('ggr_execution_time', 0):.2f} seconds")
        print(f"  Reordered Table Size: {results.get('reordered_table_size', 0)}")
        
        # Inference Results
        if results.get('inference_completed'):
            print(f"\\n🚀 vLLM INFERENCE RESULTS:")
            print(f"  Model: {results.get('model_name', 'Unknown')}")
            print(f"  Samples Processed: {results.get('inference_samples', 0)}")
            print(f"  Inference Time: {results.get('inference_time', 0):.2f} seconds")
            print(f"  Throughput: {results.get('inference_throughput', 0):.2f} prompts/second")
            
            # Resource usage
            resource_usage = results.get('resource_usage', {})
            if resource_usage:
                print(f"\\n💻 RESOURCE USAGE:")
                print(f"  Average CPU: {resource_usage.get('avg_cpu_percent', 0):.1f}%")
                print(f"  Peak CPU: {resource_usage.get('peak_cpu_percent', 0):.1f}%")
                print(f"  Average Memory: {resource_usage.get('avg_memory_percent', 0):.1f}%")
                print(f"  Monitoring Samples: {resource_usage.get('monitoring_samples', 0)}")
        elif results.get('inference_skipped'):
            print(f"\\n⚠️  vLLM INFERENCE SKIPPED (--no-inference flag used)")
        elif results.get('inference_error'):
            print(f"\\n❌ vLLM INFERENCE FAILED: {results.get('inference_error')}")
        
        # Performance Summary
        ggr_effectiveness = results.get('ggr_effectiveness', {})
        if ggr_effectiveness:
            print(f"\\n🎯 PERFORMANCE SUMMARY:")
            print(f"  GGR Total Hits: {ggr_effectiveness.get('total_hits', 0)}")
            print(f"  GGR Processing Time: {ggr_effectiveness.get('processing_time', 0):.2f}s")
            print(f"  Samples Reordered: {ggr_effectiveness.get('reordered_samples', 0)}")
            
            inference_perf = results.get('inference_performance', {})
            if inference_perf:
                print(f"  Inference Throughput: {inference_perf.get('throughput', 0):.2f} prompts/s")
                print(f"  Total Experiment Time: {results.get('total_experiment_time', 0):.2f}s")
        
        print(f"\\n📁 OUTPUT FILES:")
        print(f"  Results Directory: {args.output}")
        print(f"  Main Results: {args.name or 'auto-generated'}_complete_results.json")
        if results.get('reordered_dataset_path'):
            print(f"  Reordered Dataset: {os.path.basename(results.get('reordered_dataset_path'))}")
        
        # Recommendations
        print(f"\\n💡 NEXT STEPS:")
        if results.get('inference_completed'):
            print("  ✅ Complete experiment finished!")
            print("  📈 Check Jupyter notebook for detailed analysis")
            print("  🔄 Run baseline comparison with shuffled data")
            print("  📊 Compare prefix cache hit rates and throughput")
        else:
            print("  🔧 Install vLLM for complete inference analysis")
            print("  📝 Review GGR reordering results")
            print("  🚀 Re-run with --model flag for inference")
        
        print("="*70)
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Enhanced experiment failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
