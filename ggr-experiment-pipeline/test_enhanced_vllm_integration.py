#!/usr/bin/env python3
"""
Example script demonstrating the enhanced vLLM with stats logging.
"""

import logging
import sys
import os
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the path for local vLLM setup
sys.path.insert(0, '/Users/zhang/Desktop/huawei/so1/semantic-operators/ggr-experiment-pipeline')

def create_sample_dataset():
    """Create a small sample dataset for testing."""
    data = [
        {
            'title': 'The Shawshank Redemption',
            'description': 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
            'review': 'An absolutely masterful film that touches the heart and soul. Outstanding performances and direction.',
            'genre': 'Drama'
        },
        {
            'title': 'The Dark Knight',
            'description': 'Batman begins his war on crime with his first major enemy being Jack Napier, a criminal who becomes the clownishly homicidal Joker.',
            'review': 'Heath Ledger\'s performance as the Joker is legendary. A dark and gripping superhero movie.',
            'genre': 'Action'
        },
        {
            'title': 'Toy Story',
            'description': 'A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy\'s room.',
            'review': 'A delightful animated film that appeals to both children and adults. Creative and heartwarming.',
            'genre': 'Animation'
        }
    ]
    
    df = pd.DataFrame(data)
    sample_file = 'sample_movies.csv'
    df.to_csv(sample_file, index=False)
    logger.info(f"Created sample dataset: {sample_file}")
    return sample_file

def test_enhanced_vllm():
    """Test the enhanced vLLM functionality."""
    
    try:
        # Import the local vLLM setup
        from use_local_vllm import initialize_experiment_with_local_vllm, create_enhanced_llm
        from vllm import SamplingParams
        
        logger.info("Testing enhanced vLLM with stats logging...")
        
        # Initialize local vLLM
        initialize_experiment_with_local_vllm()
        
        # Create a small test model (you can change this to your preferred model)
        model_path = "microsoft/DialoGPT-small"  # Small model for testing
        logger.info(f"Initializing LLM with model: {model_path}")
        
        # Create enhanced LLM
        llm = create_enhanced_llm(
            model_path,
            max_model_len=512,  # Small for testing
            gpu_memory_utilization=0.7,
            enable_prefix_caching=True
        )
        
        # Test prompts
        test_prompts = [
            "The capital of France is",
            "Machine learning is a field of",
            "The best way to learn programming"
        ]
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=30
        )
        
        logger.info("Running inference with detailed stats...")
        
        # Run inference
        outputs = llm.generate(test_prompts, sampling_params, log_detailed_stats=True)
        
        # Display results
        logger.info("\nResults:")
        for i, output in enumerate(outputs):
            generated = output.outputs[0].text
            logger.info(f"  Prompt {i+1}: {test_prompts[i]}")
            logger.info(f"  Generated: {generated}")
            logger.info(f"  Tokens: {len(output.outputs[0].token_ids)}")
        
        # Get final stats
        stats = llm.get_current_stats()
        logger.info("\nFinal Stats Summary:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("✅ Enhanced vLLM test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced vLLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_experiment_pipeline():
    """Test the complete experiment pipeline."""
    
    try:
        logger.info("Testing complete experiment pipeline...")
        
        # Create sample dataset
        sample_file = create_sample_dataset()
        
        # Import the enhanced experiment runner
        sys.path.insert(0, '/Users/zhang/Desktop/huawei/so1/semantic-operators/ggr-experiment-pipeline/src/experiment')
        from run_experiment_enhanced import run_enhanced_experiment
        
        # Run a small test experiment
        results_df, detailed_stats = run_enhanced_experiment(
            csv_file=sample_file,
            query_key='filter_movies_kids',  # Test query
            model_path='microsoft/DialoGPT-small',  # Small test model
            batch_size=2,
            max_rows=3,
            output_dir='test_results',
            max_model_len=512,
            gpu_memory_utilization=0.7
        )
        
        logger.info("✅ Experiment pipeline test completed successfully!")
        logger.info(f"Results shape: {results_df.shape}")
        logger.info(f"Stats keys: {list(detailed_stats.keys())}")
        
        # Cleanup
        os.remove(sample_file)
        logger.info("Test cleanup completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Experiment pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run all tests."""
    logger.info("Starting enhanced vLLM integration tests...")
    
    # Test 1: Basic enhanced vLLM functionality
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Enhanced vLLM Basic Functionality")
    logger.info("="*60)
    
    test1_success = test_enhanced_vllm()
    
    # Test 2: Complete experiment pipeline
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Complete Experiment Pipeline")
    logger.info("="*60)
    
    test2_success = test_experiment_pipeline()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Enhanced vLLM Test: {'✅ PASS' if test1_success else '❌ FAIL'}")
    logger.info(f"Experiment Pipeline Test: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        logger.info("🎉 All tests passed! Your enhanced vLLM is ready to use.")
        logger.info("\n📝 Next steps:")
        logger.info("1. Install your modified vLLM: cd /Users/zhang/Desktop/huawei/so1/vllm && ./setup_vllm_with_stats.sh")
        logger.info("2. Run experiments: python src/experiment/run_experiment_enhanced.py your_data.csv query_key")
        logger.info("3. Check results in the output directory")
        return True
    else:
        logger.info("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
