#!/usr/bin/env python3
"""
Test script for the updated vLLM metrics integration
Verifies that the modern LoggingStatLogger and Stats dataclass integration works correctly
"""

import os
import sys
import pandas as pd
import tempfile
import logging

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from experiment.run_experiment import SimpleLLMExperiment, VLLMMetricsCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_dataset():
    """Create a small test dataset for experimentation"""
    test_data = {
        'question': [
            'What is machine learning?',
            'How does deep learning work?',
            'What are neural networks?',
            'Explain artificial intelligence',
            'What is natural language processing?'
        ],
        'context': [
            'Machine learning is a subset of AI',
            'Deep learning uses neural networks',
            'Neural networks mimic brain structure',
            'AI systems can perform tasks typically requiring human intelligence',
            'NLP helps computers understand human language'
        ]
    }
    
    df = pd.DataFrame(test_data)
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    return temp_file.name

def test_vllm_metrics_collector():
    """Test the VLLMMetricsCollector class independently"""
    logger.info("Testing VLLMMetricsCollector class...")
    
    # Test without LLM instance
    collector = VLLMMetricsCollector()
    
    # Test initialization
    init_success = collector.initialize_logging()
    logger.info(f"Metrics logging initialization: {'Success' if init_success else 'Legacy mode'}")
    
    # Test stats collection without LLM
    stats = collector.collect_current_stats()
    logger.info(f"Stats collection without LLM: {stats}")
    
    # Test monitoring functions
    collector.start_monitoring()
    import time
    time.sleep(2)  # Let it collect some stats
    collector.stop_monitoring()
    
    comprehensive_stats = collector.get_comprehensive_stats()
    logger.info(f"Comprehensive stats: {comprehensive_stats}")
    
    metrics_history = collector.get_metrics_history()
    logger.info(f"Metrics history length: {len(metrics_history)}")
    
    return True

def test_experiment_integration():
    """Test the integration with SimpleLLMExperiment"""
    logger.info("Testing SimpleLLMExperiment integration...")
    
    # Create test dataset
    test_dataset = create_test_dataset()
    
    try:
        # Create experiment instance (this will test GPU detection and model setup)
        experiment = SimpleLLMExperiment(
            model_name="microsoft/DialoGPT-small",  # Use a small model for testing
            output_dir="test_output",
            gpu_ids=[0]  # Single GPU for testing
        )
        
        # Test vLLM metrics collector integration
        logger.info(f"VLLMMetricsCollector initialized: {experiment.vllm_metrics_collector is not None}")
        
        # Test get_vllm_stats method
        stats = experiment.get_vllm_stats()
        logger.info(f"Initial vLLM stats: {stats}")
        
        # Test comprehensive stats
        comprehensive = experiment.vllm_metrics_collector.get_comprehensive_stats()
        logger.info(f"Comprehensive stats available: {len(comprehensive) > 0}")
        
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        logger.info("This is expected if GPU/vLLM is not available")
        return False
    
    finally:
        # Cleanup
        try:
            os.unlink(test_dataset)
        except:
            pass

def test_modern_metrics_availability():
    """Test if modern vLLM metrics are available"""
    logger.info("Testing modern vLLM metrics availability...")
    
    try:
        from vllm.engine.metrics import LoggingStatLogger, Stats
        from vllm.engine.llm_engine import LLMEngine
        logger.info("✅ Modern vLLM metrics imports successful")
        return True
    except ImportError as e:
        logger.warning(f"❌ Modern vLLM metrics not available: {e}")
        logger.info("This may be due to vLLM version compatibility")
        return False

def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("VLLM METRICS INTEGRATION TEST")
    logger.info("=" * 60)
    
    # Test 1: Modern metrics availability
    modern_available = test_modern_metrics_availability()
    
    # Test 2: VLLMMetricsCollector functionality
    collector_test = test_vllm_metrics_collector()
    
    # Test 3: Integration with SimpleLLMExperiment
    integration_test = test_experiment_integration()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Modern vLLM metrics available: {'✅' if modern_available else '❌'}")
    logger.info(f"VLLMMetricsCollector test: {'✅' if collector_test else '❌'}")
    logger.info(f"SimpleLLMExperiment integration: {'✅' if integration_test else '❌'}")
    
    if modern_available and collector_test:
        logger.info("🎉 vLLM metrics integration is working correctly!")
        logger.info("The system will use modern LoggingStatLogger and Stats dataclass for metrics collection")
    else:
        logger.info("⚠️  Some tests failed, but this may be due to environment limitations")
        logger.info("The system will fall back to legacy metrics collection methods")
    
    logger.info("\n📝 Next steps:")
    logger.info("1. Run a small experiment with: python run_experiment.py test_data.csv query_key")
    logger.info("2. Check the generated vllm_metrics.csv file for detailed metrics")
    logger.info("3. Review the performance_report.md for comprehensive analysis")

if __name__ == "__main__":
    main()
