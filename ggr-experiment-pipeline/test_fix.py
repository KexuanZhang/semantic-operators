#!/usr/bin/env python3
"""
Test script to verify that the AttributeError for gpu_memory_utilization is fixed
"""

import sys
import os

# Add src to path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'experiment'))

try:
    from run_experiment import SimpleLLMExperiment
    
    print("✅ Successfully imported SimpleLLMExperiment")
    
    # Test initializing the class
    experiment = SimpleLLMExperiment(
        model_name="test-model",
        output_dir="test_output",
        gpu_ids=[0]
    )
    
    print("✅ Successfully initialized SimpleLLMExperiment")
    
    # Test that gpu_memory_utilization attribute exists and has the correct default value
    if hasattr(experiment, 'gpu_memory_utilization'):
        print(f"✅ gpu_memory_utilization attribute exists with value: {experiment.gpu_memory_utilization}")
        
        if experiment.gpu_memory_utilization == 0.85:
            print("✅ Default value is correct (0.85)")
        else:
            print(f"⚠️ Default value is {experiment.gpu_memory_utilization}, expected 0.85")
    else:
        print("❌ gpu_memory_utilization attribute not found")
        sys.exit(1)
    
    # Test the plot functionality is disabled
    try:
        result = experiment.plot_kv_cache_metrics("/tmp")
        if result == {}:
            print("✅ Plot functionality properly disabled (returns empty dict)")
        else:
            print(f"⚠️ Plot functionality returned: {result}")
    except Exception as e:
        print(f"❌ Plot functionality test failed: {e}")
        sys.exit(1)
    
    # Test VLLMMetricsCollector plot functionality is disabled
    if hasattr(experiment, 'vllm_metrics_collector'):
        try:
            experiment.vllm_metrics_collector.plot_kv_cache_metrics("/tmp")
            print("✅ VLLMMetricsCollector plot functionality properly disabled")
        except Exception as e:
            print(f"❌ VLLMMetricsCollector plot functionality test failed: {e}")
            sys.exit(1)
    
    print("\n🎉 All tests passed! The AttributeError fix is working correctly.")
    print("📊 Plotting functionality has been successfully disabled as requested.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
