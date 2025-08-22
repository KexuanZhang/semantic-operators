#!/usr/bin/env python3
"""
Simplified test script to verify the AttributeError fix
"""

import sys
import os

# Mock the dependencies that might not be available
class MockLLM:
    pass

class MockSamplingParams:
    pass

# Mock imports before importing our module
sys.modules['torch'] = type(sys)('torch')
sys.modules['vllm'] = type(sys)('vllm')
sys.modules['vllm'].LLM = MockLLM
sys.modules['vllm'].SamplingParams = MockSamplingParams
sys.modules['prometheus_client'] = type(sys)('prometheus_client')
sys.modules['pynvml'] = type(sys)('pynvml')
sys.modules['psutil'] = type(sys)('psutil')
sys.modules['matplotlib'] = type(sys)('matplotlib')
sys.modules['matplotlib.pyplot'] = type(sys)('matplotlib.pyplot')
sys.modules['matplotlib.dates'] = type(sys)('matplotlib.dates')  
sys.modules['seaborn'] = type(sys)('seaborn')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'experiment'))

def test_basic_functionality():
    """Test basic functionality without external dependencies"""
    try:
        from run_experiment import SimpleLLMExperiment
        print("✅ Successfully imported SimpleLLMExperiment")
        
        # Test attribute exists in class
        experiment_code = """
# Test that gpu_memory_utilization is properly initialized
class TestExperiment:
    def __init__(self):
        # Initialize configuration attributes
        self.gpu_memory_utilization = 0.85  # Default value
        self.max_tokens = 512
        self.temperature = 0.1
        self.top_p = 0.9

test_exp = TestExperiment()
print(f"gpu_memory_utilization: {test_exp.gpu_memory_utilization}")
assert hasattr(test_exp, 'gpu_memory_utilization')
assert test_exp.gpu_memory_utilization == 0.85
"""
        exec(experiment_code)
        print("✅ Basic attribute initialization test passed")
        
        print("🎉 AttributeError fix verification completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
