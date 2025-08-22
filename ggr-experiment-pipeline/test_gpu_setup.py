#!/usr/bin/env python3
"""
GPU Setup Test Script

This script verifies that your system is ready for GPU-accelerated vLLM experiments.
"""

import sys
import subprocess
from pathlib import Path

def test_python_packages():
    """Test if required Python packages are available"""
    print("🧪 Testing Python Package Dependencies")
    print("-" * 50)
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('vllm', 'vLLM'),
        ('requests', 'HTTP Requests'),
        ('pandas', 'Pandas'),
        ('prometheus_client', 'Prometheus Client')
    ]
    
    all_good = True
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {description}: Available")
        except ImportError:
            print(f"❌ {description}: Not installed")
            all_good = False
    
    return all_good

def test_gpu_availability():
    """Test GPU availability and CUDA support"""
    print("\n🖥️  Testing GPU and CUDA Support")
    print("-" * 50)
    
    try:
        import torch
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print("✅ CUDA: Available")
            
            # Get GPU count and info
            gpu_count = torch.cuda.device_count()
            print(f"✅ GPU Count: {gpu_count}")
            
            # List GPU details
            for i in range(gpu_count):
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    print(f"   GPU {i}: {gpu_name} ({memory_gb:.1f}GB)")
                except Exception as e:
                    print(f"   GPU {i}: Info unavailable ({e})")
            
            # Test basic GPU operation
            try:
                x = torch.rand(100, 100).cuda()
                y = torch.rand(100, 100).cuda()
                z = torch.matmul(x, y)
                print("✅ GPU Operations: Working")
                return True
            except Exception as e:
                print(f"❌ GPU Operations: Failed ({e})")
                return False
        else:
            print("❌ CUDA: Not available")
            print("   Note: vLLM can still run on CPU")
            return False
            
    except ImportError:
        print("❌ PyTorch: Not installed")
        return False

def test_vllm_installation():
    """Test vLLM installation"""
    print("\n🚀 Testing vLLM Installation")
    print("-" * 50)
    
    try:
        import vllm
        print(f"✅ vLLM Version: {vllm.__version__}")
        
        # Test if we can import key components
        from vllm.entrypoints.openai.api_server import app
        print("✅ vLLM API Server: Available")
        
        from vllm import LLM
        print("✅ vLLM Engine: Available")
        
        return True
    except ImportError as e:
        print(f"❌ vLLM: Not properly installed ({e})")
        return False
    except Exception as e:
        print(f"⚠️  vLLM: Partial installation ({e})")
        return False

def test_system_resources():
    """Test system resources"""
    print("\n💻 Testing System Resources")
    print("-" * 50)
    
    try:
        import psutil
        
        # CPU info
        cpu_count = psutil.cpu_count()
        print(f"✅ CPU Cores: {cpu_count}")
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"✅ System Memory: {memory_gb:.1f}GB")
        
        # Disk space
        disk = psutil.disk_usage('/')
        free_gb = disk.free / (1024**3)
        print(f"✅ Free Disk Space: {free_gb:.1f}GB")
        
        # Check if we have enough resources
        if memory_gb < 8:
            print("⚠️  Warning: Less than 8GB RAM may limit model choices")
        if free_gb < 10:
            print("⚠️  Warning: Less than 10GB free disk space may be insufficient")
        
        return True
    except ImportError:
        print("⚠️  psutil not available - cannot check system resources")
        return False

def test_network_connectivity():
    """Test network connectivity for model downloads"""
    print("\n🌐 Testing Network Connectivity")
    print("-" * 50)
    
    try:
        import requests
        
        # Test Hugging Face connectivity
        response = requests.get("https://huggingface.co", timeout=10)
        if response.status_code == 200:
            print("✅ Hugging Face: Accessible")
        else:
            print(f"⚠️  Hugging Face: Unexpected status {response.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ Network connectivity: Failed ({e})")
        return False

def run_simple_test():
    """Run a simple vLLM test if possible"""
    print("\n🎯 Running Simple vLLM Test")
    print("-" * 50)
    
    try:
        import torch
        from vllm import LLM, SamplingParams
        
        # Only run if we have sufficient resources
        if torch.cuda.is_available():
            print("⏳ Loading small model for testing...")
            
            # Use a very small model for testing
            try:
                llm = LLM(
                    model="microsoft/DialoGPT-small",
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.3,  # Use minimal memory
                    max_model_len=512,  # Short context
                    enforce_eager=True,  # Disable optimizations for testing
                )
                
                # Generate a simple response
                sampling_params = SamplingParams(max_tokens=5, temperature=0.0)
                outputs = llm.generate(["Hello"], sampling_params)
                
                print("✅ vLLM Simple Test: Passed")
                return True
                
            except Exception as e:
                print(f"❌ vLLM Simple Test: Failed ({e})")
                print("   This might be due to model download or resource constraints")
                return False
        else:
            print("⚠️  Skipping vLLM test - no GPU available")
            return False
            
    except ImportError:
        print("⚠️  Cannot run vLLM test - missing dependencies")
        return False

def main():
    """Main test function"""
    print("🔍 GPU Setup Test for vLLM Server Experiment")
    print("=" * 60)
    
    tests = [
        test_python_packages,
        test_gpu_availability,
        test_vllm_installation,
        test_system_resources,
        test_network_connectivity,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Optional: Run simple vLLM test
    print("\nWould you like to run a simple vLLM test? (This may take a few minutes)")
    print("This will download a small model and test basic functionality.")
    choice = input("Run test? (y/N): ").strip().lower()
    
    if choice in ('y', 'yes'):
        result = run_simple_test()
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉 All tests passed! Your system is ready for GPU-accelerated vLLM experiments.")
        print("\n💡 Next steps:")
        print("   1. Create or prepare your dataset")
        print("   2. Run: python src/experiment/server_exp.py --model MODEL_NAME --dataset DATASET_PATH")
        print("   3. Check the results/ directory for outputs")
    elif passed >= total - 1:
        print("✅ Most tests passed. Your system should work with minor limitations.")
        print("\n⚠️  Check the failed tests above and consider installing missing dependencies.")
    else:
        print("❌ Several tests failed. Please address the issues before running experiments.")
        print("\n🔧 Common fixes:")
        print("   - Install missing packages: pip install -r requirements-vllm.txt")
        print("   - Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   - Install vLLM: pip install vllm")
    
    print(f"\nTest Results: {passed}/{total} passed")

if __name__ == "__main__":
    main()
