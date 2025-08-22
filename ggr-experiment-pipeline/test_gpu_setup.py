#!/usr/bin/env python3
"""
GPU Setup Test Script - Configured for GPUs 6 & 7

This script verifies that your system is ready for GPU-accelerated vLLM experiments
using GPUs 6 and 7 with tensor parallelism support.

Features:
- Tests availability of GPUs 6 and 7 specifically
- Checks GPU memory status on both GPUs
- Validates vLLM installation with tensor parallelism
- Runs comprehensive system compatibility tests
- Provides detailed troubleshooting guidance

Usage:
    python test_gpu_setup.py

The script will automatically configure CUDA_VISIBLE_DEVICES=6,7 to use only
these GPUs for testing and subsequent vLLM experiments.
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
    """Test GPU availability and CUDA support - focusing on GPUs 6 and 7"""
    print("\n🖥️  Testing GPU and CUDA Support (GPUs 6 & 7)")
    print("-" * 50)
    
    # Set CUDA_VISIBLE_DEVICES to use only GPUs 6 and 7
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
    print("🎯 Configured to use GPUs 6 and 7 only")
    
    try:
        import torch
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print("✅ CUDA: Available")
            
            # Get GPU count and info (should be 2 with CUDA_VISIBLE_DEVICES=6,7)
            gpu_count = torch.cuda.device_count()
            print(f"✅ Visible GPU Count: {gpu_count} (GPUs 6 & 7)")
            
            if gpu_count < 2:
                print(f"⚠️  Warning: Only {gpu_count} GPU(s) visible. Expected 2 (GPUs 6 & 7)")
                print("   This might be due to GPU unavailability or driver issues")
            
            # List GPU details (these will be remapped as 0, 1 due to CUDA_VISIBLE_DEVICES)
            for i in range(gpu_count):
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    actual_gpu_id = 6 + i  # Map back to actual GPU IDs
                    print(f"   GPU {i} (Physical GPU {actual_gpu_id}): {gpu_name} ({memory_gb:.1f}GB)")
                except Exception as e:
                    print(f"   GPU {i}: Info unavailable ({e})")
            
            # Test basic GPU operations on both GPUs if available
            try:
                if gpu_count >= 1:
                    # Test first GPU (GPU 6)
                    with torch.cuda.device(0):
                        x = torch.rand(100, 100).cuda()
                        y = torch.rand(100, 100).cuda()
                        z = torch.matmul(x, y)
                    print("✅ GPU 0 (Physical GPU 6): Operations working")
                
                if gpu_count >= 2:
                    # Test second GPU (GPU 7)
                    with torch.cuda.device(1):
                        x = torch.rand(100, 100).cuda()
                        y = torch.rand(100, 100).cuda()
                        z = torch.matmul(x, y)
                    print("✅ GPU 1 (Physical GPU 7): Operations working")
                
                print("✅ GPU Operations: Working on selected GPUs")
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

def check_gpu_memory_details():
    """Check detailed GPU memory usage for GPUs 6 and 7"""
    print("\n🧠 Detailed GPU Memory Analysis (GPUs 6 & 7)")
    print("-" * 50)
    
    # Ensure CUDA_VISIBLE_DEVICES is set
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False
        
        gpu_count = torch.cuda.device_count()
        print(f"🎯 Analyzing {gpu_count} visible GPUs (Physical GPUs 6 & 7)")
        
        all_good = True
        for i in range(gpu_count):
            actual_gpu_id = 6 + i  # Map back to actual GPU IDs
            print(f"\n📊 GPU {i} (Physical GPU {actual_gpu_id}) Memory Status:")
            
            # Get memory info
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            free = total_memory - cached
            
            print(f"   Total Memory: {total_memory:.2f} GiB")
            print(f"   Allocated:    {allocated:.2f} GiB")
            print(f"   Cached:       {cached:.2f} GiB")
            print(f"   Free:         {free:.2f} GiB")
            
            # Check if we have enough for vLLM with tensor parallelism
            min_required = 3.0  # 3GB minimum per GPU for tensor parallelism
            if free < min_required:
                print(f"   ⚠️  WARNING: Only {free:.2f} GiB free, need at least {min_required} GiB")
                print("   💡 Suggested fixes:")
                print("      - Kill other GPU processes: sudo fuser -k /dev/nvidia*")
                print("      - Clear GPU cache: torch.cuda.empty_cache()")
                print("      - Check processes on this GPU: nvidia-smi")
                print("      - Restart system if needed")
                all_good = False
            else:
                print(f"   ✅ Sufficient memory available ({free:.2f} GiB)")
        
        if gpu_count == 2 and all_good:
            total_free = sum(
                (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)) / (1024**3)
                for i in range(gpu_count)
            )
            print(f"\n🎯 Total free memory across both GPUs: {total_free:.2f} GiB")
            print("✅ Ready for tensor parallel vLLM deployment")
        elif gpu_count < 2:
            print(f"\n⚠️  Only {gpu_count} GPU available. Tensor parallelism requires 2 GPUs.")
            print("   Will fall back to single GPU mode.")
            all_good = all_good and (gpu_count >= 1)
        
        return all_good
        
    except Exception as e:
        print(f"❌ Error checking GPU memory: {e}")
        return False

def run_simple_test():
    """Run a simple vLLM test with GPUs 6 and 7 (tensor parallelism)"""
    print("\n🎯 Running Simple vLLM Test (GPUs 6 & 7 with Tensor Parallelism)")
    print("-" * 50)
    
    # Ensure CUDA_VISIBLE_DEVICES is set
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
    
    try:
        import torch
        
        # First check if we have sufficient GPU memory
        if not torch.cuda.is_available():
            print("⚠️  Skipping vLLM test - no GPU available")
            return False
        
        gpu_count = torch.cuda.device_count()
        print(f"📊 Using {gpu_count} GPU(s) (Physical GPUs 6 & 7)")
        
        # Check available memory before attempting test
        total_free_memory = 0
        for gpu_id in range(gpu_count):
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
            cached = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            free_memory = total_memory - cached
            total_free_memory += free_memory
            actual_gpu_id = 6 + gpu_id
            print(f"   GPU {gpu_id} (Physical GPU {actual_gpu_id}): {free_memory:.2f} GiB free out of {total_memory:.2f} GiB total")
        
        print(f"📊 Total free memory: {total_free_memory:.2f} GiB")
        
        min_required = 3.0 if gpu_count >= 2 else 1.5  # More memory needed for tensor parallelism
        if total_free_memory < min_required:
            print(f"❌ Insufficient GPU memory ({total_free_memory:.2f} GiB free, need {min_required} GiB)")
            print("   💡 Cannot run vLLM test - GPU memory is occupied")
            print("   🔧 To free GPU memory:")
            print("      1. Check GPU processes: nvidia-smi")
            print("      2. Kill GPU processes: sudo fuser -k /dev/nvidia*")
            print("      3. Clear PyTorch cache: python -c 'import torch; torch.cuda.empty_cache()'")
            return False
        
        # Configure tensor parallelism based on available GPUs
        tensor_parallel_size = min(gpu_count, 2)  # Use up to 2 GPUs
        print(f"⏳ Attempting vLLM test with tensor_parallel_size={tensor_parallel_size}...")
        
        try:
            from vllm import LLM, SamplingParams
            
            # Use conservative settings optimized for GPUs 6 & 7
            llm_kwargs = {
                "model": "facebook/opt-125m",  # Small model for testing
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": 0.2,  # Use only 20% of available memory
                "max_model_len": 512,  # Moderate context length
                "max_num_seqs": 2,    # Process 2 sequences at a time
                "enforce_eager": True,  # Disable optimizations for stability
                "trust_remote_code": True,
            }
            
            print(f"🚀 Initializing vLLM with settings:")
            for key, value in llm_kwargs.items():
                print(f"   {key}: {value}")
            
            llm = LLM(**llm_kwargs)
            
            # Generate a simple response
            sampling_params = SamplingParams(
                max_tokens=5, 
                temperature=0.0,
                top_p=1.0
            )
            prompts = ["Hello, how are you?"]
            outputs = llm.generate(prompts, sampling_params)
            
            print("✅ vLLM Test with Tensor Parallelism: Passed")
            print(f"   Input: '{prompts[0]}'")
            print(f"   Generated: '{outputs[0].outputs[0].text.strip()}'")
            print(f"   🎯 Successfully used {tensor_parallel_size} GPU(s) in tensor parallel mode")
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ vLLM Test: Failed")
            
            if "memory" in error_msg.lower() or "cuda out of memory" in error_msg.lower():
                print("   💭 Memory-related error detected")
                print("   🔧 Try these solutions:")
                print("      - Free up more GPU memory on GPUs 6 & 7")
                print("      - Use single GPU: set tensor_parallel_size=1")
                print("      - Lower gpu_memory_utilization (currently 0.2)")
                print("      - Use CPU fallback: CUDA_VISIBLE_DEVICES=''")
            elif "tensor parallel" in error_msg.lower():
                print("   💭 Tensor parallelism configuration issue")
                print("   🔧 Possible solutions:")
                print("      - Ensure both GPUs 6 & 7 are available")
                print("      - Check NCCL installation for multi-GPU communication")
                print("      - Try single GPU mode first")
            else:
                print(f"   Error: {error_msg}")
                print("   This might be due to model download, network, or configuration issues")
            
            return False
            
    except ImportError:
        print("⚠️  Cannot run vLLM test - missing dependencies")
        return False

def main():
    """Main test function - configured for GPUs 6 & 7"""
    print("🔍 GPU Setup Test for vLLM Server Experiment (GPUs 6 & 7)")
    print("=" * 65)
    
    # Set environment variable at the start
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
    print("🎯 Environment configured to use GPUs 6 and 7 only")
    print("=" * 65)
    
    tests = [
        test_python_packages,
        test_gpu_availability,
        check_gpu_memory_details,
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
    print("\nWould you like to run a vLLM test with tensor parallelism? (This may take a few minutes)")
    print("This will download a small model and test vLLM with GPUs 6 & 7 in tensor parallel mode.")
    choice = input("Run test? (y/N): ").strip().lower()
    
    if choice in ('y', 'yes'):
        result = run_simple_test()
        results.append(result)
    
    # Summary
    print("\n" + "=" * 65)
    print("📊 TEST SUMMARY (GPUs 6 & 7)")
    print("=" * 65)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉 All tests passed! Your system is ready for GPU-accelerated vLLM experiments on GPUs 6 & 7.")
        print("\n💡 Next steps:")
        print("   1. Create or prepare your dataset")
        print("   2. Run: python src/experiment/server_exp.py --model MODEL_NAME --dataset DATASET_PATH")
        print("   3. The experiment will automatically use tensor parallelism with GPUs 6 & 7")
        print("   4. Check the results/ directory for outputs")
    elif passed >= total - 1:
        print("✅ Most tests passed. Your system should work with minor limitations on GPUs 6 & 7.")
        print("\n⚠️  Check the failed tests above and consider installing missing dependencies.")
    else:
        print("❌ Several tests failed. Please address the issues before running experiments.")
        print("\n🔧 Common fixes:")
        print("   - Install missing packages: pip install -r requirements-vllm.txt")
        print("   - Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   - Install vLLM: pip install vllm")
        print("   - Ensure GPUs 6 & 7 are available: nvidia-smi")
    
    print(f"\nTest Results: {passed}/{total} passed")
    print("🎯 Configured for GPUs 6 & 7 with tensor parallelism support")

if __name__ == "__main__":
    main()
