#!/usr/bin/env python3
"""
Conservative vLLM Server Experiment Runner

This script runs the server experiment with very conservative memory settings
to work around GPU memory limitations.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_available_memory():
    """Check available GPU memory"""
    try:
        import torch
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            cached = torch.cuda.memory_reserved(0) / (1024**3)
            free_memory = total_memory - cached
            return free_memory, total_memory
        else:
            return 0, 0
    except:
        return 0, 0

def create_sample_dataset():
    """Create a small sample dataset for testing"""
    dataset_path = Path("minimal_test_dataset.csv")
    
    if not dataset_path.exists():
        print("📝 Creating minimal test dataset...")
        with open(dataset_path, 'w') as f:
            f.write("text\n")
            f.write("This product is amazing!\n")
            f.write("Terrible quality, very disappointed.\n")
            f.write("Good value for money.\n")
        print(f"✅ Created {dataset_path}")
    
    return str(dataset_path)

def run_cpu_experiment(dataset_path):
    """Run experiment on CPU as fallback"""
    print("\n🖥️  Running CPU Experiment (Fallback)")
    print("-" * 50)
    
    cmd = [
        sys.executable, "src/experiment/server_exp.py",
        "--model", "gpt2",
        "--dataset", dataset_path,
        "--max-tokens", "5",
        "--temperature", "0.0",
        "--result-dir", "results_cpu_test"
    ]
    
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
    
    print(f"Command: {' '.join(cmd)}")
    print("Environment: CUDA_VISIBLE_DEVICES=''")
    
    try:
        result = subprocess.run(cmd, env=env, timeout=300)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ CPU experiment timed out")
        return False
    except Exception as e:
        print(f"❌ CPU experiment failed: {e}")
        return False

def run_minimal_gpu_experiment(dataset_path, free_memory):
    """Run experiment with minimal GPU settings"""
    print(f"\n🎯 Running Minimal GPU Experiment ({free_memory:.1f}GB free)")
    print("-" * 50)
    
    # Choose model and settings based on available memory
    if free_memory > 3:
        model = "microsoft/DialoGPT-small"
        memory_util = "0.2"
        max_model_len = "1024"
        max_num_seqs = "32"
    elif free_memory > 1.5:
        model = "gpt2"
        memory_util = "0.1"
        max_model_len = "512"
        max_num_seqs = "16"
    else:
        print(f"❌ Insufficient GPU memory ({free_memory:.1f}GB)")
        print("💡 Falling back to CPU...")
        return run_cpu_experiment(dataset_path)
    
    cmd = [
        sys.executable, "src/experiment/server_exp.py",
        "--model", model,
        "--dataset", dataset_path,
        "--gpu-memory-utilization", memory_util,
        "--max-model-len", max_model_len,
        "--max-num-seqs", max_num_seqs,
        "--dtype", "float16",
        "--max-tokens", "5",
        "--temperature", "0.0",
        "--enforce-eager",
        "--result-dir", "results_minimal_gpu"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, timeout=600)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ GPU experiment timed out")
        return False
    except Exception as e:
        print(f"❌ GPU experiment failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Conservative vLLM Server Experiment Runner")
    print("=" * 55)
    
    # Check if we're in the right directory
    if not Path("src/experiment/server_exp.py").exists():
        print("❌ server_exp.py not found in src/experiment/")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Create minimal test dataset
    dataset_path = create_sample_dataset()
    
    # Check GPU memory
    free_memory, total_memory = check_available_memory()
    
    if total_memory > 0:
        print(f"\n📊 GPU Memory Status:")
        print(f"   Total: {total_memory:.1f} GiB")
        print(f"   Free:  {free_memory:.1f} GiB")
        
        if free_memory < 0.5:
            print("\n⚠️  WARNING: Very low GPU memory!")
            print("🔧 Recommendations:")
            print("   1. Run: python clear_gpu_memory.py")
            print("   2. Kill GPU processes: sudo fuser -k /dev/nvidia*")
            print("   3. Restart system if needed")
            
            choice = input("\nContinue anyway? (y/N): ").strip().lower()
            if choice not in ('y', 'yes'):
                print("👋 Exiting. Please free GPU memory first.")
                sys.exit(0)
    else:
        print("📊 No GPU detected, will use CPU")
    
    # Choose experiment type
    if total_memory == 0 or free_memory < 0.5:
        print("\n🖥️  Using CPU experiment")
        success = run_cpu_experiment(dataset_path)
    else:
        print("\n🎯 Using minimal GPU experiment")
        success = run_minimal_gpu_experiment(dataset_path, free_memory)
    
    # Results
    print("\n" + "=" * 55)
    if success:
        print("🎉 Experiment completed successfully!")
        print("\n📁 Results saved in:")
        if total_memory == 0 or free_memory < 0.5:
            print("   results_cpu_test/")
        else:
            print("   results_minimal_gpu/")
        
        print("\n💡 Next steps:")
        print("   1. Check the results directory for outputs")
        print("   2. If successful, try larger models with more memory")
        print("   3. Analyze the metrics and performance data")
    else:
        print("❌ Experiment failed")
        print("\n🔧 Troubleshooting:")
        print("   1. Check GPU memory: nvidia-smi")
        print("   2. Clear GPU processes: python clear_gpu_memory.py")
        print("   3. Check logs in results directory")
        print("   4. Try CPU fallback: CUDA_VISIBLE_DEVICES=''")

if __name__ == "__main__":
    main()
