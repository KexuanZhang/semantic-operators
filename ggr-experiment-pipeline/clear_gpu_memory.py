#!/usr/bin/env python3
"""
GPU Memory Cleanup Script

This script helps clear GPU memory and provides diagnostics for vLLM experiments.
"""

import os
import sys
import subprocess
import time

def check_gpu_processes():
    """Check what processes are using GPU"""
    print("🔍 Checking GPU processes...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print("❌ nvidia-smi command failed")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found. CUDA drivers may not be installed.")
        return False
    except Exception as e:
        print(f"❌ Error running nvidia-smi: {e}")
        return False

def clear_pytorch_cache():
    """Clear PyTorch GPU cache"""
    print("\n🧹 Clearing PyTorch GPU cache...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU memory before clearing: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB allocated")
            torch.cuda.empty_cache()
            print(f"GPU memory after clearing: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB allocated")
            print("✅ PyTorch cache cleared")
            return True
        else:
            print("⚠️  CUDA not available, nothing to clear")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        return False

def kill_python_processes():
    """Kill Python processes that might be using GPU"""
    print("\n⚠️  WARNING: This will kill ALL Python processes!")
    choice = input("Do you want to kill all Python processes? (y/N): ").strip().lower()
    
    if choice in ('y', 'yes'):
        try:
            print("🔪 Killing Python processes...")
            subprocess.run(['pkill', '-f', 'python'], check=False)
            time.sleep(2)
            print("✅ Python processes killed")
            return True
        except Exception as e:
            print(f"❌ Error killing processes: {e}")
            return False
    else:
        print("⏭️  Skipped killing Python processes")
        return False

def reset_gpu():
    """Reset GPU (requires sudo)"""
    print("\n⚠️  WARNING: This will reset the GPU and requires sudo!")
    choice = input("Do you want to reset the GPU? (y/N): ").strip().lower()
    
    if choice in ('y', 'yes'):
        try:
            print("🔄 Resetting GPU...")
            result = subprocess.run(['sudo', 'nvidia-smi', '--gpu-reset'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ GPU reset successful")
                return True
            else:
                print(f"❌ GPU reset failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error resetting GPU: {e}")
            return False
    else:
        print("⏭️  Skipped GPU reset")
        return False

def suggest_vllm_settings():
    """Suggest conservative vLLM settings based on available memory"""
    print("\n💡 Suggested vLLM Settings Based on Available Memory")
    print("-" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                cached = torch.cuda.memory_reserved(i) / (1024**3)
                free_memory = total_memory - cached
                
                print(f"\n🎯 GPU {i} Recommendations (Free: {free_memory:.1f}GB):")
                
                if free_memory > 10:
                    print("   Model: microsoft/DialoGPT-medium or meta-llama/Llama-2-7b-chat-hf")
                    print("   --gpu-memory-utilization 0.8")
                    print("   --max-model-len 2048")
                    print("   --dtype float16")
                elif free_memory > 5:
                    print("   Model: microsoft/DialoGPT-small")
                    print("   --gpu-memory-utilization 0.6")
                    print("   --max-model-len 1024")
                    print("   --dtype float16")
                elif free_memory > 2:
                    print("   Model: gpt2")
                    print("   --gpu-memory-utilization 0.3")
                    print("   --max-model-len 512")
                    print("   --dtype float16")
                    print("   --max-num-seqs 32")
                elif free_memory > 1:
                    print("   Model: gpt2")
                    print("   --gpu-memory-utilization 0.1")
                    print("   --max-model-len 256")
                    print("   --dtype float16")
                    print("   --max-num-seqs 16")
                else:
                    print("   ❌ Insufficient memory for GPU inference")
                    print("   💡 Use CPU fallback: CUDA_VISIBLE_DEVICES=''")
                    print("   Or free up GPU memory first")
        
    except Exception as e:
        print(f"❌ Error checking GPU memory: {e}")

def run_minimal_test():
    """Run a minimal test to see if GPU is working"""
    print("\n🧪 Running Minimal GPU Test")
    print("-" * 40)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False
        
        # Simple GPU operation
        print("🔄 Testing basic GPU operations...")
        device = torch.device('cuda:0')
        
        # Small tensor operations
        a = torch.rand(100, 100, device=device)
        b = torch.rand(100, 100, device=device)
        c = torch.matmul(a, b)
        
        print(f"✅ GPU computation successful!")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory used: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def main():
    """Main cleanup and diagnostic function"""
    print("🧹 GPU Memory Cleanup and Diagnostic Tool")
    print("=" * 50)
    
    # Check current GPU status
    gpu_available = check_gpu_processes()
    
    if not gpu_available:
        print("\n❌ No GPU detected or nvidia-smi not available")
        return
    
    # Clear PyTorch cache
    clear_pytorch_cache()
    
    # Show menu
    print("\n📋 Available Actions:")
    print("1. Kill Python processes (frees GPU memory)")
    print("2. Reset GPU (requires sudo)")
    print("3. Run minimal GPU test")
    print("4. Show recommended vLLM settings")
    print("5. Check GPU status again")
    print("6. Exit")
    
    while True:
        choice = input("\nSelect an action (1-6): ").strip()
        
        if choice == '1':
            kill_python_processes()
            check_gpu_processes()
        elif choice == '2':
            reset_gpu()
            time.sleep(3)
            check_gpu_processes()
        elif choice == '3':
            run_minimal_test()
        elif choice == '4':
            suggest_vllm_settings()
        elif choice == '5':
            check_gpu_processes()
        elif choice == '6':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-6.")

if __name__ == "__main__":
    main()
