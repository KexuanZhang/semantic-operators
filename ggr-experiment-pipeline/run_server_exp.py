#!/usr/bin/env python3
"""
Example script for running vLLM server experiments with GPU configuration

This script demonstrates various configurations for running sentiment analysis
experiments with the vLLM server.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle the output"""
    print(f"\n{description}")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
        return result.returncode == 0
    except KeyboardInterrupt:
        print(f"\n⚠️ {description} interrupted by user")
        return False
    except Exception as e:
        print(f"❌ {description} failed with error: {e}")
        return False

def main():
    """Main function to run different experiment configurations"""
    
    # Check if server_exp.py exists
    server_script = Path("src/experiment/server_exp.py")
    if not server_script.exists():
        print(f"❌ Server script not found: {server_script}")
        print("Please ensure you're running this from the project root directory")
        sys.exit(1)
    
    # Check if sample dataset exists
    sample_dataset = Path("sample_dataset.csv")
    if not sample_dataset.exists():
        print(f"❌ Sample dataset not found: {sample_dataset}")
        print("Please create a sample dataset first")
        sys.exit(1)
    
    print("🚀 vLLM Server Experiment Runner")
    print("This script will run various GPU configurations for sentiment analysis")
    
    # Configuration options
    experiments = [
        {
            "name": "Single GPU with Small Model",
            "cmd": [
                sys.executable, str(server_script),
                "--model", "microsoft/DialoGPT-small",
                "--dataset", str(sample_dataset),
                "--max-tokens", "10",
                "--temperature", "0.0",
                "--tensor-parallel-size", "1",
                "--gpu-memory-utilization", "0.8",
                "--dtype", "auto"
            ]
        },
        {
            "name": "Single GPU with Memory Optimization",
            "cmd": [
                sys.executable, str(server_script),
                "--model", "microsoft/DialoGPT-medium",
                "--dataset", str(sample_dataset),
                "--max-tokens", "10",
                "--temperature", "0.0",
                "--tensor-parallel-size", "1",
                "--gpu-memory-utilization", "0.9",
                "--dtype", "float16",
                "--enable-chunked-prefill"
            ]
        },
        {
            "name": "Multi-GPU Setup (if available)",
            "cmd": [
                sys.executable, str(server_script),
                "--model", "microsoft/DialoGPT-medium",
                "--dataset", str(sample_dataset),
                "--max-tokens", "10",
                "--temperature", "0.0",
                "--tensor-parallel-size", "2",
                "--gpu-memory-utilization", "0.8",
                "--dtype", "float16"
            ]
        },
        {
            "name": "CPU Fallback Test",
            "cmd": [
                sys.executable, str(server_script),
                "--model", "microsoft/DialoGPT-small",
                "--dataset", str(sample_dataset),
                "--max-tokens", "5",
                "--temperature", "0.0",
                "--tensor-parallel-size", "1",
                "--gpu-memory-utilization", "0.8",
                "--dtype", "float32"
            ],
            "env": {"CUDA_VISIBLE_DEVICES": ""}
        }
    ]
    
    print(f"\nFound {len(experiments)} experiment configurations to run:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}")
    
    print("\nSelect which experiments to run:")
    print("  a) Run all experiments")
    print("  1-4) Run specific experiment")
    print("  q) Quit")
    
    choice = input("\nEnter your choice: ").strip().lower()
    
    if choice == 'q':
        print("👋 Goodbye!")
        return
    
    if choice == 'a':
        selected_experiments = experiments
    else:
        try:
            exp_num = int(choice)
            if 1 <= exp_num <= len(experiments):
                selected_experiments = [experiments[exp_num - 1]]
            else:
                print("❌ Invalid choice")
                return
        except ValueError:
            print("❌ Invalid choice")
            return
    
    # Run selected experiments
    results = []
    for exp in selected_experiments:
        print(f"\n🔄 Running: {exp['name']}")
        
        # Set environment if specified
        if 'env' in exp:
            import os
            for key, value in exp['env'].items():
                os.environ[key] = value
        
        success = run_command(exp['cmd'], exp['name'])
        results.append((exp['name'], success))
    
    # Summary
    print("\n" + "="*80)
    print("📊 EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    for name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {name}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {successful}/{total} experiments completed successfully")
    
    if successful > 0:
        print("\n📁 Check the 'results/' directory for detailed experiment outputs")
        print("📈 Analyze the metrics data to understand GPU performance")

if __name__ == "__main__":
    main()
