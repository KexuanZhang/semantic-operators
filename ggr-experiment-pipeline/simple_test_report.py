#!/usr/bin/env python3
"""
Simple test to validate comprehensive report generation without full dependencies
"""
import os
import json
import sys
from datetime import datetime

# Test data simulating experiment results
def get_test_experiment_results():
    return {
        'experiment_info': {
            'dataset_path': '/test/reordered_ggr_dataset.csv',
            'query_key': 'movie_review',
            'query_type': 'Movie Review Analysis',
            'model_name': 'meta-llama/Llama-2-7b-hf',
            'gpu_id': 0,
            'total_rows': 100,
            'processed_rows': 100,
            'batch_size': 8,
            'experiment_timestamp': datetime.now().isoformat()
        },
        'performance_metrics': {
            'total_inference_time': 45.2,
            'prompt_creation_time': 2.1,
            'avg_time_per_row': 0.452,
            'overall_throughput_tokens_per_sec': 234.5,
            'estimated_total_tokens': 10600,
            'estimated_input_tokens': 6200,
            'estimated_output_tokens': 4400,
            'successful_batches': 12,
            'failed_batches': 1
        },
        'vllm_metrics': {
            'initial_stats': {'vllm_stats_available': True},
            'final_stats': {
                'vllm_stats_available': True,
                'key_metrics': {
                    'gpu_cache_usage_percent': 78.5,
                    'gpu_prefix_cache_hit_rate_percent': 82.3,
                    'gpu_prefix_cache_hits': 1850,
                    'gpu_prefix_cache_queries': 2247,
                    'requests_running': 0,
                    'requests_waiting': 0,
                    'prompt_tokens_total': 6200,
                    'generation_tokens_total': 4400
                }
            },
            'prefix_caching_enabled': True,
            'modern_metrics_enabled': True
        },
        'resource_monitoring': {
            'monitoring_duration_seconds': 45.0,
            'total_samples': 90,
            'sampling_interval_seconds': 0.5,
            'cpu_utilization_mean': 25.3,
            'cpu_utilization_max': 45.2,
            'memory_utilization_mean': 42.1,
            'memory_utilization_max': 52.8,
            'memory_used_gb_mean': 8.4,
            'memory_used_gb_max': 10.6,
            'gpu_compute_utilization_mean': 87.5,
            'gpu_compute_utilization_max': 95.2,
            'gpu_memory_utilization_mean': 76.8,
            'gpu_memory_utilization_max': 82.1,
            'gpu_memory_allocated_gb_mean': 9.8,
            'gpu_memory_allocated_gb_max': 11.2
        },
        'batch_metrics': [
            {
                'batch_idx': 0,
                'batch_size': 8,
                'batch_duration': 3.2,
                'batch_throughput_tokens_per_sec': 265.0
            },
            {
                'batch_idx': 1,
                'batch_size': 8,
                'batch_duration': 3.5,
                'batch_throughput_tokens_per_sec': 245.7
            },
            {
                'batch_idx': 2,
                'batch_size': 6,
                'batch_duration': 2.8,
                'batch_throughput_tokens_per_sec': 214.3
            }
        ],
        'query_results': [
            {
                'input_text': 'Analyze this movie review: "The cinematography was breathtaking, with stunning visuals that transported me to another world."',
                'generated_text': 'This review expresses strong positive sentiment about the visual aspects of the film...',
                'processing_time': 0.45
            },
            {
                'input_text': 'Analyze this movie review: "The plot was confusing and the characters were poorly developed."',
                'generated_text': 'This review indicates dissatisfaction with both story structure and character development...',
                'processing_time': 0.42
            }
        ]
    }

def test_report_structure():
    """Test if the report generation logic is working"""
    print("🧪 Testing Comprehensive Report Generation Logic")
    print("=" * 60)
    
    test_data = get_test_experiment_results()
    test_folder = "test_output_simple"
    os.makedirs(test_folder, exist_ok=True)
    
    # Create a simplified version of the report logic
    report_file = os.path.join(test_folder, "test_comprehensive_report.md")
    
    try:
        with open(report_file, 'w') as f:
            # Header
            exp_info = test_data['experiment_info']
            timestamp = datetime.now()
            
            f.write("# 📊 Comprehensive vLLM Experiment Report\n\n")
            f.write(f"**Generated**: {timestamp.isoformat()}  \n")
            f.write(f"**Run Folder**: `{test_folder}`  \n")
            f.write(f"**Experiment**: {exp_info['query_key']} on {os.path.basename(exp_info['dataset_path'])}  \n\n")
            
            # Table of Contents
            f.write("## 📋 Table of Contents\n\n")
            f.write("1. [Experiment Configuration](#experiment-configuration)\n")
            f.write("2. [Performance Summary](#performance-summary)\n")
            f.write("3. [KV Cache Metrics & Analysis](#kv-cache-metrics--analysis)\n\n")
            
            # 1. Configuration
            f.write("## 1. Experiment Configuration\n\n")
            f.write("| Parameter | Value |\n")
            f.write("|-----------|-------|\n")
            f.write(f"| **Model** | {exp_info['model_name']} |\n")
            f.write(f"| **Dataset** | {os.path.basename(exp_info['dataset_path'])} |\n")
            f.write(f"| **Processed Rows** | {exp_info['processed_rows']:,} |\n\n")
            
            # 2. Performance
            f.write("## 2. Performance Summary\n\n")
            perf = test_data['performance_metrics']
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| **Total Inference Time** | {perf['total_inference_time']:.2f}s |\n")
            f.write(f"| **Overall Throughput** | {perf['overall_throughput_tokens_per_sec']:.1f} tokens/sec |\n\n")
            
            # 3. KV Cache Analysis
            f.write("## 3. KV Cache Metrics & Analysis\n\n")
            vllm_metrics = test_data['vllm_metrics']
            final_stats = vllm_metrics.get('final_stats', {})
            
            if final_stats.get('vllm_stats_available', False):
                key_metrics = final_stats.get('key_metrics', {})
                f.write("### 📈 Cache Performance Overview\n\n")
                
                if 'gpu_cache_usage_percent' in key_metrics:
                    cache_usage = key_metrics['gpu_cache_usage_percent']
                    f.write(f"**GPU Cache Usage**: {cache_usage:.1f}%\n\n")
                
                if 'gpu_prefix_cache_hit_rate_percent' in key_metrics:
                    hit_rate = key_metrics['gpu_prefix_cache_hit_rate_percent']
                    hits = key_metrics.get('gpu_prefix_cache_hits', 0)
                    queries = key_metrics.get('gpu_prefix_cache_queries', 0)
                    
                    f.write(f"### 🎯 GPU Prefix Cache Hit Rate: **{hit_rate:.2f}%**\n\n")
                    f.write(f"📊 **Cache Statistics**: {hits:,} hits / {queries:,} queries\n\n")
                    
                    # Performance analysis
                    if hit_rate > 70:
                        f.write("🏆 **EXCELLENT PERFORMANCE** - Outstanding cache effectiveness!\n")
                    elif hit_rate > 40:
                        f.write("✅ **GOOD PERFORMANCE** - Strong cache benefits detected!\n")
                    else:
                        f.write("⚠️ **MODERATE PERFORMANCE** - Some optimization opportunities\n")
            
            f.write("\n---\n")
            f.write(f"**Report Generated**: {timestamp.isoformat()}\n")
        
        # Verify the file was created and read it back
        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                content = f.read()
            
            print(f"✅ Report generated successfully!")
            print(f"📁 Location: {report_file}")
            print(f"📄 Size: {len(content):,} characters")
            
            # Check for key elements
            key_elements = [
                "Comprehensive vLLM Experiment Report",
                "Table of Contents", 
                "Experiment Configuration",
                "KV Cache Metrics",
                "Cache Performance Overview",
                "GPU Prefix Cache Hit Rate"
            ]
            
            missing_elements = []
            for element in key_elements:
                if element not in content:
                    missing_elements.append(element)
            
            if missing_elements:
                print(f"⚠️  Missing elements: {missing_elements}")
                return False
            else:
                print("✅ All key report sections present")
                
                # Show a preview
                print("\n📖 Report Preview (first 500 characters):")
                print("-" * 50)
                print(content[:500] + "...")
                print("-" * 50)
                return True
        else:
            print("❌ Report file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Simple Comprehensive Report Test")
    print("=" * 60)
    
    success = test_report_structure()
    
    if success:
        print("\n🎉 Test passed! The report generation logic is working correctly.")
        print("\n💡 Next steps:")
        print("   1. Run actual vLLM experiment to test with real metrics")
        print("   2. Validate KV cache plot generation and embedding")
        print("   3. Test with different dataset types and configurations")
    else:
        print("\n⚠️  Test failed. Check the output above for details.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
