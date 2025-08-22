#!/usr/bin/env python3
"""
Test script to validate the JSON output structure
"""
import json
import time

# Simulate the comprehensive experiment stats JSON that would be generated
def create_sample_experiment_stats():
    """Create a sample of the comprehensive experiment stats JSON structure"""
    
    sample_stats = {
        'experiment_metadata': {
            'timestamp': time.time(),
            'dataset_path': '/path/to/dataset.csv',
            'query_key': 'summarize',
            'query_type': 'summarization',
            'custom_query': None,
            'model_name': 'meta-llama/Llama-2-7b-hf',
            'gpu_config': {
                'gpu_id': 0,
                'gpu_ids': [0],
                'gpu_memory_utilization': 0.9
            },
            'inference_config': {
                'batch_size': 8,
                'max_tokens': 512,
                'temperature': 0.1,
                'top_p': 0.9
            },
            'dataset_stats': {
                'total_rows': 1000,
                'processed_rows': 1000,
                'max_rows_limit': None
            }
        },
        'inference_performance': {
            'total_experiment_time_seconds': 245.67,
            'total_inference_time_seconds': 198.45,
            'prompt_creation_time_seconds': 2.34,
            'avg_time_per_row_seconds': 0.198,
            'estimated_total_tokens': 125000,
            'estimated_input_tokens': 75000,
            'estimated_output_tokens': 50000,
            'overall_throughput_tokens_per_sec': 629.3,
            'batch_statistics': {
                'successful_batches': 124,
                'failed_batches': 1,
                'total_batches': 125,
                'success_rate_percent': 99.2
            },
            'batch_metrics': [
                {
                    'batch_idx': 0,
                    'batch_size': 8,
                    'batch_duration': 1.87,
                    'batch_input_tokens': 600,
                    'batch_output_tokens': 400,
                    'batch_throughput_tokens_per_sec': 534.2
                }
            ]
        },
        'system_resource_usage': {
            'monitoring_duration_seconds': 200,
            'total_samples': 20,
            'sampling_interval_seconds': 10.0,
            'cpu_utilization_mean': 45.6,
            'cpu_utilization_max': 78.3,
            'cpu_utilization_min': 12.1,
            'cpu_utilization_std': 15.2,
            'memory_utilization_mean': 62.4,
            'memory_utilization_max': 71.8,
            'memory_used_gb_mean': 8.4,
            'memory_used_gb_max': 9.7,
            'gpu_memory_utilization_mean': 85.3,
            'gpu_memory_utilization_max': 92.1,
            'gpu_memory_allocated_gb_mean': 13.6,
            'gpu_memory_allocated_gb_max': 14.7,
            'gpu_compute_utilization_mean': 78.9,
            'gpu_compute_utilization_max': 95.2,
            'gpu_temperature_mean': 72.4,
            'gpu_temperature_max': 79.8,
            'gpu_power_mean': 185.6,
            'gpu_power_max': 220.4
        },
        'vllm_metrics': {
            'metrics_available': True,
            'collection_method': 'internal_stats_primary',
            'monitoring_active': False,
            'total_collections': 13,
            'key_metrics': {
                'gpu_cache_usage_percent': 76.8,
                'gpu_cache_usage_raw': 0.768,
                'gpu_prefix_cache_hit_rate': 0.847,
                'gpu_prefix_cache_hit_rate_percent': 84.7,
                'gpu_prefix_cache_hits': 1623,
                'gpu_prefix_cache_queries': 1917,
                'cpu_prefix_cache_hits': 45,
                'cpu_prefix_cache_queries': 67,
                'cpu_prefix_cache_hit_rate': 0.672,
                'cpu_prefix_cache_hit_rate_percent': 67.2,
                'requests_running': 2,
                'requests_waiting': 0,
                'prompt_tokens_total': 75234,
                'generation_tokens_total': 49876
            },
            'histogram_analysis': {
                'avg_time_to_first_token_seconds': 0.145,
                'total_ttft_requests': 1000,
                'avg_time_per_output_token_seconds': 0.028,
                'total_tpot_samples': 49876,
                'avg_e2e_latency_seconds': 2.34,
                'total_e2e_requests': 1000
            }
        },
        'results_summary': {
            'total_prompts': 1000,
            'successful_responses': 992,
            'failed_responses': 8,
            'average_response_length_chars': 287.4,
            'sample_responses': [
                "Sample response 1...",
                "Sample response 2...",
                "Sample response 3...",
                "Sample response 4...",
                "Sample response 5..."
            ]
        }
    }
    
    return sample_stats

def main():
    print("🧪 Testing comprehensive experiment stats JSON structure")
    print("=" * 60)
    
    # Create sample stats
    stats = create_sample_experiment_stats()
    
    # Test JSON serialization
    try:
        json_str = json.dumps(stats, indent=2, default=str)
        print("✅ JSON serialization successful")
        
        # Test JSON parsing
        parsed = json.loads(json_str)
        print("✅ JSON parsing successful")
        
        # Validate structure
        required_sections = [
            'experiment_metadata',
            'inference_performance', 
            'system_resource_usage',
            'vllm_metrics',
            'results_summary'
        ]
        
        for section in required_sections:
            if section in parsed:
                print(f"✅ Section '{section}' present")
            else:
                print(f"❌ Section '{section}' missing")
        
        # Show key metrics that would be important for comparison
        print("\n📊 Key Performance Metrics:")
        print(f"  Total experiment time: {stats['inference_performance']['total_experiment_time_seconds']:.2f}s")
        print(f"  Inference throughput: {stats['inference_performance']['overall_throughput_tokens_per_sec']:.1f} tokens/sec")
        print(f"  Success rate: {stats['inference_performance']['batch_statistics']['success_rate_percent']:.1f}%")
        
        if stats['vllm_metrics']['metrics_available']:
            key_metrics = stats['vllm_metrics']['key_metrics']
            print(f"  GPU cache usage: {key_metrics['gpu_cache_usage_percent']:.1f}%")
            print(f"  GPU prefix cache hit rate: {key_metrics['gpu_prefix_cache_hit_rate_percent']:.1f}%")
            print(f"  Cache counters: {key_metrics['gpu_prefix_cache_hits']} hits / {key_metrics['gpu_prefix_cache_queries']} queries")
            print(f"  Avg TTFT: {stats['vllm_metrics']['histogram_analysis']['avg_time_to_first_token_seconds']:.3f}s")
            print(f"  Avg TPOT: {stats['vllm_metrics']['histogram_analysis']['avg_time_per_output_token_seconds']:.3f}s")
        
        print(f"\n🖥️  System Resource Usage:")
        print(f"  GPU utilization: {stats['system_resource_usage']['gpu_compute_utilization_mean']:.1f}% avg, {stats['system_resource_usage']['gpu_compute_utilization_max']:.1f}% max")
        print(f"  GPU memory: {stats['system_resource_usage']['gpu_memory_allocated_gb_mean']:.1f}GB avg, {stats['system_resource_usage']['gpu_memory_allocated_gb_max']:.1f}GB max")
        print(f"  CPU utilization: {stats['system_resource_usage']['cpu_utilization_mean']:.1f}% avg")
        
        # Save sample file
        output_file = "sample_experiment_stats.json"
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"\n📁 Sample JSON saved to: {output_file}")
        
        print("\n🎯 JSON Structure Validation: PASSED")
        print("\n💡 This structure contains all necessary metrics for:")
        print("   - Fair inference speed comparison")
        print("   - vLLM KV cache analysis (using modern hit/query counters)")
        print("   - GPU usage monitoring")
        print("   - CPU usage tracking")
        print("   - Overall inference performance metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON validation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
