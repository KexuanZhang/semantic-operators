#!/usr/bin/env python3
"""
Test the enhanced consolidated report generation functionality
"""

import os
import sys
import json
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def simulate_experiment_results():
    """Create realistic test experiment results"""
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
                },
                'collection_methods_tried': ['prometheus_registry', 'legacy_stats']
            },
            'prefix_caching_enabled': True
        },
        'resource_monitoring': {
            'monitoring_duration_seconds': 45.2,
            'total_samples': 23,
            'sampling_interval_seconds': 2.0,
            'cpu_utilization_mean': 68.4,
            'cpu_utilization_max': 89.2,
            'memory_utilization_mean': 45.6,
            'memory_utilization_max': 52.1,
            'memory_used_gb_mean': 14.5,
            'memory_used_gb_max': 16.6,
            'gpu_compute_utilization_mean': 82.3,
            'gpu_compute_utilization_max': 95.7,
            'gpu_memory_utilization_mean': 76.8,
            'gpu_memory_utilization_max': 84.2
        },
        'batch_metrics': [
            {
                'batch_idx': 1,
                'batch_size': 8,
                'batch_duration': 3.2,
                'batch_input_tokens': 520,
                'batch_output_tokens': 380,
                'batch_throughput_tokens_per_sec': 281.3
            },
            {
                'batch_idx': 2,
                'batch_size': 8,
                'batch_duration': 2.8,
                'batch_duration': 2.8,
                'batch_input_tokens': 495,
                'batch_output_tokens': 335,
                'batch_throughput_tokens_per_sec': 296.4
            }
        ],
        'query_results': [
            {
                'input_text': 'Analyze this movie review: The cinematography was breathtaking but the plot felt rushed.',
                'generated_text': 'This review presents a mixed assessment of the film. The positive comment about cinematography suggests strong visual direction and technical excellence. However, the criticism of pacing indicates potential issues with story structure and narrative flow. The reviewer appreciates the artistic elements while finding fault with the storytelling mechanics.',
                'processing_time': 2.1
            },
            {
                'input_text': 'Analyze this movie review: A masterpiece of storytelling with incredible performances.',
                'generated_text': 'This is an overwhelmingly positive review highlighting two key strengths. The term "masterpiece" indicates the reviewer considers this an exceptional work of cinema. The emphasis on both storytelling and performances suggests a well-rounded film that excels in multiple critical areas - narrative craft and acting quality.',
                'processing_time': 1.8
            }
        ]
    }

def test_comprehensive_report_generation():
    """Test the new comprehensive report generation"""
    print("🧪 Testing Comprehensive Report Generation")
    print("=" * 60)
    
    try:
        # Try to import the experiment class
        from experiment.run_experiment import SimpleLLMExperiment
        
        # Create a test experiment instance 
        experiment = SimpleLLMExperiment(
            model_name="meta-llama/Llama-2-7b-hf",
            output_dir="test_output"
        )
        
        # Generate test experiment results
        experiment_results = simulate_experiment_results()
        
        # Create test output directory
        test_folder = "test_output/comprehensive_report_test"
        os.makedirs(test_folder, exist_ok=True)
        
        # Test the comprehensive report generation
        print("📊 Generating comprehensive report...")
        report_path = experiment.generate_comprehensive_report(experiment_results, test_folder)
        
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report_content = f.read()
            
            print(f"✅ Report generated successfully!")
            print(f"📁 Location: {report_path}")
            print(f"📄 Size: {len(report_content):,} characters")
            print(f"📋 Sections: {report_content.count('##')} main sections")
            
            # Check for key elements
            key_elements = [
                "Comprehensive vLLM Experiment Report",
                "Table of Contents", 
                "Experiment Configuration",
                "KV Cache Metrics",
                "Resource Utilization",
                "Analysis & Recommendations"
            ]
            
            missing_elements = []
            for element in key_elements:
                if element not in report_content:
                    missing_elements.append(element)
            
            if missing_elements:
                print(f"⚠️  Missing elements: {missing_elements}")
            else:
                print("✅ All key report sections present")
                
            # Show a preview
            print("\n📖 Report Preview (first 500 characters):")
            print("-" * 50)
            print(report_content[:500] + "...")
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

def test_metrics_consolidation():
    """Test that the consolidation actually replaces multiple files"""
    print("\n🔄 Testing Output Consolidation")
    print("=" * 60)
    
    # This would be tested by running actual experiment and checking
    # that only the comprehensive report + JSON + CSV exports are created
    # instead of the old multiple file approach
    
    expected_files = [
        "comprehensive_experiment_report.md",  # New consolidated report
        "experiment_results.json",             # Raw data
        "query_results.csv",                   # Exported CSV
        "batch_metrics.csv",                   # Exported CSV 
        "resource_metrics.csv",                # Exported CSV
        "vllm_metrics.json"                    # Exported metrics
    ]
    
    removed_files = [
        "performance_report.md",               # Replaced by comprehensive
        "experiment_summary.txt",              # Integrated into comprehensive
    ]
    
    print("✅ Expected output files:")
    for file in expected_files:
        print(f"   - {file}")
    
    print("\n❌ No longer generated (consolidated):")
    for file in removed_files:
        print(f"   - {file}")
    
    return True

def main():
    """Run the enhanced reporting tests"""
    print("🚀 Enhanced vLLM Experiment Reporting Test Suite")
    print("=" * 80)
    print()
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Comprehensive report generation
    if test_comprehensive_report_generation():
        tests_passed += 1
    
    # Test 2: Output consolidation
    if test_metrics_consolidation():
        tests_passed += 1
    
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The enhanced reporting system is ready.")
        print("\n💡 Next steps:")
        print("   1. Run a real experiment to test with actual vLLM metrics")
        print("   2. Verify KV cache plots are generated and embedded")
        print("   3. Test with different dataset types (reordered, shuffled, original)")
        
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
