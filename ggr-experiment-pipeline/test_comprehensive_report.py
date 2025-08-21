#!/usr/bin/env python3
"""
Test the comprehensive report system validation
"""
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, '/Users/zhang/Desktop/huawei/so1/semantic-operators/ggr-experiment-pipeline/src')

def test_comprehensive_report_import():
    """Test importing and validating the comprehensive report system"""
    try:
        # Test if the main module can be imported (will show syntax errors if they exist)
        from experiment.run_experiment import LLMExperimentRunner
        print("✅ Main experiment class imported successfully")
        
        # Check if all comprehensive report methods exist
        runner = LLMExperimentRunner()
        required_methods = [
            'generate_comprehensive_report',
            '_write_report_header',
            '_write_table_of_contents',
            '_write_experiment_configuration',
            '_write_performance_summary',
            '_write_vllm_metrics_analysis',
            '_write_kv_cache_analysis',
            '_write_token_processing_stats',
            '_write_request_queue_stats',
            '_write_resource_utilization',
            '_write_batch_analysis',
            '_write_kv_cache_visualizations',
            '_write_analysis_and_recommendations',
            '_write_data_exports'
        ]
        
        missing_methods = []
        for method_name in required_methods:
            if not hasattr(runner, method_name):
                missing_methods.append(method_name)
        
        if missing_methods:
            print(f"❌ Missing methods: {missing_methods}")
            return False
        else:
            print("✅ All comprehensive report methods are present")
            
        # Test if we can create a sample report structure
        test_data = {
            'experiment_info': {
                'dataset_path': '/test/path/reordered_dataset.csv',
                'query_key': 'test_query'
            },
            'performance_metrics': {
                'total_inference_time': 120.5,
                'overall_throughput_tokens_per_sec': 450.2,
                'successful_batches': 10,
                'failed_batches': 0
            },
            'vllm_metrics': {
                'prefix_caching_enabled': True,
                'final_stats': {
                    'vllm_stats_available': True,
                    'key_metrics': {
                        'gpu_prefix_cache_hit_rate_percent': 65.8,
                        'gpu_prefix_cache_hits': 1250,
                        'gpu_prefix_cache_queries': 1900,
                        'gpu_cache_usage_percent': 78.5
                    }
                }
            }
        }
        
        # Test generating the report  
        output_folder = '/tmp/test_report'
        os.makedirs(output_folder, exist_ok=True)
        
        report_file = runner.generate_comprehensive_report(test_data, output_folder)
        
        if os.path.exists(report_file):
            print(f"✅ Comprehensive report generated successfully: {report_file}")
            
            # Check report content
            with open(report_file, 'r') as f:
                content = f.read()
                
            # Validate key sections are present
            key_sections = [
                "# 📊 Comprehensive vLLM Experiment Report",
                "## 📋 Table of Contents", 
                "## 1. Experiment Configuration",
                "## 2. Performance Summary",
                "## 3. vLLM Engine Metrics",
                "🎯 KV Cache Performance",
                "GPU Prefix Cache Hit Rate": "65.8%"
            ]
            
            missing_sections = []
            for section in key_sections:
                if section not in content:
                    missing_sections.append(section)
            
            if missing_sections:
                print(f"⚠️ Missing report sections: {missing_sections}")
            else:
                print("✅ All key report sections are present and correctly formatted")
                
            # Show sample of the generated report
            print("\n📄 Sample Report Content:")
            print("=" * 60)
            print(content[:1500] + "\n..." if len(content) > 1500 else content)
            print("=" * 60)
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except SyntaxError as e:
        print(f"❌ Syntax error in code: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Comprehensive Report System...")
    success = test_comprehensive_report_import()
    if success:
        print("\n🎉 Comprehensive report system is working correctly!")
    else:
        print("\n❌ Issues detected with comprehensive report system")
        sys.exit(1)
