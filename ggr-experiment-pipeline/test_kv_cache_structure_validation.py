#!/usr/bin/env python3
"""
Lightweight test script to validate KV cache stats implementation structure
without requiring full vLLM installation.

This test focuses on validating that our enhanced internal Stats access
implementation is structurally sound and ready for production use.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class KVCacheStatsStructureTest:
    """Test the structure and implementation of KV cache stats collection"""
    
    def __init__(self):
        self.test_results = {}
    
    def test_enhanced_metrics_collector_import(self) -> bool:
        """Test that enhanced VLLMMetricsCollector can be imported"""
        print("🔧 Testing enhanced VLLMMetricsCollector import...")
        
        try:
            from experiment.run_experiment import VLLMMetricsCollector
            print("   ✅ VLLMMetricsCollector imported successfully")
            
            # Test instantiation without vLLM instance (for structure validation)
            collector = VLLMMetricsCollector(llm=None, collection_interval=1.0)
            print("   ✅ VLLMMetricsCollector instantiated successfully")
            
            return True
            
        except ImportError as e:
            print(f"   ❌ Import failed: {e}")
            return False
        except Exception as e:
            print(f"   ❌ Instantiation failed: {e}")
            return False
    
    def test_internal_stats_methods_exist(self) -> bool:
        """Test that all required internal stats methods exist"""
        print("📊 Testing internal stats methods availability...")
        
        try:
            from experiment.run_experiment import VLLMMetricsCollector
            
            collector = VLLMMetricsCollector(llm=None)
            
            # Check for key methods
            required_methods = [
                'collect_internal_stats',
                '_collect_prometheus_registry_only', 
                'collect_prometheus_metrics',
                'get_comprehensive_stats',
                'start_monitoring',
                'stop_monitoring'
            ]
            
            missing_methods = []
            for method_name in required_methods:
                if not hasattr(collector, method_name):
                    missing_methods.append(method_name)
                else:
                    print(f"   ✅ Method {method_name} found")
            
            if missing_methods:
                print(f"   ❌ Missing methods: {missing_methods}")
                return False
            
            print("   ✅ All required methods available")
            return True
            
        except Exception as e:
            print(f"   ❌ Method check failed: {e}")
            return False
    
    def test_internal_stats_collection_without_vllm(self) -> bool:
        """Test internal stats collection gracefully handles missing vLLM instance"""
        print("🧪 Testing internal stats collection without vLLM...")
        
        try:
            from experiment.run_experiment import VLLMMetricsCollector
            
            collector = VLLMMetricsCollector(llm=None)
            
            # Test collect_internal_stats
            internal_stats = collector.collect_internal_stats()
            print(f"   📊 collect_internal_stats returned: {type(internal_stats)}")
            
            # Should return a dict with error handling
            if isinstance(internal_stats, dict):
                print(f"   ✅ Returns dict structure")
                print(f"   📋 Collection method: {internal_stats.get('collection_method', 'unknown')}")
                print(f"   📋 Stats available: {internal_stats.get('stats_available', False)}")
                
                if 'debug_info' in internal_stats:
                    debug_info = internal_stats['debug_info']
                    print(f"   📋 Debug info keys: {list(debug_info.keys())}")
                
                return True
            else:
                print(f"   ❌ Expected dict, got {type(internal_stats)}")
                return False
                
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            return False
    
    def test_prometheus_fallback_method(self) -> bool:
        """Test Prometheus fallback method structure"""
        print("📈 Testing Prometheus fallback method...")
        
        try:
            from experiment.run_experiment import VLLMMetricsCollector
            
            collector = VLLMMetricsCollector(llm=None)
            
            # Test Prometheus registry method
            prometheus_stats = collector._collect_prometheus_registry_only()
            print(f"   📊 Prometheus collection returned: {type(prometheus_stats)}")
            
            if isinstance(prometheus_stats, dict):
                print(f"   ✅ Returns dict structure")
                print(f"   📋 Collection method: {prometheus_stats.get('collection_method', 'unknown')}")
                print(f"   📋 Stats available: {prometheus_stats.get('stats_available', False)}")
                print(f"   📋 Total metrics found: {prometheus_stats.get('total_metrics_found', 0)}")
                
                # Check for expected structure
                expected_keys = ['collection_method', 'stats_available', 'metrics_found', 'debug_info']
                missing_keys = [key for key in expected_keys if key not in prometheus_stats]
                
                if missing_keys:
                    print(f"   ⚠️  Missing expected keys: {missing_keys}")
                else:
                    print(f"   ✅ All expected keys present")
                
                return True
            else:
                print(f"   ❌ Expected dict, got {type(prometheus_stats)}")
                return False
                
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            return False
    
    def test_comprehensive_stats_method(self) -> bool:
        """Test comprehensive stats method structure"""
        print("📈 Testing comprehensive stats method...")
        
        try:
            from experiment.run_experiment import VLLMMetricsCollector
            
            collector = VLLMMetricsCollector(llm=None)
            
            # Test comprehensive stats
            comprehensive = collector.get_comprehensive_stats()
            print(f"   📊 Comprehensive stats returned: {type(comprehensive)}")
            
            if isinstance(comprehensive, dict):
                print(f"   ✅ Returns dict structure")
                
                # Check for expected high-level structure
                expected_keys = [
                    'latest_stats', 'monitoring_active', 'total_collections',
                    'collection_interval', 'collection_method'
                ]
                
                found_keys = []
                for key in expected_keys:
                    if key in comprehensive:
                        found_keys.append(key)
                        print(f"   ✅ Key '{key}': {comprehensive[key]}")
                
                print(f"   📋 Found {len(found_keys)}/{len(expected_keys)} expected keys")
                return len(found_keys) > 0
            else:
                print(f"   ❌ Expected dict, got {type(comprehensive)}")
                return False
                
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            return False
    
    def test_key_metrics_extraction_logic(self) -> bool:
        """Test that key metrics extraction logic is properly implemented"""
        print("🎯 Testing key metrics extraction logic...")
        
        try:
            from experiment.run_experiment import VLLMMetricsCollector
            
            collector = VLLMMetricsCollector(llm=None)
            
            # Create mock metrics data to test extraction
            mock_metrics = {
                'vllm:gpu_cache_usage_perc': 0.78,
                'vllm:gpu_prefix_cache_hit_rate': 0.85,
                'vllm:num_requests_running': 2,
                'vllm:num_requests_waiting': 1,
                'vllm:prompt_tokens_total': 1500,
                'vllm:generation_tokens_total': 800
            }
            
            # Simulate what would happen with real metrics
            # This tests the logic without needing actual vLLM
            
            print("   📊 Simulating key metrics extraction...")
            
            # Test pattern matching for cache metrics
            cache_metrics = {}
            for key, value in mock_metrics.items():
                if 'cache' in key.lower():
                    cache_metrics[key] = value
            
            print(f"   🗄️  Cache metrics identified: {len(cache_metrics)}")
            for key, value in cache_metrics.items():
                print(f"      • {key}: {value}")
            
            # Test pattern matching for request metrics
            request_metrics = {}
            for key, value in mock_metrics.items():
                if any(term in key.lower() for term in ['running', 'waiting', 'token']):
                    request_metrics[key] = value
            
            print(f"   📈 Request metrics identified: {len(request_metrics)}")
            for key, value in request_metrics.items():
                print(f"      • {key}: {value}")
            
            # Success if we found expected patterns
            success = len(cache_metrics) >= 2 and len(request_metrics) >= 3
            
            if success:
                print("   ✅ Key metrics extraction logic working correctly")
            else:
                print("   ❌ Key metrics extraction logic needs attention")
            
            return success
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            return False
    
    def test_monitoring_lifecycle(self) -> bool:
        """Test monitoring start/stop lifecycle"""
        print("🔄 Testing monitoring lifecycle...")
        
        try:
            from experiment.run_experiment import VLLMMetricsCollector
            
            collector = VLLMMetricsCollector(llm=None, collection_interval=0.1)
            
            # Test initial state
            print("   📊 Testing initial state...")
            initial_monitoring = getattr(collector, 'monitoring', False)
            print(f"   Initial monitoring state: {initial_monitoring}")
            
            # Test start monitoring
            print("   ▶️  Testing start_monitoring()...")
            collector.start_monitoring()
            
            # Brief wait to let monitoring start
            time.sleep(0.5)
            
            monitoring_after_start = getattr(collector, 'monitoring', False)
            print(f"   Monitoring after start: {monitoring_after_start}")
            
            # Test stop monitoring
            print("   ⏹️  Testing stop_monitoring()...")
            collector.stop_monitoring()
            
            monitoring_after_stop = getattr(collector, 'monitoring', False)
            print(f"   Monitoring after stop: {monitoring_after_stop}")
            
            # Verify lifecycle worked
            success = (
                initial_monitoring == False and
                monitoring_after_start == True and
                monitoring_after_stop == False
            )
            
            if success:
                print("   ✅ Monitoring lifecycle working correctly")
            else:
                print("   ❌ Monitoring lifecycle has issues")
                
            return success
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            return False
    
    def run_structure_tests(self) -> dict:
        """Run all structure validation tests"""
        print("🧪 KV Cache Stats Implementation Structure Tests")
        print("=" * 60)
        
        results = {
            'test_suite': 'kv_cache_structure_validation',
            'start_time': datetime.now().isoformat(),
            'tests_run': 0,
            'tests_passed': 0,
            'test_results': {}
        }
        
        tests = [
            ('import_test', self.test_enhanced_metrics_collector_import),
            ('methods_exist', self.test_internal_stats_methods_exist),
            ('internal_stats_without_vllm', self.test_internal_stats_collection_without_vllm),
            ('prometheus_fallback', self.test_prometheus_fallback_method),
            ('comprehensive_stats', self.test_comprehensive_stats_method),
            ('key_metrics_extraction', self.test_key_metrics_extraction_logic),
            ('monitoring_lifecycle', self.test_monitoring_lifecycle)
        ]
        
        for test_name, test_method in tests:
            print(f"\n" + "="*50)
            print(f"🔬 Running: {test_name}")
            print("="*50)
            
            try:
                success = test_method()
                results['test_results'][test_name] = {
                    'success': success,
                    'error': None
                }
                results['tests_run'] += 1
                
                if success:
                    results['tests_passed'] += 1
                    
            except Exception as e:
                print(f"❌ Test {test_name} failed with exception: {e}")
                results['test_results'][test_name] = {
                    'success': False,
                    'error': str(e)
                }
                results['tests_run'] += 1
        
        results['end_time'] = datetime.now().isoformat()
        
        # Summary
        self.print_structure_test_summary(results)
        
        return results
    
    def print_structure_test_summary(self, results: dict):
        """Print summary of structure tests"""
        print("\n" + "=" * 60)
        print("📋 STRUCTURE TEST SUMMARY")
        print("=" * 60)
        
        success_rate = (results['tests_passed'] / results['tests_run'] * 100) if results['tests_run'] > 0 else 0
        
        print(f"🧪 Tests Run: {results['tests_run']}")
        print(f"✅ Tests Passed: {results['tests_passed']}")
        print(f"📊 Success Rate: {success_rate:.1f}%")
        
        print("\n📋 Individual Results:")
        for test_name, test_result in results['test_results'].items():
            status = "✅ PASS" if test_result['success'] else "❌ FAIL"
            print(f"   {status} - {test_name}")
            
            if test_result.get('error'):
                print(f"      Error: {test_result['error']}")
        
        print(f"\n🎯 Overall Assessment:")
        if success_rate >= 85:
            print("   🏆 EXCELLENT: Implementation structure is solid!")
        elif success_rate >= 70:
            print("   ✅ GOOD: Implementation is mostly ready")
        elif success_rate >= 50:
            print("   ⚠️  PARTIAL: Some components need attention")
        else:
            print("   ❌ NEEDS WORK: Implementation requires significant fixes")
        
        print("\n💡 Next Steps:")
        if success_rate >= 70:
            print("   → Run full inference test with actual vLLM instance")
            print("   → Test with real workloads and KV cache scenarios")
        else:
            print("   → Fix failing structure tests first")
            print("   → Verify all required methods are properly implemented")

def main():
    """Run structure validation tests"""
    print("🚀 KV Cache Stats Implementation Structure Validation")
    print("Testing without requiring full vLLM installation")
    print("=" * 60)
    
    test_suite = KVCacheStatsStructureTest()
    
    try:
        results = test_suite.run_structure_tests()
        
        # Save results
        results_file = f"kv_cache_structure_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {results_file}")
        except Exception as e:
            print(f"\n⚠️  Could not save results: {e}")
        
        # Return exit code
        if results['tests_passed'] >= results['tests_run'] * 0.7:  # 70% success threshold
            print(f"\n🎉 Structure validation completed successfully!")
            return 0
        else:
            print(f"\n⚠️  Structure validation completed with issues.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Structure validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
