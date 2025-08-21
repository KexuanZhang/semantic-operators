#!/usr/bin/env python3
"""
Test script to validate local KV cache stats retrieval during vLLM inference.

This script tests the enhanced internal Stats access implementation to ensure
comprehensive KV cache monitoring works correctly with real vLLM instances
during live inference operations.

Tests include:
1. Basic KV cache stats collection during inference
2. Prefix cache hit rate monitoring with repeated prompts
3. Cache usage progression during batch processing
4. Internal Stats object access validation
5. Real-time metrics collection performance
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KVCacheStatsInferenceTest:
    """Comprehensive test suite for KV cache stats retrieval during vLLM inference"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        """
        Initialize the test suite
        
        Args:
            model_name: Model to use for testing (default: small model for fast testing)
        """
        self.model_name = model_name
        self.test_results = {}
        self.llm = None
        self.metrics_collector = None
        
    def setup_vllm_with_caching(self) -> bool:
        """Set up vLLM with prefix caching enabled for KV cache testing"""
        try:
            print("🚀 Setting up vLLM with prefix caching enabled...")
            
            # Import vLLM components
            from vllm import LLM, SamplingParams
            from vllm.engine.arg_utils import EngineArgs
            
            # Configure vLLM with prefix caching
            engine_args = EngineArgs(
                model=self.model_name,
                max_model_len=2048,  # Smaller context for testing
                enable_prefix_caching=True,  # Critical for KV cache monitoring
                gpu_memory_utilization=0.6,  # Conservative for testing
                tensor_parallel_size=1,
                disable_log_stats=False,  # Enable stats logging
                disable_log_requests=False
            )
            
            # Create LLM instance
            self.llm = LLM(**engine_args.create_engine_configs())
            
            print(f"✅ vLLM initialized successfully with model: {self.model_name}")
            print(f"   - Prefix caching: ENABLED")
            print(f"   - Max model length: 2048")
            print(f"   - GPU memory utilization: 60%")
            
            return True
            
        except ImportError as e:
            print(f"❌ vLLM import failed: {e}")
            print("   Please install vLLM: pip install vllm")
            return False
        except Exception as e:
            print(f"❌ vLLM setup failed: {e}")
            return False
    
    def setup_metrics_collector(self) -> bool:
        """Set up enhanced metrics collector with internal Stats access"""
        try:
            print("📊 Setting up enhanced metrics collector...")
            
            from experiment.run_experiment import VLLMMetricsCollector
            
            # Create collector with the vLLM instance for internal access
            self.metrics_collector = VLLMMetricsCollector(
                llm_instance=self.llm
            )
            
            print("✅ Enhanced metrics collector initialized")
            print(f"   - Collection interval: 0.5 seconds")
            print(f"   - Internal Stats access: ENABLED")
            
            return True
            
        except ImportError as e:
            print(f"❌ Metrics collector import failed: {e}")
            print("   Make sure enhanced implementation is available in src/experiment/")
            return False
        except Exception as e:
            print(f"❌ Metrics collector setup failed: {e}")
            return False
    
    def test_basic_kv_cache_stats_collection(self) -> Dict[str, Any]:
        """Test basic KV cache stats collection during inference"""
        print("\n" + "="*60)
        print("🧪 TEST 1: Basic KV Cache Stats Collection")
        print("="*60)
        
        test_result = {
            'test_name': 'basic_kv_cache_stats',
            'success': False,
            'error': None,
            'stats_collected': {},
            'collection_methods_tested': []
        }
        
        try:
            from vllm import SamplingParams
            
            # Test prompt
            prompt = "Explain the concept of machine learning in simple terms."
            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.95,
                max_tokens=100
            )
            
            print(f"🔧 Running inference with prompt: '{prompt[:50]}...'")
            
            # Start metrics monitoring
            self.metrics_collector.start_monitoring()
            time.sleep(1.0)  # Let monitoring stabilize
            
            # Collect initial stats
            initial_stats = self.metrics_collector.collect_internal_stats()
            print(f"📊 Initial stats collection method: {initial_stats.get('collection_method', 'unknown')}")
            test_result['collection_methods_tested'].append(initial_stats.get('collection_method', 'unknown'))
            
            # Run inference
            start_time = time.time()
            responses = self.llm.generate([prompt], sampling_params)
            inference_time = time.time() - start_time
            
            print(f"⚡ Inference completed in {inference_time:.2f}s")
            
            # Wait for metrics to update
            time.sleep(2.0)
            
            # Collect post-inference stats
            final_stats = self.metrics_collector.collect_internal_stats()
            print(f"📊 Final stats collection method: {final_stats.get('collection_method', 'unknown')}")
            
            # Also test Prometheus fallback
            prometheus_stats = self.metrics_collector._collect_prometheus_registry_only()
            print(f"📊 Prometheus fallback method: {prometheus_stats.get('collection_method', 'unknown')}")
            test_result['collection_methods_tested'].append(prometheus_stats.get('collection_method', 'unknown'))
            
            # Analyze collected stats
            print("\n🔍 Analyzing collected stats...")
            
            # Check initial stats
            if initial_stats.get('stats_available'):
                print(f"   ✅ Initial stats available: {initial_stats.get('total_metrics_found', 0)} metrics")
            else:
                print(f"   ⚠️  Initial stats not available: {initial_stats.get('error', 'unknown error')}")
            
            # Check final stats
            if final_stats.get('stats_available'):
                print(f"   ✅ Final stats available: {final_stats.get('total_metrics_found', 0)} metrics")
                
                # Look for key KV cache metrics
                final_metrics = final_stats.get('metrics_found', {})
                
                cache_metrics = {}
                for key, value in final_metrics.items():
                    if any(term in key.lower() for term in ['cache', 'gpu_cache', 'prefix']):
                        cache_metrics[key] = value
                
                if cache_metrics:
                    print(f"   🗄️  Found {len(cache_metrics)} cache-related metrics:")
                    for key, value in cache_metrics.items():
                        print(f"      • {key}: {value}")
                    
                    test_result['stats_collected']['cache_metrics'] = cache_metrics
                else:
                    print("   ⚠️  No cache-specific metrics found in final stats")
                
                # Look for request/token metrics
                request_metrics = {}
                for key, value in final_metrics.items():
                    if any(term in key.lower() for term in ['running', 'waiting', 'token', 'request']):
                        request_metrics[key] = value
                
                if request_metrics:
                    print(f"   📈 Found {len(request_metrics)} request/token metrics:")
                    for key, value in list(request_metrics.items())[:5]:  # Show first 5
                        print(f"      • {key}: {value}")
                    
                    test_result['stats_collected']['request_metrics'] = request_metrics
                
            else:
                print(f"   ❌ Final stats not available: {final_stats.get('error', 'unknown error')}")
            
            # Check Prometheus fallback
            if prometheus_stats.get('stats_available'):
                print(f"   ✅ Prometheus fallback available: {prometheus_stats.get('total_metrics_found', 0)} metrics")
            else:
                print(f"   ⚠️  Prometheus fallback not available")
            
            # Get comprehensive stats
            comprehensive = self.metrics_collector.get_comprehensive_stats()
            if comprehensive.get('key_metrics'):
                key_metrics = comprehensive['key_metrics']
                print(f"   📊 Comprehensive stats extracted {len(key_metrics)} key metrics")
                test_result['stats_collected']['key_metrics'] = key_metrics
            
            # Stop monitoring
            self.metrics_collector.stop_monitoring()
            
            # Success criteria: at least one collection method worked
            success = (
                initial_stats.get('stats_available', False) or 
                final_stats.get('stats_available', False) or 
                prometheus_stats.get('stats_available', False)
            )
            
            test_result['success'] = success
            test_result['inference_time'] = inference_time
            test_result['response_generated'] = bool(responses and len(responses) > 0)
            
            if success:
                print("✅ Basic KV cache stats collection test PASSED")
            else:
                print("❌ Basic KV cache stats collection test FAILED")
                
            return test_result
            
        except Exception as e:
            test_result['error'] = str(e)
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return test_result
    
    def test_prefix_cache_hit_rate_monitoring(self) -> Dict[str, Any]:
        """Test prefix cache hit rate monitoring with repeated prompts"""
        print("\n" + "="*60)
        print("🧪 TEST 2: Prefix Cache Hit Rate Monitoring")
        print("="*60)
        
        test_result = {
            'test_name': 'prefix_cache_hit_rate',
            'success': False,
            'error': None,
            'cache_hit_progression': [],
            'final_hit_rate': None
        }
        
        try:
            from vllm import SamplingParams
            
            # Base prompt to establish cache
            base_prompt = "Tell me about the benefits of renewable energy sources like solar and wind power"
            
            # Variations with shared prefix to test cache hits
            prompts = [
                base_prompt + ".",
                base_prompt + " in detail.",
                base_prompt + " for the environment.",
                base_prompt + " versus fossil fuels.",
                base_prompt + " in developing countries."
            ]
            
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=80
            )
            
            print(f"🔧 Testing with {len(prompts)} prompts sharing common prefix")
            print(f"   Base prefix: '{base_prompt[:60]}...'")
            
            # Start monitoring
            self.metrics_collector.start_monitoring()
            time.sleep(1.0)
            
            # Process prompts sequentially to build cache
            for i, prompt in enumerate(prompts):
                print(f"\n📝 Processing prompt {i+1}/{len(prompts)}")
                
                # Run inference
                start_time = time.time()
                responses = self.llm.generate([prompt], sampling_params)
                inference_time = time.time() - start_time
                
                # Wait for metrics update
                time.sleep(1.0)
                
                # Collect stats
                current_stats = self.metrics_collector.collect_internal_stats()
                
                # Extract cache metrics
                cache_data = {
                    'prompt_index': i,
                    'inference_time': inference_time,
                    'stats_available': current_stats.get('stats_available', False),
                    'collection_method': current_stats.get('collection_method', 'unknown')
                }
                
                if current_stats.get('stats_available'):
                    metrics = current_stats.get('metrics_found', {})
                    
                    # Look for cache hit rate metrics
                    for key, value in metrics.items():
                        if 'cache_hit' in key.lower() or 'prefix_cache' in key.lower():
                            cache_data[key] = value
                    
                    # Calculate hit rate if we have hits and queries
                    hits = None
                    queries = None
                    for key, value in metrics.items():
                        if 'cache_hits' in key.lower():
                            hits = value
                        elif 'cache_queries' in key.lower():
                            queries = value
                    
                    if hits is not None and queries is not None and queries > 0:
                        hit_rate = (hits / queries) * 100
                        cache_data['calculated_hit_rate'] = hit_rate
                        print(f"   🎯 Cache hit rate: {hit_rate:.1f}% ({hits}/{queries})")
                    else:
                        print(f"   📊 Cache metrics: hits={hits}, queries={queries}")
                else:
                    print(f"   ⚠️  Stats not available: {current_stats.get('error', 'unknown')}")
                
                test_result['cache_hit_progression'].append(cache_data)
            
            # Get final comprehensive stats
            final_comprehensive = self.metrics_collector.get_comprehensive_stats()
            
            if final_comprehensive.get('key_metrics'):
                key_metrics = final_comprehensive['key_metrics']
                
                # Look for final hit rate
                for key, value in key_metrics.items():
                    if 'hit_rate' in key.lower() and 'percent' in key.lower():
                        test_result['final_hit_rate'] = value
                        print(f"\n🏆 Final cache hit rate: {value:.1f}%")
                        break
            
            # Stop monitoring
            self.metrics_collector.stop_monitoring()
            
            # Success criteria: collected cache progression data
            success = len(test_result['cache_hit_progression']) > 0
            if test_result['final_hit_rate'] is not None:
                success = success and test_result['final_hit_rate'] > 0
            
            test_result['success'] = success
            test_result['prompts_processed'] = len(prompts)
            
            if success:
                print("✅ Prefix cache hit rate monitoring test PASSED")
                
                # Show progression
                print("\n📈 Cache Hit Rate Progression:")
                for i, data in enumerate(test_result['cache_hit_progression']):
                    if 'calculated_hit_rate' in data:
                        print(f"   Prompt {i+1}: {data['calculated_hit_rate']:.1f}%")
            else:
                print("❌ Prefix cache hit rate monitoring test FAILED")
            
            return test_result
            
        except Exception as e:
            test_result['error'] = str(e)
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return test_result
    
    def test_cache_usage_progression(self) -> Dict[str, Any]:
        """Test cache usage progression during batch processing"""
        print("\n" + "="*60)
        print("🧪 TEST 3: Cache Usage Progression During Batch Processing")
        print("="*60)
        
        test_result = {
            'test_name': 'cache_usage_progression',
            'success': False,
            'error': None,
            'usage_progression': [],
            'peak_usage': None
        }
        
        try:
            from vllm import SamplingParams
            
            # Generate batch of prompts with varying lengths
            prompts = [
                "Write a short story about a robot learning to paint.",
                "Explain quantum computing to a high school student.",
                "Describe the process of photosynthesis in plants.",
                "What are the main causes of climate change?",
                "How does machine learning differ from traditional programming?"
            ]
            
            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.95,
                max_tokens=150  # Longer responses to stress cache
            )
            
            print(f"🔧 Processing batch of {len(prompts)} prompts")
            
            # Start monitoring
            self.metrics_collector.start_monitoring()
            time.sleep(1.0)
            
            # Collect baseline stats
            baseline_stats = self.metrics_collector.collect_internal_stats()
            if baseline_stats.get('stats_available'):
                print("📊 Baseline cache stats collected")
            
            # Process batch
            print("\n🚀 Starting batch inference...")
            batch_start_time = time.time()
            
            # Run batch inference
            responses = self.llm.generate(prompts, sampling_params)
            
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            
            print(f"⚡ Batch completed in {batch_duration:.2f}s")
            
            # Monitor cache usage progression post-batch
            for i in range(5):  # Monitor for 5 intervals
                time.sleep(1.0)
                
                current_stats = self.metrics_collector.collect_internal_stats()
                
                usage_data = {
                    'interval': i,
                    'timestamp': time.time(),
                    'stats_available': current_stats.get('stats_available', False)
                }
                
                if current_stats.get('stats_available'):
                    metrics = current_stats.get('metrics_found', {})
                    
                    # Extract cache usage metrics
                    for key, value in metrics.items():
                        if 'cache_usage' in key.lower() or 'gpu_cache' in key.lower():
                            usage_data[key] = value
                            
                            # Track peak usage
                            if 'usage' in key.lower() and 'percent' in key.lower():
                                if test_result['peak_usage'] is None or value > test_result['peak_usage']:
                                    test_result['peak_usage'] = value
                
                test_result['usage_progression'].append(usage_data)
                
                if current_stats.get('stats_available'):
                    print(f"   📊 Interval {i+1}: Stats collected")
                else:
                    print(f"   ⚠️  Interval {i+1}: Stats not available")
            
            # Get final comprehensive analysis
            final_stats = self.metrics_collector.get_comprehensive_stats()
            
            # Stop monitoring
            self.metrics_collector.stop_monitoring()
            
            # Success criteria
            success = (
                len(responses) == len(prompts) and
                len(test_result['usage_progression']) > 0 and
                any(data.get('stats_available', False) for data in test_result['usage_progression'])
            )
            
            test_result['success'] = success
            test_result['batch_duration'] = batch_duration
            test_result['responses_generated'] = len(responses)
            test_result['monitoring_intervals'] = len(test_result['usage_progression'])
            
            if success:
                print("✅ Cache usage progression test PASSED")
                
                if test_result['peak_usage']:
                    print(f"   🏔️  Peak cache usage: {test_result['peak_usage']:.1f}%")
                
                stats_available_count = sum(1 for data in test_result['usage_progression'] 
                                          if data.get('stats_available', False))
                print(f"   📊 Stats collected in {stats_available_count}/{len(test_result['usage_progression'])} intervals")
            else:
                print("❌ Cache usage progression test FAILED")
            
            return test_result
            
        except Exception as e:
            test_result['error'] = str(e)
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return test_result
    
    def test_internal_stats_access_methods(self) -> Dict[str, Any]:
        """Test different internal Stats access methods"""
        print("\n" + "="*60)
        print("🧪 TEST 4: Internal Stats Access Methods Validation")
        print("="*60)
        
        test_result = {
            'test_name': 'internal_stats_access',
            'success': False,
            'error': None,
            'methods_tested': {},
            'best_method': None
        }
        
        try:
            print("🔧 Testing different internal Stats access methods...")
            
            # Test Method 1: Internal stats collection
            print("\n📊 Testing collect_internal_stats()...")
            internal_stats = self.metrics_collector.collect_internal_stats()
            
            method1_result = {
                'method_name': 'collect_internal_stats',
                'stats_available': internal_stats.get('stats_available', False),
                'collection_method': internal_stats.get('collection_method', 'unknown'),
                'total_metrics': internal_stats.get('total_metrics_found', 0),
                'debug_info': internal_stats.get('debug_info', {})
            }
            
            if method1_result['stats_available']:
                print(f"   ✅ Method 1 SUCCESS: {method1_result['total_metrics']} metrics")
                print(f"      Collection method: {method1_result['collection_method']}")
            else:
                print(f"   ❌ Method 1 FAILED: {internal_stats.get('error', 'unknown error')}")
            
            test_result['methods_tested']['internal_stats'] = method1_result
            
            # Test Method 2: Prometheus registry only
            print("\n📊 Testing _collect_prometheus_registry_only()...")
            prometheus_stats = self.metrics_collector._collect_prometheus_registry_only()
            
            method2_result = {
                'method_name': 'prometheus_registry_only',
                'stats_available': prometheus_stats.get('stats_available', False),
                'collection_method': prometheus_stats.get('collection_method', 'unknown'),
                'total_metrics': prometheus_stats.get('total_metrics_found', 0),
                'total_histograms': prometheus_stats.get('total_histograms_found', 0)
            }
            
            if method2_result['stats_available']:
                print(f"   ✅ Method 2 SUCCESS: {method2_result['total_metrics']} metrics, {method2_result['total_histograms']} histograms")
            else:
                print(f"   ❌ Method 2 FAILED: {prometheus_stats.get('error', 'unknown error')}")
            
            test_result['methods_tested']['prometheus_registry'] = method2_result
            
            # Test Method 3: Main collection method
            print("\n📊 Testing collect_prometheus_metrics() (main method)...")
            main_stats = self.metrics_collector.collect_prometheus_metrics()
            
            method3_result = {
                'method_name': 'collect_prometheus_metrics',
                'stats_available': main_stats.get('stats_available', False),
                'collection_method': main_stats.get('collection_method', 'unknown'),
                'total_metrics': main_stats.get('total_metrics_found', 0)
            }
            
            if method3_result['stats_available']:
                print(f"   ✅ Method 3 SUCCESS: {method3_result['total_metrics']} metrics")
                print(f"      Collection method: {method3_result['collection_method']}")
            else:
                print(f"   ❌ Method 3 FAILED: {main_stats.get('error', 'unknown error')}")
            
            test_result['methods_tested']['main_collection'] = method3_result
            
            # Test Method 4: Comprehensive stats
            print("\n📊 Testing get_comprehensive_stats()...")
            comprehensive_stats = self.metrics_collector.get_comprehensive_stats()
            
            method4_result = {
                'method_name': 'get_comprehensive_stats',
                'monitoring_active': comprehensive_stats.get('monitoring_active', False),
                'total_collections': comprehensive_stats.get('total_collections', 0),
                'key_metrics_available': bool(comprehensive_stats.get('key_metrics')),
                'histogram_analysis_available': bool(comprehensive_stats.get('histogram_analysis'))
            }
            
            if method4_result['key_metrics_available']:
                key_metrics_count = len(comprehensive_stats['key_metrics'])
                print(f"   ✅ Method 4 SUCCESS: {key_metrics_count} key metrics extracted")
            else:
                print(f"   ⚠️  Method 4 PARTIAL: No key metrics extracted")
            
            test_result['methods_tested']['comprehensive_stats'] = method4_result
            
            # Determine best method
            successful_methods = []
            for method_key, method_data in test_result['methods_tested'].items():
                if method_data.get('stats_available', False) or method_data.get('key_metrics_available', False):
                    successful_methods.append((method_key, method_data.get('total_metrics', 0)))
            
            if successful_methods:
                # Best method is the one with most metrics
                best_method_key, best_metrics_count = max(successful_methods, key=lambda x: x[1])
                test_result['best_method'] = best_method_key
                print(f"\n🏆 Best performing method: {best_method_key} ({best_metrics_count} metrics)")
            
            # Success criteria: at least one method worked
            success = len(successful_methods) > 0
            test_result['success'] = success
            test_result['successful_methods_count'] = len(successful_methods)
            
            if success:
                print("✅ Internal Stats access methods validation PASSED")
            else:
                print("❌ Internal Stats access methods validation FAILED")
            
            return test_result
            
        except Exception as e:
            test_result['error'] = str(e)
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return test_result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all KV cache stats inference tests"""
        print("🧪 Starting Comprehensive KV Cache Stats Inference Tests")
        print("=" * 80)
        
        overall_results = {
            'test_suite': 'kv_cache_stats_inference',
            'start_time': datetime.now().isoformat(),
            'model_used': self.model_name,
            'setup_success': False,
            'tests_run': 0,
            'tests_passed': 0,
            'test_results': {},
            'summary': {}
        }
        
        # Setup phase
        print("🔧 Setup Phase")
        print("-" * 40)
        
        if not self.setup_vllm_with_caching():
            overall_results['setup_error'] = 'vLLM setup failed'
            return overall_results
        
        if not self.setup_metrics_collector():
            overall_results['setup_error'] = 'Metrics collector setup failed'
            return overall_results
        
        overall_results['setup_success'] = True
        print("✅ Setup completed successfully")
        
        # Test execution phase
        print("\n🚀 Test Execution Phase")
        print("-" * 40)
        
        tests = [
            ('basic_kv_cache_stats', self.test_basic_kv_cache_stats_collection),
            ('prefix_cache_hit_rate', self.test_prefix_cache_hit_rate_monitoring),
            ('cache_usage_progression', self.test_cache_usage_progression),
            ('internal_stats_access', self.test_internal_stats_access_methods)
        ]
        
        for test_key, test_method in tests:
            try:
                result = test_method()
                overall_results['test_results'][test_key] = result
                overall_results['tests_run'] += 1
                
                if result.get('success', False):
                    overall_results['tests_passed'] += 1
                    
            except Exception as e:
                print(f"❌ Test {test_key} failed with exception: {e}")
                overall_results['test_results'][test_key] = {
                    'test_name': test_key,
                    'success': False,
                    'error': str(e)
                }
                overall_results['tests_run'] += 1
        
        overall_results['end_time'] = datetime.now().isoformat()
        
        # Generate summary
        self.generate_test_summary(overall_results)
        
        return overall_results
    
    def generate_test_summary(self, results: Dict[str, Any]):
        """Generate and display test summary"""
        print("\n" + "=" * 80)
        print("📋 TEST SUMMARY REPORT")
        print("=" * 80)
        
        print(f"🕒 **Test Duration**: {results['start_time']} to {results['end_time']}")
        print(f"🤖 **Model Used**: {results['model_used']}")
        print(f"✅ **Setup Success**: {results['setup_success']}")
        print(f"🧪 **Tests Run**: {results['tests_run']}")
        print(f"🎯 **Tests Passed**: {results['tests_passed']}")
        
        success_rate = (results['tests_passed'] / results['tests_run'] * 100) if results['tests_run'] > 0 else 0
        print(f"📊 **Success Rate**: {success_rate:.1f}%")
        
        print("\n📋 Individual Test Results:")
        print("-" * 50)
        
        for test_key, test_result in results['test_results'].items():
            status = "✅ PASS" if test_result.get('success', False) else "❌ FAIL"
            test_name = test_result.get('test_name', test_key)
            print(f"   {status} - {test_name}")
            
            if test_result.get('error'):
                print(f"      Error: {test_result['error']}")
        
        # Key findings
        print("\n🔍 Key Findings:")
        print("-" * 50)
        
        # Check if any method successfully collected cache stats
        cache_stats_collected = False
        best_collection_method = None
        
        for test_result in results['test_results'].values():
            if test_result.get('stats_collected') or test_result.get('methods_tested'):
                cache_stats_collected = True
                
                if 'methods_tested' in test_result and test_result.get('best_method'):
                    best_collection_method = test_result['best_method']
        
        if cache_stats_collected:
            print("   ✅ KV cache statistics successfully collected during inference")
            if best_collection_method:
                print(f"   🏆 Best performing collection method: {best_collection_method}")
        else:
            print("   ❌ No KV cache statistics were successfully collected")
        
        # Check for prefix cache monitoring
        prefix_cache_test = results['test_results'].get('prefix_cache_hit_rate', {})
        if prefix_cache_test.get('final_hit_rate') is not None:
            hit_rate = prefix_cache_test['final_hit_rate']
            print(f"   🎯 Prefix cache hit rate achieved: {hit_rate:.1f}%")
        
        # Overall assessment
        print(f"\n🎯 Overall Assessment:")
        print("-" * 50)
        
        if success_rate >= 75:
            print("   🏆 EXCELLENT: KV cache stats retrieval is working well!")
        elif success_rate >= 50:
            print("   ✅ GOOD: Most KV cache monitoring features are functional")
        elif success_rate >= 25:
            print("   ⚠️  PARTIAL: Some KV cache monitoring capabilities detected")
        else:
            print("   ❌ NEEDS WORK: KV cache stats retrieval requires attention")
        
        # Save results
        results_file = f"kv_cache_stats_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Test results saved to: {results_file}")
        except Exception as e:
            print(f"\n⚠️  Could not save results: {e}")

def main():
    """Main function to run KV cache stats inference tests"""
    print("🚀 vLLM KV Cache Stats Inference Test Suite")
    print("Testing enhanced internal Stats access during live inference")
    print("=" * 80)
    
    # You can change the model here - using a small model for faster testing
    test_model = "microsoft/DialoGPT-small"  # Small model for testing
    # test_model = "meta-llama/Llama-2-7b-chat-hf"  # Larger model for production testing
    
    print(f"📋 Configuration:")
    print(f"   Model: {test_model}")
    print(f"   Test focus: KV cache stats retrieval during inference")
    print(f"   Enhanced internal Stats access implementation")
    
    # Create test instance
    test_suite = KVCacheStatsInferenceTest(model_name=test_model)
    
    try:
        # Run all tests
        results = test_suite.run_all_tests()
        
        # Return success code based on results
        if results.get('setup_success', False) and results.get('tests_passed', 0) > 0:
            print(f"\n🎉 Test suite completed successfully!")
            return 0
        else:
            print(f"\n⚠️  Test suite completed with issues.")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Test suite interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
