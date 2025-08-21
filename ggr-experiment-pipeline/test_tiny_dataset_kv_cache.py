#!/usr/bin/env python3
"""
Test script to validate KV cache stats output during actual vLLM inference
using a tiny dataset. This test focuses on verifying we get real values for
KV cache hit rates and other metrics during live inference.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TinyDatasetKVCacheTest:
    """Test KV cache stats with a tiny dataset during actual inference"""
    
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-small"  # Small model for fast testing
        self.llm = None
        self.metrics_collector = None
        self.test_results = {}
        
        # Tiny test dataset - designed to trigger cache hits with shared prefixes
        self.tiny_dataset = [
            # Group 1: Movie reviews (shared prefix for cache testing)
            "Analyze this movie review: 'The cinematography was absolutely stunning.'",
            "Analyze this movie review: 'The cinematography was mediocre at best.'", 
            "Analyze this movie review: 'The cinematography captured every emotion.'",
            
            # Group 2: Restaurant reviews (different shared prefix)
            "Evaluate this restaurant review: 'The service was exceptional.'",
            "Evaluate this restaurant review: 'The service was disappointing.'",
            "Evaluate this restaurant review: 'The service exceeded expectations.'",
            
            # Group 3: Product reviews (another shared prefix)
            "Rate this product review: 'The quality is outstanding.'",
            "Rate this product review: 'The quality is questionable.'",
            "Rate this product review: 'The quality surpassed my expectations.'"
        ]
    
    def setup_vllm(self) -> bool:
        """Set up vLLM with optimal settings for KV cache testing"""
        print("🚀 Setting up vLLM for KV cache testing...")
        
        try:
            from vllm import LLM, SamplingParams
            
            # Configure vLLM with prefix caching and small context for testing
            self.llm = LLM(
                model=self.model_name,
                max_model_len=1024,  # Small context for faster testing
                enable_prefix_caching=True,  # Essential for KV cache hit testing
                gpu_memory_utilization=0.5,  # Conservative for testing
                tensor_parallel_size=1,
                disable_log_stats=False  # Enable internal stats
                # Note: disable_log_requests removed for compatibility
            )
            
            print(f"✅ vLLM initialized successfully")
            print(f"   Model: {self.model_name}")
            print(f"   Prefix caching: ENABLED")
            print(f"   Max context: 1024 tokens")
            print(f"   Test dataset size: {len(self.tiny_dataset)} prompts")
            
            return True
            
        except ImportError as e:
            print(f"❌ vLLM import failed: {e}")
            print("   Install with: pip install vllm")
            return False
        except Exception as e:
            print(f"❌ vLLM setup failed: {e}")
            return False
    
    def setup_metrics_collector(self) -> bool:
        """Set up the enhanced metrics collector"""
        print("📊 Setting up KV cache metrics collector...")
        
        try:
            from experiment.run_experiment import VLLMMetricsCollector
            
            # Create collector with the vLLM instance
            self.metrics_collector = VLLMMetricsCollector(llm_instance=self.llm)
            
            print("✅ Metrics collector initialized")
            print("   Internal Stats access: ENABLED")
            
            return True
            
        except ImportError as e:
            print(f"❌ Metrics collector import failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Metrics collector setup failed: {e}")
            return False
    
    def run_inference_and_collect_stats(self) -> dict:
        """Run inference on tiny dataset and collect KV cache stats"""
        print("\n🧪 Running inference with KV cache monitoring...")
        print("=" * 60)
        
        test_result = {
            'test_name': 'tiny_dataset_kv_cache_inference',
            'success': False,
            'inference_results': [],
            'stats_snapshots': [],
            'final_analysis': {}
        }
        
        try:
            from vllm import SamplingParams
            
            # Sampling parameters for consistent testing
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=50,  # Short responses for faster testing
                stop=["\n", ".", "!"]  # Stop early to focus on cache behavior
            )
            
            print(f"🔧 Inference settings:")
            print(f"   Max tokens per response: {sampling_params.max_tokens}")
            print(f"   Temperature: {sampling_params.temperature}")
            print(f"   Total prompts: {len(self.tiny_dataset)}")
            
            # Start monitoring
            self.metrics_collector.start_monitoring()
            time.sleep(0.5)  # Let monitoring stabilize
            
            # Collect baseline stats
            baseline_stats = self.metrics_collector.collect_internal_stats()
            test_result['stats_snapshots'].append({
                'stage': 'baseline',
                'stats': baseline_stats
            })
            
            print(f"\n📊 Baseline stats collected")
            print(f"   Collection method: {baseline_stats.get('collection_method', 'unknown')}")
            print(f"   Stats available: {baseline_stats.get('stats_available', False)}")
            
            # Process each prompt and collect stats
            print(f"\n🚀 Processing {len(self.tiny_dataset)} prompts...")
            
            for i, prompt in enumerate(self.tiny_dataset):
                print(f"\n📝 Prompt {i+1}/{len(self.tiny_dataset)}: {prompt[:40]}...")
                
                # Run single inference
                start_time = time.time()
                responses = self.llm.generate([prompt], sampling_params)
                inference_time = time.time() - start_time
                
                # Extract response
                response_text = responses[0].outputs[0].text if responses and responses[0].outputs else ""
                
                inference_result = {
                    'prompt_index': i,
                    'prompt': prompt,
                    'response': response_text,
                    'inference_time': inference_time
                }
                test_result['inference_results'].append(inference_result)
                
                print(f"   ⚡ Completed in {inference_time:.2f}s")
                print(f"   📝 Response: {response_text[:30]}...")
                
                # Wait and collect stats after this inference
                time.sleep(0.5)
                current_stats = self.metrics_collector.collect_internal_stats()
                
                stats_snapshot = {
                    'stage': f'after_prompt_{i+1}',
                    'prompt_index': i,
                    'stats': current_stats
                }
                test_result['stats_snapshots'].append(stats_snapshot)
                
                # Extract key metrics for immediate feedback
                if current_stats.get('stats_available'):
                    metrics = current_stats.get('metrics_found', {})
                    
                    # Look for cache hit rate metrics
                    hit_rate_found = False
                    for key, value in metrics.items():
                        if 'cache_hit' in key.lower() or 'hit_rate' in key.lower():
                            print(f"   🎯 {key}: {value}")
                            hit_rate_found = True
                    
                    # Look for cache usage metrics
                    usage_found = False
                    for key, value in metrics.items():
                        if 'cache_usage' in key.lower() or ('cache' in key.lower() and 'perc' in key.lower()):
                            print(f"   🗄️  {key}: {value}")
                            usage_found = True
                    
                    if not hit_rate_found and not usage_found:
                        print(f"   📊 Stats collected but no cache metrics found yet")
                else:
                    print(f"   ⚠️  Stats not available: {current_stats.get('error', 'unknown')}")
            
            # Final comprehensive stats collection
            print(f"\n📈 Collecting final comprehensive stats...")
            time.sleep(1.0)  # Wait for final metrics to update
            
            final_stats = self.metrics_collector.get_comprehensive_stats()
            test_result['stats_snapshots'].append({
                'stage': 'final_comprehensive',
                'stats': final_stats
            })
            
            # Stop monitoring
            self.metrics_collector.stop_monitoring()
            
            # Analyze results
            analysis = self.analyze_kv_cache_results(test_result)
            test_result['final_analysis'] = analysis
            
            # Determine success
            test_result['success'] = (
                len(test_result['inference_results']) == len(self.tiny_dataset) and
                len(test_result['stats_snapshots']) > 0 and
                analysis.get('any_cache_metrics_found', False)
            )
            
            return test_result
            
        except Exception as e:
            test_result['error'] = str(e)
            print(f"❌ Inference test failed: {e}")
            import traceback
            traceback.print_exc()
            return test_result
    
    def analyze_kv_cache_results(self, test_result: dict) -> dict:
        """Analyze the collected KV cache stats and provide insights"""
        print("\n📊 Analyzing KV cache statistics...")
        print("=" * 60)
        
        analysis = {
            'any_cache_metrics_found': False,
            'cache_hit_rates': [],
            'cache_usage_values': [],
            'stats_progression': [],
            'key_findings': [],
            'recommendations': []
        }
        
        # Analyze each stats snapshot
        for snapshot in test_result['stats_snapshots']:
            stage = snapshot['stage']
            stats = snapshot['stats']
            
            stage_analysis = {
                'stage': stage,
                'stats_available': stats.get('stats_available', False),
                'collection_method': stats.get('collection_method', 'unknown'),
                'cache_metrics': {}
            }
            
            if stats.get('stats_available'):
                metrics = stats.get('metrics_found', {})
                
                # Extract cache-related metrics
                for key, value in metrics.items():
                    if any(term in key.lower() for term in ['cache', 'hit', 'prefix']):
                        stage_analysis['cache_metrics'][key] = value
                        analysis['any_cache_metrics_found'] = True
                        
                        # Categorize specific metrics
                        if 'hit_rate' in key.lower() or 'hit' in key.lower():
                            if isinstance(value, (int, float)) and 0 <= value <= 1:
                                analysis['cache_hit_rates'].append({
                                    'stage': stage,
                                    'metric': key,
                                    'value': value,
                                    'percentage': value * 100
                                })
                        
                        if 'usage' in key.lower() or ('cache' in key.lower() and 'perc' in key.lower()):
                            if isinstance(value, (int, float)):
                                analysis['cache_usage_values'].append({
                                    'stage': stage,
                                    'metric': key,
                                    'value': value
                                })
                
                # Check comprehensive stats if available
                if 'key_metrics' in stats:
                    key_metrics = stats['key_metrics']
                    for key, value in key_metrics.items():
                        if 'cache' in key.lower():
                            stage_analysis['cache_metrics'][f'comprehensive_{key}'] = value
                            analysis['any_cache_metrics_found'] = True
            
            analysis['stats_progression'].append(stage_analysis)
        
        # Generate findings
        print("🔍 Analysis Results:")
        
        if analysis['any_cache_metrics_found']:
            print("   ✅ KV cache metrics successfully collected!")
            analysis['key_findings'].append("✅ KV cache metrics are accessible during inference")
            
            # Analyze hit rates
            if analysis['cache_hit_rates']:
                hit_rates = [item['percentage'] for item in analysis['cache_hit_rates']]
                max_hit_rate = max(hit_rates)
                avg_hit_rate = sum(hit_rates) / len(hit_rates)
                
                print(f"   🎯 Cache hit rates found: {len(hit_rates)} measurements")
                print(f"      • Maximum hit rate: {max_hit_rate:.1f}%")
                print(f"      • Average hit rate: {avg_hit_rate:.1f}%")
                
                analysis['key_findings'].append(f"🎯 Cache hit rate: max {max_hit_rate:.1f}%, avg {avg_hit_rate:.1f}%")
                
                if max_hit_rate > 0:
                    analysis['key_findings'].append("🏆 Positive cache hit rate detected - prefix caching is working!")
                else:
                    analysis['key_findings'].append("⚠️ Zero hit rate - may need more similar prompts for cache benefits")
            
            # Analyze usage values
            if analysis['cache_usage_values']:
                usage_values = [item['value'] for item in analysis['cache_usage_values']]
                max_usage = max(usage_values)
                
                print(f"   🗄️  Cache usage values found: {len(usage_values)} measurements")
                print(f"      • Maximum usage: {max_usage}")
                
                analysis['key_findings'].append(f"🗄️ Cache usage monitored: max {max_usage}")
            
            # Stats progression analysis
            stages_with_cache = len([s for s in analysis['stats_progression'] if s['cache_metrics']])
            total_stages = len(analysis['stats_progression'])
            
            print(f"   📈 Cache stats progression: {stages_with_cache}/{total_stages} stages")
            analysis['key_findings'].append(f"📈 Cache monitoring coverage: {stages_with_cache}/{total_stages} stages")
            
        else:
            print("   ❌ No KV cache metrics found")
            analysis['key_findings'].append("❌ No KV cache metrics detected during inference")
            
            # Diagnose potential issues
            stats_available_count = sum(1 for s in analysis['stats_progression'] if s['stats_available'])
            if stats_available_count == 0:
                analysis['key_findings'].append("🔧 Issue: No stats collection working - check vLLM setup")
            else:
                analysis['key_findings'].append("🔧 Issue: Stats collection working but no cache metrics found")
        
        # Generate recommendations
        if analysis['any_cache_metrics_found']:
            if len(analysis['cache_hit_rates']) > 0:
                avg_hit_rate = sum(item['percentage'] for item in analysis['cache_hit_rates']) / len(analysis['cache_hit_rates'])
                if avg_hit_rate < 10:
                    analysis['recommendations'].append("💡 Try more prompts with longer shared prefixes to improve hit rates")
                elif avg_hit_rate > 50:
                    analysis['recommendations'].append("🚀 Excellent cache performance! Consider scaling to larger datasets")
                else:
                    analysis['recommendations'].append("👍 Good cache performance detected")
            
            analysis['recommendations'].append("✅ KV cache monitoring is working - ready for production experiments")
        else:
            analysis['recommendations'].append("🔧 Check vLLM prefix caching configuration")
            analysis['recommendations'].append("🔍 Verify enhanced metrics collector setup")
            analysis['recommendations'].append("📊 Consider using different collection methods")
        
        return analysis
    
    def print_final_report(self, test_result: dict):
        """Print comprehensive test report"""
        print("\n" + "=" * 80)
        print("📋 TINY DATASET KV CACHE TEST REPORT")
        print("=" * 80)
        
        print(f"🕒 Test completed at: {datetime.now().isoformat()}")
        print(f"🤖 Model used: {self.model_name}")
        print(f"📊 Dataset size: {len(self.tiny_dataset)} prompts")
        
        success = test_result.get('success', False)
        print(f"✅ Overall success: {'PASS' if success else 'FAIL'}")
        
        # Inference results summary
        inference_results = test_result.get('inference_results', [])
        if inference_results:
            total_time = sum(r['inference_time'] for r in inference_results)
            avg_time = total_time / len(inference_results)
            
            print(f"\n⚡ Inference Performance:")
            print(f"   • Total inference time: {total_time:.2f}s")
            print(f"   • Average per prompt: {avg_time:.2f}s")
            print(f"   • Prompts processed: {len(inference_results)}/{len(self.tiny_dataset)}")
        
        # KV cache analysis summary
        analysis = test_result.get('final_analysis', {})
        
        print(f"\n🎯 KV Cache Analysis:")
        key_findings = analysis.get('key_findings', [])
        for finding in key_findings:
            print(f"   {finding}")
        
        print(f"\n💡 Recommendations:")
        recommendations = analysis.get('recommendations', [])
        for rec in recommendations:
            print(f"   {rec}")
        
        # Show sample responses
        if inference_results:
            print(f"\n📝 Sample Responses:")
            for i, result in enumerate(inference_results[:3]):  # Show first 3
                prompt = result['prompt'][:50] + "..." if len(result['prompt']) > 50 else result['prompt']
                response = result['response'][:50] + "..." if len(result['response']) > 50 else result['response']
                print(f"   {i+1}. Prompt: {prompt}")
                print(f"      Response: {response}")
                print(f"      Time: {result['inference_time']:.2f}s")
        
        print(f"\n🎯 Test Result: {'✅ SUCCESS - KV cache monitoring is working!' if success else '❌ NEEDS ATTENTION - Check setup and configuration'}")
    
    def run_complete_test(self) -> dict:
        """Run the complete tiny dataset KV cache test"""
        print("🧪 Tiny Dataset KV Cache Stats Test")
        print("=" * 60)
        print("Testing real KV cache values during vLLM inference")
        print("Using shared prefixes to trigger cache hits")
        
        overall_result = {
            'test_suite': 'tiny_dataset_kv_cache',
            'start_time': datetime.now().isoformat(),
            'setup_success': False,
            'test_results': {}
        }
        
        # Setup phase
        print("\n🔧 Setup Phase")
        print("-" * 30)
        
        if not self.setup_vllm():
            overall_result['setup_error'] = 'vLLM setup failed'
            return overall_result
        
        if not self.setup_metrics_collector():
            overall_result['setup_error'] = 'Metrics collector setup failed'
            return overall_result
        
        overall_result['setup_success'] = True
        
        # Test execution
        print("\n🚀 Test Execution Phase")
        print("-" * 30)
        
        try:
            test_result = self.run_inference_and_collect_stats()
            overall_result['test_results'] = test_result
            
            overall_result['end_time'] = datetime.now().isoformat()
            overall_result['overall_success'] = test_result.get('success', False)
            
            # Print final report
            self.print_final_report(test_result)
            
            # Save detailed results
            self.save_test_results(overall_result)
            
            return overall_result
            
        except Exception as e:
            overall_result['execution_error'] = str(e)
            print(f"❌ Test execution failed: {e}")
            return overall_result
    
    def save_test_results(self, results: dict):
        """Save test results to JSON file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f"tiny_dataset_kv_cache_test_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n💾 Test results saved to: {results_file}")
            
        except Exception as e:
            print(f"\n⚠️  Could not save results: {e}")

def main():
    """Main function to run tiny dataset KV cache test"""
    print("🚀 Tiny Dataset KV Cache Stats Validation")
    print("Testing real KV cache values with shared prefixes")
    print("=" * 60)
    
    test_suite = TinyDatasetKVCacheTest()
    
    try:
        results = test_suite.run_complete_test()
        
        success = results.get('overall_success', False)
        setup_success = results.get('setup_success', False)
        
        if setup_success and success:
            print(f"\n🎉 All tests completed successfully!")
            print(f"✅ KV cache stats are working with real inference!")
            return 0
        elif setup_success:
            print(f"\n⚠️  Tests completed but with issues.")
            print(f"🔧 Check the analysis above for recommendations.")
            return 1
        else:
            print(f"\n❌ Setup failed - cannot run inference tests.")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Test interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
