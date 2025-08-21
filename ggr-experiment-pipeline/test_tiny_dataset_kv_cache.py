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
    
    def __init__(self, gpu_id: int = 0):
        # Try multiple fallback models in case of download issues
        self.model_options = [
            "gpt2",  # Most likely to be cached locally
            "microsoft/DialoGPT-small",  # Original choice
            "distilgpt2",  # Smaller alternative
        ]
        self.model_name = None  # Will be set during setup
        self.gpu_id = gpu_id  # GPU device to use
        self.llm = None
        self.metrics_collector = None
        self.test_results = {}
        
        # Set GPU device via environment variable (most reliable method)
        self.set_gpu_device(gpu_id)
        
        # High KV cache potential dataset - designed to maximize cache hits with very long shared prefixes
        self.tiny_dataset = [
            # Group 1: Very long shared prefixes with detailed context (should trigger significant cache hits)
            "Please analyze this detailed movie review in the context of modern cinema trends and audience expectations, considering both technical aspects and emotional impact: 'The cinematography was absolutely stunning and created an immersive experience.'",
            "Please analyze this detailed movie review in the context of modern cinema trends and audience expectations, considering both technical aspects and emotional impact: 'The cinematography was mediocre at best and failed to engage viewers.'",
            "Please analyze this detailed movie review in the context of modern cinema trends and audience expectations, considering both technical aspects and emotional impact: 'The cinematography captured every emotion perfectly and elevated the storytelling.'",
            "Please analyze this detailed movie review in the context of modern cinema trends and audience expectations, considering both technical aspects and emotional impact: 'The cinematography was innovative and pushed creative boundaries.'",
            
            # Group 2: Another set with very long shared prefix but different domain
            "Given the following comprehensive restaurant review, please provide a detailed evaluation covering service quality, food presentation, ambiance, and overall dining experience: 'The service was exceptional with attentive staff and prompt delivery.'",
            "Given the following comprehensive restaurant review, please provide a detailed evaluation covering service quality, food presentation, ambiance, and overall dining experience: 'The service was disappointing with long waits and inattentive staff.'",
            "Given the following comprehensive restaurant review, please provide a detailed evaluation covering service quality, food presentation, ambiance, and overall dining experience: 'The service exceeded all expectations with personalized attention.'",
            "Given the following comprehensive restaurant review, please provide a detailed evaluation covering service quality, food presentation, ambiance, and overall dining experience: 'The service was professional and created a memorable dining experience.'",
            
            # Group 3: Extremely long shared prefix to maximize cache potential
            "As a professional product analyst, please conduct a thorough evaluation of this consumer review, analyzing sentiment, identifying key concerns or praise points, assessing the reviewer's experience level, and providing insights about product quality and customer satisfaction: 'The build quality is outstanding and exceeds industry standards.'",
            "As a professional product analyst, please conduct a thorough evaluation of this consumer review, analyzing sentiment, identifying key concerns or praise points, assessing the reviewer's experience level, and providing insights about product quality and customer satisfaction: 'The build quality is questionable with several manufacturing defects.'",
            "As a professional product analyst, please conduct a thorough evaluation of this consumer review, analyzing sentiment, identifying key concerns or praise points, assessing the reviewer's experience level, and providing insights about product quality and customer satisfaction: 'The build quality surpassed expectations with premium materials.'",
            "As a professional product analyst, please conduct a thorough evaluation of this consumer review, analyzing sentiment, identifying key concerns or praise points, assessing the reviewer's experience level, and providing insights about product quality and customer satisfaction: 'The build quality demonstrates excellent craftsmanship and attention to detail.'",
            
            # Group 4: Identical prefixes with minimal variation (maximum cache hit potential)
            "Translate and analyze the sentiment of this customer feedback for our quality assurance team: 'The product delivered exactly what was promised.'",
            "Translate and analyze the sentiment of this customer feedback for our quality assurance team: 'The product failed to meet basic expectations.'",
            "Translate and analyze the sentiment of this customer feedback for our quality assurance team: 'The product exceeded our initial requirements.'",
            "Translate and analyze the sentiment of this customer feedback for our quality assurance team: 'The product represents excellent value for money.'",
            
            # Group 5: Repeated processing of same prefix pattern (should show progressive cache hits)
            "Process this business review following our standard evaluation protocol: 'Management was responsive and addressed all concerns promptly.'",
            "Process this business review following our standard evaluation protocol: 'Management was unresponsive and ignored customer complaints.'",
            "Process this business review following our standard evaluation protocol: 'Management demonstrated strong leadership and customer focus.'",
        ]
    
    def set_gpu_device(self, gpu_id: int):
        """Set GPU device for vLLM to use"""
        import os
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_id >= gpu_count:
                print(f"⚠️  Warning: GPU {gpu_id} not available. Available GPUs: 0-{gpu_count-1}")
                print(f"   Falling back to GPU 0")
                gpu_id = 0
            
            # Set CUDA_VISIBLE_DEVICES to specify which GPU to use
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            print(f"🎯 GPU device set to: GPU {gpu_id}")
            if gpu_count > 0:
                try:
                    gpu_name = torch.cuda.get_device_name(gpu_id)
                    gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
                    print(f"   GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f} GB)")
                except:
                    print(f"   GPU {gpu_id}: Available")
        else:
            print(f"❌ CUDA not available - will use CPU (very slow)")
            self.gpu_id = -1  # Indicate CPU usage
    
    def setup_vllm(self) -> bool:
        """Set up vLLM with optimal settings for KV cache testing"""
        print("🚀 Setting up vLLM for KV cache testing...")
        
        try:
            from vllm import LLM, SamplingParams
            
            # Try different models in case of download issues
            for model_candidate in self.model_options:
                try:
                    print(f"🔄 Attempting to load model: {model_candidate}")
                    
                    # Configure vLLM with optimal settings for maximum KV cache hit potential
                    self.llm = LLM(
                        model=model_candidate,
                        max_model_len=2048,  # Larger context to accommodate long shared prefixes
                        enable_prefix_caching=True,  # Essential for KV cache hit testing
                        gpu_memory_utilization=0.6,  # Higher utilization for more cache space
                        tensor_parallel_size=1,
                        disable_log_stats=False,  # Enable internal stats
                        # Force prefix caching explicitly with optimal settings
                        enable_chunked_prefill=False,  # May conflict with prefix caching in some versions
                        enforce_eager=True,  # Avoid compilation issues
                        # Cache-optimized configuration
                        block_size=16,  # Standard block size for prefix caching
                        swap_space=8,  # More CPU offloading to support larger cache
                        max_num_seqs=1,  # Process one at a time to maximize cache reuse
                        # GPU device selection - MODIFY THIS LINE TO SET YOUR GPU
                        # Option 1: Single GPU (replace 0 with your desired GPU ID)
                        # device="cuda:0",  # Use GPU 0
                        # Option 2: Multiple GPUs  
                        # device="cuda:0,1",  # Use GPUs 0 and 1
                        # Option 3: Auto-detect (default)
                        # device="auto",  # Let vLLM choose
                        # Note: disable_log_requests removed for compatibility
                    )
                    
                    self.model_name = model_candidate  # Record successful model
                    print(f"✅ vLLM initialized successfully with model: {model_candidate}")
                    
                    # Verify prefix caching is actually enabled
                    try:
                        if hasattr(self.llm, 'llm_engine'):
                            engine = self.llm.llm_engine
                            if hasattr(engine, 'model_config'):
                                prefix_caching_enabled = getattr(engine.model_config, 'enable_prefix_caching', False)
                                print(f"   Prefix caching verified: {'✅ ENABLED' if prefix_caching_enabled else '❌ DISABLED'}")
                                if not prefix_caching_enabled:
                                    print("   ⚠️  WARNING: Prefix caching is not enabled in model config!")
                                    print("   This may be due to model compatibility or vLLM version")
                            else:
                                print("   Prefix caching status: Cannot verify (no model_config)")
                        else:
                            print("   Prefix caching status: Cannot verify (no llm_engine)")
                    except Exception as e:
                        print(f"   Prefix caching verification failed: {e}")
                    
                    print(f"   Max context: 1024 tokens")
                    print(f"   Test dataset size: {len(self.tiny_dataset)} prompts")
                    
                    return True
                    
                except Exception as model_error:
                    print(f"⚠️  Failed to load {model_candidate}: {model_error}")
                    continue
            
            # If all models failed
            print("❌ All model options failed to load")
            return False
            
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
            
            # Sampling parameters optimized for KV cache testing
            sampling_params = SamplingParams(
                temperature=0.0,  # Deterministic for consistent cache behavior
                top_p=1.0,        # No top-p filtering to ensure consistent token generation
                max_tokens=30,    # Moderate length to allow cache benefits without being too short
                # Removed stop tokens to allow natural completion and better cache utilization
            )
            
            print(f"🔧 Inference settings (optimized for KV cache testing):")
            print(f"   Max tokens per response: {sampling_params.max_tokens}")
            print(f"   Temperature: {sampling_params.temperature} (deterministic)")
            print(f"   Total prompts: {len(self.tiny_dataset)}")
            print(f"   Expected high cache hit rate due to long shared prefixes")
            
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
    
    def analyze_cache_potential(self, test_result: dict):
        """Analyze cache hit potential and effectiveness for high-overlap prompts"""
        print("\n🔬 High KV Cache Potential Analysis")
        print("-" * 50)
        
        # Analyze prompt structure for cache potential
        shared_prefix_analysis = {}
        
        # Group prompts by their long shared prefixes
        prefix_groups = {}
        for i, prompt in enumerate(self.tiny_dataset):
            # Find shared prefixes of different lengths
            for prefix_len in [50, 100, 150, 200]:
                if len(prompt) >= prefix_len:
                    prefix = prompt[:prefix_len]
                    if prefix not in prefix_groups:
                        prefix_groups[prefix] = []
                    prefix_groups[prefix].append(i)
        
        # Find the most promising cache groups
        cache_groups = [(prefix, indices) for prefix, indices in prefix_groups.items() if len(indices) >= 2]
        cache_groups.sort(key=lambda x: len(x[1]), reverse=True)
        
        print(f"📈 Cache Potential Analysis:")
        print(f"   • Total prompts: {len(self.tiny_dataset)}")
        print(f"   • Shared prefix groups found: {len(cache_groups)}")
        
        total_cache_potential = 0
        for i, (prefix, indices) in enumerate(cache_groups[:5]):  # Top 5 groups
            potential_hits = len(indices) - 1  # First is cache miss, rest are hits
            total_cache_potential += potential_hits
            print(f"   • Group {i+1}: {len(indices)} prompts, {potential_hits} potential cache hits")
            print(f"     Prefix: '{prefix[:60]}...'")
        
        expected_hit_rate = (total_cache_potential / len(self.tiny_dataset)) * 100
        print(f"   • Expected cache hit rate: {expected_hit_rate:.1f}%")
        print(f"   • Theoretical maximum hits: {total_cache_potential}/{len(self.tiny_dataset)}")
        
        # Analyze actual results
        stats_snapshots = test_result.get('stats_snapshots', [])
        final_stats = None
        
        for snapshot in reversed(stats_snapshots):
            if snapshot.get('stats', {}).get('stats_available', False):
                final_stats = snapshot['stats']
                break
        
        if final_stats and 'metrics_found' in final_stats:
            metrics = final_stats['metrics_found']
            
            # Look for cache hit metrics
            cache_metrics = {}
            for key, value in metrics.items():
                if 'cache' in key.lower():
                    cache_metrics[key] = value
            
            print(f"\n🎯 Actual Cache Performance:")
            if cache_metrics:
                for metric, value in cache_metrics.items():
                    if 'hit' in metric.lower() and 'rate' in metric.lower():
                        actual_rate = value * 100 if value <= 1 else value
                        print(f"   • {metric}: {actual_rate:.2f}%")
                    elif 'usage' in metric.lower():
                        usage_val = value * 100 if value <= 1 else value
                        print(f"   • {metric}: {usage_val:.2f}%")
                    else:
                        print(f"   • {metric}: {value}")
                
                # Performance comparison
                print(f"\n📊 Performance Analysis:")
                hit_rate_keys = [k for k in cache_metrics.keys() if 'hit' in k.lower() and 'rate' in k.lower()]
                if hit_rate_keys:
                    actual_hit_rate = cache_metrics[hit_rate_keys[0]]
                    actual_hit_rate = actual_hit_rate * 100 if actual_hit_rate <= 1 else actual_hit_rate
                    efficiency = (actual_hit_rate / max(expected_hit_rate, 1)) * 100
                    
                    print(f"   • Expected hit rate: {expected_hit_rate:.1f}%")
                    print(f"   • Actual hit rate: {actual_hit_rate:.1f}%") 
                    print(f"   • Cache efficiency: {efficiency:.1f}%")
                    
                    if efficiency > 80:
                        print("   🏆 EXCELLENT: Cache is working at near-optimal efficiency!")
                    elif efficiency > 50:
                        print("   ✅ GOOD: Cache is providing significant benefits")
                    elif efficiency > 20:
                        print("   ⚠️  MODERATE: Some cache benefits but room for improvement")
                    else:
                        print("   ❌ LOW: Cache not performing as expected - check configuration")
                else:
                    print("   ⚠️  No cache hit rate metrics found")
            else:
                print("   ❌ No cache-related metrics found")
        else:
            print(f"\n⚠️  No final stats available for analysis")
        
        return expected_hit_rate, cache_groups

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
    import argparse
    
    # Add command line argument for GPU selection
    parser = argparse.ArgumentParser(description='Test KV cache stats with tiny dataset')
    parser.add_argument('--gpu', type=int, default=0, 
                       help='GPU device ID to use (default: 0)')
    args = parser.parse_args()
    
    print("🚀 Tiny Dataset KV Cache Stats Validation")
    print("Testing real KV cache values with shared prefixes")
    print("=" * 60)
    print(f"🎯 Using GPU: {args.gpu}")
    
    test_suite = TinyDatasetKVCacheTest(gpu_id=args.gpu)
    
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
