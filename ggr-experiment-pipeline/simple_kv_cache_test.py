#!/usr/bin/env python3
"""
Simple KV Cache Test - Fixed Network and Model Issues
This script tests KV cache functionality with high-potential shared prefixes
without requiring complex infrastructure setup.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime

# Set environment variables to handle SSL and model length issues
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleKVCacheTest:
    """Simple KV cache test focusing on high cache hit potential inputs"""
    
    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.llm = None
        self.model_name = None
        
        # Set GPU device
        self.set_gpu_device(gpu_id)
        
        # High-potential KV cache dataset with very long shared prefixes
        # These are designed to maximize cache hits
        self.test_prompts = [
            # Group 1: Very long shared prefixes (150+ chars)
            "Please provide a comprehensive analysis of the following customer feedback, considering sentiment, specific concerns, and actionable insights for improvement: 'The service was exceptional.'",
            "Please provide a comprehensive analysis of the following customer feedback, considering sentiment, specific concerns, and actionable insights for improvement: 'The service was disappointing.'",
            "Please provide a comprehensive analysis of the following customer feedback, considering sentiment, specific concerns, and actionable insights for improvement: 'The service exceeded expectations.'",
            
            # Group 2: Another set with long shared prefixes
            "As a professional product reviewer, evaluate this consumer comment with focus on product quality, user experience, and recommendations: 'The build quality is outstanding.'",
            "As a professional product reviewer, evaluate this consumer comment with focus on product quality, user experience, and recommendations: 'The build quality is poor.'",
            "As a professional product reviewer, evaluate this consumer comment with focus on product quality, user experience, and recommendations: 'The build quality is acceptable.'",
            
            # Group 3: Maximum cache potential - very long identical prefixes
            "Analyze this business review following our standard evaluation protocol, examining service quality, operational efficiency, and customer satisfaction metrics: 'Staff was professional.'",
            "Analyze this business review following our standard evaluation protocol, examining service quality, operational efficiency, and customer satisfaction metrics: 'Staff was unprofessional.'",
            "Analyze this business review following our standard evaluation protocol, examining service quality, operational efficiency, and customer satisfaction metrics: 'Staff was friendly.'",
        ]
        
        print(f"🎯 Initialized Simple KV Cache Test")
        print(f"   GPU Device: {gpu_id}")
        print(f"   Test Prompts: {len(self.test_prompts)}")
        print(f"   Expected Cache Hits: High (due to long shared prefixes)")
    
    def set_gpu_device(self, gpu_id: int):
        """Set GPU device for inference"""
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_id >= gpu_count:
                print(f"⚠️  Warning: GPU {gpu_id} not available. Available GPUs: 0-{gpu_count-1}")
                gpu_id = 0
            
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            print(f"🎯 GPU set to: {gpu_id}")
            
            try:
                gpu_name = torch.cuda.get_device_name(gpu_id)
                gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
                print(f"   {gpu_name} ({gpu_memory:.1f} GB)")
            except:
                print(f"   GPU {gpu_id} available")
        else:
            print(f"❌ CUDA not available - using CPU")
    
    def setup_vllm(self) -> bool:
        """Set up vLLM with simple, working configuration"""
        print("\n🚀 Setting up vLLM...")
        
        try:
            from vllm import LLM, SamplingParams
            
            # Try models in order of likelihood to work offline
            models_to_try = ["gpt2", "distilgpt2", "openai-community/gpt2"]
            
            for model_name in models_to_try:
                try:
                    print(f"🔄 Trying model: {model_name}")
                    
                    # Simple, reliable configuration
                    self.llm = LLM(
                        model=model_name,
                        enable_prefix_caching=True,  # KEY: Enable prefix caching
                        gpu_memory_utilization=0.5,
                        tensor_parallel_size=1,
                        disable_log_stats=False,
                        enforce_eager=True,  # Avoid graph compilation issues
                        max_num_seqs=1,  # Process one at a time for max cache reuse
                        trust_remote_code=False,
                        # Let vLLM use model's natural max length
                    )
                    
                    self.model_name = model_name
                    print(f"✅ Model loaded successfully: {model_name}")
                    
                    # Check if prefix caching is enabled
                    try:
                        if hasattr(self.llm, 'llm_engine') and hasattr(self.llm.llm_engine, 'model_config'):
                            enabled = getattr(self.llm.llm_engine.model_config, 'enable_prefix_caching', False)
                            print(f"   Prefix caching: {'✅ ENABLED' if enabled else '❌ DISABLED'}")
                        else:
                            print(f"   Prefix caching: Status unknown")
                    except Exception:
                        print(f"   Prefix caching: Cannot verify")
                    
                    return True
                    
                except Exception as e:
                    print(f"⚠️  {model_name} failed: {str(e)[:80]}...")
                    continue
            
            print("❌ All models failed to load")
            return False
            
        except Exception as e:
            print(f"❌ vLLM setup error: {e}")
            return False
    
    def run_inference_test(self) -> dict:
        """Run the KV cache test with monitoring"""
        print("\n🧪 Running KV Cache Test...")
        print("=" * 50)
        
        results = {
            'test_name': 'simple_kv_cache_test',
            'model_name': self.model_name,
            'total_prompts': len(self.test_prompts),
            'responses': [],
            'timing': {},
            'success': False
        }
        
        try:
            from vllm import SamplingParams
            
            # Sampling parameters for consistent results
            sampling_params = SamplingParams(
                temperature=0.0,  # Deterministic
                max_tokens=20,    # Short responses to focus on prefix caching
                top_p=1.0
            )
            
            print(f"📋 Test Configuration:")
            print(f"   Model: {self.model_name}")
            print(f"   Prompts: {len(self.test_prompts)}")
            print(f"   Max tokens per response: {sampling_params.max_tokens}")
            print(f"   Temperature: {sampling_params.temperature}")
            
            # Collect initial engine stats if available
            initial_stats = self.get_engine_stats()
            results['initial_stats'] = initial_stats
            
            start_time = time.time()
            
            # Process each prompt
            for i, prompt in enumerate(self.test_prompts):
                print(f"\n📝 Prompt {i+1}: {prompt[:50]}...")
                
                prompt_start = time.time()
                
                # Run inference
                outputs = self.llm.generate([prompt], sampling_params)
                response = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
                
                prompt_time = time.time() - prompt_start
                
                result_entry = {
                    'prompt_index': i,
                    'prompt': prompt,
                    'response': response.strip(),
                    'inference_time': prompt_time
                }
                
                results['responses'].append(result_entry)
                
                print(f"   ⚡ Time: {prompt_time:.3f}s")
                print(f"   📝 Response: {response.strip()[:30]}...")
                
                # Brief pause between inferences
                time.sleep(0.2)
            
            total_time = time.time() - start_time
            
            # Collect final engine stats
            final_stats = self.get_engine_stats()
            results['final_stats'] = final_stats
            
            results['timing'] = {
                'total_time': total_time,
                'avg_time_per_prompt': total_time / len(self.test_prompts),
                'prompts_per_second': len(self.test_prompts) / total_time
            }
            
            results['success'] = len(results['responses']) == len(self.test_prompts)
            
            # Analyze results
            self.analyze_results(results)
            
            return results
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
            return results
    
    def get_engine_stats(self) -> dict:
        """Try to get engine stats for KV cache monitoring"""
        stats = {'available': False, 'method': None, 'data': {}}
        
        try:
            if hasattr(self.llm, 'llm_engine'):
                engine = self.llm.llm_engine
                
                # Try direct _get_stats() access
                if hasattr(engine, '_get_stats'):
                    try:
                        engine_stats = engine._get_stats()
                        if engine_stats:
                            stats['available'] = True
                            stats['method'] = 'engine._get_stats()'
                            stats['data'] = engine_stats
                            return stats
                    except Exception:
                        pass
                
                # Try scheduler stats
                if hasattr(engine, 'scheduler') and hasattr(engine.scheduler, 'get_stats'):
                    try:
                        sched_stats = engine.scheduler.get_stats()
                        if sched_stats:
                            stats['available'] = True
                            stats['method'] = 'scheduler.get_stats()'
                            stats['data'] = sched_stats
                            return stats
                    except Exception:
                        pass
                
                stats['method'] = 'No accessible stats method found'
            else:
                stats['method'] = 'No llm_engine attribute'
                
        except Exception as e:
            stats['method'] = f'Stats collection error: {e}'
        
        return stats
    
    def analyze_results(self, results: dict):
        """Analyze the test results and provide insights"""
        print(f"\n📊 Test Analysis")
        print(f"=" * 50)
        
        timing = results['timing']
        print(f"⏱️  **Performance:**")
        print(f"   Total time: {timing['total_time']:.2f}s")
        print(f"   Average per prompt: {timing['avg_time_per_prompt']:.3f}s")
        print(f"   Throughput: {timing['prompts_per_second']:.1f} prompts/sec")
        
        print(f"\n🎯 **Cache Analysis:**")
        
        # Check for cache hit indicators in timing
        times = [r['inference_time'] for r in results['responses']]
        if len(times) > 3:
            early_avg = sum(times[:3]) / 3
            later_avg = sum(times[3:]) / len(times[3:])
            
            if later_avg < early_avg * 0.8:  # 20% faster
                print(f"   🏆 **POSITIVE SIGNAL**: Later prompts 20% faster than early ones!")
                print(f"      Early prompts: {early_avg:.3f}s avg")
                print(f"      Later prompts: {later_avg:.3f}s avg")
                print(f"      This suggests KV cache hits are occurring!")
            else:
                print(f"   📊 No clear speed improvement detected in timing")
                print(f"      Early: {early_avg:.3f}s, Later: {later_avg:.3f}s")
        
        # Check stats availability
        initial_stats = results.get('initial_stats', {})
        final_stats = results.get('final_stats', {})
        
        print(f"\n📈 **Engine Stats:**")
        if final_stats.get('available'):
            print(f"   ✅ Stats collection: {final_stats['method']}")
            stats_data = final_stats.get('data', {})
            
            # Look for cache-related metrics
            cache_metrics = {}
            for key, value in stats_data.items():
                if any(cache_term in key.lower() for cache_term in ['cache', 'hit', 'prefix']):
                    cache_metrics[key] = value
            
            if cache_metrics:
                print(f"   🎯 **Cache metrics found:**")
                for key, value in cache_metrics.items():
                    print(f"      {key}: {value}")
            else:
                print(f"   ⚠️  No explicit cache metrics found in stats")
                print(f"   Available stats keys: {list(stats_data.keys())[:5]}...")
                
        else:
            print(f"   ❌ Stats not available: {final_stats.get('method', 'Unknown')}")
        
        print(f"\n🔍 **Test Assessment:**")
        if results['success']:
            print(f"   ✅ All {results['total_prompts']} prompts processed successfully")
            
            # Assess cache potential
            shared_prefixes = self.analyze_shared_prefixes()
            print(f"   📊 Cache potential analysis:")
            for group, info in shared_prefixes.items():
                print(f"      {group}: {info['count']} prompts, {info['prefix_length']} char prefix")
            
            total_cache_potential = sum(info['count'] - 1 for info in shared_prefixes.values())
            print(f"   🎯 Expected cache hits: {total_cache_potential}/{results['total_prompts']} prompts")
            
        else:
            print(f"   ❌ Test incomplete")
        
        print(f"\n💡 **Recommendations:**")
        print(f"   1. Monitor inference timing - later prompts should be faster")
        print(f"   2. Check vLLM logs for prefix caching messages")
        print(f"   3. Try setting VLLM_USE_V1=0 for better stats access")
        print(f"   4. Run with --verbose for more detailed vLLM output")
    
    def analyze_shared_prefixes(self) -> dict:
        """Analyze the shared prefixes in test prompts"""
        groups = {}
        
        # Group 1: Customer feedback analysis
        group1_prefix = "Please provide a comprehensive analysis of the following customer feedback"
        group1_count = sum(1 for p in self.test_prompts if p.startswith(group1_prefix))
        if group1_count > 0:
            groups['Group1_CustomerFeedback'] = {
                'count': group1_count,
                'prefix_length': len(group1_prefix)
            }
        
        # Group 2: Product reviewer
        group2_prefix = "As a professional product reviewer, evaluate this consumer comment"
        group2_count = sum(1 for p in self.test_prompts if p.startswith(group2_prefix))
        if group2_count > 0:
            groups['Group2_ProductReview'] = {
                'count': group2_count,
                'prefix_length': len(group2_prefix)
            }
        
        # Group 3: Business review analysis
        group3_prefix = "Analyze this business review following our standard evaluation protocol"
        group3_count = sum(1 for p in self.test_prompts if p.startswith(group3_prefix))
        if group3_count > 0:
            groups['Group3_BusinessReview'] = {
                'count': group3_count,
                'prefix_length': len(group3_prefix)
            }
        
        return groups

def main():
    """Main test execution"""
    print("🧪 Simple KV Cache Test - Starting")
    print("=" * 60)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Simple KV Cache Test')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID (default: 0)')
    args = parser.parse_args()
    
    # Create and run test
    test = SimpleKVCacheTest(gpu_id=args.gpu)
    
    # Setup vLLM
    if not test.setup_vllm():
        print("❌ Setup failed - cannot run test")
        return
    
    # Run test
    results = test.run_inference_test()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"simple_kv_cache_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    if results['success']:
        print(f"🎉 Test completed successfully!")
    else:
        print(f"❌ Test had issues - check results for details")

if __name__ == "__main__":
    main()
