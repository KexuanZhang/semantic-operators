# KV Cache Stats Inference Testing Suite

This directory contains comprehensive test scripts to validate local KV cache stats retrieval during vLLM inference, ensuring the enhanced internal Stats object access implementation works correctly.

## Test Scripts Overview

### 1. `test_kv_cache_structure_validation.py`
**Lightweight Structure Validation (No vLLM Required)**

This test validates the implementation structure without requiring a full vLLM installation:

```bash
python test_kv_cache_structure_validation.py
```

**What it tests:**
- ✅ Enhanced VLLMMetricsCollector import and instantiation
- ✅ Required internal stats methods exist and are callable
- ✅ Graceful error handling when vLLM instance is not available
- ✅ Prometheus fallback method structure and behavior
- ✅ Comprehensive stats method structure
- ✅ Key metrics extraction logic validation
- ✅ Monitoring lifecycle (start/stop) functionality

**Use this test to:**
- Verify implementation structure before installing vLLM
- Quick validation after code changes
- CI/CD pipeline integration for structure checks

### 2. `test_kv_cache_stats_inference.py`
**Full Inference Testing (Requires vLLM Installation)**

This test validates actual KV cache stats collection during live vLLM inference:

```bash
python test_kv_cache_stats_inference.py
```

**What it tests:**
- 🚀 Complete vLLM setup with prefix caching enabled
- 📊 Real-time KV cache stats collection during inference
- 🎯 Prefix cache hit rate monitoring with repeated prompts
- 📈 Cache usage progression during batch processing
- 🔧 Internal Stats access methods validation with live engine
- 🏆 Performance comparison of different collection methods

**Use this test to:**
- Validate production readiness with actual vLLM instances
- Test KV cache monitoring during real workloads
- Performance benchmarking of different stats collection approaches

## Prerequisites

### For Structure Validation Test
```bash
# Minimal requirements (already in your environment)
pip install pandas numpy matplotlib seaborn
```

### For Full Inference Test
```bash
# Full vLLM installation required
pip install -r requirements-vllm.txt

# Or install vLLM directly:
pip install vllm torch transformers
```

## Running the Tests

### Quick Structure Check (Recommended First)
```bash
# Test implementation structure without vLLM
python test_kv_cache_structure_validation.py
```

Expected output:
```
🚀 KV Cache Stats Implementation Structure Validation
🔧 Testing enhanced VLLMMetricsCollector import...
   ✅ VLLMMetricsCollector imported successfully
   ✅ VLLMMetricsCollector instantiated successfully

📊 Testing internal stats methods availability...
   ✅ Method collect_internal_stats found
   ✅ Method _collect_prometheus_registry_only found
   ✅ All required methods available

📋 STRUCTURE TEST SUMMARY
🏆 EXCELLENT: Implementation structure is solid!
```

### Full Inference Testing
```bash
# Test with actual vLLM inference (requires vLLM installation)
python test_kv_cache_stats_inference.py
```

Expected output:
```
🚀 vLLM KV Cache Stats Inference Test Suite
🔧 Setup Phase
✅ vLLM initialized successfully with model: microsoft/DialoGPT-small
   - Prefix caching: ENABLED
   - Max model length: 2048

📊 Setting up enhanced metrics collector...
✅ Enhanced metrics collector initialized
   - Internal Stats access: ENABLED

🧪 TEST 1: Basic KV Cache Stats Collection
⚡ Inference completed in 2.45s
📊 Final stats collection method: internal_engine_stats
   🗄️  Found 8 cache-related metrics:
      • gpu_cache_usage_sys: 0.745
      • gpu_prefix_cache_hit_rate: 0.823

🎯 Overall Assessment:
🏆 EXCELLENT: KV cache stats retrieval is working well!
```

## Test Configuration

### Model Selection
The tests use small models by default for fast execution:
- **Default**: `microsoft/DialoGPT-small` (fast testing)
- **Production**: Change to larger models like `meta-llama/Llama-2-7b-chat-hf`

```python
# Edit in test file to use different model
test_model = "microsoft/DialoGPT-small"  # Fast testing
# test_model = "meta-llama/Llama-2-7b-chat-hf"  # Production testing
```

### vLLM Configuration
Tests configure vLLM with optimal settings for KV cache monitoring:
```python
engine_args = EngineArgs(
    model=model_name,
    max_model_len=2048,
    enable_prefix_caching=True,  # Critical for KV cache stats
    gpu_memory_utilization=0.6,
    disable_log_stats=False      # Enable internal stats
)
```

## Understanding Test Results

### Success Indicators
- ✅ **Structure Validation**: All methods present and callable
- ✅ **Stats Collection**: KV cache metrics successfully retrieved
- ✅ **Internal Access**: Direct engine Stats object access working
- ✅ **Cache Monitoring**: Hit rates and usage metrics tracked

### Key Metrics Monitored
The tests validate collection of these critical metrics:
- `gpu_cache_usage_sys` - GPU cache utilization percentage
- `gpu_prefix_cache_hit_rate` - Prefix cache effectiveness  
- `num_running_sys` / `num_waiting_sys` - Request queue status
- `prompt_tokens_total` / `generation_tokens_total` - Token processing
- Performance histograms (TTFT, TPOT, E2E latency)

### Troubleshooting Common Issues

**Import Errors:**
```bash
❌ Import error: No module named 'vllm'
```
Solution: Install vLLM with `pip install vllm`

**GPU Memory Issues:**
```bash
❌ vLLM setup failed: CUDA out of memory
```
Solution: Reduce `gpu_memory_utilization` in test configuration

**No Stats Available:**
```bash
⚠️ No vLLM metrics found in any collection method
```
Solution: Ensure `enable_prefix_caching=True` and `disable_log_stats=False`

## Integration with Your Workflow

### Development Workflow
1. **Code Changes** → Run structure validation test
2. **Structure OK** → Run full inference test  
3. **All Tests Pass** → Ready for production

### CI/CD Integration
```bash
# Add to your CI pipeline
python test_kv_cache_structure_validation.py || exit 1

# Optional: Full test if vLLM available
if command -v nvidia-smi &> /dev/null; then
    python test_kv_cache_stats_inference.py || exit 1
fi
```

### Production Validation
Before deploying to production:
1. Run tests with your target model
2. Verify all collection methods work
3. Check performance impact of monitoring
4. Validate key metrics are being collected

## Test Results Output

Both tests generate JSON results files:
- `kv_cache_structure_test_results_YYYYMMDD_HHMMSS.json`
- `kv_cache_stats_test_results_YYYYMMDD_HHMMSS.json`

These files contain detailed test execution data for analysis and debugging.

## Related Files

- `src/experiment/run_experiment.py` - Enhanced VLLMMetricsCollector implementation
- `INTERNAL_STATS_IMPLEMENTATION.md` - Detailed implementation documentation  
- `test_internal_stats_access.py` - Additional implementation validation

## Support

For issues with the test suite:
1. Check test output for specific error messages
2. Verify all prerequisites are installed
3. Review implementation documentation
4. Ensure vLLM is properly configured with prefix caching

The test suite provides comprehensive validation that your KV cache stats retrieval implementation is production-ready! 🚀
