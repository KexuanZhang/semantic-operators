# KVTuner + vLLM Integration

This document describes the integration of KVTuner's layer-wise mixed precision KV cache quantization with vLLM for memory-efficient production serving.

## Overview

KVTuner provides sophisticated per-layer mixed precision quantization (2-4-6-8 bits) for KV caches, while vLLM provides high-performance serving infrastructure. This integration combines both to enable:

- **Memory Efficiency**: Significant memory reduction through per-layer mixed precision quantization
- **Performance**: vLLM's optimized serving capabilities with PagedAttention
- **Flexibility**: Preset configurations for different models and quality/memory trade-offs
- **Production Ready**: Seamless integration with vLLM's API and serving infrastructure

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Request  │───▶│   vLLM Engine    │───▶│  Model Executor │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Cache Config    │    │ Attention Layer │
                       │  +KVTuner params │    │ +KVTuner Cache  │
                       └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ KVTuner Config   │───▶│FlexibleQuantized│
                       │ (YAML presets)   │    │     Cache       │
                       └──────────────────┘    └─────────────────┘
```

## Installation

### 1. Install Modified vLLM

```bash
# Clone and install the modified vLLM with KVTuner support
cd /path/to/modified/vllm
pip install -e .
```

### 2. Install KVTuner Dependencies

```bash
# Add KVTuner to your environment
export KVTUNER_PATH="/home/data/so2/KVTuner"
export PYTHONPATH="$KVTUNER_PATH:$PYTHONPATH"

# Install KVTuner flexible quantization
cd $KVTUNER_PATH/flexible_quant
pip install -e .
```

## Quick Start

### Basic Usage

```python
from vllm import LLM, SamplingParams

# Initialize vLLM with KVTuner quantization
llm = LLM(
    model="/path/to/model",
    quantization="kvtuner",
    kvtuner_config_path="/path/to/kvtuner_preset.yaml",
    kvtuner_scheme="per_token",
    kvtuner_backend="vanilla"
)

# Generate text as usual
prompts = ["Hello, how are you?"]
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### Command Line Interface

```bash
# Using the integrated script
python kvtuner_vllm_inference.py \
    --model /path/to/model \
    --quantization kvtuner \
    --kvtuner-config /path/to/preset.yaml \
    --kvtuner-scheme per_token \
    --kvtuner-backend vanilla \
    --prompt "Hello, how are you?"

# Using vLLM serve with KVTuner
vllm serve /path/to/model \
    --quantization kvtuner \
    --kvtuner-config-path /path/to/preset.yaml \
    --kvtuner-scheme per_token \
    --kvtuner-backend vanilla
```

## Configuration

### KVTuner Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `kvtuner_config_path` | Path to YAML preset configuration | `None` | Any valid file path |
| `kvtuner_scheme` | Quantization scheme | `"per_token"` | `"per_token"`, `"per_channel"` |
| `kvtuner_backend` | Quantization backend | `"vanilla"` | `"vanilla"`, `"quanto"`, `"hqq"` |

### Preset Configurations

KVTuner provides pre-calibrated configurations for various models:

```yaml
# Example: Qwen2.5-3B-Instruct_pertoken_KVTuner4_0.yaml
0:
  nbits_key: 8
  nbits_value: 4
1:
  nbits_key: 4
  nbits_value: 2
2:
  nbits_key: 4
  nbits_value: 2
# ... per-layer configurations
```

Available presets:
- `Meta-Llama-3.1-8B-Instruct_*`
- `Qwen2.5-3B-Instruct_*`
- `Mistral-7B-Instruct-v0.3_*`

Naming convention: `{model}_{scheme}_KVTuner{avg_bits}_{variant}.yaml`

## Advanced Usage

### Custom Configuration

```python
# Create custom KVTuner configuration
from vllm.model_executor.layers.quantization.kvtuner import KVTunerConfig

config = KVTunerConfig(
    kvtuner_scheme="per_channel",
    kvtuner_backend="quanto",
    force_quant=True,
    residual_length=128,
    q_group_size=64
)

# Use with vLLM
llm = LLM(
    model="/path/to/model",
    quantization="kvtuner",
    kvtuner_config_path="/path/to/custom_config.yaml"
)
```

### Benchmark Performance

```python
# Run benchmark to compare performance
python kvtuner_vllm_inference.py \
    --model /path/to/model \
    --quantization kvtuner \
    --kvtuner-config /path/to/preset.yaml \
    --benchmark \
    --verbose
```

### Memory Usage Monitoring

```python
import torch
from vllm import LLM

# Monitor GPU memory usage
def print_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

# Before initialization
print_memory_usage()

# With KVTuner quantization
llm = LLM(
    model="/path/to/model",
    quantization="kvtuner",
    kvtuner_config_path="/path/to/preset.yaml"
)

# After initialization
print_memory_usage()
```

## Integration Details

### Modified Components

1. **Quantization System**
   - Added `kvtuner.py` quantization config
   - Updated `__init__.py` to register KVTuner
   - Extended quantization methods enum

2. **Configuration System**
   - Extended `CacheConfig` with KVTuner fields
   - Updated `EngineArgs` for parameter passing
   - Modified config creation pipeline

3. **Engine Integration**
   - Updated `LLM` class constructor
   - Added parameter forwarding
   - Integrated with existing infrastructure

4. **Cache Management**
   - Created `kvtuner_cache.py` integration module
   - Bridge between KVTuner and vLLM caching
   - Maintains compatibility with PagedAttention

### Architecture Flow

1. **Initialization**: User specifies KVTuner parameters
2. **Configuration**: Parameters flow through EngineArgs → CacheConfig
3. **Engine Creation**: vLLM engine initializes with KVTuner support
4. **Runtime**: KVTuner cache manager handles quantized KV caching
5. **Inference**: Standard vLLM inference with memory-efficient caching

## Performance Characteristics

### Memory Savings

- **4-bit average**: ~50% memory reduction
- **6-bit average**: ~25% memory reduction
- **Mixed precision**: Optimized quality/memory trade-off per layer

### Throughput Impact

- **Minimal overhead**: KVTuner quantization is optimized for speed
- **Maintained compatibility**: Works with vLLM's PagedAttention
- **Scalable**: Supports tensor parallel and pipeline parallel configurations

### Quality Preservation

- **Calibrated presets**: Pre-optimized for minimal quality loss
- **Layer-wise optimization**: Different precision per layer based on sensitivity
- **Extensive validation**: Tested on standard benchmarks

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure KVTuner path is in PYTHONPATH
   export PYTHONPATH="/home/data/so2/KVTuner:$PYTHONPATH"
   ```

2. **Configuration Not Found**
   ```bash
   # Check preset path
   ls /home/data/so2/KVTuner/calibration_presets/
   ```

3. **Memory Issues**
   ```python
   # Reduce GPU memory utilization if needed
   llm = LLM(
       model="/path/to/model",
       quantization="kvtuner",
       gpu_memory_utilization=0.8
   )
   ```

### Debugging

```python
# Enable verbose logging
import logging
logging.getLogger("vllm").setLevel(logging.DEBUG)

# Check quantization status
from vllm.model_executor.layers.kvtuner_cache import create_kvtuner_cache_manager
manager = create_kvtuner_cache_manager(vllm_config)
print(f"KVTuner enabled: {manager.is_quantized()}")
```

## Examples

### Example Scripts

1. **Basic Inference**: `kvtuner_vllm_inference.py`
2. **Integration Test**: `test_kvtuner_integration.py`
3. **Validation**: `validate_integration.py`

### Use Cases

1. **Production Serving**: Deploy models with reduced memory footprint
2. **Research**: Experiment with different quantization configurations
3. **Benchmarking**: Compare performance across configurations
4. **Development**: Integrate into existing vLLM workflows

## Contributing

### Adding New Presets

1. Calibrate using KVTuner tools
2. Save configuration as YAML
3. Add to `calibration_presets/` directory
4. Test with integration scripts

### Extending Backends

1. Implement in KVTuner's flexible_quant
2. Add backend option to `kvtuner_backend` choices
3. Update cache manager logic
4. Test integration

## License

This integration maintains the licenses of both constituent projects:
- vLLM: Apache 2.0 License
- KVTuner: [Check KVTuner repository for license]

## References

- **KVTuner Paper**: "Sensitivity-Aware Layer-wise Mixed Precision KV Cache Quantization"
- **vLLM**: High-performance LLM serving system
- **Integration**: Combines memory efficiency with serving performance
