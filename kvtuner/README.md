# KVTuner Integration Scripts for Qwen2.5-3B-Instruct

Complete set of scripts for running inference with KVTuner quantization on the Qwen2.5-3B-Instruct model.

## Available Scripts

1. **`llm_inference.py`** - **NEW**: Complete LLM inference script with KVTuner integration (same interface as original)
2. **`kvtuner_inference.py`** - Comprehensive inference script with full configuration options
3. **`kvtuner_simple.py`** - Simplified runner for quick testing and basic dataset processing  
4. **`run_kvtuner_inference.py`** - Advanced inference script with multiple model support
5. **`run_inference.sh`** - Interactive shell script runner with convenient options
6. **`test_kvtuner_inference.py`** - Test script to create sample data and verify functionality
7. **`run_examples.sh`** - Example usage script with different configuration options

## Quick Start

### Option 1: NEW LLM Inference with KVTuner (Recommended)

The new `llm_inference.py` script provides the same interface as the original LLM inference script but with KVTuner quantization integrated:

```bash
# Basic usage with pertoken scheme
python llm_inference.py \
    --dataset "/path/to/your/dataset.csv" \
    --model "/Users/zhang/Desktop/huawei/untitled folder 6/Qwen2.5-3B-Instruct" \
    --kvtuner_scheme pertoken \
    --include_columns text_content \
    --max_rows 50

# Using KiVi quantization scheme  
python llm_inference.py \
    --dataset "/path/to/your/dataset.csv" \
    --model "/Users/zhang/Desktop/huawei/untitled folder 6/Qwen2.5-3B-Instruct" \
    --kvtuner_scheme kivi \
    --include_columns text_content \
    --prompt_template "Analyze this text: {text_content}" \
    --max_new_tokens 150

# Create test dataset and see usage examples
python test_kvtuner_inference.py

# Run interactive examples
./run_examples.sh
```

### Option 2: Interactive Shell Script

```bash
# Make executable and run
chmod +x run_inference.sh

# Simple test with predefined prompts
./run_inference.sh simple-test

# Process a dataset interactively
./run_inference.sh simple-dataset

# Full configuration options
./run_inference.sh full-dataset
```

### Option 3: Simple Python Runner

```bash
# Quick test with a few prompts
python kvtuner_simple.py \
    --model_path "/Users/zhang/Desktop/huawei/untitled folder 6/Qwen2.5-3B-Instruct" \
    --test

# Process a small dataset
python kvtuner_simple.py \
    --model_path "/Users/zhang/Desktop/huawei/untitled folder 6/Qwen2.5-3B-Instruct" \
    --dataset "/path/to/dataset.csv" \
    --max_rows 10 \
    --scheme pertoken
```

### Option 4: Comprehensive Inference

```bash
python kvtuner_inference.py \
    --model_path "/Users/zhang/Desktop/huawei/untitled folder 6/Qwen2.5-3B-Instruct" \
    --dataset "/path/to/dataset.csv" \
    --scheme pertoken \
    --text_column "text" \
    --max_new_tokens 256
```

## Configuration

### Quantization Schemes

1. **pertoken** (default): Per-token quantization for both keys and values
   - Uses: `Qwen2.5-3B-Instruct_pertoken_KVTuner4_0.yaml`
   - More memory efficient, good general-purpose option

2. **kivi**: KiVi quantization scheme  
   - Uses: `Qwen2.5-3B-Instruct_kivi_KVTuner4_0.yaml`
   - Per-channel for keys, per-token for values
   - May provide better quality in some cases

### Available Preset Configurations

The scripts automatically use configurations from the KVTuner calibration presets:

- `Qwen2.5-3B-Instruct_pertoken_KVTuner4_0.yaml` (~4-bit average)
- `Qwen2.5-3B-Instruct_pertoken_KVTuner4_1.yaml` 
- `Qwen2.5-3B-Instruct_pertoken_KVTuner6_0.yaml` (~6-bit average)
- `Qwen2.5-3B-Instruct_kivi_KVTuner4_0.yaml` (~4-bit average)
- `Qwen2.5-3B-Instruct_kivi_KVTuner4_1.yaml`
- `Qwen2.5-3B-Instruct_kivi_KVTuner6_0.yaml` (~6-bit average)

## Parameters Reference

### Common Parameters
- `--model_path`: Path to local Qwen2.5-3B-Instruct model
- `--dataset`: CSV file with text data to process
- `--scheme`: Quantization scheme (`pertoken` or `kivi`)
- `--text_column`: Column name containing text (default: "text")
- `--max_rows`: Limit number of rows to process (optional)
- `--max_new_tokens`: Maximum tokens to generate (default: 256)

### Advanced Parameters  
- `--prompt_template`: Template with placeholders like `{text}` (default: "Analyze the following text: {text}")
- `--gpu_ids`: Specific GPUs to use, e.g., "0,1" (optional)
- `--output_dir`: Directory for results (default: ./results)
- `--debug`: Enable verbose logging

## Dataset Format

CSV files with text data. The scripts will:
- Auto-detect text columns if `--text_column` not specified
- Try common column names: text, content, review, comment, description
- Fall back to first column if no text column found

Example CSV:
```csv
text,label
"This is a sample text to analyze",positive
"Another example for processing",neutral
```

## Output Format

Results saved to timestamped directories under `./results/`:

- **`results.json`**: Complete results with metadata
- **`results.csv`**: Results in CSV format for analysis
- **`statistics.json`**: Performance timing and statistics

Example result structure:
```json
{
  "id": 0,
  "input_text": "Original text",
  "prompt": "Formatted prompt sent to model", 
  "completion": "Model response",
  "inference_time": 1.23
}
```

## Usage Examples

### Example 1: Quick Test
```bash
# Test with simple prompts using pertoken scheme
./run_inference.sh simple-test

# Test with KiVi scheme  
./run_inference.sh kivi-test
```

### Example 2: Small Dataset Processing
```bash
# Process 20 rows with KiVi quantization
python kvtuner_simple.py \
    --model_path "/Users/zhang/Desktop/huawei/untitled folder 6/Qwen2.5-3B-Instruct" \
    --dataset "/path/to/data.csv" \
    --scheme kivi \
    --max_rows 20 \
    --max_new_tokens 128
```

### Example 3: Custom Prompt Template
```bash
# Sentiment analysis with custom template
python kvtuner_inference.py \
    --model_path "/Users/zhang/Desktop/huawei/untitled folder 6/Qwen2.5-3B-Instruct" \
    --dataset "reviews.csv" \
    --text_column "review_text" \
    --prompt_template "Classify the sentiment of this review as positive, negative, or neutral: {review_text}" \
    --scheme pertoken \
    --max_new_tokens 50
```

### Example 4: Full Dataset with Multiple GPUs
```bash
python kvtuner_inference.py \
    --model_path "/Users/zhang/Desktop/huawei/untitled folder 6/Qwen2.5-3B-Instruct" \
    --dataset "large_dataset.csv" \
    --scheme kivi \
    --gpu_ids "0,1" \
    --max_new_tokens 256 \
    --debug
```

## Requirements

- **Python 3.8+**
- **PyTorch** with CUDA support
- **Transformers** (version with CacheConfig and QuantizedCacheConfig)
- **KVTuner** installed at `/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner`
- **Dependencies**: pandas, tqdm, pyyaml

## Installation Check

```bash
# Test if imports work correctly
python -c "from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig; print('KVTuner imports OK')"

# Check transformers version
python -c "from transformers.cache_utils import CacheConfig; print('Transformers cache support OK')"
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'flexible_quant'
   ```
   - Ensure KVTuner is installed: `pip install -e /path/to/KVTuner`
   - Check the path in scripts matches your KVTuner location

2. **Model Loading Issues**
   ```
   FileNotFoundError: Model path not found
   ```
   - Verify the model path in scripts or shell configuration
   - Ensure you have sufficient disk space and memory

3. **CUDA/GPU Errors**
   ```
   CUDA out of memory
   ```
   - Reduce `--max_new_tokens` or `--max_rows`
   - Use fewer GPUs or smaller batch sizes
   - Monitor GPU memory with `nvidia-smi`

4. **Configuration File Missing**
   ```
   Warning: Config file not found
   ```
   - Scripts will use default settings if preset files aren't found
   - Check that KVTuner calibration presets are available

### Debug Mode

Enable detailed logging with `--debug`:
```bash
python kvtuner_inference.py --debug [other args]
```

## Performance Notes

- **Memory**: Each inference creates a fresh KV cache
- **Speed**: Varies by quantization scheme (pertoken typically faster)
- **Quality**: KiVi may provide better output quality in some cases
- **Monitoring**: Progress bars show real-time processing status

## File Structure

```
kvtuner/
├── kvtuner_inference.py      # Main comprehensive script
├── kvtuner_simple.py         # Simple runner for quick tasks
├── run_kvtuner_inference.py  # Advanced multi-model script  
├── run_inference.sh          # Interactive shell runner
└── README.md                 # This documentation
```
