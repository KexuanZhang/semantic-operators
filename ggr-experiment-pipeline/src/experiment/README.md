# Sentiment Analysis Experiment with vLLM

This module provides a comprehensive framework for performing sentiment analysis experiments using vLLM (Very Large Language Models). It allows you to load local LLM models, process datasets for sentiment analysis, and save detailed results including inference times.

## Features

- 🚀 **vLLM Integration**: Efficient loading and inference with local LLM models
- 📊 **Dataset Processing**: Batch processing of CSV datasets with configurable text columns
- ⏱️ **Performance Tracking**: Detailed timing information for each inference and total experiment time
- 💾 **Result Storage**: Saves results in multiple formats (JSON, CSV) with comprehensive metadata
- 🎯 **Sentiment Analysis**: Binary sentiment classification (POSITIVE/NEGATIVE)
- 🛠️ **Configurable**: Command-line interface with flexible parameters

## Installation

First, ensure you have the required dependencies installed:

```bash
pip install -r requirements-vllm.txt
```

Key dependencies:
- `vllm>=0.3.0`
- `torch>=2.0.0`  
- `transformers>=4.30.0`
- `pandas>=1.5.0`
- `tqdm`

## Quick Start

### Command Line Usage

Basic usage:
```bash
python run_exp.py --model-path /path/to/your/model --dataset-path your_dataset.csv
```

With custom parameters:
```bash
python run_exp.py \
  --model-path /path/to/your/model \
  --dataset-path your_dataset.csv \
  --text-column "review_text" \
  --results-dir ./my_results \
  --experiment-name sentiment_test_2024
```

### Programmatic Usage

```python
from run_exp import SentimentAnalysisExperiment

# Initialize experiment
experiment = SentimentAnalysisExperiment(
    model_path="/path/to/your/model",
    results_dir="results"
)

# Load model
experiment.load_model()

# Process dataset
results = experiment.process_dataset(
    dataset_path="your_dataset.csv",
    text_column="text"
)

# Save results
experiment.save_results(results, "my_experiment")

# Cleanup
experiment.cleanup()
```

## Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model-path` | Yes | - | Path to the local LLM model directory or HuggingFace model ID |
| `--dataset-path` | Yes | - | Path to the CSV dataset file |
| `--text-column` | No | `"text"` | Name of the column containing text to analyze |
| `--results-dir` | No | `"results"` | Directory to save experiment results |
| `--experiment-name` | No | Auto-generated | Custom name for the experiment files |
| `--batch-size` | No | `1` | Number of samples to process at once |

## Dataset Format

Your dataset should be a CSV file with at least one text column. Example:

```csv
text,label
"I love this product!",positive
"This is terrible.",negative
"Great service and fast delivery.",positive
```

The experiment will automatically include all other columns in the results with an `original_` prefix.

## Output Files

The experiment generates three types of output files:

### 1. Detailed Results JSON (`{experiment_name}_results.json`)
Contains complete experiment data including individual predictions and timing:

```json
{
  "results": [
    {
      "index": 0,
      "text": "I love this product!",
      "predicted_sentiment": "POSITIVE",
      "inference_time": 0.234,
      "timestamp": "2024-01-15T10:30:45.123456",
      "original_label": "positive"
    }
  ],
  "metadata": {
    "model_path": "/path/to/model",
    "total_samples": 100,
    "positive_predictions": 65,
    "negative_predictions": 35,
    "total_inference_time_seconds": 23.45,
    "average_inference_time_seconds": 0.2345
  }
}
```

### 2. Results CSV (`{experiment_name}_results.csv`)
Tabular format for easy analysis:

| index | text | predicted_sentiment | inference_time | timestamp | original_label |
|-------|------|-------------------|----------------|-----------|----------------|
| 0 | I love this product! | POSITIVE | 0.234 | 2024-01-15T10:30:45 | positive |

### 3. Metadata JSON (`{experiment_name}_metadata.json`)
Experiment configuration and summary statistics.

## Model Compatibility

The experiment works with:

- **Local models**: Any model directory compatible with vLLM
- **HuggingFace models**: Model IDs like `meta-llama/Llama-2-7b-chat-hf`
- **Popular architectures**: LLaMA, Mistral, Qwen, and other transformer models

## Performance Optimization

The experiment is configured for optimal performance:

- **GPU Memory**: Uses 80% of available GPU memory
- **Temperature**: Low temperature (0.1) for consistent sentiment analysis
- **Max Tokens**: Limited to 50 tokens for efficiency
- **Stop Sequences**: Natural stopping points for cleaner responses

## Examples

### Example 1: Movie Review Dataset
```bash
python run_exp.py \
  --model-path meta-llama/Llama-2-7b-chat-hf \
  --dataset-path movie_reviews.csv \
  --text-column "review_text" \
  --experiment-name movie_sentiment_analysis
```

### Example 2: Product Reviews
```bash
python run_exp.py \
  --model-path /local/models/mistral-7b \
  --dataset-path product_reviews.csv \
  --text-column "comment" \
  --results-dir ./product_analysis_results
```

### Example 3: Social Media Posts
```bash
python run_exp.py \
  --model-path /local/models/qwen-7b \
  --dataset-path social_media.csv \
  --text-column "post_content" \
  --experiment-name social_media_sentiment
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `gpu_memory_utilization` in the code or use a smaller model
2. **Model Loading Error**: Verify the model path exists and is compatible with vLLM
3. **Column Not Found**: Check that the specified `--text-column` exists in your CSV
4. **Permission Error**: Ensure write permissions for the results directory

### Memory Management

The experiment automatically:
- Clears CUDA cache before starting
- Uses environment variables for caching optimization
- Cleans up resources after completion

## Contributing

To extend the experiment:

1. **Custom Prompts**: Modify `create_sentiment_prompt()` method
2. **Different Tasks**: Extend the class for other NLP tasks
3. **Batch Processing**: Implement batch inference for better throughput
4. **Additional Metrics**: Add evaluation metrics in the results

## License

This project follows the same license as the parent vLLM project.
