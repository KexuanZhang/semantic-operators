# Simple LLM Inference Script

This script focuses only on the essential LLM inference tasks with minimal overhead:
1. Load a dataset
2. Initialize an LLM model
3. Run inference on each row using a specified template
4. Save the raw outputs and statistics

## Key Features

- **Simple Interface**: Minimalist design focused on core functionality
- **Content Detection**: Automatically finds relevant content columns
- **Raw Output**: Saves unprocessed model responses
- **Basic Statistics**: Records timing information
- **GPU Selection**: Supports specific GPU IDs
- **Tensor Parallelism**: Can use multiple GPUs for larger models

## Usage Examples

### Basic Usage with Default Template

```bash
python simple_llm_inference.py --dataset your_dataset.csv --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

This uses the default prompt template: 
```
Based on this review: {content}, is this movie suitable for children under 12? Answer with Yes or No.
```

### Custom Prompt Template

```bash
python simple_llm_inference.py --dataset your_dataset.csv --model your_model_name --prompt_template "Summarize this review: {content}"
```

### Specify Content Column and GPU IDs

```bash
python simple_llm_inference.py --dataset your_dataset.csv --model your_model_name --content_column review_content --gpu_ids "6,7" --tp_size 2
```

### Test with Limited Rows

```bash
python simple_llm_inference.py --dataset your_dataset.csv --model your_model_name --max_rows 10
```

## Output Structure

Results are saved in a timestamped directory under `simple_inference_results/`:

- `*_results.json`: Raw inference results including prompts and responses
- `*_stats.json`: Timing statistics
- `*_summary.txt`: Human-readable summary of the experiment

## Example for Movie Review Dataset

```bash
python simple_llm_inference.py --dataset rotten_tomatoes_critic_reviews.csv --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --content_column review_content --prompt_template "Based on this review: {content}, is this movie suitable for children under 12? Answer with only Yes or No." --gpu_ids "6,7" --tp_size 2
```

## Example for Movie Review Dataset with Summarization

```bash
python simple_llm_inference.py --dataset rotten_tomatoes_critic_reviews.csv --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --content_column review_content --prompt_template "Summarize the following movie review in one short sentence: {content}" --gpu_ids "6,7" --tp_size 2
```
