# GGR Algorithm Experiment Pipeline

This project provides a comprehensive experiment pipeline for running the Greedy Group Recursion (GGR) algorithm on datasets. The pipeline includes data preprocessing, functional dependency discovery, and result analysis capabilities.

## Features

- **Data Preprocessing**: Load and clean CSV datasets
- **Functional Dependency Discovery**: Automatically discover functional dependencies in datasets
- **GGR Algorithm**: Apply the Greedy Group Recursion algorithm for data reordering
- **vLLM Integration**: Run inference with vLLM on reordered datasets
- **Resource Monitoring**: Track GPU, CPU, and memory usage during inference
- **KV Cache Analysis**: Monitor prefix cache hit rates and utilization
- **Performance Metrics**: Comprehensive throughput and latency analysis
- **Result Export**: Save reordered tables, metrics, and experiment metadata
- **Command-line Interface**: Easy-to-use CLI for running experiments
- **Jupyter Notebook**: Interactive analysis and visualization
- **Configurable Parameters**: Customize algorithm behavior and preprocessing options

## Installation

1. Clone or download this project
2. Install basic requirements:

```bash
pip install -r requirements.txt
```

3. For enhanced experiments with vLLM inference:

```bash
pip install -r requirements-vllm.txt
```

**Note**: vLLM requires CUDA-compatible GPU for optimal performance.

## Quick Start

### Basic GGR Experiment

Run the GGR algorithm on a dataset with automatic functional dependency discovery:

```bash
python main.py path/to/your/dataset.csv
```

### Enhanced Experiment with vLLM Inference

Run the complete pipeline including GGR reordering and vLLM inference monitoring:

```bash
python main_enhanced.py path/to/your/dataset.csv --model microsoft/DialoGPT-medium
```

### Interactive Analysis

For detailed analysis and visualization, use the Jupyter notebook:

```bash
jupyter notebook ggr_vllm_inference_experiment.ipynb
```

### Advanced Usage

Specify functional dependencies manually:

```bash
python main.py data/movies.csv --fds "movie_id->title,movie_id->genre"
```

Select specific columns and customize parameters:

```bash
python main.py data/movies.csv \
    --columns "movie_title,review_content,rating" \
    --max-depth 50 \
    --output results/ \
    --name my_experiment
```

## Enhanced Experiment Workflow

The enhanced experiment pipeline (`main_enhanced.py`) provides a complete workflow:

### 1. **GGR Phase**
- Load and preprocess dataset
- Discover functional dependencies (if not provided)
- Apply GGR algorithm for optimal data reordering
- Save reordered dataset for inference

### 2. **vLLM Inference Phase** 
- Initialize vLLM model with prefix caching enabled
- Generate prompts from reordered dataset
- Run inference while monitoring resources
- Collect KV cache metrics and performance data

### 3. **Analysis Phase**
- Analyze prefix cache hit rates
- Calculate throughput improvements
- Monitor GPU/CPU/Memory utilization
- Generate comprehensive performance reports

### Key Metrics Monitored

- **KV Cache Performance**: Hit rates, cache utilization, prefix reuse
- **Inference Performance**: Throughput (prompts/second), latency, batch efficiency  
- **System Resources**: GPU utilization, memory usage, CPU load
- **GGR Effectiveness**: Total hits, reordering time, data organization quality

### Expected Benefits of GGR

- **Higher Prefix Cache Hit Rate**: 20% → 80%+ improvement
- **Better Throughput**: 1.5-4× faster inference (literature reports)
- **Consistent Performance**: Lower variance in processing times
- **Efficient Resource Usage**: Higher GPU utilization, better memory efficiency

## Command Line Options

```
usage: main.py [-h] [--fds FDS] [--columns COLUMNS] [--output OUTPUT]
               [--name NAME] [--max-depth MAX_DEPTH] [--no-discover-fds]
               [--fd-confidence FD_CONFIDENCE] [--handle-missing {drop,fill,keep}]
               [--verbose]
               dataset_path

positional arguments:
  dataset_path          Path to the input CSV dataset

optional arguments:
  -h, --help            show this help message and exit
  --fds FDS, --functional-dependencies FDS
                        Functional dependencies as comma-separated pairs 
                        (e.g., "col1->col2,col3->col4")
  --columns COLUMNS, --columns-of-interest COLUMNS
                        Comma-separated list of columns to analyze
  --output OUTPUT, --output-dir OUTPUT
                        Output directory for results (default: results)
  --name NAME, --experiment-name NAME
                        Name for this experiment (auto-generated if not specified)
  --max-depth MAX_DEPTH
                        Maximum recursion depth for GGR algorithm (default: 100)
  --no-discover-fds     Disable automatic discovery of functional dependencies
  --fd-confidence FD_CONFIDENCE
                        Confidence threshold for functional dependency discovery 
                        (default: 0.95)
  --handle-missing {drop,fill,keep}
                        How to handle missing values: drop, fill, or keep 
                        (default: drop)
  --verbose, -v         Enable verbose logging
```

## Output Files

The pipeline generates the following output files:

1. **Reordered Table CSV**: `{experiment_name}_reordered_table.csv`
   - Contains the data reordered by the GGR algorithm
   
2. **Experiment Metadata JSON**: `{experiment_name}_metadata.json`
   - Contains experiment parameters, results, and performance metrics

## Project Structure

```
ggr-experiment-pipeline/
├── main.py                              # Basic GGR experiment CLI
├── main_enhanced.py                     # Enhanced experiment with vLLM inference
├── ggr_vllm_inference_experiment.ipynb # Interactive Jupyter notebook
├── test.py                             # Test script with sample data
├── requirements.txt                    # Basic Python dependencies
├── requirements-vllm.txt              # Enhanced dependencies (includes vLLM)
├── README.md                          # This file
├── src/                               # Source code
│   ├── __init__.py
│   ├── ggr_algorithm.py              # GGR algorithm implementation
│   ├── data_preprocessing.py         # Data loading and preprocessing
│   ├── experiment_runner.py          # Basic experiment orchestration
│   └── enhanced_experiment_runner.py # Enhanced experiment with vLLM
├── data/                             # Input datasets (create this directory)
├── results/                          # Basic experiment outputs
├── enhanced_results/                 # Enhanced experiment outputs
└── inference_results/                # Jupyter notebook outputs
```

## API Usage

You can also use the pipeline programmatically:

```python
from src.experiment_runner import GGRExperiment

# Create experiment
experiment = GGRExperiment(
    dataset_path="data/my_dataset.csv",
    output_dir="results",
    experiment_name="my_test"
)

# Run experiment
results = experiment.run_experiment(
    functional_dependencies=[("movie_id", "title"), ("movie_id", "genre")],
    columns_of_interest=["movie_title", "review_content"],
    max_depth=100
)

print(f"Total hits: {results['total_hits']}")
```

## Functional Dependencies

Functional dependencies define relationships between columns in your dataset. They can be:

1. **Auto-discovered**: The pipeline can automatically discover FDs with a configurable confidence threshold
2. **Manually specified**: Provide FDs in the format `"source->target,source2->target2"`

### Examples of Functional Dependencies

- `"movie_id->title"`: Movie ID determines movie title
- `"isbn->book_title,isbn->author"`: ISBN determines both book title and author
- `"student_id->name,student_id->email"`: Student ID determines name and email

## Performance Considerations

- **Large datasets**: Consider using `--max-depth` to limit recursion depth
- **Memory usage**: For very large datasets, consider selecting specific columns with `--columns`
- **Missing values**: Choose appropriate handling strategy with `--handle-missing`

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce dataset size or limit columns
2. **No functional dependencies found**: Lower `--fd-confidence` threshold
3. **CSV parsing errors**: Ensure CSV file is properly formatted

### Getting Help

Run with `--verbose` flag for detailed logging:

```bash
python main.py dataset.csv --verbose
```

## Example Workflows

### Movie Reviews Analysis

```bash
# Analyze movie reviews with automatic FD discovery
python main.py rotten_tomatoes_data.csv \
    --columns "movie_title,review_content,critic_name" \
    --output movie_analysis/ \
    --name movie_review_experiment
```

### Custom Dataset with Known Dependencies

```bash
# Use known functional dependencies
python main.py customer_data.csv \
    --fds "customer_id->name,customer_id->email,order_id->product" \
    --no-discover-fds \
    --handle-missing fill
```

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0 (for preprocessing utilities)

See `requirements.txt` for complete list.

## License

This project is provided as-is for research and educational purposes.
