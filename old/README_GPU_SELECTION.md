# GPU Specific Device Selection

You can now run the experiment script on specific GPU devices by using the `--gpu_ids` argument.

## Examples:

### Use GPUs 6 and 7 specifically:
```bash
python reorder_inference_experiment.py --dataset path/to/dataset.csv --gpu_ids "6,7" --tp_size 2 --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### Use only GPU 0:
```bash
python reorder_inference_experiment.py --dataset path/to/dataset.csv --gpu_ids "0" --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### Use provided shell script with specific GPUs:
```bash
./run_experiment_with_gpus.sh path/to/dataset.csv [max_rows]
```

## Notes:

- The `--gpu_ids` parameter accepts a comma-separated list of GPU IDs
- Make sure to set `--tp_size` equal to the number of GPUs you're using for tensor parallelism
- You can combine this with other parameters like `--reorder`, `--no_sort`, etc.
- For larger models, using multiple GPUs is recommended

This feature is particularly useful in multi-GPU environments where specific GPUs may have different memory capacities or loads.
