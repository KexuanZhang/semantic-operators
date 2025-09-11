#!/usr/bin/env python3
"""
Usage Examples for vLLM + KVTuner Inference Script

This script shows various ways to use the updated llm_inference.py script
with vLLM and KVTuner integration.
"""

import os
import pandas as pd
import tempfile

def create_sample_dataset():
    """Create a sample dataset for testing"""
    
    sample_data = {
        'text_content': [
            "This is a positive review about a great product.",
            "This is a negative review about poor service.",
            "This is a neutral review about an average experience.",
            "This product exceeded my expectations completely!",
            "The quality was disappointing and not worth the price."
        ],
        'sentiment_label': ['positive', 'negative', 'neutral', 'positive', 'negative'],
        'category': ['product', 'service', 'general', 'product', 'product']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Save to CSV
    csv_path = "/tmp/sample_dataset.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Created sample dataset: {csv_path}")
    print("Dataset preview:")
    print(df.head())
    print()
    
    return csv_path

def show_usage_examples():
    """Show various usage examples"""
    
    print("🚀 vLLM + KVTuner Inference Usage Examples")
    print("=" * 60)
    
    # Create sample dataset
    sample_csv = create_sample_dataset()
    
    examples = [
        {
            "name": "Basic Usage - Qwen Model with KVTuner",
            "description": "Use Qwen model with per-token KVTuner quantization",
            "command": f"""python llm_inference.py \\
    --dataset {sample_csv} \\
    --model Qwen/Qwen2.5-3B-Instruct \\
    --kvtuner_scheme pertoken \\
    --include_columns text_content \\
    --prompt_template "Analyze the sentiment of this text: {{text_content}}. Is it positive, negative, or neutral?" \\
    --max_new_tokens 100 \\
    --output_prefix qwen_sentiment"""
        },
        
        {
            "name": "Llama Model with KiVi Scheme",
            "description": "Use Llama model with KiVi quantization scheme",
            "command": f"""python llm_inference.py \\
    --dataset {sample_csv} \\
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \\
    --kvtuner_scheme kivi \\
    --include_columns text_content \\
    --prompt_template "Classify this review: {{text_content}}" \\
    --max_new_tokens 50 \\
    --gpu_ids "0" \\
    --output_prefix llama_classification"""
        },
        
        {
            "name": "Mistral Model with Custom Prompt",
            "description": "Use Mistral model with custom prompt template",
            "command": f"""python llm_inference.py \\
    --dataset {sample_csv} \\
    --model mistralai/Mistral-7B-Instruct-v0.3 \\
    --kvtuner_scheme pertoken \\
    --include_columns text_content category \\
    --prompt_template "Category: {{category}}\\nText: {{text_content}}\\nSentiment:" \\
    --max_new_tokens 20 \\
    --output_prefix mistral_category_sentiment"""
        },
        
        {
            "name": "Limited Testing Run",
            "description": "Test with only a few rows for quick validation",
            "command": f"""python llm_inference.py \\
    --dataset {sample_csv} \\
    --model Qwen/Qwen2.5-3B-Instruct \\
    --kvtuner_scheme pertoken \\
    --max_rows 2 \\
    --max_new_tokens 30 \\
    --output_prefix quick_test"""
        },
        
        {
            "name": "Custom KVTuner Directory",
            "description": "Specify custom KVTuner directory path",
            "command": f"""python llm_inference.py \\
    --dataset {sample_csv} \\
    --model Qwen/Qwen2.5-3B-Instruct \\
    --kvtuner_scheme pertoken \\
    --kvtuner_dir "/path/to/your/KVTuner" \\
    --output_prefix custom_kvtuner"""
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n📝 Example {i}: {example['name']}")
        print(f"Description: {example['description']}")
        print("Command:")
        print(example['command'])
        print("-" * 50)
    
    print("\n🔧 Key Parameters:")
    print("--dataset: Path to your CSV dataset")
    print("--model: HuggingFace model name or local path")
    print("--kvtuner_scheme: 'pertoken' or 'kivi' quantization scheme")
    print("--include_columns: Column names to use in prompt template")
    print("--prompt_template: Template with {column_name} placeholders")
    print("--max_new_tokens: Maximum tokens to generate per response")
    print("--max_rows: Limit number of rows for testing")
    print("--gpu_ids: Specific GPU IDs (e.g., '0' or '0,1')")
    print("--output_prefix: Prefix for result files")
    
    print("\n📊 Output Files:")
    print("- {prefix}_kvtuner_results.csv: Results in CSV format")
    print("- {prefix}_kvtuner_inference_results.json: Detailed JSON results")
    print("- {prefix}_kvtuner_stats.json: Performance statistics")
    print("- {prefix}_kvtuner_summary.txt: Human-readable summary")
    
    print("\n💡 Performance Tips:")
    print("1. Use smaller models for testing (e.g., Qwen2.5-3B)")
    print("2. Start with --max_rows 5-10 for initial validation")
    print("3. KVTuner provides ~4.6x memory reduction on average")
    print("4. pertoken scheme generally works well for most use cases")
    print("5. kivi scheme may be better for specific model architectures")
    
    print(f"\n🗑️ Clean up: rm {sample_csv}")

if __name__ == "__main__":
    show_usage_examples()
