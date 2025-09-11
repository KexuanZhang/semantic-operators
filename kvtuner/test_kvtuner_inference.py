#!/usr/bin/env python3
"""
Test script for KVTuner LLM inference

This script creates a simple test dataset and runs inference to verify
that the KVTuner integration is working correctly.
"""

import pandas as pd
import os
import tempfile

def create_test_dataset():
    """Create a simple test dataset"""
    test_data = {
        'text_content': [
            "This movie was absolutely amazing! Great acting and story.",
            "The product quality is terrible and arrived broken.",
            "Neutral review about an average restaurant experience.",
            "I love this book! It's incredibly well written.",
            "The service was poor and the staff was rude."
        ],
        'category': ['movie', 'product', 'restaurant', 'book', 'service'],
        'id': [1, 2, 3, 4, 5]
    }
    
    df = pd.DataFrame(test_data)
    
    # Save to temporary CSV
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return temp_file.name, df

def main():
    print("Creating test dataset...")
    test_file, test_df = create_test_dataset()
    
    print("Test dataset created:")
    print(test_df)
    print(f"\nSaved to: {test_file}")
    
    print("\nTo run the KVTuner inference test, use:")
    print(f"""
python llm_inference.py \\
    --dataset "{test_file}" \\
    --model "/Users/zhang/Desktop/huawei/untitled folder 6/Qwen2.5-3B-Instruct" \\
    --kvtuner_scheme pertoken \\
    --include_columns text_content \\
    --prompt_template "Analyze this text and determine if it's positive, negative, or neutral: {{text_content}}" \\
    --max_new_tokens 50 \\
    --max_rows 3
""")
    
    print("\nOr with KiVi scheme:")
    print(f"""
python llm_inference.py \\
    --dataset "{test_file}" \\
    --model "/Users/zhang/Desktop/huawei/untitled folder 6/Qwen2.5-3B-Instruct" \\
    --kvtuner_scheme kivi \\
    --include_columns text_content \\
    --prompt_template "Classify the sentiment: {{text_content}}" \\
    --max_new_tokens 30 \\
    --max_rows 2
""")
    
    print(f"\nRemember to clean up the test file when done: rm {test_file}")

if __name__ == "__main__":
    main()
