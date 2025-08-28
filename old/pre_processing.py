import pandas as pd
from vllm import LLM, SamplingParams
import torch
import torch.distributed as dist

# Clear GPU memory
torch.cuda.empty_cache()
    
def calculate_scores(df):
    """Calculate scores for each column based on average string length and cardinality."""
    column_scores = {}
    total_length = len(df)  # Total number of rows in the DataFrame
    avg_string_length = {col: df[col].astype(str).str.len().mean() for col in df.columns}

    for col in df.columns:
        cardinality = df[col].nunique()
        if cardinality > 0:  # Avoid division by zero
            score = avg_string_length[col] * (total_length / cardinality)
            column_scores[col] = score
    return column_scores

def reorder_columns(df):
    """Reorder columns based on precomputed scores."""
    column_scores = calculate_scores(df)  # Calculate scores for all columns once
    reordered_columns = []
    current_columns = list(df.columns)
    
    # Print the original column order
    print("Original column order:")
    print(current_columns)

    while current_columns:
        # Select column with max score
        selected_column = max(column_scores, key=column_scores.get)
        reordered_columns.append(selected_column)
        current_columns.remove(selected_column)
        
        # Remove the selected column's score from the dictionary
        del column_scores[selected_column]
        
        # Recalculate scores only for the remaining columns if necessary
        # For now, if needed, add specific recalculations here based on requirements

    print("reordered columns")
    print(reordered_columns)
    return df[reordered_columns]


def sort_rows_by_prefix(df):
    """Sort rows based on the concatenated string of all values in a row."""
    df['combined'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    sorted_df = df.sort_values(by='combined')
    sorted_df = sorted_df.drop(columns=['combined'])  # Drop the combined column after sorting
    return sorted_df

def deduplicate_rows(df):
    """Remove duplicate rows based on the 'review_content' column."""
    deduplicated_df = df.drop_duplicates(subset=['review_content'])
    return deduplicated_df


# Load data from CSV files
print("loading critic reviews data")
critic_reviews_df = pd.read_csv('/Users/zhang/Desktop/huawei/so/data/rotten_tomatoes_critic_reviews.csv')
print("loading movies data")
movies_df = pd.read_csv('data/rotten_tomatoes_movies.csv')

# Merge the two DataFrames on the 'rotten_tomatoes_link' column
print("merging the data")
merged_df = pd.merge(critic_reviews_df, movies_df, on='rotten_tomatoes_link', how='inner')

# select only first 15000 rows from the merged data frame
merged_small_df = merged_df.head(15000)

# Step 1: Reorder columns
print("reorder columns")
reordered_df = reorder_columns(merged_small_df)

# Step 2: Sort rows by prefix
print("sort rows ")
sorted_df = sort_rows_by_prefix(reordered_df)

# Step 3: Deduplicate rows
print("deduplicate rows ")
deduplicated_df = deduplicate_rows(sorted_df)

# Store the deduplicated DataFrame in a CSV file
output_file = 'data/optimized_review_table_15k.csv'  # Specify the output filename
deduplicated_df.to_csv(output_file)  # Save to CSV without the index


# Output results
print("Reordered DataFrame:")
print(reordered_df.head())  # Display first few rows for brevity
print("\nSorted DataFrame:")
print(sorted_df.head())  # Display first few rows for brevity
print("\nDeduplicated DataFrame:")
print(deduplicated_df.head())  # Display first few rows for brevity

print(f"\nDeduplicated DataFrame saved to {output_file}")
