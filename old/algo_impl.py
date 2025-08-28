import pandas as pd

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

# def invoke_llm(df):
#     """Simulate invoking a large language model (LLM) for each review."""
#     results = []
#     for index, row in df.iterrows():
#         #prompt = f"Analyze the review: {row['review_content']} for movie: {row['movie_title']}"
#         # Here we would call our LLM API
#         # response = call_llm_api(prompt)
#        response = f"Processed: {prompt}"  # Placeholder for actual LLM response
#        results.append(response)
#     return results

# Load data from CSV files
print("loading critic reviews data")
critic_reviews_df = pd.read_csv('rotten_tomatoes_critic_reviews.csv')
print("loading movies data")
movies_df = pd.read_csv('rotten_tomatoes_movies.csv')

# Merge the two DataFrames on the 'rotten_tomatoes_link' column
print("merging the data")
merged_df = pd.merge(critic_reviews_df, movies_df, on='rotten_tomatoes_link', how='inner')

# Step 1: Reorder columns
print("reorder columns")
reordered_df = reorder_columns(merged_df)

# Step 2: Sort rows by prefix
print("sort rows ")
sorted_df = sort_rows_by_prefix(reordered_df)

# Step 3: Deduplicate rows
print("deduplicate rows ")
deduplicated_df = deduplicate_rows(sorted_df)

# Step 4: Invoke LLM
#llm_results = invoke_llm(deduplicated_df)

# Output results
print("Reordered DataFrame:")
print(reordered_df.head())  # Display first few rows for brevity
print("\nSorted DataFrame:")
print(sorted_df.head())  # Display first few rows for brevity
print("\nDeduplicated DataFrame:")
print(deduplicated_df.head())  # Display first few rows for brevity
#print("\nLLM Results:")
#print(llm_results[:5])  # Display first few results for brevity
