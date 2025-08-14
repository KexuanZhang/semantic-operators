import math
import pandas as pd


# Function to calculate hit count for a value in a field
def calculate_hit_count(value, field, table, functional_dependencies):
    if isinstance(value, float) and math.isnan(value):
        rows_with_value = table[
            table[field].isna()  # This is a condition, not real rows
        ]
    else:
        rows_with_value = table[table[field] == value]
    inferred_columns = [
        col for (source, col) in functional_dependencies if source == field
    ]

    total_length = len(str(value))**2 + sum(
        rows_with_value[col].apply(len).sum() for col in inferred_columns
    ) / len(rows_with_value)
    hit_count = total_length * (len(rows_with_value) - 1)

    return hit_count, [field] + inferred_columns

# Greedy Group Recursion (GGR) function
def ggr(table, functional_dependencies, depth=0, max_depth=100):
    print(f"GGR: Depth {depth}, Table Size: {table.shape}")
    
    # Base conditions
    if table.shape[0] == 1:  # Single row case
        return 0, table.iloc[0].tolist()
    if table.shape[1] == 1:  # Single column case
        sorted_table = table.sort_values(by=table.columns[0])
        return sum(
            3**2 if isinstance(value, float) and math.isnan(value)  # 'nan'
            else len(value)**2
            for value in sorted_table.iloc[:, 0]
        ), sorted_table.values.tolist()
    
    # Prevent excessive recursion
    if depth >= max_depth:
        print("GGR: Maximum recursion depth reached")
        return 0, []

    max_hit_count, best_value, best_field, best_cols = -1, None, None, []
    print("GGR: for loop")

    for field in table.columns:
        for value in table[field].unique():
            hit_count, cols = calculate_hit_count(value, field, table, functional_dependencies)
            if hit_count > max_hit_count:
                max_hit_count, best_value, best_field, best_cols = hit_count, value, field, cols

    print("GGR: for loop end")

    if best_field is None:  # No valid field found
        print("GGR: No valid field found, returning 0")
        return 0, []

    print("GGR: extracting rows")
    rows_with_value = table[table[best_field] == best_value]
    remaining_rows = table[table[best_field] != best_value]

    # Recursive calls
    print("GGR: recursive calls")
    hit_count_A, reordered_A = ggr(remaining_rows, functional_dependencies, depth + 1, max_depth)
    hit_count_B, reordered_B = ggr(rows_with_value.drop(columns=best_cols), functional_dependencies, depth + 1, max_depth)

    # Combine results
    print("GGR: combine results")
    total_hit_count = hit_count_A + hit_count_B + max_hit_count
    if len(reordered_B) == 0:
        reordered_list = [[best_value] + reordered_B] + reordered_A
    else:
        reordered_list = [[best_value] + reordered_B[i] for i in range(len(rows_with_value))] + reordered_A

    return total_hit_count, reordered_list
# LLM inference function
def llm_inference(system_prompt, user_prompt, row):
    # Example: Construct a JSON-style prompt
    row_prompt = {f"field_{i+1}": str(value) for i, value in enumerate(row)}
    prompt = f"{system_prompt}\nUser Question: {user_prompt}\nRow Data: {row_prompt}"
    return prompt # Replace with actual LLM call



# Load data from CSV files
print("Loading critic reviews data")
critic_reviews_df = pd.read_csv('rotten_tomatoes_critic_reviews.csv')
print("Loading movies data")
movies_df = pd.read_csv('rotten_tomatoes_movies.csv')

# Merge the two DataFrames on the 'rotten_tomatoes_link' column
print("Merging the data")
merged_df = pd.merge(critic_reviews_df, movies_df, on='rotten_tomatoes_link', how='inner')

# Convert the merged DataFrame to a list of lists for GGR
# Assuming we want to use specific columns for the GGR algorithm
# You can modify the columns based on your requirements
columns_of_interest = ['movie_info', 'review_content', 'movie_title']  # Example columns
merged_table = merged_df[columns_of_interest]#.values.tolist()
    
# Define functional dependencies (example, adjust based on schema)
functional_dependencies = [("movie_id", "title"), ("movie_id", "genre")]

# Apply GGR algorithm
print("call GGR algo")
total_hits, reordered_table = ggr(merged_table, functional_dependencies)
print("Total Prefix Hits:", total_hits)
print("Reordered Table:", reordered_table)

# Perform LLM inference
system_prompt = "Analyze the following row from the dataset:"
user_prompt = "What can you infer about the critic's sentiment?"
for row in reordered_table:
    prompt = llm_inference(system_prompt, user_prompt, row)
    print("Generated Prompt for LLM:", prompt)
