"""
GGR Algorithm Implementation
Copied and adapted from the original GGR_algo_bugfixed.py
"""
import math
import pandas as pd


def calculate_hit_count(value, field, table, functional_dependencies):
    """Calculate hit count for a value in a field"""
    if isinstance(value, float) and math.isnan(value):
        rows_with_value = table[table[field].isna()]
    else:
        rows_with_value = table[table[field] == value]
    
    inferred_columns = [
        col for (source, col) in functional_dependencies if source == field
    ]

    total_length = len(str(value))**2 + sum(
        rows_with_value[col].apply(len).sum() for col in inferred_columns
    ) / len(rows_with_value) if len(rows_with_value) > 0 else 0
    
    hit_count = total_length * (len(rows_with_value) - 1)

    return hit_count, [field] + inferred_columns


def ggr(table, functional_dependencies, depth=0, max_depth=100):
    """Greedy Group Recursion (GGR) function"""
    print(f"GGR: Depth {depth}, Table Size: {table.shape}")
    
    # Base conditions
    if table.shape[0] == 1:  # Single row case
        return 0, table.iloc[0].tolist()
    if table.shape[1] == 1:  # Single column case
        sorted_table = table.sort_values(by=table.columns[0])
        return sum(
            3**2 if isinstance(value, float) and math.isnan(value)  # 'nan'
            else len(str(value))**2
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
    if isinstance(best_value, float) and math.isnan(best_value):
        rows_with_value = table[table[best_field].isna()]
        remaining_rows = table[~table[best_field].isna()]
    else:
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
