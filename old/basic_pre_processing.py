import pandas as pd

# Load data from CSV files
print("loading critic reviews data")
critic_reviews_df = pd.read_csv('rotten_tomatoes_critic_reviews.csv')
print("loading movies data")
movies_df = pd.read_csv('rotten_tomatoes_movies.csv')

# Merge the two DataFrames on the 'rotten_tomatoes_link' column
print("merging the data")
merged_df = pd.merge(critic_reviews_df, movies_df, on='rotten_tomatoes_link', how='inner')


# select only first 15000 rows from the merged data frame
merged_small_df = merged_df.head(15000)

# Store the deduplicated DataFrame in a CSV file
output_file = 'merged_movie_reviews_table_15k.csv'  # Specify the output filename
merged_small_df.to_csv(output_file)  # Save to CSV without the index


print(f"\n Merged DataFrame saved to {output_file}")
