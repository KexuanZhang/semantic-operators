import os
import pandas as pd
from vllm import LLM, SamplingParams
import torch
import torch.distributed as dist
import time 
from collections import defaultdict

# Set environment variable for cached outputs
os.environ["VLLM_USE_CACHED_OUTPUTS"] = "True"

# Clear GPU memory
torch.cuda.empty_cache()

# Load the model (provide the correct path or model checkpoint)
model_path = "/data/llama_7b/llama2_7B/zhouyucheng/llama2_7B/llama2_7B"  # Replace with your local path

# Initialize the LLM 
llm = LLM(model=model_path)

# Define sampling parameters
sampling_params = SamplingParams(temperature=0.7, top_p=0.95)

# Function to check current GPU memory usage
def get_memory_usage():
    return torch.cuda.memory_allocated() / (1024 ** 2)  # Convert bytes to MB

# Function to generate text using vLLM
def llm_inference(prompt):
    # Generate text using vLLM
    output = llm.generate(prompt, sampling_params)
    return output[0].outputs[0].text

# Generic function to get movie recommendations
import pandas as pd
from collections import defaultdict

def get_movie_recommendations(df):
    all_recommendations = []  # List to collect movie recommendations
    seen_titles = set()  # Initialize seen_titles as an empty set
    results = defaultdict(list)  # Initialize results as a defaultdict to collect responses

    for index, row in df.iterrows():
        movie_info = row['movie_info']
        review_content = row['review_content']
        movie_title = row['movie_title']
        
        # Construct the prompt
        prompt = f"Analyze whether this movie would be suitable for kids based on {movie_info} and {review_content}."
        
        # Generate recommendations
        is_suitable = llm_inference(prompt).strip()  # Get the response and strip whitespace
        
        # Append the response to the results for the specific movie title
        results[movie_title].append(is_suitable)

    # Now consolidate the responses for each movie title
    consolidated_results = []
    for movie_title, responses in results.items():
        # Count the occurrences of 'Yes' and 'No'
        yes_count = responses.count("Yes")
        no_count = responses.count("No")
        
        # Determine the majority response
        majority_response = 'Yes' if yes_count > no_count else 'No'
        
        # Append the consolidated result
        consolidated_results.append({
            'movie_title': movie_title,
            'recommended_for_children': majority_response
        })

    # Convert consolidated results to DataFrame for better visualization
    results_df = pd.DataFrame(consolidated_results)  # Create DataFrame from consolidated results list

    # Print the results
    print(results_df)
    return results_df  # Return the DataFrame for further use

# Start timing
start_time = time.time()  # Record the start time

# Load data from CSV files
print("Loading data")
movie_df = pd.read_csv('optimized_review_table_15k.csv')

# Filter the DataFrame for reviews where review_type is "Fresh"
filtered_movie_df = movie_df[movie_df['review_type'] == 'Fresh']

# Get movie analysis
analysis_result = get_movie_recommendations(filtered_movie_df)

# Print the results
print("Analysis result: ")
print(analysis_result)

# End timing
end_time = time.time()  # Record the end time
execution_time = end_time - start_time  # Calculate the total execution time
print(f"Total execution time: {execution_time:.2f} seconds")  # Print the execution time

# Cleanup
if dist.is_initialized():
    dist.destroy_process_group() 