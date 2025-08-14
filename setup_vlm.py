from vllm import LLM, SamplingParams

# Load the model (provide the correct path or model checkpoint)
model_path = "/data/llama_7b/llama2_7B/zhouyucheng/llama2_7B/llama2_7B"  # Replace with your local path
llm = LLM(model=model_path)

# Define sampling parameters
sampling_params = SamplingParams(temperature=0.7, top_p=0.95)

# Function to generatae text using vLLM
def generate_text(prompt):
    # Generate text using vLLM
    output = llm.generate(prompt, sampling_params)
    return output[0].outputs[0].text

# Generate text using vLLM
prompt = "Once upon a time,"
result = generate_text(prompt)

# Print the result
print("+++++++Result+++++++++")
print(result)

# Continue generating text with a new prompt
next_prompt = "The kingdom was filled with"
next_result = generate_text(next_prompt)

# Print the next result
print("+++++++Next Result+++++++++")
print(next_result)
