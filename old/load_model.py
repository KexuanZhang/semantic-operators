from transformers import AutoTokenizer, AutoModelForCausalLM

local_model_path = "/data/llama_7b/llama2_7B/zhouyucheng/llama2_7B/llama2_7B"  # Replace with your local path

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)
