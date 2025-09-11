#!/usr/bin/env python3
"""
Prompt Formation Analysis for llm_inference.py

This document explains how prompts are constructed in the LLM inference script.
The prompt formation happens in multiple stages with different formatting for different models.
"""

def explain_prompt_formation():
    print("=" * 70)
    print("PROMPT FORMATION IN LLM_INFERENCE.PY")
    print("=" * 70)
    
    print("\n📋 STAGE 1: BASE PROMPT CREATION")
    print("─" * 50)
    print("Location: process_dataset() function")
    print("Purpose: Create the initial prompt from dataset rows")
    
    print("\n1. Template-based Prompt Construction:")
    print("   • Default template: 'Analyze the following text: {text_content}. Is this a positive or negative text?'")
    print("   • Custom template via --prompt_template argument")
    print("   • Uses Python .format() with column names as placeholders")
    
    print("\n   Example:")
    print("   Template: 'Analyze the following text: {text_content}. Is this a positive or negative text?'")
    print("   Row data: {'text_content': 'This movie is amazing!'}")
    print("   Result: 'Analyze the following text: This movie is amazing!. Is this a positive or negative text?'")
    
    print("\n2. Column Detection:")
    print("   • Uses columns specified in --include_columns")
    print("   • Auto-detects content columns: 'content', 'text', 'review', 'description', 'comment', 'body'")
    print("   • Falls back to first available column if none found")
    
    print("\n3. Fallback Handling:")
    print("   • If template formatting fails: 'Analyze this: {row_data}'")
    print("   • Ensures every row gets processed even with template mismatches")
    
    print("\n🎯 STAGE 2: MODEL-SPECIFIC FORMATTING")
    print("─" * 50)
    print("Location: llm_inference() function")
    print("Purpose: Apply model-specific conversation formatting")
    
    print("\n1. Qwen Models:")
    print("   Format: <|im_start|>system\\nYou are a helpful assistant...\\n<|im_end|>")
    print("          <|im_start|>user\\n{prompt}\\n<|im_end|>")
    print("          <|im_start|>assistant\\n")
    
    print("\n2. LLaMA/Mistral/Vicuna Models:")
    print("   Format: <s>[INST] {prompt} [/INST]")
    
    print("\n3. ChatGLM Models:")
    print("   Format: [gMASK]system\\nYou are a helpful assistant...\\n\\nuser\\n{prompt}\\n\\nassistant\\n")
    
    print("\n4. Gemma Models:")
    print("   Format: <start_of_turn>user\\n{prompt}<end_of_turn>\\n<start_of_turn>model\\n")
    
    print("\n5. TinyLlama Chat Models:")
    print("   Format: <|system|>\\nYou are a helpful assistant...\\n<|user|>\\n{prompt}\\n<|assistant|>")
    
    print("\n6. Yi Models:")
    print("   Format: <|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n")
    
    print("\n7. GPT/OPT Models:")
    print("   Format: User: {prompt}\\nAssistant:")
    
    print("\n8. BERT/T5/Flan Models:")
    print("   Format: {prompt} (direct, no formatting)")
    
    print("\n9. Generic/Unknown Models:")
    print("   Format: ### Instruction:\\n{prompt}\\n\\n### Response:\\n")
    
    print("\n10. No Model Specified:")
    print("    Format: Answer the following question directly and concisely: {prompt}\\n")
    
    print("\n📊 COMPLETE PROMPT FLOW EXAMPLE")
    print("─" * 50)
    
    print("\n1. Dataset Row:")
    print("   {'text_content': 'The product quality is excellent!'}")
    
    print("\n2. Template Application:")
    print("   Template: 'Analyze the following text: {text_content}. Is this positive or negative?'")
    print("   Base Prompt: 'Analyze the following text: The product quality is excellent!. Is this positive or negative?'")
    
    print("\n3. Model-Specific Formatting (e.g., Qwen):")
    print("   Final Prompt:")
    print("   '<|im_start|>system")
    print("   You are a helpful assistant that provides clear, concise, and accurate answers.")
    print("   <|im_end|>")
    print("   <|im_start|>user")
    print("   Analyze the following text: The product quality is excellent!. Is this positive or negative?")
    print("   <|im_end|>")
    print("   <|im_start|>assistant")
    print("   '")
    
    print("\n🔧 CUSTOMIZATION OPTIONS")
    print("─" * 50)
    print("1. --prompt_template: Custom template with {column_name} placeholders")
    print("2. --include_columns: Specify which columns to use in prompts")
    print("3. Model auto-detection: Automatic formatting based on model name")
    
    print("\n💡 KEY FEATURES")
    print("─" * 50)
    print("• Template validation with dummy data")
    print("• Automatic fallback for template errors")
    print("• Model-specific conversation formatting")
    print("• Flexible column selection")
    print("• Error-resistant prompt construction")

if __name__ == "__main__":
    explain_prompt_formation()
