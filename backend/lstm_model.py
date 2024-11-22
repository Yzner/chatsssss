from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import torch

# Load GPT-2 model and tokenizer
model_name = "gpt2"  # You can replace with a fine-tuned model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define the function to enhance explanations
def enhance_explanation(input_sentence, max_length=100, temperature=0.7, top_p=0.9):
    """
    Enhance the explanation of a sentence using GPT-2.
    
    Args:
    - input_sentence (str): The sentence to enhance.
    - max_length (int): The maximum length of the output.
    - temperature (float): Sampling temperature for diversity.
    - top_p (float): Nucleus sampling for more focused generation.
    
    Returns:
    - str: The enhanced explanation.
    """
    # Add a prompt to guide GPT-2 for explanation enhancement
    prompt = f"Explain the following sentence in detail: {input_sentence}"
    
    # Encode input
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate output using model
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode output and return enhanced explanation
    explanation = tokenizer.decode(output[0], skip_special_tokens=True)
    return explanation

# Example usage
input_sentence = "Palawan State University originated in 1965 when President Diosdado P. Macapagal signed R.A. 4303 on June 19, establishing Palawan Teachers’ College, which was later renamed Palawan State University."
enhanced_explanation = enhance_explanation(input_sentence)
print("Enhanced Explanation:\n", enhanced_explanation)
