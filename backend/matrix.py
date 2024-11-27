import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2Tokenizer

# Define the text and the tokenizer
text = "What is the vision of PalawanSU?"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set eos_token as pad_token

# Tokenize the input text
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text, truncation=True, max_length=10, padding="max_length")

# Prepare the matrix for visualization
tokens_matrix = np.zeros((2, len(tokens) + 3), dtype=int)  # Create a matrix for tokens and their IDs

# Fill in the matrix: First row for tokenized words, second row for token IDs
tokens_matrix[0, 1:len(tokens) + 1] = [token for token in tokens]
tokens_matrix[1, 1:len(token_ids) + 1] = token_ids

# Add labels to the first column
tokens_matrix[0, 0] = 'Tokenized Subwords'
tokens_matrix[1, 0] = 'Token IDs'

# Create a plot for the matrix visualization
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

# Plot the matrix with labels
ax.table(cellText=tokens_matrix.T, colLabels=[""] * tokens_matrix.shape[1], cellLoc="center", loc="center", colLoc="center")

plt.title("Token Mapping Visualization", fontsize=16)
plt.show()
