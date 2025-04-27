from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bert_score import score  # Import BERTScore

# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Candidate and reference sentences
candidate = "PalawanSU was transformed into a university on November 12, 1994, through Republic Act 7818."
reference = "On November 12, 1994, Palawan Teachers’ College was converted into a university through R.A. 7818."


# Tokenize sentences and get embeddings
def get_embeddings(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the last hidden state as embeddings (shape: [batch_size, seq_len, hidden_dim])
    return outputs.last_hidden_state.squeeze(0), inputs["input_ids"].squeeze(0)

candidate_embeddings, candidate_ids = get_embeddings(candidate)
reference_embeddings, reference_ids = get_embeddings(reference)

# Compute cosine similarity
cosine_similarity = torch.nn.functional.cosine_similarity(
    candidate_embeddings.unsqueeze(1), reference_embeddings.unsqueeze(0), dim=2
)

# Convert to numpy for visualization
similarity_matrix = cosine_similarity.numpy()
candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_ids)
reference_tokens = tokenizer.convert_ids_to_tokens(reference_ids)

# Display results
print("Candidate Tokens:", candidate_tokens)
print("Reference Tokens:", reference_tokens)
print("Similarity Matrix Shape:", similarity_matrix.shape)

# Plot similarity matrix with a softer color palette
plt.figure(figsize=(16, 10))  # Increase figure size
sns.heatmap(
    similarity_matrix, 
    xticklabels=reference_tokens, 
    yticklabels=candidate_tokens, 
    cmap="Blues",  # Use a softer blue palette
    annot=True, 
    fmt=".4f", 
    annot_kws={"size": 7}  # Adjust annotation font size
)
plt.title("Token Similarity Matrix", fontsize=14)
plt.xlabel("Reference Tokens", fontsize=12)
plt.ylabel("Candidate Tokens", fontsize=12)
plt.xticks(rotation=90, fontsize=10)  # Rotate x-axis labels for better readability
plt.yticks(fontsize=10)  # Adjust y-axis label font size
plt.show()

# Compute BERTScore (precision, recall, F1)
P, R, F1 = score([candidate], [reference], lang="en")

# Print the BERTScore precision, recall, and F1
print("\nBERTScore Results:")
print(f"Precision: {P.item():.4f}")
print(f"Recall: {R.item():.4f}")
print(f"F1 Score: {F1.item():.4f}")
