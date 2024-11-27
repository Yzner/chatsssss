from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, brevity_penalty

# Candidate and reference sentences
candidate = "PalawanSU was transformed into a university on November 12, 1994, through Republic Act 7818."
reference = "On November 12, 1994, Palawan Teachers’ College was converted into a university through R.A. 7818."

# Tokenize the sentences and normalize to lowercase
candidate_tokens = [word.lower() for word in candidate.split()]
reference_tokens = [word.lower() for word in reference.split()]

# Function to generate n-grams
def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Function to find matched n-grams between candidate and reference
def get_matched_ngrams(candidate_ngrams, reference_ngrams):
    return set(candidate_ngrams).intersection(reference_ngrams)

# Function to compute n-gram precision
def ngram_precision(candidate_tokens, reference_tokens, n):
    # Generate n-grams for both candidate and reference
    candidate_ngrams = generate_ngrams(candidate_tokens, n)
    reference_ngrams = generate_ngrams(reference_tokens, n)
    
    # Find matched n-grams
    matched_ngrams = get_matched_ngrams(candidate_ngrams, reference_ngrams)
    
    # Compute precision as matched n-grams divided by total candidate n-grams
    precision = len(matched_ngrams) / len(candidate_ngrams) if len(candidate_ngrams) > 0 else 0
    return precision, matched_ngrams, candidate_ngrams

# Compute n-gram precisions for 1-gram, 2-gram, 3-gram, and 4-gram
bleu_1, matched_1gram, total_1gram = ngram_precision(candidate_tokens, reference_tokens, 1)
bleu_2, matched_2gram, total_2gram = ngram_precision(candidate_tokens, reference_tokens, 2)
bleu_3, matched_3gram, total_3gram = ngram_precision(candidate_tokens, reference_tokens, 3)
bleu_4, matched_4gram, total_4gram = ngram_precision(candidate_tokens, reference_tokens, 4)

# Print n-grams and precision results
print("1-gram splits:")
print("Matched 1-gram:", matched_1gram)
print("Total 1-gram:", total_1gram)
print("Precision 1-gram:", bleu_1)

print("\n2-gram splits:")
print("Matched 2-gram:", matched_2gram)
print("Total 2-gram:", total_2gram)
print("Precision 2-gram:", bleu_2)

print("\n3-gram splits:")
print("Matched 3-gram:", matched_3gram)
print("Total 3-gram:", total_3gram)
print("Precision 3-gram:", bleu_3)

print("\n4-gram splits:")
print("Matched 4-gram:", matched_4gram)
print("Total 4-gram:", total_4gram)
print("Precision 4-gram:", bleu_4)

# Compute BLEU score using the BLEU formula (geometric average of n-gram precisions)
geo_avg_precision = (bleu_1 * bleu_2 * bleu_3 * bleu_4) ** (1/4)

# Compute brevity penalty
bp = brevity_penalty(reference_tokens, candidate_tokens)

# Calculate the total BLEU score
total_bleu = bp * geo_avg_precision

# Print final results
print(f"\nGeometric Average Precision: {geo_avg_precision:.4f}")
print(f"Brevity Penalty: {bp:.4f}")
print(f"Total BLEU Score: {total_bleu:.4f}")
