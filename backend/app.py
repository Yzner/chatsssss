# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from spellchecker import SpellChecker

# app = Flask(__name__)
# CORS(app)

# spell = SpellChecker()

# def load_models():
#     gpt_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
#     gpt_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")
#     return gpt_model, gpt_tokenizer

# gpt_model, gpt_tokenizer = load_models()

# def correct_spelling(user_input):
#     corrected_input = []
#     for word in user_input.split():
#         corrected_input.append(spell.correction(word))
#     return " ".join(corrected_input)

# def generate_gpt_response(user_input, gpt_model, gpt_tokenizer):
#     inputs = gpt_tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
    
#     # Generate response with restrictions
#     chat_history_ids = gpt_model.generate(
#         inputs['input_ids'],
#         max_length=150,
#         temperature=0.1,       # Lower temperature for less randomness
#         top_p=0.9,             # Nucleus sampling to limit creativity
#         do_sample=False,       # Disable sampling for more deterministic output
#         num_beams=5,           # Use beam search for better output
#         early_stopping=True,   # Stop generation early if needed
#         no_repeat_ngram_size=3 # Avoid repeating the same n-grams
#     )
    
#     return gpt_tokenizer.decode(chat_history_ids[:, inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True)

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json.get('user_input')
#     corrected_input = correct_spelling(user_input)
    
#     # Generate response using fine-tuned GPT model
#     gpt_response = generate_gpt_response(corrected_input, gpt_model, gpt_tokenizer)
    
#     return jsonify({'response': gpt_response if gpt_response else "I'm sorry, I don't have an answer for that."})

# if __name__ == '__main__':
#     app.run(port=5000)



from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from spellchecker import SpellChecker
from sentence_transformers import SentenceTransformer, util
import mysql.connector

app = Flask(__name__)
CORS(app)

spell = SpellChecker()

# Load the GPT-2 model and tokenizer
def load_models():
    gpt_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")
    retrieval_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Using a lightweight retrieval model
    return gpt_model, gpt_tokenizer, retrieval_model

gpt_model, gpt_tokenizer, retrieval_model = load_models()

# Spell checker temporarily disabled for testing
def correct_spelling(user_input):
    return user_input  # Temporarily bypass spell checker to avoid interference

# Database connection
def connect_to_db():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='root',
            database='chat'
        )
        return connection
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

# Fetch FAQ data from the database
def fetch_faq_data(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT category, question, answer FROM faq")
    faq_data = cursor.fetchall()
    cursor.close()
    return faq_data

# Perform retrieval-based FAQ matching before generating a response
def retrieve_faq_response(user_input, faq_data):
    faq_questions = [item[1] for item in faq_data]  # Extract only the questions
    faq_embeddings = retrieval_model.encode(faq_questions, convert_to_tensor=True)
    user_embedding = retrieval_model.encode(user_input, convert_to_tensor=True)

    # Find the most similar question from the FAQ
    similarity_scores = util.pytorch_cos_sim(user_embedding, faq_embeddings)
    best_match_idx = similarity_scores.argmax()
    best_match_question = faq_data[best_match_idx][1]
    best_match_answer = faq_data[best_match_idx][2]

    return best_match_question, best_match_answer

# Generate response using fine-tuned GPT model with constraints
def generate_gpt_response(user_input, gpt_model, gpt_tokenizer):
    inputs = gpt_tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
    
    # Generate response with restrictions to limit creativity
    chat_history_ids = gpt_model.generate(
        inputs['input_ids'],
        max_length=150,
        temperature=0.1,  # Lower temperature for less random responses
        top_p=0.9,        # Nucleus sampling for controlling randomness
        num_beams=5,      # Beam search for better, more reliable results
        no_repeat_ngram_size=3,  # Prevent repetition of phrases
        early_stopping=True      # Stop early when reaching a good solution
    )
    
    return gpt_tokenizer.decode(chat_history_ids[:, inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('user_input')
    corrected_input = correct_spelling(user_input)
    
    # Fetch FAQ data
    connection = connect_to_db()
    if connection is None:
        return jsonify({'response': "Error connecting to the database."})
    
    faq_data = fetch_faq_data(connection)
    connection.close()

    # Perform retrieval
    best_question, best_answer = retrieve_faq_response(corrected_input, faq_data)

    # Optionally, feed this retrieved answer into GPT for refinement
    gpt_response = generate_gpt_response(f"FAQ Response: {best_answer}", gpt_model, gpt_tokenizer)
    
    return jsonify({'response': gpt_response if gpt_response else "I'm sorry, I don't have an answer for that."})

if __name__ == '__main__':
    app.run(port=5000)
