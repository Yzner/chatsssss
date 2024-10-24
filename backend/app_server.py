# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import mysql.connector
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertModel
# from fuzzywuzzy import process
# from spellchecker import SpellChecker

# app = Flask(__name__)
# CORS(app)

# spell = SpellChecker()

# def connect_to_db():
#     connection = mysql.connector.connect(
#         host='localhost',
#         user='root',
#         password='root',
#         database='chat'
#     )
#     return connection

# def load_models():
#     gpt_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
#     gpt_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")

#     bert_model = BertModel.from_pretrained("bert-base-uncased")
#     bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#     return gpt_model, gpt_tokenizer, bert_model, bert_tokenizer

# gpt_model, gpt_tokenizer, bert_model, bert_tokenizer = load_models()

# def correct_spelling(user_input):
#     exceptions = {"palawansu"}  # Set of words to not correct
#     corrected_input = []
    
#     for word in user_input.split():
#         if word.lower() in exceptions:  # Check if the word is in the exceptions (case-insensitive)
#             corrected_input.append(word)  # Keep the original word
#         else:
#             corrected_word = spell.correction(word)  # Attempt to correct the word
#             # Only append if the corrected word is not None or empty
#             if corrected_word:  
#                 corrected_input.append(corrected_word)
#             else:
#                 corrected_input.append(word)  # Append the original word if no correction found
                
#     return " ".join(corrected_input)


# def fetch_faq_data():
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT question, answer FROM faq")
#     faq_data = cursor.fetchall()
#     cursor.close()
#     connection.close()
#     return faq_data

# def generate_bert_embeddings(user_input, bert_model, bert_tokenizer):
#     inputs = bert_tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = bert_model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1)

# def generate_gpt_response(user_input, context_embedding, gpt_model, gpt_tokenizer):
#     inputs = gpt_tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
#     chat_history_ids = gpt_model.generate(inputs['input_ids'], max_length=150)
#     return gpt_tokenizer.decode(chat_history_ids[:, inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True)

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json.get('user_input')
#     corrected_input = correct_spelling(user_input)
#     faq_data = fetch_faq_data()

#     questions = [q[0] for q in faq_data]
#     answers = {q[0]: q[1] for q in faq_data}

#     best_match, score = process.extractOne(corrected_input, questions)
#     if score >= 70:
#         return jsonify({'response': answers[best_match]})
#     else:
#         context_embedding = generate_bert_embeddings(corrected_input, bert_model, bert_tokenizer)
#         gpt_response = generate_gpt_response(corrected_input, context_embedding, gpt_model, gpt_tokenizer)
#         return jsonify({'response': gpt_response if gpt_response else "I'm sorry, I don't have an answer for that."})
    
    

# # Read admin
# @app.route('/faqs', methods=['GET'])
# def get_faqs():
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT * FROM faq")
#     faq_data = cursor.fetchall()
#     cursor.close()
#     connection.close()
    
#     faqs = [
#         {'id': row[0], 'category': row[1], 'question': row[2], 'answer': row[3]} 
#         for row in faq_data
#     ]
    
#     return jsonify(faqs)


# # Edit admin
# @app.route('/faqs/<int:id>', methods=['PUT'])
# def edit_faq(id):
#     data = request.json
#     category = data.get('category')
#     question = data.get('question')
#     answer = data.get('answer')

#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("UPDATE faq SET category=%s, question=%s, answer=%s WHERE id=%s", 
#                    (category, question, answer, id))
#     connection.commit()
#     cursor.close()
#     connection.close()
    
#     return jsonify({'message': 'FAQ updated successfully'})


# # Delete admin
# @app.route('/faqs/<int:id>', methods=['DELETE'])
# def delete_faq(id):
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("DELETE FROM faq WHERE id=%s", (id,))
#     connection.commit()
#     cursor.close()
#     connection.close()
    
#     return jsonify({'message': 'FAQ deleted successfully'})


# # Add admin
# @app.route('/faqs', methods=['POST'])
# def add_faq():
#     data = request.json
#     category = data.get('category')
#     question = data.get('question')
#     answer = data.get('answer')

#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("INSERT INTO faq (category, question, answer) VALUES (%s, %s, %s)", 
#                    (category, question, answer))
#     connection.commit()
#     cursor.close()
#     connection.close()

#     return jsonify({'message': 'FAQ added successfully'}), 201


# if __name__ == '__main__':
#     app.run(port=5000)






# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import mysql.connector
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertModel, pipeline
# from fuzzywuzzy import process
# from spellchecker import SpellChecker
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# app = Flask(__name__)
# CORS(app)

# spell = SpellChecker()

# def connect_to_db():
#     connection = mysql.connector.connect(
#         host='localhost',
#         user='root',
#         password='root',
#         database='chat'
#     )
#     return connection

# def load_models():
#     gpt_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
#     gpt_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")

#     bert_model = BertModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
#     bert_tokenizer = BertTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")

#     return gpt_model, gpt_tokenizer, bert_model, bert_tokenizer

# gpt_model, gpt_tokenizer, bert_model, bert_tokenizer = load_models()

# def correct_spelling(user_input):
#     corrected_input = []
#     for word in user_input.split():
#         corrected_input.append(spell.correction(word))
#     return " ".join(corrected_input)

# def fetch_faq_data():
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT question, answer FROM faq")
#     faq_data = cursor.fetchall()
#     cursor.close()
#     connection.close()
#     return faq_data

# # Generate embeddings using BERT or a similar model for semantic matching
# def generate_bert_embeddings(user_input, bert_model, bert_tokenizer):
#     inputs = bert_tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = bert_model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1)  # Shape: (batch_size, hidden_size)

# # Generate GPT-2 response
# def generate_gpt_response(user_input, previous_conversation, gpt_model, gpt_tokenizer):
#     input_text = previous_conversation + f"\nUser: {user_input}\nBot:"
#     inputs = gpt_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    
#     chat_history_ids = gpt_model.generate(
#         inputs['input_ids'], 
#         attention_mask=inputs['attention_mask'], 
#         max_length=200,
#         no_repeat_ngram_size=3,  
#         temperature=0.7,  
#         top_p=0.9,
#         do_sample=True,   
#         num_return_sequences=1,
#         pad_token_id=gpt_tokenizer.eos_token_id  # Explicitly set pad_token_id
#     )

    
#     return gpt_tokenizer.decode(chat_history_ids[:, inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True)

# # Rerank GPT-2 response based on FAQ similarity using BERT embeddings
# def rerank_gpt2_answer(gpt_answer, faq_answers, bert_model, bert_tokenizer):
#     gpt_embedding = generate_bert_embeddings(gpt_answer, bert_model, bert_tokenizer).cpu().numpy()
#     faq_embeddings = [generate_bert_embeddings(a, bert_model, bert_tokenizer).cpu().numpy() for a in faq_answers]

#     similarities = cosine_similarity([gpt_embedding], faq_embeddings).flatten()
#     best_match_score = max(similarities)
    
#     if best_match_score >= 0.7:  
#         best_faq_answer = faq_answers[np.argmax(similarities)]
#         return best_faq_answer
#     else:
#         return gpt_answer

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json.get('user_input')
    
#     # Spell-check and feedback
#     corrected_input = correct_spelling(user_input)
#     if corrected_input != user_input:
#         feedback = f"Did you mean: '{corrected_input}'? Using corrected input."
#     else:
#         feedback = None

#     # Fetch FAQ data
#     faq_data = fetch_faq_data()
#     questions = [q[0] for q in faq_data]
#     answers = {q[0]: q[1] for q in faq_data}

#     # Try matching user input with FAQ using fuzzy matching
#     best_match, score = process.extractOne(corrected_input, questions)
    
#     if score >= 70:
#         response = answers[best_match]
#         if feedback:
#             response = f"{feedback}\n\n{response}"
#         return jsonify({'response': response})
#     else:
#         # Generate GPT-2 response if no close FAQ match
#         gpt_response = generate_gpt_response(corrected_input, "", gpt_model, gpt_tokenizer)

#         # Rerank GPT-2 response using FAQ similarity with BERT
#         faq_answers = [answer for _, answer in faq_data]
#         reranked_answer = rerank_gpt2_answer(gpt_response, faq_answers, bert_model, bert_tokenizer)

#         # If reranking didn't yield a confident answer, append that it's AI-generated
#         if reranked_answer == gpt_response:
#             if feedback:
#                 response = f"{feedback}\n\n{gpt_response} (AI-generated response)"
#             else:
#                 response = f"{gpt_response} (AI-generated response)"
#         else:
#             if feedback:
#                 response = f"{feedback}\n\n{reranked_answer}"
#             else:
#                 response = reranked_answer
        
#         return jsonify({'response': response})
    

# if __name__ == '__main__':
#     app.run(debug=True)






# from flask import Flask, request, jsonify, session
# from flask_cors import CORS
# import mysql.connector
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertModel, pipeline
# from fuzzywuzzy import process
# from spellchecker import SpellChecker
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# app = Flask(__name__)
# CORS(app)
# app.secret_key = 'your_secret_key'  # Needed for session handling

# spell = SpellChecker()

# def connect_to_db():
#     connection = mysql.connector.connect(
#         host='localhost',
#         user='root',
#         password='root',
#         database='chat'
#     )
#     return connection

# def load_models():
#     gpt_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
#     gpt_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")

#     bert_model = BertModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
#     bert_tokenizer = BertTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")

#     return gpt_model, gpt_tokenizer, bert_model, bert_tokenizer

# gpt_model, gpt_tokenizer, bert_model, bert_tokenizer = load_models()

# def correct_spelling(user_input):
#     corrected_input = []
#     for word in user_input.split():
#         corrected_input.append(spell.correction(word))
#     return " ".join(corrected_input)

# def fetch_faq_data():
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT question, answer FROM faq")
#     faq_data = cursor.fetchall()
#     cursor.close()
#     connection.close()
#     return faq_data

# # Generate embeddings using BERT or a similar model for semantic matching
# def generate_bert_embeddings(user_input, bert_model, bert_tokenizer):
#     inputs = bert_tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = bert_model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1)

# # Generate GPT-2 response
# def generate_gpt_response(user_input, gpt_model, gpt_tokenizer, previous_conversation=None):
#     input_text = f"{previous_conversation}\nUser: {user_input}\nBot:" if previous_conversation else f"User: {user_input}\nBot:"
#     inputs = gpt_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    
#     chat_history_ids = gpt_model.generate(
#         inputs['input_ids'], 
#         attention_mask=inputs['attention_mask'], 
#         max_length=200,
#         no_repeat_ngram_size=3,  
#         temperature=0.7,  
#         top_p=0.9,
#         do_sample=True,   
#         num_return_sequences=1,
#         pad_token_id=gpt_tokenizer.eos_token_id  
#     )
    
#     return gpt_tokenizer.decode(chat_history_ids[:, inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True)

# # Rerank GPT-2 response based on FAQ similarity using BERT embeddings
# def rerank_gpt2_answer(gpt_answer, faq_answers, bert_model, bert_tokenizer):
#     gpt_embedding = generate_bert_embeddings(gpt_answer, bert_model, bert_tokenizer).cpu().numpy()
#     faq_embeddings = [generate_bert_embeddings(a, bert_model, bert_tokenizer).cpu().numpy() for a in faq_answers]

#     similarities = cosine_similarity([gpt_embedding], faq_embeddings).flatten()
#     best_match_score = max(similarities)
    
#     if best_match_score >= 0.7:  
#         best_faq_answer = faq_answers[np.argmax(similarities)]
#         return best_faq_answer
#     else:
#         return gpt_answer

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json.get('user_input')
    
#     corrected_input = correct_spelling(user_input)
#     faq_data = fetch_faq_data()

#     questions = [q[0] for q in faq_data]
#     answers = {q[0]: q[1] for q in faq_data}

#     # Get previous conversation from session
#     previous_conversation = session.get('conversation', '')

#     # Find closest match from FAQ
#     best_match, score = process.extractOne(corrected_input, questions)
#     if score >= 70:
#         response = answers[best_match]
#     else:
#         # Generate GPT-2 response based on conversation history
#         gpt_response = generate_gpt_response(corrected_input, gpt_model, gpt_tokenizer, previous_conversation)
#         response = rerank_gpt2_answer(gpt_response, [a for _, a in faq_data], bert_model, bert_tokenizer)

#     # Update conversation history in session
#     session['conversation'] = f"{previous_conversation}\nUser: {user_input}\nBot: {response}"

#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(debug=True)








# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel

# app = Flask(__name__)
# CORS(app)

# # Load the fine-tuned GPT-2 model and tokenizer
# def load_gpt_model():
#     gpt_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
#     gpt_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")
#     return gpt_model, gpt_tokenizer

# gpt_model, gpt_tokenizer = load_gpt_model()

# # Correct spelling function (optional, can remove if not needed)
# def correct_spelling(user_input):
#     # Implement spell checking here if required
#     # Placeholder for any spell-checking logic you want to retain
#     return user_input

# # Generate GPT response based on the user input and the fine-tuned GPT-2 model
# def generate_gpt_response(user_input, gpt_model, gpt_tokenizer):
#     inputs = gpt_tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
#     chat_history_ids = gpt_model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)
#     return gpt_tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)

# @app.route('/chat', methods=['POST'])
# def chat():
#     # Get the user input from the request
#     user_input = request.json.get('user_input')
    
#     # Optionally correct the spelling (you can disable this if not needed)
#     corrected_input = correct_spelling(user_input)
    
#     # Generate a response from the fine-tuned GPT-2 model
#     gpt_response = generate_gpt_response(corrected_input, gpt_model, gpt_tokenizer)
    
#     # Return the response as JSON
#     return jsonify({'response': gpt_response if gpt_response else "I'm sorry, I don't have an answer for that."})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

#best







# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from sentence_transformers import SentenceTransformer, util  # To compare sentence similarity
# import mysql.connector

# app = Flask(__name__)
# CORS(app)

# # Load the fine-tuned GPT-2 model and tokenizer
# def load_gpt_model():
#     gpt_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
#     gpt_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")
#     return gpt_model, gpt_tokenizer

# gpt_model, gpt_tokenizer = load_gpt_model()

# # Load a sentence transformer model for FAQ matching
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Connect to the database
# def connect_to_db():
#     try:
#         connection = mysql.connector.connect(
#             host='localhost',
#             user='root',
#             password='root',
#             database='chat'
#         )
#         return connection
#     except mysql.connector.Error as err:
#         print(f"Error: {err}")
#         return None

# # Fetch FAQ data from the database
# def fetch_faq_data(connection):
#     cursor = connection.cursor()
#     cursor.execute("SELECT question, answer FROM faq")
#     faq_data = cursor.fetchall()
#     cursor.close()
#     return faq_data

# # Function to find the closest FAQ match based on similarity
# def get_faq_match(user_input, faq_data):
#     # Extract FAQ questions
#     faq_questions = [faq[0] for faq in faq_data]
    
#     # Embed user input and FAQ questions
#     user_embedding = sentence_model.encode(user_input, convert_to_tensor=True)
#     faq_embeddings = sentence_model.encode(faq_questions, convert_to_tensor=True)
    
#     # Compute similarity between user input and FAQ questions
#     similarities = util.pytorch_cos_sim(user_embedding, faq_embeddings)[0]
    
#     # Find the index of the most similar FAQ question
#     best_match_idx = similarities.argmax().item()
#     best_match_score = similarities[best_match_idx].item()
    
#     # Set a similarity threshold (e.g., 0.7) to decide if the match is good enough
#     if best_match_score > 0.7:
#         return faq_data[best_match_idx][1]  # Return the answer
#     return None

# # Correct spelling function (optional, can remove if not needed)
# def correct_spelling(user_input):
#     # Placeholder for any spell-checking logic you want to retain
#     return user_input

# # Generate GPT response based on the user input and the fine-tuned GPT-2 model
# def generate_gpt_response(user_input, gpt_model, gpt_tokenizer):
#     inputs = gpt_tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
#     chat_history_ids = gpt_model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)
#     return gpt_tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)

# @app.route('/chat', methods=['POST'])
# def chat():
#     # Get the user input from the request
#     user_input = request.json.get('user_input')
    
#     # Optionally correct the spelling (you can disable this if not needed)
#     corrected_input = correct_spelling(user_input)
    
#     # Connect to the database
#     connection = connect_to_db()
#     if connection:
#         faq_data = fetch_faq_data(connection)
#         connection.close()
    
#         # Try to find an FAQ match first
#         faq_answer = get_faq_match(corrected_input, faq_data)
    
#         if faq_answer:
#             return jsonify({faq_answer})  # Return the FAQ answer if found

#     # If no FAQ match is found, fall back to the GPT-2 model
#     gpt_response = generate_gpt_response(corrected_input, gpt_model, gpt_tokenizer)
    
#     # Return the response as JSON
#     return jsonify({'response': gpt_response if gpt_response else "I'm sorry, I don't have an answer for that."})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

#goods but need improvement




from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from fuzzywuzzy import fuzz

app = Flask(__name__)
CORS(app)

# Load the fine-tuned GPT-2 model and tokenizer
def load_gpt_model():
    gpt_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")
    return gpt_model, gpt_tokenizer

gpt_model, gpt_tokenizer = load_gpt_model()

# Connect to MySQL database
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
def fetch_faq_data():
    connection = connect_to_db()
    if connection is None:
        return []

    cursor = connection.cursor()
    cursor.execute("SELECT question, answer FROM faq")
    faq_data = cursor.fetchall()
    cursor.close()
    connection.close()
    return faq_data

faq_data = fetch_faq_data()  # Cache FAQ data for quick lookup

# Find the best match in the FAQ using fuzzy string matching
def find_best_faq_match(user_input, faq_data, threshold=80):
    best_match = None
    highest_score = 0

    for question, answer in faq_data:
        score = fuzz.ratio(user_input.lower(), question.lower())
        if score > highest_score:
            highest_score = score
            best_match = answer

    # Return the answer if it meets the threshold, otherwise return None
    return best_match if highest_score >= threshold else None

# Correct spelling function (optional, can remove if not needed)
def correct_spelling(user_input):
    # Implement spell checking here if required
    # Placeholder for any spell-checking logic you want to retain
    return user_input

# Generate GPT response based on the user input and the fine-tuned GPT-2 model
def generate_gpt_response(user_input, gpt_model, gpt_tokenizer):
    inputs = gpt_tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
    chat_history_ids = gpt_model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)
    return gpt_tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)

@app.route('/chat', methods=['POST'])
def chat():
    # Get the user input from the request
    user_input = request.json.get('user_input')
    
    # Optionally correct the spelling (you can disable this if not needed)
    corrected_input = correct_spelling(user_input)

    # Try to find the best FAQ match first
    faq_answer = find_best_faq_match(corrected_input, faq_data)
    
    # If an FAQ answer is found, return it
    if faq_answer:
        return jsonify({'response': faq_answer})

    # If no FAQ match is found, generate a response from the fine-tuned GPT-2 model
    gpt_response = generate_gpt_response(corrected_input, gpt_model, gpt_tokenizer)
    
    # Return the GPT-2 response as JSON
    return jsonify({'response': gpt_response if gpt_response else "I'm sorry, I don't have an answer for that."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
