
# #so goodd
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import mysql.connector
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from fuzzywuzzy import fuzz
# import subprocess
# from werkzeug.security import generate_password_hash, check_password_hash
# import jwt
# import datetime
# import torch.nn as nn
# import nltk
# from nltk.corpus import words

# app = Flask(__name__)
# CORS(app)
# SECRET_KEY = 'your_secret_key'
# nltk.download('words')
# # Define Memory LSTM
# class MemoryLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(MemoryLSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x, hidden):
#         out, hidden = self.lstm(x, hidden)
#         out = self.fc(out[:, -1, :])
#         return out, hidden

#     def init_hidden(self, batch_size):
#         return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
#                 torch.zeros(self.num_layers, batch_size, self.hidden_size))

# gpt_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
# gpt_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")
# memory_lstm = MemoryLSTM(input_size=768, hidden_size=128, num_layers=2, output_size=1)


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

# def fetch_faq_data():
#     connection = connect_to_db()
#     if connection is None:
#         return []

#     cursor = connection.cursor()
#     cursor.execute("SELECT question, simple_answer FROM faq_data")
#     faq_data = cursor.fetchall()
#     cursor.close()
#     connection.close()
#     return faq_data

# faq_data = fetch_faq_data()  


# def find_best_faq_match(user_input, faq_data, threshold=85):
#     best_match = None
#     highest_score = 0

#     for question, simple_answer in faq_data:
#         score = fuzz.ratio(user_input.lower(), question.lower())
#         if score > highest_score:
#             highest_score = score
#             best_match = simple_answer
#     return best_match if highest_score >= threshold else None


# def correct_spelling(user_input):
#     return user_input

# def save_message(session_id, user_id, message, sender):
#     connection = connect_to_db()
#     if connection is None:
#         return
#     cursor = connection.cursor()
#     cursor.execute(
#         "INSERT INTO conversations (session_id, user_id, message, sender, timestamp) VALUES (%s, %s, %s, %s, NOW())",
#         (session_id, user_id, message, sender)
#     )
#     connection.commit()
#     cursor.close()
#     connection.close()

# def retrieve_past_response(session_id, user_id):
#     connection = connect_to_db()
#     if connection is None:
#         return None
#     cursor = connection.cursor()
#     query = """
#     SELECT message FROM conversations
#     WHERE session_id = %s AND user_id = %s AND sender = 'bot'
#     ORDER BY timestamp DESC LIMIT 1
#     """
#     cursor.execute(query, (session_id, user_id))
#     result = cursor.fetchone()
#     cursor.close()
#     connection.close()
#     return result[0] if result else None


# def detect_repeat_request(user_input):
#     repeat_phrases = ["repeat", "again", "say that again", "can you repeat", "can you say that again", "what did you say"]
#     for phrase in repeat_phrases:
#         if fuzz.ratio(user_input.lower(), phrase) > 80:
#             return True
#     return False

# def detect_clarification_request(user_input):
#     clarification_phrases = ["clarify", "more details", "explain further", "could you elaborate", "what do you mean"]
#     for phrase in clarification_phrases:
#         if fuzz.ratio(user_input.lower(), phrase.lower()) > 80:
#             return True
#     return False


# def detect_gibberish(user_input, threshold=0.6):
#     # Check if the message contains recognizable English words
#     words_list = user_input.split()
#     english_words = set(words.words())
#     score = 0
#     for word in words_list:
#         if word.lower() in english_words:
#             score += 1
#     return (score / len(words_list)) < threshold  # Return True if less than the threshold are valid words


# def generate_gpt_response(user_input, gpt_model, gpt_tokenizer, lstm_model, lstm_hidden, session_id, user_id, confidence_threshold=0.5): 
#     # Detect gibberish input
#     if detect_gibberish(user_input):
#         return "I'm sorry, but I couldn't understand your message. Could you please clarify or rephrase your question?"
    
#     if detect_clarification_request(user_input):
#         # Retrieve the last response directly and ask for more clarification
#         past_response = retrieve_past_response(session_id, user_id)
#         return past_response + " Could you specify what you'd like me to clarify further?"
    
#     # Check if the user is asking for a repetition
#     if detect_repeat_request(user_input):
#         # Retrieve the last response directly without generating a new one
#         past_response = retrieve_past_response(session_id, user_id)
#         return "Sure, here is what I said earlier, " + (past_response if past_response else "I'm sorry, I don't have a recent response to repeat.")

#     # Tokenize and get embeddings for the user input
#     inputs = gpt_tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
#     input_ids = inputs['input_ids']
#     embeddings = gpt_model.transformer.wte(input_ids)  # Get embeddings

#     # Use LSTM model to check if further clarification is needed
#     lstm_output, lstm_hidden = lstm_model(embeddings, lstm_hidden)
#     clarify_needed = (torch.sigmoid(lstm_output) > 0.5).item()

#     # If clarification is detected, retrieve the past response
#     if clarify_needed:
#         past_response = retrieve_past_response(session_id, user_id)
#         return past_response if past_response else "I'm sorry, I don't have a recent response to clarify."

#     # Otherwise, generate a new response with GPT-2
#     chat_history_ids = gpt_model.generate(input_ids, max_length=150, num_return_sequences=1, return_dict_in_generate=True, output_scores=True)
#     generated_tokens = chat_history_ids['sequences'][0]
#     token_scores = chat_history_ids['scores']

#     total_log_prob = 0
#     for score in token_scores:
#         total_log_prob += torch.log_softmax(score, dim=-1).max().item()
#     avg_log_prob = total_log_prob / len(token_scores)
#     confidence = torch.exp(torch.tensor(avg_log_prob)).item()
#     if confidence < confidence_threshold:
#         return "Sorry, I can't answer that. Could you please clarify or rephrase your question? I can only answer questions related to Palawan State University and the College of Science."
#     return gpt_tokenizer.decode(generated_tokens, skip_special_tokens=True)


# # @app.route('/chat', methods=['POST'])
# # def chat():
# #     user_input = request.json.get('user_input')
# #     session_id = request.json.get('session_id')
# #     user_id = request.json.get('user_id')

# #     # Save the user's message to the conversation history
# #     save_message(session_id, user_id, user_input, 'user')
    
# #     # Initialize LSTM hidden state with batch dimension
# #     lstm_hidden = memory_lstm.init_hidden(1)  # Batch size 1

# #     # Generate response
# #     response = generate_gpt_response(user_input, gpt_model, gpt_tokenizer, memory_lstm, lstm_hidden, session_id, user_id)
    
# #     # Save the bot's response to the conversation history
# #     save_message(session_id, user_id, response, 'bot')
    
# #     return jsonify({'response': response})



# # #GOOD
# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json.get('user_input')
#     session_id = request.json.get('session_id')
#     user_id = request.json.get('user_id')

#     # Save the user's message to the conversation history
#     save_message(session_id, user_id, user_input, 'user')
    
#     # Fetch FAQ data and attempt a match
#     faq_match = find_best_faq_match(user_input, faq_data)
#     if faq_match:
#         response = faq_match
#     else:
#         # Initialize LSTM hidden state with batch dimension
#         lstm_hidden = memory_lstm.init_hidden(1)  # Batch size 1
#         response = generate_gpt_response(user_input, gpt_model, gpt_tokenizer, memory_lstm, lstm_hidden, session_id, user_id)

#     # Save the bot's response to the conversation history
#     save_message(session_id, user_id, response, 'bot')
    
#     return jsonify({'response': response})



# # Read admin
# @app.route('/faqs', methods=['GET'])
# def get_faqs():
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT * FROM data")
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
#     cursor.execute("UPDATE data SET category=%s, question=%s, answer=%s WHERE id=%s", 
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
#     cursor.execute("DELETE FROM data WHERE id=%s", (id,))
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
#     cursor.execute("INSERT INTO data (category, question, answer) VALUES (%s, %s, %s)", 
#                    (category, question, answer))
#     connection.commit()
#     cursor.close()
#     connection.close()

#     return jsonify({'message': 'FAQ added successfully'}), 201

# #train button
# @app.route('/train', methods=['POST'])
# def train_data():
#     try:
#         subprocess.run(['python', 'fine_tune.py'], check=True)
#         return jsonify({'message': 'Training started successfully'}), 200
#     except subprocess.CalledProcessError as e:
#         print(f"Error: {e}")
#         return jsonify({'message': 'Failed to start training'}), 500
    


# #Login User
# def create_user(first_name, last_name, email, password, role):
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     hashed_password = generate_password_hash(password)
#     cursor.execute("INSERT INTO users (first_name, last_name, email, password, role) VALUES (%s, %s, %s, %s, %s)", 
#                    (first_name, last_name, email, hashed_password, role))
#     connection.commit()
#     cursor.close()
#     connection.close()
# def authenticate_user(email, password):
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT password FROM users WHERE email = %s", (email,))
#     result = cursor.fetchone()
#     cursor.close()
#     connection.close()
#     if result and check_password_hash(result[0], password):
#         return True
#     return False
# @app.route('/signup', methods=['POST'])
# def signup():
#     data = request.json
#     first_name = data.get('firstName')
#     last_name = data.get('lastName')
#     email = data.get('email')
#     password = data.get('password')
#     role = data.get('role')
#     create_user(first_name, last_name, email, password, role)
#     return jsonify({'success': True, 'message': 'User created successfully'}), 201
# @app.route('/login', methods=['POST'])
# def login():
#     data = request.json
#     email = data.get('email')
#     password = data.get('password')

#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT id, password FROM users WHERE email = %s", (email,))
#     result = cursor.fetchone()
#     cursor.close()
#     connection.close()

#     if result and check_password_hash(result[1], password):
#         token = jwt.encode({
#             'email': email,
#             'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
#         }, SECRET_KEY)
#         return jsonify({'success': True, 'token': token, 'userId': result[0]})  # Return user ID
#     return jsonify({'success': False, 'message': 'Invalid email or password'}), 401




# #admin
# @app.route('/adminsignup', methods=['POST'])
# def admin_signup():
#     data = request.json
#     first_name = data.get('firstName')
#     last_name = data.get('lastName')
#     email = data.get('email')
#     password = data.get('password')
#     department = data.get('department')
#     role = "pending" 

#     connection = connect_to_db()
#     cursor = connection.cursor()
#     hashed_password = generate_password_hash(password)
#     cursor.execute("INSERT INTO admin (first_name, last_name, email, password, role, department) VALUES (%s, %s, %s, %s, %s, %s)", 
#                    (first_name, last_name, email, hashed_password, role, department))
#     connection.commit()
#     cursor.close()
#     connection.close()

#     return jsonify({'success': True, 'message': 'Sign-up request submitted. Awaiting main admin approval.'}), 201

# # Admin 
# @app.route('/adminlogin', methods=['POST'])
# def admin_login():
#     data = request.json
#     email = data.get('email')
#     password = data.get('password')

#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT password, role FROM admin WHERE email = %s", (email,))
#     result = cursor.fetchone()
#     cursor.close()
#     connection.close()

#     if result and check_password_hash(result[0], password):
#         if result[1] == "approved" or result[1] == "main_admin":
#             token = jwt.encode({
#                 'email': email,
#                 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
#             }, SECRET_KEY)
#             return jsonify({'success': True, 'token': token})
#         else:
#             return jsonify({'success': False, 'message': 'Account pending main admin approval.'}), 403
#     return jsonify({'success': False, 'message': 'Invalid email or password'}), 401


# # Approve admin request 
# @app.route('/approve_admin', methods=['POST'])
# def approve_admin():
#     data = request.json
#     email = data.get('email')
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("UPDATE admin SET role = 'approved' WHERE email = %s AND role = 'pending'", (email,))
#     connection.commit()
#     cursor.close()
#     connection.close()

#     return jsonify({'message': 'Admin approved successfully.'})


# @app.route('/pending_admins', methods=['GET'])
# def get_pending_admins():
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT first_name, last_name, email, department FROM admin WHERE role = 'pending'")
#     pending_admins = cursor.fetchall()
#     cursor.close()
#     connection.close()

#     return jsonify([{'firstName': admin[0], 'lastName': admin[1], 'email': admin[2], 'department': admin[3]} for admin in pending_admins])

# @app.route('/decline_admin', methods=['POST'])
# def decline_admin():
#     data = request.json
#     email = data.get('email')
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("DELETE FROM admin WHERE email = %s AND role = 'pending'", (email,))
#     connection.commit()
#     cursor.close()
#     connection.close()

#     return jsonify({'message': 'Admin request declined successfully.'})


# @app.route('/conversation_history', methods=['GET'])
# def get_conversation_history():
#     session_id = request.args.get('session_id')
#     user_id = request.args.get('user_id')

#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute(
#         "SELECT message, sender FROM conversations WHERE session_id = %s AND user_id = %s ORDER BY timestamp",
#         (session_id, user_id)
#     )
#     history = cursor.fetchall()
#     cursor.close()
#     connection.close()

#     return jsonify([
#         {'sender': row[1], 'message': row[0]}
#         for row in history
#     ])


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)



























# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import mysql.connector
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from fuzzywuzzy import fuzz
# import subprocess
# from werkzeug.security import generate_password_hash, check_password_hash
# import jwt
# import datetime
# import torch.nn as nn
# import nltk
# from nltk.corpus import words
# import torch.nn.functional as F


# app = Flask(__name__)
# CORS(app)
# SECRET_KEY = 'your_secret_key'
# nltk.download('words')
# # Define Memory LSTM
# class LSTMContextEncoder(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers):
#         super(LSTMContextEncoder, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)

#     def forward(self, inputs, hidden_state=None):
#         outputs, hidden = self.lstm(inputs, hidden_state)
#         logits = self.fc(outputs[:, -1, :])  # Only the last output
#         return logits, hidden

#     def init_hidden(self, batch_size):
#         return (torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size),
#                 torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size))

# gpt_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
# gpt_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")
# gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

# # Load LSTM model
# lstm_model = LSTMContextEncoder(input_size=768, hidden_size=128, num_layers=2)
# lstm_hidden = lstm_model.init_hidden(batch_size=1)


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

# def fetch_faq_data():
#     connection = connect_to_db()
#     if connection is None:
#         return []

#     cursor = connection.cursor()
#     cursor.execute("SELECT question, simple_answer FROM faq_data")
#     faq_data = cursor.fetchall()
#     cursor.close()
#     connection.close()
#     return faq_data

# faq_data = fetch_faq_data()  


# def find_best_faq_match(user_input, faq_data, threshold=85):
#     best_match = None
#     highest_score = 0

#     for question, simple_answer in faq_data:
#         score = fuzz.ratio(user_input.lower(), question.lower())
#         if score > highest_score:
#             highest_score = score
#             best_match = simple_answer
#     return best_match if highest_score >= threshold else None


# def correct_spelling(user_input):
#     return user_input

# def save_message(session_id, user_id, message, sender):
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute(
#         "INSERT INTO conversations (session_id, user_id, message, sender, timestamp) VALUES (%s, %s, %s, %s, NOW())",
#         (session_id, user_id, message, sender)
#     )
#     connection.commit()
#     cursor.close()
#     connection.close()

# def retrieve_conversation_history(session_id, user_id):
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute(
#         "SELECT message FROM conversations WHERE session_id = %s AND user_id = %s ORDER BY timestamp",
#         (session_id, user_id)
#     )
#     history = [row[0] for row in cursor.fetchall()]
#     cursor.close()
#     connection.close()
#     return history

# def retrieve_past_response(session_id, user_id):
#     connection = connect_to_db()
#     if connection is None:
#         return None
#     cursor = connection.cursor()
#     query = """
#     SELECT message FROM conversations
#     WHERE session_id = %s AND user_id = %s AND sender = 'bot'
#     ORDER BY timestamp DESC LIMIT 1
#     """
#     cursor.execute(query, (session_id, user_id))
#     result = cursor.fetchone()
#     cursor.close()
#     connection.close()
#     return result[0] if result else None


# def detect_repeat_request(user_input):
#     repeat_phrases = ["repeat", "again", "say that again", "can you repeat", "can you say that again", "what did you say"]
#     for phrase in repeat_phrases:
#         if fuzz.ratio(user_input.lower(), phrase) > 80:
#             return True
#     return False

# def detect_clarification_request(user_input):
#     clarification_phrases = ["clarify", "more details", "explain further", "could you elaborate", "what do you mean"]
#     for phrase in clarification_phrases:
#         if fuzz.ratio(user_input.lower(), phrase.lower()) > 80:
#             return True
#     return False


# def detect_gibberish(user_input, threshold=0.6):
#     # Check if the message contains recognizable English words
#     words_list = user_input.split()
#     english_words = set(words.words())
#     score = 0
#     for word in words_list:
#         if word.lower() in english_words:
#             score += 1
#     return (score / len(words_list)) < threshold  # Return True if less than the threshold are valid words


# def tokenize_conversation_history(history, tokenizer):
#     tokenized = tokenizer(
#         history, 
#         padding="max_length", 
#         truncation=True, 
#         max_length=150, 
#         return_tensors="pt"
#     )
#     return tokenized.input_ids

# # Generate response
# def generate_response(user_input, session_id, user_id, tokenizer, gpt_model, lstm_model, lstm_hidden):
#     # Check for FAQ match
#     faq_match = find_best_faq_match(user_input, faq_data)
#     if faq_match:
#         return faq_match

#     # Prepare conversation history for LSTM
#     past_responses = retrieve_conversation_history(session_id, user_id)
#     conversation_context = " ".join(past_responses + [user_input])
#     context_tokens = tokenize_conversation_history([conversation_context], tokenizer)

#     # Use LSTM to assess if clarification is needed
#     embeddings = gpt_model.transformer.wte(context_tokens)
#     lstm_output, lstm_hidden = lstm_model(embeddings, lstm_hidden)
#     clarify_needed = torch.sigmoid(lstm_output).item() > 0.5

#     if clarify_needed or detect_clarification_request(user_input):
#         detailed_answer = "Here’s a more detailed explanation: [Add detailed answer based on context or FAQ data]."
#         return detailed_answer

#     # Generate GPT-2 response
#     inputs = tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
#     output = gpt_model.generate(
#         inputs['input_ids'], 
#         max_length=150, 
#         num_return_sequences=1, 
#         pad_token_id=tokenizer.eos_token_id
#     )
#     return tokenizer.decode(output[0], skip_special_tokens=True)

# # @app.route('/chat', methods=['POST'])
# # def chat():
# #     user_input = request.json.get('user_input')
# #     session_id = request.json.get('session_id')
# #     user_id = request.json.get('user_id')

# #     # Save the user's message to the conversation history
# #     save_message(session_id, user_id, user_input, 'user')
    
# #     # Initialize LSTM hidden state with batch dimension
# #     lstm_hidden = memory_lstm.init_hidden(1)  # Batch size 1

# #     # Generate response
# #     response = generate_gpt_response(user_input, gpt_model, gpt_tokenizer, memory_lstm, lstm_hidden, session_id, user_id)
    
# #     # Save the bot's response to the conversation history
# #     save_message(session_id, user_id, response, 'bot')
    
# #     return jsonify({'response': response})



# # #GOOD
# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.json
#     user_input = data.get('user_input')
#     session_id = data.get('session_id')
#     user_id = data.get('user_id')

#     save_message(session_id, user_id, user_input, 'user')
#     response = generate_response(user_input, session_id, user_id, gpt_tokenizer, gpt_model, lstm_model, lstm_hidden)
#     save_message(session_id, user_id, response, 'bot')

#     return jsonify({'response': response})



# # Read admin
# @app.route('/faqs', methods=['GET'])
# def get_faqs():
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT * FROM data")
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
#     cursor.execute("UPDATE data SET category=%s, question=%s, answer=%s WHERE id=%s", 
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
#     cursor.execute("DELETE FROM data WHERE id=%s", (id,))
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
#     cursor.execute("INSERT INTO data (category, question, answer) VALUES (%s, %s, %s)", 
#                    (category, question, answer))
#     connection.commit()
#     cursor.close()
#     connection.close()

#     return jsonify({'message': 'FAQ added successfully'}), 201

# #train button
# @app.route('/train', methods=['POST'])
# def train_data():
#     try:
#         subprocess.run(['python', 'fine_tune.py'], check=True)
#         return jsonify({'message': 'Training started successfully'}), 200
#     except subprocess.CalledProcessError as e:
#         print(f"Error: {e}")
#         return jsonify({'message': 'Failed to start training'}), 500
    


# #Login User
# def create_user(first_name, last_name, email, password, role):
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     hashed_password = generate_password_hash(password)
#     cursor.execute("INSERT INTO users (first_name, last_name, email, password, role) VALUES (%s, %s, %s, %s, %s)", 
#                    (first_name, last_name, email, hashed_password, role))
#     connection.commit()
#     cursor.close()
#     connection.close()
# def authenticate_user(email, password):
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT password FROM users WHERE email = %s", (email,))
#     result = cursor.fetchone()
#     cursor.close()
#     connection.close()
#     if result and check_password_hash(result[0], password):
#         return True
#     return False
# @app.route('/signup', methods=['POST'])
# def signup():
#     data = request.json
#     first_name = data.get('firstName')
#     last_name = data.get('lastName')
#     email = data.get('email')
#     password = data.get('password')
#     role = data.get('role')
#     create_user(first_name, last_name, email, password, role)
#     return jsonify({'success': True, 'message': 'User created successfully'}), 201
# @app.route('/login', methods=['POST'])
# def login():
#     data = request.json
#     email = data.get('email')
#     password = data.get('password')

#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT id, password FROM users WHERE email = %s", (email,))
#     result = cursor.fetchone()
#     cursor.close()
#     connection.close()

#     if result and check_password_hash(result[1], password):
#         token = jwt.encode({
#             'email': email,
#             'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
#         }, SECRET_KEY)
#         return jsonify({'success': True, 'token': token, 'userId': result[0]})  # Return user ID
#     return jsonify({'success': False, 'message': 'Invalid email or password'}), 401




# #admin
# @app.route('/adminsignup', methods=['POST'])
# def admin_signup():
#     data = request.json
#     first_name = data.get('firstName')
#     last_name = data.get('lastName')
#     email = data.get('email')
#     password = data.get('password')
#     department = data.get('department')
#     role = "pending" 

#     connection = connect_to_db()
#     cursor = connection.cursor()
#     hashed_password = generate_password_hash(password)
#     cursor.execute("INSERT INTO admin (first_name, last_name, email, password, role, department) VALUES (%s, %s, %s, %s, %s, %s)", 
#                    (first_name, last_name, email, hashed_password, role, department))
#     connection.commit()
#     cursor.close()
#     connection.close()

#     return jsonify({'success': True, 'message': 'Sign-up request submitted. Awaiting main admin approval.'}), 201

# # Admin 
# @app.route('/adminlogin', methods=['POST'])
# def admin_login():
#     data = request.json
#     email = data.get('email')
#     password = data.get('password')

#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT password, role FROM admin WHERE email = %s", (email,))
#     result = cursor.fetchone()
#     cursor.close()
#     connection.close()

#     if result and check_password_hash(result[0], password):
#         if result[1] == "approved" or result[1] == "main_admin":
#             token = jwt.encode({
#                 'email': email,
#                 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
#             }, SECRET_KEY)
#             return jsonify({'success': True, 'token': token})
#         else:
#             return jsonify({'success': False, 'message': 'Account pending main admin approval.'}), 403
#     return jsonify({'success': False, 'message': 'Invalid email or password'}), 401


# # Approve admin request 
# @app.route('/approve_admin', methods=['POST'])
# def approve_admin():
#     data = request.json
#     email = data.get('email')
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("UPDATE admin SET role = 'approved' WHERE email = %s AND role = 'pending'", (email,))
#     connection.commit()
#     cursor.close()
#     connection.close()

#     return jsonify({'message': 'Admin approved successfully.'})


# @app.route('/pending_admins', methods=['GET'])
# def get_pending_admins():
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT first_name, last_name, email, department FROM admin WHERE role = 'pending'")
#     pending_admins = cursor.fetchall()
#     cursor.close()
#     connection.close()

#     return jsonify([{'firstName': admin[0], 'lastName': admin[1], 'email': admin[2], 'department': admin[3]} for admin in pending_admins])

# @app.route('/decline_admin', methods=['POST'])
# def decline_admin():
#     data = request.json
#     email = data.get('email')
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("DELETE FROM admin WHERE email = %s AND role = 'pending'", (email,))
#     connection.commit()
#     cursor.close()
#     connection.close()

#     return jsonify({'message': 'Admin request declined successfully.'})


# @app.route('/conversation_history', methods=['GET'])
# def get_conversation_history():
#     session_id = request.args.get('session_id')
#     user_id = request.args.get('user_id')

#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute(
#         "SELECT message, sender FROM conversations WHERE session_id = %s AND user_id = %s ORDER BY timestamp",
#         (session_id, user_id)
#     )
#     history = cursor.fetchall()
#     cursor.close()
#     connection.close()

#     return jsonify([
#         {'sender': row[1], 'message': row[0]}
#         for row in history
#     ])


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)



















# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import mysql.connector
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from fuzzywuzzy import fuzz
# import subprocess
# from werkzeug.security import generate_password_hash, check_password_hash
# import jwt
# import datetime
# import torch.nn as nn

# app = Flask(__name__)
# CORS(app)
# SECRET_KEY = 'your_secret_key'

# class MemoryLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(MemoryLSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x, hidden):
#         out, hidden = self.lstm(x, hidden)
#         out = self.fc(out[:, -1, :])
#         return out, hidden

#     def init_hidden(self, batch_size):
#         return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
#                 torch.zeros(self.num_layers, batch_size, self.hidden_size))


# gpt_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
# gpt_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")
# memory_lstm = MemoryLSTM(input_size=768, hidden_size=128, num_layers=2, output_size=1)
# memory_lstm.load_state_dict(torch.load("lstm_weights.pth"))
# memory_lstm.eval()


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

# def fetch_faq_data():
#     connection = connect_to_db()
#     if connection is None:
#         return []

#     cursor = connection.cursor()
#     cursor.execute("SELECT question, answer FROM data")
#     faq_data = cursor.fetchall()
#     cursor.close()
#     connection.close()
#     return faq_data

# faq_data = fetch_faq_data()  
# def find_best_faq_match(user_input, faq_data, threshold=80):
#     best_match = None
#     highest_score = 0

#     for question, answer in faq_data:
#         score = fuzz.ratio(user_input.lower(), question.lower())
#         if score > highest_score:
#             highest_score = score
#             best_match = answer
#     return best_match if highest_score >= threshold else None

# def correct_spelling(user_input):
#     return user_input

# def save_message(session_id, user_id, message, sender):
#     connection = connect_to_db()
#     if connection is None:
#         return
#     cursor = connection.cursor()
#     cursor.execute(
#         "INSERT INTO conversations (session_id, user_id, message, sender, timestamp) VALUES (%s, %s, %s, %s, NOW())",
#         (session_id, user_id, message, sender)
#     )
#     connection.commit()
#     cursor.close()
#     connection.close()

# def retrieve_past_response(session_id, user_id):
#     connection = connect_to_db()
#     if connection is None:
#         return None
#     cursor = connection.cursor()
#     query = """
#     SELECT message FROM conversations
#     WHERE session_id = %s AND user_id = %s AND sender = 'bot'
#     ORDER BY timestamp DESC LIMIT 1
#     """
#     cursor.execute(query, (session_id, user_id))
#     result = cursor.fetchone()
#     cursor.close()
#     connection.close()
#     return result[0] if result else None


# def detect_repeat_request(user_input):
#     repeat_phrases = ["repeat", "again", "say that again", "can you repeat", "can you say that again", "what did you say"]
#     for phrase in repeat_phrases:
#         if fuzz.ratio(user_input.lower(), phrase) > 80:
#             return True
#     return False


# def retrieve_last_bot_response(session_id, user_id):
#     connection = connect_to_db()
#     if connection is None:
#         return None
#     cursor = connection.cursor()
#     cursor.execute(
#         "SELECT message FROM conversations WHERE session_id = %s AND user_id = %s AND sender = 'bot' ORDER BY timestamp DESC LIMIT 1",
#         (session_id, user_id)
#     )
#     result = cursor.fetchone()
#     cursor.close()
#     connection.close()
#     return result[0] if result else None


# def detect_clarification_request(lstm_model, embeddings, lstm_hidden):
#     lstm_output, lstm_hidden = lstm_model(embeddings, lstm_hidden)
#     clarify_needed = (torch.sigmoid(lstm_output) > 0.5).item()
#     return clarify_needed, lstm_hidden


# def generate_gpt_response(user_input, gpt_model, gpt_tokenizer, lstm_model, lstm_hidden, session_id, user_id):
#     # Check if the user is asking for a repetition
#     if detect_repeat_request(user_input):
#         past_response = retrieve_past_response(session_id, user_id)
#         return past_response if past_response else "I'm sorry, I don't have a recent response to repeat."

#     # Tokenize and get embeddings for the user input
#     inputs = gpt_tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
#     input_ids = inputs['input_ids']
#     embeddings = gpt_model.transformer.wte(input_ids)  # Get embeddings

#     # Use LSTM model to check if further clarification is needed
#     lstm_output, lstm_hidden = lstm_model(embeddings, lstm_hidden)
#     clarify_needed = (torch.sigmoid(lstm_output) > 0.5).item()

#     # If clarification is detected, retrieve the past response
#     if clarify_needed:
#         past_response = retrieve_past_response(session_id, user_id)
#         return past_response if past_response else "I'm sorry, I don't have a recent response to clarify."

#     # Otherwise, generate a new response with GPT-2
#     chat_history_ids = gpt_model.generate(input_ids, max_length=150, num_return_sequences=1)
#     return gpt_tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json.get('user_input')
#     session_id = request.json.get('session_id')
#     user_id = request.json.get('user_id')

#     # Save the user's message to the conversation history
#     save_message(session_id, user_id, user_input, 'user')
    
#     # Initialize LSTM hidden state with batch dimension
#     lstm_hidden = memory_lstm.init_hidden(1)  # Batch size 1

#     # Generate response
#     response = generate_gpt_response(user_input, gpt_model, gpt_tokenizer, memory_lstm, lstm_hidden, session_id, user_id)
    
#     # Save the bot's response to the conversation history
#     save_message(session_id, user_id, response, 'bot')
    
#     return jsonify({'response': response})



# # Read admin
# @app.route('/faqs', methods=['GET'])
# def get_faqs():
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT * FROM data")
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
#     cursor.execute("UPDATE data SET category=%s, question=%s, answer=%s WHERE id=%s", 
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
#     cursor.execute("DELETE FROM data WHERE id=%s", (id,))
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
#     cursor.execute("INSERT INTO data (category, question, answer) VALUES (%s, %s, %s)", 
#                    (category, question, answer))
#     connection.commit()
#     cursor.close()
#     connection.close()

#     return jsonify({'message': 'FAQ added successfully'}), 201

# #train button
# @app.route('/train', methods=['POST'])
# def train_data():
#     try:
#         subprocess.run(['python', 'fine_tune.py'], check=True)
#         return jsonify({'message': 'Training started successfully'}), 200
#     except subprocess.CalledProcessError as e:
#         print(f"Error: {e}")
#         return jsonify({'message': 'Failed to start training'}), 500
    


# #Login User
# def create_user(first_name, last_name, email, password, role):
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     hashed_password = generate_password_hash(password)
#     cursor.execute("INSERT INTO users (first_name, last_name, email, password, role) VALUES (%s, %s, %s, %s, %s)", 
#                    (first_name, last_name, email, hashed_password, role))
#     connection.commit()
#     cursor.close()
#     connection.close()
# def authenticate_user(email, password):
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT password FROM users WHERE email = %s", (email,))
#     result = cursor.fetchone()
#     cursor.close()
#     connection.close()
#     if result and check_password_hash(result[0], password):
#         return True
#     return False
# @app.route('/signup', methods=['POST'])
# def signup():
#     data = request.json
#     first_name = data.get('firstName')
#     last_name = data.get('lastName')
#     email = data.get('email')
#     password = data.get('password')
#     role = data.get('role')
#     create_user(first_name, last_name, email, password, role)
#     return jsonify({'success': True, 'message': 'User created successfully'}), 201
# @app.route('/login', methods=['POST'])
# def login():
#     data = request.json
#     email = data.get('email')
#     password = data.get('password')

#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT id, password FROM users WHERE email = %s", (email,))
#     result = cursor.fetchone()
#     cursor.close()
#     connection.close()

#     if result and check_password_hash(result[1], password):
#         token = jwt.encode({
#             'email': email,
#             'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
#         }, SECRET_KEY)
#         return jsonify({'success': True, 'token': token, 'userId': result[0]})  # Return user ID
#     return jsonify({'success': False, 'message': 'Invalid email or password'}), 401




# #admin
# @app.route('/adminsignup', methods=['POST'])
# def admin_signup():
#     data = request.json
#     first_name = data.get('firstName')
#     last_name = data.get('lastName')
#     email = data.get('email')
#     password = data.get('password')
#     department = data.get('department')
#     role = "pending" 

#     connection = connect_to_db()
#     cursor = connection.cursor()
#     hashed_password = generate_password_hash(password)
#     cursor.execute("INSERT INTO admin (first_name, last_name, email, password, role, department) VALUES (%s, %s, %s, %s, %s, %s)", 
#                    (first_name, last_name, email, hashed_password, role, department))
#     connection.commit()
#     cursor.close()
#     connection.close()

#     return jsonify({'success': True, 'message': 'Sign-up request submitted. Awaiting main admin approval.'}), 201

# # Admin 
# @app.route('/adminlogin', methods=['POST'])
# def admin_login():
#     data = request.json
#     email = data.get('email')
#     password = data.get('password')

#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT password, role FROM admin WHERE email = %s", (email,))
#     result = cursor.fetchone()
#     cursor.close()
#     connection.close()

#     if result and check_password_hash(result[0], password):
#         if result[1] == "approved" or result[1] == "main_admin":
#             token = jwt.encode({
#                 'email': email,
#                 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
#             }, SECRET_KEY)
#             return jsonify({'success': True, 'token': token})
#         else:
#             return jsonify({'success': False, 'message': 'Account pending main admin approval.'}), 403
#     return jsonify({'success': False, 'message': 'Invalid email or password'}), 401


# # Approve admin request 
# @app.route('/approve_admin', methods=['POST'])
# def approve_admin():
#     data = request.json
#     email = data.get('email')
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("UPDATE admin SET role = 'approved' WHERE email = %s AND role = 'pending'", (email,))
#     connection.commit()
#     cursor.close()
#     connection.close()

#     return jsonify({'message': 'Admin approved successfully.'})


# @app.route('/pending_admins', methods=['GET'])
# def get_pending_admins():
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT first_name, last_name, email, department FROM admin WHERE role = 'pending'")
#     pending_admins = cursor.fetchall()
#     cursor.close()
#     connection.close()

#     return jsonify([{'firstName': admin[0], 'lastName': admin[1], 'email': admin[2], 'department': admin[3]} for admin in pending_admins])

# @app.route('/decline_admin', methods=['POST'])
# def decline_admin():
#     data = request.json
#     email = data.get('email')
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("DELETE FROM admin WHERE email = %s AND role = 'pending'", (email,))
#     connection.commit()
#     cursor.close()
#     connection.close()

#     return jsonify({'message': 'Admin request declined successfully.'})


# @app.route('/conversation_history', methods=['GET'])
# def get_conversation_history():
#     session_id = request.args.get('session_id')
#     user_id = request.args.get('user_id')

#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute(
#         "SELECT message, sender FROM conversations WHERE session_id = %s AND user_id = %s ORDER BY timestamp",
#         (session_id, user_id)
#     )
#     history = cursor.fetchall()
#     cursor.close()
#     connection.close()

#     return jsonify([
#         {'sender': row[1], 'message': row[0]}
#         for row in history
#     ])


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)












from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from fuzzywuzzy import fuzz
import subprocess
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
import torch.nn as nn
import logging


app = Flask(__name__)
CORS(app)
SECRET_KEY = 'your_secret_key'

class MemoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MemoryLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


gpt_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")
memory_lstm = MemoryLSTM(input_size=768, hidden_size=128, num_layers=2, output_size=1)
memory_lstm.load_state_dict(torch.load("lstm_weights.pth"))
memory_lstm.eval()


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

def fetch_faq_data():
    connection = connect_to_db()
    if connection is None:
        return []

    cursor = connection.cursor()
    cursor.execute("SELECT question, answer FROM data")
    faq_data = cursor.fetchall()
    cursor.close()
    connection.close()
    return faq_data

faq_data = fetch_faq_data()  
def find_best_faq_match(user_input, faq_data, threshold=80):
    best_match = None
    highest_score = 0

    for question, answer in faq_data:
        score = fuzz.ratio(user_input.lower(), question.lower())
        if score > highest_score:
            highest_score = score
            best_match = answer
    return best_match if highest_score >= threshold else None

def correct_spelling(user_input):
    return user_input

def save_message(session_id, user_id, message, sender):
    connection = connect_to_db()
    if connection is None:
        return
    cursor = connection.cursor()
    cursor.execute(
        "INSERT INTO conversations (session_id, user_id, message, sender, timestamp) VALUES (%s, %s, %s, %s, NOW())",
        (session_id, user_id, message, sender)
    )
    connection.commit()
    cursor.close()
    connection.close()

def retrieve_past_response(session_id, user_id):
    connection = connect_to_db()
    if connection is None:
        return None
    cursor = connection.cursor()
    query = """
    SELECT message FROM conversations
    WHERE session_id = %s AND user_id = %s AND sender = 'bot'
    ORDER BY timestamp DESC LIMIT 1
    """
    cursor.execute(query, (session_id, user_id))
    result = cursor.fetchone()
    cursor.close()
    connection.close()
    return result[0] if result else None


def detect_repeat_request(user_input):
    repeat_phrases = ["repeat", "again", "say that again", "can you repeat", "can you say that again", "what did you say"]
    for phrase in repeat_phrases:
        if fuzz.ratio(user_input.lower(), phrase) > 80:
            return True
    return False


# def retrieve_detailed_response(session_id, user_id):
#     connection = connect_to_db()
#     if connection is None:
#         return None
#     cursor = connection.cursor()
#     cursor.execute(
#         "SELECT message FROM conversations WHERE session_id = %s AND user_id = %s AND sender = 'bot' ORDER BY timestamp DESC LIMIT 1",
#         (session_id, user_id)
#     )
#     result = cursor.fetchone()
#     cursor.close()
#     connection.close()
#     return result[0] if result else None

def retrieve_detailed_response(session_id, user_id, detailed_answer):
    """
    Retrieve the detailed answer from the 'faq_data' table that matches the bot's most recent response.
    """
    connection = connect_to_db()
    if connection is None:
        return None
    
    try:
        cursor = connection.cursor()
        
        # Query to get the latest bot message and match it with 'faq_data.detailed_answer'
        query = """
        SELECT faq_data.detailed_answer
        FROM conversations
        JOIN faq_data ON conversations.message = faq_data.detailed_answer
        WHERE conversations.session_id = %s AND conversations.user_id = %s AND conversations.sender = 'bot'
        ORDER BY conversations.timestamp DESC
        LIMIT 1
        """
        
        cursor.execute(query, (session_id, user_id, detailed_answer))
        result = cursor.fetchone()
        
    except mysql.connector.Error as e:
        logging.error("Error retrieving detailed response: %s", e)
        result = None

    finally:
        cursor.close()
        connection.close()
    
    return result[0] if result else None



def detect_clarification_request(lstm_model, embeddings, lstm_hidden):
    lstm_output, lstm_hidden = lstm_model(embeddings, lstm_hidden)
    clarify_needed = (torch.sigmoid(lstm_output) > 0.5).item()
    return clarify_needed, lstm_hidden


def generate_gpt_response(user_input, gpt_model, gpt_tokenizer, lstm_model, lstm_hidden, session_id, user_id):
    if detect_repeat_request(user_input):
        past_response = retrieve_past_response(session_id, user_id)
        return past_response if past_response else "I'm sorry, I don't have a recent response to repeat."

    inputs = gpt_tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids']
    embeddings = gpt_model.transformer.wte(input_ids)  

    lstm_output, lstm_hidden = lstm_model(embeddings, lstm_hidden)
    clarify_needed = (torch.sigmoid(lstm_output) > 0.5).item()
    if clarify_needed:
        past_response = retrieve_past_response(session_id, user_id)
        return past_response if past_response else "I'm sorry, I don't have a clarify."

    chat_history_ids = gpt_model.generate(input_ids, max_length=150, num_return_sequences=1)
    return gpt_tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('user_input')
    session_id = request.json.get('session_id')
    user_id = request.json.get('user_id')
    save_message(session_id, user_id, user_input, 'user')
    lstm_hidden = memory_lstm.init_hidden(1)  
    response = generate_gpt_response(user_input, gpt_model, gpt_tokenizer, memory_lstm, lstm_hidden, session_id, user_id)
    save_message(session_id, user_id, response, 'bot')
    return jsonify({'response': response})



# Read admin
@app.route('/faqs', methods=['GET'])
def get_faqs():
    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM data")
    faq_data = cursor.fetchall()
    cursor.close()
    connection.close()
    
    faqs = [
        {'id': row[0], 'category': row[1], 'question': row[2], 'answer': row[3]} 
        for row in faq_data
    ]
    
    return jsonify(faqs)


# Edit admin
@app.route('/faqs/<int:id>', methods=['PUT'])
def edit_faq(id):
    data = request.json
    category = data.get('category')
    question = data.get('question')
    answer = data.get('answer')

    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute("UPDATE data SET category=%s, question=%s, answer=%s WHERE id=%s", 
                   (category, question, answer, id))
    connection.commit()
    cursor.close()
    connection.close()
    
    return jsonify({'message': 'FAQ updated successfully'})


# Delete admin
@app.route('/faqs/<int:id>', methods=['DELETE'])
def delete_faq(id):
    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute("DELETE FROM data WHERE id=%s", (id,))
    connection.commit()
    cursor.close()
    connection.close()
    
    return jsonify({'message': 'FAQ deleted successfully'})


# Add admin
@app.route('/faqs', methods=['POST'])
def add_faq():
    data = request.json
    category = data.get('category')
    question = data.get('question')
    answer = data.get('answer')

    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute("INSERT INTO data (category, question, answer) VALUES (%s, %s, %s)", 
                   (category, question, answer))
    connection.commit()
    cursor.close()
    connection.close()

    return jsonify({'message': 'FAQ added successfully'}), 201

#train button
@app.route('/train', methods=['POST'])
def train_data():
    try:
        subprocess.run(['python', 'fine_tune.py'], check=True)
        return jsonify({'message': 'Training started successfully'}), 200
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return jsonify({'message': 'Failed to start training'}), 500
    


#Login User
def create_user(first_name, last_name, email, password, role):
    connection = connect_to_db()
    cursor = connection.cursor()
    hashed_password = generate_password_hash(password)
    cursor.execute("INSERT INTO users (first_name, last_name, email, password, role) VALUES (%s, %s, %s, %s, %s)", 
                   (first_name, last_name, email, hashed_password, role))
    connection.commit()
    cursor.close()
    connection.close()
def authenticate_user(email, password):
    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute("SELECT password FROM users WHERE email = %s", (email,))
    result = cursor.fetchone()
    cursor.close()
    connection.close()
    if result and check_password_hash(result[0], password):
        return True
    return False
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    first_name = data.get('firstName')
    last_name = data.get('lastName')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role')
    create_user(first_name, last_name, email, password, role)
    return jsonify({'success': True, 'message': 'User created successfully'}), 201
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute("SELECT id, password FROM users WHERE email = %s", (email,))
    result = cursor.fetchone()
    cursor.close()
    connection.close()

    if result and check_password_hash(result[1], password):
        token = jwt.encode({
            'email': email,
            'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
        }, SECRET_KEY)
        return jsonify({'success': True, 'token': token, 'userId': result[0]})  # Return user ID
    return jsonify({'success': False, 'message': 'Invalid email or password'}), 401




#admin
@app.route('/adminsignup', methods=['POST'])
def admin_signup():
    data = request.json
    first_name = data.get('firstName')
    last_name = data.get('lastName')
    email = data.get('email')
    password = data.get('password')
    department = data.get('department')
    role = "pending" 

    connection = connect_to_db()
    cursor = connection.cursor()
    hashed_password = generate_password_hash(password)
    cursor.execute("INSERT INTO admin (first_name, last_name, email, password, role, department) VALUES (%s, %s, %s, %s, %s, %s)", 
                   (first_name, last_name, email, hashed_password, role, department))
    connection.commit()
    cursor.close()
    connection.close()

    return jsonify({'success': True, 'message': 'Sign-up request submitted. Awaiting main admin approval.'}), 201

# Admin 
@app.route('/adminlogin', methods=['POST'])
def admin_login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute("SELECT password, role FROM admin WHERE email = %s", (email,))
    result = cursor.fetchone()
    cursor.close()
    connection.close()

    if result and check_password_hash(result[0], password):
        if result[1] == "approved" or result[1] == "main_admin":
            token = jwt.encode({
                'email': email,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
            }, SECRET_KEY)
            return jsonify({'success': True, 'token': token})
        else:
            return jsonify({'success': False, 'message': 'Account pending main admin approval.'}), 403
    return jsonify({'success': False, 'message': 'Invalid email or password'}), 401


# Approve admin request 
@app.route('/approve_admin', methods=['POST'])
def approve_admin():
    data = request.json
    email = data.get('email')
    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute("UPDATE admin SET role = 'approved' WHERE email = %s AND role = 'pending'", (email,))
    connection.commit()
    cursor.close()
    connection.close()

    return jsonify({'message': 'Admin approved successfully.'})


@app.route('/pending_admins', methods=['GET'])
def get_pending_admins():
    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute("SELECT first_name, last_name, email, department FROM admin WHERE role = 'pending'")
    pending_admins = cursor.fetchall()
    cursor.close()
    connection.close()

    return jsonify([{'firstName': admin[0], 'lastName': admin[1], 'email': admin[2], 'department': admin[3]} for admin in pending_admins])

@app.route('/decline_admin', methods=['POST'])
def decline_admin():
    data = request.json
    email = data.get('email')
    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute("DELETE FROM admin WHERE email = %s AND role = 'pending'", (email,))
    connection.commit()
    cursor.close()
    connection.close()

    return jsonify({'message': 'Admin request declined successfully.'})


@app.route('/conversation_history', methods=['GET'])
def get_conversation_history():
    session_id = request.args.get('session_id')
    user_id = request.args.get('user_id')

    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute(
        "SELECT message, sender FROM conversations WHERE session_id = %s AND user_id = %s ORDER BY timestamp",
        (session_id, user_id)
    )
    history = cursor.fetchall()
    cursor.close()
    connection.close()

    return jsonify([
        {'sender': row[1], 'message': row[0]}
        for row in history
    ])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
