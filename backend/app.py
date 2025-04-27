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


# def generate_gpt_response(user_input, gpt_model, gpt_tokenizer, lstm_model, lstm_hidden, session_id, user_id):
#     # Check if the user is asking for a repetition
#     if detect_repeat_request(user_input):
#         # Retrieve the last response directly without generating a new one
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
# import logging
# import nltk
# from nltk.corpus import words
# from langdetect import detect, LangDetectException

# app = Flask(__name__)
# CORS(app)
# SECRET_KEY = 'your_secret_key'
# nltk.download('words')

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
#     cursor.execute("SELECT question, answer, category, detailed_answer FROM faq_data")
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

# def retrieve_detailed_response(session_id, user_id):
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
#     last_bot_message = cursor.fetchone()
#     cursor.close()
#     connection.close()

#     if not last_bot_message:
#         return None

#     last_bot_message = last_bot_message[0]
#     best_match = None
#     highest_score = 0

#     for _, answer, _, detailed_answer in faq_data:
#         score = fuzz.ratio(last_bot_message.lower(), answer.lower())
#         logging.info(f"Matching '{last_bot_message}' with '{answer}': Score {score}")
#         if score > highest_score:
#             highest_score = score
#             best_match = detailed_answer

#     threshold = 75 
#     if highest_score >= threshold:
#         return best_match if best_match else "I don't have additional details at the moment. Let me know if there's something specific you'd like to clarify."
#     return None



# def retrieve_category_response(session_id, user_id):
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
#     last_bot_message = cursor.fetchone()
#     cursor.close()
#     connection.close()

#     if not last_bot_message:
#         return None

#     last_bot_message = last_bot_message[0]
#     best_match = None
#     highest_score = 0

#     for _, answer, category, _ in faq_data:
#         score = fuzz.ratio(last_bot_message.lower(), answer.lower())
#         logging.info(f"Matching '{last_bot_message}' with '{answer}': Score {score}")
#         if score > highest_score:
#             highest_score = score
#             best_match = category

#     threshold = 75 
#     if highest_score >= threshold:
#         return best_match if best_match else "I don't have a category match at the moment. Let me know if there's something specific you'd like to clarify."
#     return None



# def detect_clarification_request(lstm_model, embeddings, lstm_hidden):
#     lstm_output, lstm_hidden = lstm_model(embeddings, lstm_hidden)
#     clarify_needed = (torch.sigmoid(lstm_output) > 0.6).item()
#     return clarify_needed, lstm_hidden



# # def generate_gpt_response(user_input, gpt_model, gpt_tokenizer, lstm_model, lstm_hidden, session_id, user_id):

# #     inputs = gpt_tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
# #     input_ids = inputs['input_ids']
# #     embeddings = gpt_model.transformer.wte(input_ids)  

# #     lstm_output, lstm_hidden = lstm_model(embeddings, lstm_hidden)
# #     clarify_needed = (torch.sigmoid(lstm_output) > 0.5).item()

# #     if clarify_needed:
# #         category_response = retrieve_category_response(session_id, user_id)
# #         detailed_response = retrieve_detailed_response(session_id, user_id)
# #         chat_history_ids = gpt_model.generate(input_ids, max_length=150, num_return_sequences=1)
# #         if detailed_response and category_response:
# #             return f"Here is the detailed answer about your question {category_response}. {detailed_response}"
# #         else:
# #             return gpt_tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)



# def detect_gibberish(user_input, threshold=0.6):
#     # Check if the message contains recognizable English words
#     words_list = user_input.split()
#     english_words = set(words.words())
#     score = 0
#     for word in words_list:
#         if word.lower() in english_words:
#             score += 1
#     return (score / len(words_list)) < threshold  


# BAD_WORDS = ["bobo", "bwesit", "Not good"]  
# def contains_bad_words(user_input):
#     user_words = user_input.lower().split() 
#     for word in user_words:
#         if word in BAD_WORDS:
#             return True
#     return False


# GOOD_WORDS = ["Thank you", "Thanks", "Thank You so much"]  
# def contains_good_words(user_input):
#     user_input = user_input.lower()
#     for phrase in GOOD_WORDS:
#         if phrase.lower() in user_input:  
#             return True
#     return False

# PRAISE_WORDS = ["Helpful", "Nice", "Good"]  
# def contains_praise_words(user_input):
#     user_input = user_input.lower()
#     for phrase in PRAISE_WORDS:
#         if phrase.lower() in user_input:  
#             return True
#     return False





# def handle_special_cases(user_input):
#     """Handle special cases like gibberish or bad/good words."""
#     if contains_good_words(user_input):
#         return "You're very welcome! I'm glad I could help😊. If you have any more questions or need further assistance, feel free to ask."
#     if detect_gibberish(user_input):
#         return "I'm sorry, but I couldn't understand your message😕. Could you please clarify or rephrase your question?💡"
#     if contains_bad_words(user_input):
#         return "I'm sorry if my response wasn't helpful😔. If you need a better solution or clarification, please let me know how I can assist you!📚"
#     if contains_praise_words(user_input):
#         return "I'm glad you liked it! 😊 If you need anything else or have more questions, feel free to ask. I'm here to help! 💡📚"
#     return None


# def generate_gpt_response(
#     user_input, gpt_model, gpt_tokenizer, lstm_model, lstm_hidden,
#     session_id, user_id, confidence_threshold=0.55
# ):
#     try:
#         detected_language = detect(user_input)
#     except LangDetectException:
#         detected_language = 'unknown'

#     # If the language is Tagalog, ask the user to rephrase in English
#     if detected_language == 'tl':  # 'tl' is the language code for Tagalog
#         return "Sorry, I only understand English. Can you please rephrase your question in English?"

#     # Detect gibberish input
#     special_case_response = handle_special_cases(user_input)
#     if special_case_response:
#         return special_case_response

#     inputs = gpt_tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
#     input_ids = inputs['input_ids']
#     attention_mask = inputs['attention_mask']

#     # Generate embeddings for LSTM
#     embeddings = gpt_model.transformer.wte(input_ids)
#     lstm_output, lstm_hidden = lstm_model(embeddings, lstm_hidden)

#     # Check if clarification is needed
#     clarify_needed = (torch.sigmoid(lstm_output) > 0.6).item()
#     if clarify_needed:
#         category_response = retrieve_category_response(session_id, user_id)
#         detailed_response = retrieve_detailed_response(session_id, user_id)
#         if category_response and detailed_response:
#             return f"Here is the detailed answer about your question {category_response}. {detailed_response}"

#     # Generate response with output scores
#     chat_history_ids = gpt_model.generate(
#         input_ids,
#         attention_mask=attention_mask,
#         max_length=150,
#         num_return_sequences=1,
#         pad_token_id=gpt_tokenizer.eos_token_id,
#         output_scores=True,
#         return_dict_in_generate=True
#     )

#     # Extract generated text and scores
#     sequences = chat_history_ids['sequences']
#     token_scores = chat_history_ids['scores']

#     # Compute confidence
#     if token_scores:
#         total_log_prob = sum(torch.log_softmax(score, dim=-1).max().item() for score in token_scores)
#         avg_log_prob = total_log_prob / len(token_scores)
#         confidence = torch.exp(torch.tensor(avg_log_prob)).item()

#         if confidence < confidence_threshold:
#             return (
#                 "Sorry, I can't answer that. Could you please clarify or rephrase your question? "
#                 "I can only answer questions related to Palawan State University and the College of Science."
#             )

#     # Decode and return response
#     return gpt_tokenizer.decode(sequences[0], skip_special_tokens=True)



# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json.get('user_input')
#     session_id = request.json.get('session_id')
#     user_id = request.json.get('user_id')

#     save_message(session_id, user_id, user_input, 'user')

#     lstm_hidden = memory_lstm.init_hidden(1) 

#     response = generate_gpt_response(user_input, gpt_model, gpt_tokenizer, memory_lstm, lstm_hidden, session_id, user_id)

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















# def retrieve_detailed_response(session_id, user_id):
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
#     last_bot_message = cursor.fetchone()
#     cursor.close()
#     connection.close()

#     if not last_bot_message:
#         return None

#     last_bot_message = last_bot_message[0]
#     best_match = None
#     highest_score = 0

#     for _, answer, _, detailed_answer in faq_data:
#         score = fuzz.ratio(last_bot_message.lower(), answer.lower())
#         logging.info(f"Matching '{last_bot_message}' with '{answer}': Score {score}")
#         if score > highest_score:
#             highest_score = score
#             best_match = detailed_answer

#     threshold = 75 
#     if highest_score >= threshold:
#         return best_match if best_match else "I don't have additional details at the moment. Let me know if there's something specific you'd like to clarify."
#     return None



# def retrieve_category_response(session_id, user_id):
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
#     last_bot_message = cursor.fetchone()
#     cursor.close()
#     connection.close()

#     if not last_bot_message:
#         return None

#     last_bot_message = last_bot_message[0]
#     best_match = None
#     highest_score = 0

#     for _, answer, category, _ in faq_data:
#         score = fuzz.ratio(last_bot_message.lower(), answer.lower())
#         logging.info(f"Matching '{last_bot_message}' with '{answer}': Score {score}")
#         if score > highest_score:
#             highest_score = score
#             best_match = category

#     threshold = 75 
#     if highest_score >= threshold:
#         return best_match if best_match else "I don't have a category match at the moment. Let me know if there's something specific you'd like to clarify."
#     return None





# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import mysql.connector
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from fuzzywuzzy import fuzz
# from werkzeug.security import generate_password_hash, check_password_hash
# import jwt
# import datetime
# import torch.nn as nn
# import logging
# import nltk
# from nltk.corpus import words
# from langdetect import detect, LangDetectException
# import subprocess


# app = Flask(__name__)
# CORS(app)
# SECRET_KEY = 'your_secret_key'
# nltk.download('words')
# session_states = {}


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
# checkpoint = torch.load("lstm_weights.pth", weights_only=True)

# memory_lstm = MemoryLSTM(
#     input_size=checkpoint['config']['input_size'],
#     hidden_size=checkpoint['config']['hidden_size'],
#     num_layers=checkpoint['config']['num_layers'],
#     output_size=checkpoint['config']['output_size']
# )

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
#     cursor.execute("SELECT category, question, answer FROM faqq")
#     faq_data = cursor.fetchall()
#     cursor.close()
#     connection.close()
#     return faq_data
# faq_data = fetch_faq_data()


# def save_message(session_id, user_id, message, sender):
#     connection = connect_to_db()
#     if connection is None:
#         app.logger.error("Failed to connect to the database.")
#         return False
#     try:
#         cursor = connection.cursor()
#         query = """
#             INSERT INTO conversations (session_id, user_id, message, sender, timestamp)
#             VALUES (%s, %s, %s, %s, NOW())
#         """
#         cursor.execute(query, (session_id, user_id, message, sender))
#         connection.commit()
#         cursor.close()
#         return True
#     except mysql.connector.Error as err:
#         app.logger.error(f"Error saving message: {err}")
#         return False
#     finally:
#         connection.close()


# def retrieve_detailed_response(session_id, user_id):
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
#     last_bot_message = cursor.fetchone()
#     cursor.close()
#     connection.close()

#     if not last_bot_message:
#         return None

#     last_bot_message = last_bot_message[0]
#     best_match = None
#     highest_score = 0

#     for _, answer in faq_data:
#         score = fuzz.ratio(last_bot_message.lower(), answer.lower())
#         logging.info(f"Matching '{last_bot_message}' with '{answer}': Score {score}")
#         if score > highest_score:
#             highest_score = score
#             best_match = answer

#     threshold = 70 
#     if highest_score >= threshold:
#         return best_match if best_match else "I don't have additional details at the moment. Let me know if there's something specific you'd like to clarify."
#     return None



# def retrieve_category_response(session_id, user_id):
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
#     last_bot_message = cursor.fetchone()
#     cursor.close()
#     connection.close()

#     if not last_bot_message:
#         return None

#     last_bot_message = last_bot_message[0]
#     best_match = None
#     highest_score = 0

#     for category, _, answer, _ in faq_data:
#         score = fuzz.ratio(last_bot_message.lower(), answer.lower())
#         logging.info(f"Matching '{last_bot_message}' with '{answer}': Score {score}")
#         if score > highest_score:
#             highest_score = score
#             best_match = category

#     threshold = 70 
#     if highest_score >= threshold:
#         return best_match if best_match else "I don't have a category match at the moment. Let me know if there's something specific you'd like to clarify."
#     return None


# def generate_response(user_input, session_hidden, session_id, user_id, faq_data):

#     special_case_response = handle_special_cases(user_input, session_id, user_id)
#     if special_case_response:
#         return special_case_response, session_hidden
#     inputs = gpt_tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
#     input_ids = inputs['input_ids']
#     embeddings = gpt_model.transformer.wte(input_ids)
#     lstm_output, session_hidden = memory_lstm(embeddings, session_hidden)
#     clarification_needed = (torch.sigmoid(lstm_output) > 0.7).item()
        
#     if clarification_needed:
#         return "Could you please clarify your question?" , session_hidden

#     chat_history_ids = gpt_model.generate(input_ids, max_length=150, pad_token_id=gpt_tokenizer.eos_token_id)
#     response = gpt_tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)
#     return response, session_hidden


# @app.route('/chat', methods=['POST'])
# def chat():
#     try:
#         user_input = request.json.get('user_input')
#         session_id = request.json.get('session_id')
#         user_id = request.json.get('user_id')  

#         if not user_input or not session_id:
#             return jsonify({'error': 'Invalid input'}), 400

#         if session_id not in session_states:
#             session_states[session_id] = memory_lstm.init_hidden(1)

#         session_hidden = session_states[session_id]
#         response, session_hidden = generate_response(user_input, session_hidden, session_id, user_id, faq_data)
#         session_states[session_id] = session_hidden

#         save_message(session_id, user_id, user_input, "user")
#         save_message(session_id, user_id, response, "bot")

#         return jsonify({'response': response})
#     except Exception as e:
#         app.logger.error(f"Error in /chat: {e}")
#         return jsonify({'error': 'An error occurred while processing your request.'}), 500


# # def detect_gibberish(user_input, threshold=0.7):
# #     words_list = [word for word in user_input.split() if word.strip()]
# #     if not words_list:
# #         return True
# #     english_words = set(words.words())
# #     valid_word_count = sum(1 for word in words_list if word.lower() in english_words)
# #     ratio = valid_word_count / len(words_list)
# #     return ratio < threshold 



# repeat_phrases = ["repeat", "again", "say that again", "can you repeat", "can you say that again", "what did you say"]
# def detect_clarify_request(user_input):
#     for phrase in repeat_phrases:
#         match_score = fuzz.partial_ratio(user_input.lower(), phrase.lower())
#         if match_score > 75:  
#             return True
#     return False

# def handle_special_cases(user_input):
#     bad_words = ["bobo", "bwesit"]
#     good_words = ["thank you", "thanks"]

#     if any(bad_word in user_input.lower() for bad_word in bad_words):
#         return "Please refrain from using offensive language."
#     if any(good_word in user_input.lower() for good_word in good_words):
#         return "You're welcome! Let me know if there's anything else I can help you with."
#     return None
    
# BAD_WORDS = ["bobo", "bwesit", "Not good"]  
# def contains_bad_words(user_input):
#     user_words = user_input.lower().split() 
#     for word in user_words:
#         if word in BAD_WORDS:
#             return True
#     return False


# GOOD_WORDS = ["Thank you", "Thanks", "Thank You so much"]  
# def contains_good_words(user_input):
#     user_input = user_input.lower()
#     for phrase in GOOD_WORDS:
#         if phrase.lower() in user_input:  
#             return True
#     return False

# PRAISE_WORDS = ["Helpful", "Nice", "Good"]  
# def contains_praise_words(user_input):
#     user_input = user_input.lower()
#     for phrase in PRAISE_WORDS:
#         if phrase.lower() in user_input:  
#             return True
#     return False


# def handle_special_cases(user_input, session_id, user_id):
#     """Handle special cases like gibberish or bad/good words."""
#     try:
#         detected_language = detect(user_input)
#     except LangDetectException:
#         detected_language = 'unknown'

#     if detect_clarify_request(user_input):
#         category_response = retrieve_category_response(session_id, user_id)
#         detailed_response = retrieve_detailed_response(session_id, user_id)
#         if category_response and detailed_response:
#             return f"Here is the detailed answer about your question: {category_response}. {detailed_response}"
#         return f"Here is the detailed answer about your question: {category_response}. {detailed_response}"

#     if detected_language == 'tl':  
#         return "Sorry, I only understand English. Can you please rephrase your question in English?"
#     if contains_good_words(user_input):
#         return "You're very welcome! I'm glad I could help😊. If you have any more questions or need further assistance, feel free to ask."
#     # if detect_gibberish(user_input):
#     #     return "I'm sorry, but I couldn't understand your message😕. Could you please clarify or rephrase your question?💡"
#     if contains_bad_words(user_input):
#         return "I'm sorry if my response wasn't helpful😔. If you need a better solution or clarification, please let me know how I can assist you!📚"
#     if contains_praise_words(user_input):
#         return "I'm glad you liked it! 😊 If you need anything else or have more questions, feel free to ask. I'm here to help! 💡📚"
#     return None


# # Read admin
# @app.route('/faqs', methods=['GET'])
# def get_faqs():
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT * FROM faqq")
#     faq_data = cursor.fetchall()
#     cursor.close()
#     connection.close()
    
#     faqs = [
#         {'id': row[0], 'category': row[1], 'question': row[2], 'answer': row[3], 'detailed_answer': row[4]} 
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
#     detailed_answer = data.get('detailed_answer')

#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("UPDATE faq_data SET category=%s, question=%s, answer=%s, detailed_answer=%s, updated_at=NOW() WHERE id=%s", 
#                    (category, question, answer, detailed_answer, id))
#     connection.commit()
#     cursor.close()
#     connection.close()
    
#     return jsonify({'message': 'FAQ updated successfully'})


# # Delete admin
# @app.route('/faqs/<int:id>', methods=['DELETE'])
# def delete_faq(id):
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("DELETE FROM faq_data WHERE id=%s", (id,))
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
#     detailed_answer = data.get('detailed_answer')

#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("INSERT INTO faq_data (category, question, answer, detailed_answer,updated_at) VALUES (%s, %s, %s, %s, NOW())", 
#                    (category, question, answer, detailed_answer))
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





# try again hay nakuuuu, this is final pero ayaw gumana pag lowercase ang query



from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from fuzzywuzzy import fuzz
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
import torch.nn as nn
import logging
import nltk
from nltk.corpus import words
from langdetect import detect, LangDetectException
import subprocess


app = Flask(__name__)
CORS(app)
SECRET_KEY = 'your_secret_key'
nltk.download('words')
session_states = {}


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
checkpoint = torch.load("lstm_weights.pth", weights_only=True)

memory_lstm = MemoryLSTM(
    input_size=checkpoint['config']['input_size'],
    hidden_size=checkpoint['config']['hidden_size'],
    num_layers=checkpoint['config']['num_layers'],
    output_size=checkpoint['config']['output_size']
)

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
    cursor.execute("SELECT category, question, answer FROM faqq")
    faq_data = cursor.fetchall()
    cursor.close()
    connection.close()
    return faq_data
faq_data = fetch_faq_data()


def save_message(session_id, user_id, message, sender):
    connection = connect_to_db()
    if connection is None:
        app.logger.error("Failed to connect to the database.")
        return False
    try:
        cursor = connection.cursor()
        query = """
            INSERT INTO conversations (session_id, user_id, message, sender, timestamp)
            VALUES (%s, %s, %s, %s, NOW())
        """
        cursor.execute(query, (session_id, user_id, message, sender))
        connection.commit()
        cursor.close()
        return True
    except mysql.connector.Error as err:
        app.logger.error(f"Error saving message: {err}")
        return False
    finally:
        connection.close()

def retrieve_detailed_response(session_id, user_id):
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
    last_bot_message = cursor.fetchone()
    cursor.close()
    connection.close()

    if not last_bot_message:
        return None

    last_bot_message = last_bot_message[0]
    best_match = None
    highest_score = 0

    for _, answer in faq_data:
        score = fuzz.ratio(last_bot_message.lower(), answer.lower())
        logging.info(f"Matching '{last_bot_message}' with '{answer}': Score {score}")
        if score > highest_score:
            highest_score = score
            best_match = answer

    threshold = 70 
    if highest_score >= threshold:
        return best_match if best_match else "I don't have additional details at the moment. Let me know if there's something specific you'd like to clarify."
    return None



def retrieve_category_response(session_id, user_id):
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
    last_bot_message = cursor.fetchone()
    cursor.close()
    connection.close()

    if not last_bot_message:
        return None

    last_bot_message = last_bot_message[0]
    best_match = None
    highest_score = 0

    for category, _, answer, _ in faq_data:
        score = fuzz.ratio(last_bot_message.lower(), answer.lower())
        logging.info(f"Matching '{last_bot_message}' with '{answer}': Score {score}")
        if score > highest_score:
            highest_score = score
            best_match = category

    threshold = 70 
    if highest_score >= threshold:
        return best_match if best_match else "I don't have a category match at the moment. Let me know if there's something specific you'd like to clarify."
    return None


# def generate_response(user_input, session_hidden, session_id, user_id, faq_data):

#     special_case_response = handle_special_cases(user_input, session_id, user_id)
#     if special_case_response:
#         return special_case_response, session_hidden
#     inputs = gpt_tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
#     input_ids = inputs['input_ids']
#     embeddings = gpt_model.transformer.wte(input_ids)
#     lstm_output, session_hidden = memory_lstm(embeddings, session_hidden)
#     clarification_needed = (torch.sigmoid(lstm_output) > 0.7).item()
        
#     if clarification_needed:
#         return "Could you please clarify your question?" , session_hidden

#     chat_history_ids = gpt_model.generate(input_ids, max_length=150, pad_token_id=gpt_tokenizer.eos_token_id)
#     response = gpt_tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)
#     return response, session_hidden

def generate_response(user_input, session_hidden, session_id, user_id, faq_data):
    user_input = user_input.strip()
    inputs = gpt_tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids']
    
    chat_history_ids = gpt_model.generate(input_ids, max_length=150, pad_token_id=gpt_tokenizer.eos_token_id)
    response = gpt_tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)

    # Remove the question if it's included in the response
    if response.startswith(user_input):
        response = response[len(user_input):].strip()

    return response, session_hidden



# @app.route('/chat', methods=['POST'])
# def chat():
#     try:
#         user_input = request.json.get('user_input')
#         session_id = request.json.get('session_id')
#         user_id = request.json.get('user_id')  

#         if not user_input or not session_id:
#             return jsonify({'error': 'Invalid input'}), 400

#         if session_id not in session_states:
#             session_states[session_id] = memory_lstm.init_hidden(1)

#         session_hidden = session_states[session_id]
#         response, session_hidden = generate_response(user_input, session_hidden, session_id, user_id, faq_data)
#         session_states[session_id] = session_hidden

#         save_message(session_id, user_id, user_input, "user")
#         save_message(session_id, user_id, response, "bot")

#         return jsonify({'response': response})
#     except Exception as e:
#         app.logger.error(f"Error in /chat: {e}")
#         return jsonify({'error': 'An error occurred while processing your request.'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('user_input')
        session_id = request.json.get('session_id')
        user_id = request.json.get('user_id')  


        if not user_input or not session_id:
            return jsonify({'error': 'Invalid input'}), 400

        # 🔹 Handle special cases BEFORE generating response
        special_case_response = handle_special_cases(user_input, session_id, user_id)
        if special_case_response:
            save_message(session_id, user_id, user_input, "user")
            save_message(session_id, user_id, special_case_response, "bot")
            return jsonify({'response': special_case_response})

        # 🔹 Proceed with GPT + LSTM logic if no special case
        if session_id not in session_states:
            session_states[session_id] = memory_lstm.init_hidden(1)

        session_hidden = session_states[session_id]
        response, session_hidden = generate_response(user_input, session_hidden, session_id, user_id, faq_data)
        session_states[session_id] = session_hidden

        save_message(session_id, user_id, user_input, "user")
        save_message(session_id, user_id, response, "bot")

        return jsonify({'response': response})
    
    except Exception as e:
        app.logger.error(f"Error in /chat: {e}")
        return jsonify({'error': 'An error occurred while processing your request.'}), 500



# def detect_gibberish(user_input, threshold=0.7):
#     words_list = [word for word in user_input.split() if word.strip()]
#     if not words_list:
#         return True
#     english_words = set(words.words())
#     valid_word_count = sum(1 for word in words_list if word.lower() in english_words)
#     ratio = valid_word_count / len(words_list)
#     return ratio < threshold 



repeat_phrases = ["repeat", "again", "say that again", "can you repeat", "can you say that again", "what did you say"]
def detect_clarify_request(user_input):
    for phrase in repeat_phrases:
        match_score = fuzz.partial_ratio(user_input.lower(), phrase.lower())
        if match_score > 75:  
            return True
    return False

def handle_special_cases(user_input):
    bad_words = ["bobo", "bwesit"]
    good_words = ["thank you", "thanks"]

    if any(bad_word in user_input.lower() for bad_word in bad_words):
        return "Please refrain from using offensive language."
    if any(good_word in user_input.lower() for good_word in good_words):
        return "You're welcome! Let me know if there's anything else I can help you with."
    return None
    
BAD_WORDS = ["bobo", "bwesit", "Not good"]  
def contains_bad_words(user_input):
    user_words = user_input.lower().split() 
    for word in user_words:
        if word in BAD_WORDS:
            return True
    return False


GOOD_WORDS = ["Thank you", "Thanks", "Thank You so much"]  
def contains_good_words(user_input):
    user_input = user_input.lower()
    for phrase in GOOD_WORDS:
        if phrase.lower() in user_input:  
            return True
    return False

PRAISE_WORDS = ["Helpful", "Nice", "Good"]  
def contains_praise_words(user_input):
    user_input = user_input.lower()
    for phrase in PRAISE_WORDS:
        if phrase.lower() in user_input:  
            return True
    return False

def handle_special_cases(user_input, session_id, user_id):
    """Handle special cases like gibberish or bad/good words."""
    try:
        detected_language = detect(user_input)
    except LangDetectException:
        detected_language = 'unknown'

    if detect_clarify_request(user_input):
        category_response = retrieve_category_response(session_id, user_id)
        detailed_response = retrieve_detailed_response(session_id, user_id)
        if category_response and detailed_response:
            return f"Here is the detailed answer about your question: {category_response}. {detailed_response}"
        return f"Here is the detailed answer about your question: {category_response}. {detailed_response}"

    if detected_language == 'tl':  
        return "Sorry, I only understand English. Can you please rephrase your question in English?"
    if contains_good_words(user_input):
        return "You're very welcome! I'm glad I could help😊. If you have any more questions or need further assistance, feel free to ask."
    # if detect_gibberish(user_input):
    #     return "I'm sorry, but I couldn't understand your message😕. Could you please clarify or rephrase your question?💡"
    if contains_bad_words(user_input):
        return "I'm sorry if my response wasn't helpful😔. If you need a better solution or clarification, please let me know how I can assist you!📚"
    if contains_praise_words(user_input):
        return "I'm glad you liked it! 😊 If you need anything else or have more questions, feel free to ask. I'm here to help! 💡📚"
    return None



# Read admin
@app.route('/faqs', methods=['GET'])
def get_faqs():
    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM faq_data")
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
    cursor.execute("UPDATE faq_data SET category=%s, question=%s, answer=%s, updated_at=NOW() WHERE id=%s", 
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
    cursor.execute("DELETE FROM faq_data WHERE id=%s", (id,))
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
    cursor.execute("INSERT INTO faq_data (category, question, answer, updated_at) VALUES (%s, %s, %s, %s, NOW())", 
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
#Login User
def create_user(first_name, last_name, email, password):
    print(f"Creating user: {first_name} {last_name}, {email}")
    connection = connect_to_db()
    cursor = connection.cursor()
    hashed_password = generate_password_hash(password)
    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
    if cursor.fetchone():
        raise Exception("Email already registered")

    cursor.execute("INSERT INTO users (first_name, last_name, email, password) VALUES (%s, %s, %s, %s)", 
                   (first_name, last_name, email, hashed_password))
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
    first_name = data.get('firstname')
    last_name = data.get('lastname')
    email = data.get('email')
    password = data.get('password')
    try:
        create_user(first_name, last_name, email, password)
        return jsonify({'success': True, 'message': 'User created successfully'}), 201
    except Exception as e:
        print("Signup error:", e)
        if str(e) == "Email already registered":
            return jsonify({'success': False, 'message': 'Email already registered'}), 409
        return jsonify({'success': False, 'message': 'Signup failed. Please try again.'}), 500



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
        return jsonify({'success': True, 'token': token, 'userId': result[0]})  
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





# # lowercase chuchu




# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import mysql.connector
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from fuzzywuzzy import fuzz
# from werkzeug.security import generate_password_hash, check_password_hash
# import jwt
# import datetime
# import torch.nn as nn
# import logging
# import nltk
# from nltk.corpus import words
# from langdetect import detect, LangDetectException
# import subprocess


# app = Flask(__name__)
# CORS(app)
# SECRET_KEY = 'your_secret_key'
# nltk.download('words')
# session_states = {}


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
# checkpoint = torch.load("lstm_weights.pth", weights_only=True)

# memory_lstm = MemoryLSTM(
#     input_size=checkpoint['config']['input_size'],
#     hidden_size=checkpoint['config']['hidden_size'],
#     num_layers=checkpoint['config']['num_layers'],
#     output_size=checkpoint['config']['output_size']
# )

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
#     cursor.execute("SELECT category, question, answer FROM faqq")
#     faq_data = cursor.fetchall()
#     cursor.close()
#     connection.close()
#     return faq_data
# faq_data = fetch_faq_data()


# def save_message(session_id, user_id, message, sender):
#     connection = connect_to_db()
#     if connection is None:
#         app.logger.error("Failed to connect to the database.")
#         return False
#     try:
#         cursor = connection.cursor()
#         query = """
#             INSERT INTO conversations (session_id, user_id, message, sender, timestamp)
#             VALUES (%s, %s, %s, %s, NOW())
#         """
#         cursor.execute(query, (session_id, user_id, message, sender))
#         connection.commit()
#         cursor.close()
#         return True
#     except mysql.connector.Error as err:
#         app.logger.error(f"Error saving message: {err}")
#         return False
#     finally:
#         connection.close()

# # palit sa ni comment sa baba 4/20/25
# def retrieve_best_response(user_input, session_id, user_id):
#     detailed_response = retrieve_detailed_response(user_input, session_id, user_id)
#     category_response = retrieve_category_response(user_input, session_id, user_id)

#     if detailed_response and category_response and detailed_response != category_response:
#         return f"{detailed_response} {category_response}"
    
#     return detailed_response or category_response or "I'm sorry, I couldn't find the information you're looking for."


# # start comment 4/20/25
# def retrieve_detailed_response(user_input, session_id, user_id):
#     # 1) Normalize the user's input to lowercase
#     normalized_input = user_input.strip().lower()

#     best_answer = None
#     highest_score = 0

#     # 2) Loop over faq_data, unpacking exactly (category, question, answer)
#     for category, question, answer in faq_data:
#         # 3) Normalize the question to lowercase too
#         normalized_question = question.strip().lower()

#         # 4) Score against the normalized question
#         score = fuzz.ratio(normalized_input, normalized_question)
#         logging.info(f"[DETAILED] Matching '{normalized_input}' with '{normalized_question}': Score {score}")

#         if score > highest_score:
#             highest_score = score
#             best_answer = answer

#     logging.info(f"→ Highest fuzzy score: {highest_score}, Best match answer: {best_answer}")

#     # 5) Return only if it passes your threshold
#     if highest_score >= 70:
#         return best_answer
#     return None


# def retrieve_category_response(user_input, session_id, user_id):
#     normalized_input = user_input.strip().lower()

#     best_category = None
#     highest_score = 0

#     for category, question, answer in faq_data:
#         normalized_question = question.strip().lower()
#         score = fuzz.ratio(normalized_input, normalized_question)
#         logging.info(f"[CATEGORY] Matching '{normalized_input}' with '{normalized_question}': Score {score}")

#         if score > highest_score:
#             highest_score = score
#             best_category = category

#     logging.info(f"→ Highest fuzzy score: {highest_score}, Best match category: {best_category}")

#     if highest_score >= 70:
#         return best_category
#     return None



# # ^end of the comment 4/20/25/


# def generate_response(user_input, session_hidden, session_id, user_id, faq_data):
#     user_input = user_input.strip()
#     inputs = gpt_tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
#     input_ids = inputs['input_ids']
    
#     chat_history_ids = gpt_model.generate(input_ids, max_length=150, pad_token_id=gpt_tokenizer.eos_token_id)
#     response = gpt_tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)

#     # Remove the question if it's included in the response
#     if response.startswith(user_input):
#         response = response[len(user_input):].strip()

#     return response, session_hidden

# @app.route('/chat', methods=['POST'])
# def chat():
#     try:
#         user_input = request.json.get('user_input')
#         session_id = request.json.get('session_id')
#         user_id = request.json.get('user_id')  


#         if not user_input or not session_id:
#             return jsonify({'error': 'Invalid input'}), 400

#         # 🔹 Handle special cases BEFORE generating response
#         special_case_response = handle_special_cases(user_input, session_id, user_id)
#         if special_case_response:
#             save_message(session_id, user_id, user_input, "user")
#             save_message(session_id, user_id, special_case_response, "bot")
#             return jsonify({'response': special_case_response})

#         # 🔹 Proceed with GPT + LSTM logic if no special case
#         if session_id not in session_states:
#             session_states[session_id] = memory_lstm.init_hidden(1)

#         session_hidden = session_states[session_id]
#         response, session_hidden = generate_response(user_input, session_hidden, session_id, user_id, faq_data)
#         session_states[session_id] = session_hidden

#         save_message(session_id, user_id, user_input, "user")
#         save_message(session_id, user_id, response, "bot")

#         return jsonify({'response': response})
    
#     except Exception as e:
#         app.logger.error(f"Error in /chat: {e}")
#         return jsonify({'error': 'An error occurred while processing your request.'}), 500



# repeat_phrases = ["repeat", "again", "say that again", "can you repeat", "can you say that again", "what did you say"]
# def detect_clarify_request(user_input):
#     for phrase in repeat_phrases:
#         match_score = fuzz.partial_ratio(user_input.lower(), phrase.lower())
#         if match_score > 75:  
#             return True
#     return False

# def handle_special_cases(user_input):
#     bad_words = ["bobo", "bwesit"]
#     good_words = ["thank you", "thanks"]

#     if any(bad_word in user_input.lower() for bad_word in bad_words):
#         return "Please refrain from using offensive language."
#     if any(good_word in user_input.lower() for good_word in good_words):
#         return "You're welcome! Let me know if there's anything else I can help you with."
#     return None
    
# BAD_WORDS = ["bobo", "bwesit", "Not good"]  
# def contains_bad_words(user_input):
#     user_words = user_input.lower().split() 
#     for word in user_words:
#         if word in BAD_WORDS:
#             return True
#     return False


# GOOD_WORDS = ["Thank you", "Thanks", "Thank You so much"]  
# def contains_good_words(user_input):
#     user_input = user_input.lower()
#     for phrase in GOOD_WORDS:
#         if phrase.lower() in user_input:  
#             return True
#     return False

# PRAISE_WORDS = ["Helpful", "Nice", "Good"]  
# def contains_praise_words(user_input):
#     user_input = user_input.lower()
#     for phrase in PRAISE_WORDS:
#         if phrase.lower() in user_input:  
#             return True
#     return False
# # palit sa comment sa baba
# def handle_special_cases(user_input, session_id, user_id):
#     """Handle special cases like gibberish or bad/good words."""
#     try:
#         detected_language = detect(user_input)
#     except LangDetectException:
#         detected_language = 'unknown'

#     if detect_clarify_request(user_input):
#         answer = retrieve_best_response(user_input, session_id, user_id)
#         return f"Here is the detailed answer to your question: {answer}"

#     if detected_language == 'tl':  
#         return "Sorry, I only understand English. Can you please rephrase your question in English?"

#     if contains_good_words(user_input):
#         return "You're very welcome! I'm glad I could help😊. If you have any more questions or need further assistance, feel free to ask."

#     # if detect_gibberish(user_input):
#     #     return "I'm sorry, but I couldn't understand your message😕. Could you please clarify or rephrase your question?💡"

#     if contains_bad_words(user_input):
#         return "I'm sorry if my response wasn't helpful😔. If you need a better solution or clarification, please let me know how I can assist you!📚"

#     if contains_praise_words(user_input):
#         return "I'm glad you liked it! 😊 If you need anything else or have more questions, feel free to ask. I'm here to help! 💡📚"

#     return None


# # Read admin
# @app.route('/faqs', methods=['GET'])
# def get_faqs():
#     connection = connect_to_db()
#     cursor = connection.cursor()
#     cursor.execute("SELECT * FROM faq_data")
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
#     cursor.execute("UPDATE faq_data SET category=%s, question=%s, answer=%s, updated_at=NOW() WHERE id=%s", 
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
#     cursor.execute("DELETE FROM faq_data WHERE id=%s", (id,))
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
#     cursor.execute("INSERT INTO faq_data (category, question, answer, updated_at) VALUES (%s, %s, %s, %s, NOW())", 
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
#     logging.basicConfig(level=logging.INFO)
