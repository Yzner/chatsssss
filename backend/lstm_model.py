# # from flask import Flask, request, jsonify
# # from flask_cors import CORS
# # import mysql.connector
# # import torch
# # from transformers import GPT2Tokenizer, GPT2LMHeadModel
# # from fuzzywuzzy import fuzz
# # import subprocess
# # from werkzeug.security import generate_password_hash, check_password_hash
# # import jwt
# # import datetime
# # import torch.nn as nn

# # app = Flask(__name__)
# # CORS(app)
# # SECRET_KEY = 'your_secret_key'

# # # Define Memory LSTM
# # class MemoryLSTM(nn.Module):
# #     def __init__(self, input_size, hidden_size, num_layers, output_size):
# #         super(MemoryLSTM, self).__init__()
# #         self.hidden_size = hidden_size
# #         self.num_layers = num_layers
# #         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
# #         self.fc = nn.Linear(hidden_size, output_size)

# #     def forward(self, x, hidden):
# #         out, hidden = self.lstm(x, hidden)
# #         out = self.fc(out[:, -1, :])
# #         return out, hidden

# #     def init_hidden(self, batch_size):
# #         return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
# #                 torch.zeros(self.num_layers, batch_size, self.hidden_size))

# # gpt_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
# # gpt_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")
# # memory_lstm = MemoryLSTM(input_size=768, hidden_size=128, num_layers=2, output_size=1)


# # def connect_to_db():
# #     try:
# #         connection = mysql.connector.connect(
# #             host='localhost',
# #             user='root',
# #             password='root',
# #             database='chat'
# #         )
# #         return connection
# #     except mysql.connector.Error as err:
# #         print(f"Error: {err}")
# #         return None

# # def fetch_faq_data():
# #     connection = connect_to_db()
# #     if connection is None:
# #         return []

# #     cursor = connection.cursor()
# #     cursor.execute("SELECT question, answer FROM data")
# #     faq_data = cursor.fetchall()
# #     cursor.close()
# #     connection.close()
# #     return faq_data

# # faq_data = fetch_faq_data()  
# # def find_best_faq_match(user_input, faq_data, threshold=80):
# #     best_match = None
# #     highest_score = 0

# #     for question, answer in faq_data:
# #         score = fuzz.ratio(user_input.lower(), question.lower())
# #         if score > highest_score:
# #             highest_score = score
# #             best_match = answer
# #     return best_match if highest_score >= threshold else None

# # def correct_spelling(user_input):
# #     return user_input

# # def save_message(session_id, user_id, message, sender):
# #     connection = connect_to_db()
# #     if connection is None:
# #         return
# #     cursor = connection.cursor()
# #     cursor.execute(
# #         "INSERT INTO conversations (session_id, user_id, message, sender, timestamp) VALUES (%s, %s, %s, %s, NOW())",
# #         (session_id, user_id, message, sender)
# #     )
# #     connection.commit()
# #     cursor.close()
# #     connection.close()

# # def retrieve_past_response(session_id, user_id):
# #     connection = connect_to_db()
# #     if connection is None:
# #         return None
# #     cursor = connection.cursor()
# #     query = """
# #     SELECT message FROM conversations
# #     WHERE session_id = %s AND user_id = %s AND sender = 'bot'
# #     ORDER BY timestamp DESC LIMIT 1
# #     """
# #     cursor.execute(query, (session_id, user_id))
# #     result = cursor.fetchone()
# #     cursor.close()
# #     connection.close()
# #     return result[0] if result else None


# # def detect_repeat_request(user_input):
# #     repeat_phrases = ["repeat", "again", "say that again", "can you repeat", "can you say that again", "what did you say"]
# #     for phrase in repeat_phrases:
# #         if fuzz.ratio(user_input.lower(), phrase) > 80:
# #             return True
# #     return False


# # def generate_gpt_response(user_input, gpt_model, gpt_tokenizer, lstm_model, lstm_hidden, session_id, user_id):
# #     # Check if the user is asking for a repetition
# #     if detect_repeat_request(user_input):
# #         # Retrieve the last response directly without generating a new one
# #         past_response = retrieve_past_response(session_id, user_id)
# #         return past_response if past_response else "I'm sorry, I don't have a recent response to repeat."

# #     # Tokenize and get embeddings for the user input
# #     inputs = gpt_tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
# #     input_ids = inputs['input_ids']
# #     embeddings = gpt_model.transformer.wte(input_ids)  # Get embeddings

# #     # Use LSTM model to check if further clarification is needed
# #     lstm_output, lstm_hidden = lstm_model(embeddings, lstm_hidden)
# #     clarify_needed = (torch.sigmoid(lstm_output) > 0.5).item()

# #     # If clarification is detected, retrieve the past response
# #     if clarify_needed:
# #         past_response = retrieve_past_response(session_id, user_id)
# #         return past_response if past_response else "I'm sorry, I don't have a recent response to clarify."

# #     # Otherwise, generate a new response with GPT-2
# #     chat_history_ids = gpt_model.generate(input_ids, max_length=150, num_return_sequences=1)
# #     return gpt_tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)



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



# # # Read admin
# # @app.route('/faqs', methods=['GET'])
# # def get_faqs():
# #     connection = connect_to_db()
# #     cursor = connection.cursor()
# #     cursor.execute("SELECT * FROM data")
# #     faq_data = cursor.fetchall()
# #     cursor.close()
# #     connection.close()
    
# #     faqs = [
# #         {'id': row[0], 'category': row[1], 'question': row[2], 'answer': row[3]} 
# #         for row in faq_data
# #     ]
    
# #     return jsonify(faqs)


# # # Edit admin
# # @app.route('/faqs/<int:id>', methods=['PUT'])
# # def edit_faq(id):
# #     data = request.json
# #     category = data.get('category')
# #     question = data.get('question')
# #     answer = data.get('answer')

# #     connection = connect_to_db()
# #     cursor = connection.cursor()
# #     cursor.execute("UPDATE data SET category=%s, question=%s, answer=%s WHERE id=%s", 
# #                    (category, question, answer, id))
# #     connection.commit()
# #     cursor.close()
# #     connection.close()
    
# #     return jsonify({'message': 'FAQ updated successfully'})


# # # Delete admin
# # @app.route('/faqs/<int:id>', methods=['DELETE'])
# # def delete_faq(id):
# #     connection = connect_to_db()
# #     cursor = connection.cursor()
# #     cursor.execute("DELETE FROM data WHERE id=%s", (id,))
# #     connection.commit()
# #     cursor.close()
# #     connection.close()
    
# #     return jsonify({'message': 'FAQ deleted successfully'})


# # # Add admin
# # @app.route('/faqs', methods=['POST'])
# # def add_faq():
# #     data = request.json
# #     category = data.get('category')
# #     question = data.get('question')
# #     answer = data.get('answer')

# #     connection = connect_to_db()
# #     cursor = connection.cursor()
# #     cursor.execute("INSERT INTO data (category, question, answer) VALUES (%s, %s, %s)", 
# #                    (category, question, answer))
# #     connection.commit()
# #     cursor.close()
# #     connection.close()

# #     return jsonify({'message': 'FAQ added successfully'}), 201

# # #train button
# # @app.route('/train', methods=['POST'])
# # def train_data():
# #     try:
# #         subprocess.run(['python', 'fine_tune.py'], check=True)
# #         return jsonify({'message': 'Training started successfully'}), 200
# #     except subprocess.CalledProcessError as e:
# #         print(f"Error: {e}")
# #         return jsonify({'message': 'Failed to start training'}), 500
    


# # #Login User
# # def create_user(first_name, last_name, email, password, role):
# #     connection = connect_to_db()
# #     cursor = connection.cursor()
# #     hashed_password = generate_password_hash(password)
# #     cursor.execute("INSERT INTO users (first_name, last_name, email, password, role) VALUES (%s, %s, %s, %s, %s)", 
# #                    (first_name, last_name, email, hashed_password, role))
# #     connection.commit()
# #     cursor.close()
# #     connection.close()
# # def authenticate_user(email, password):
# #     connection = connect_to_db()
# #     cursor = connection.cursor()
# #     cursor.execute("SELECT password FROM users WHERE email = %s", (email,))
# #     result = cursor.fetchone()
# #     cursor.close()
# #     connection.close()
# #     if result and check_password_hash(result[0], password):
# #         return True
# #     return False
# # @app.route('/signup', methods=['POST'])
# # def signup():
# #     data = request.json
# #     first_name = data.get('firstName')
# #     last_name = data.get('lastName')
# #     email = data.get('email')
# #     password = data.get('password')
# #     role = data.get('role')
# #     create_user(first_name, last_name, email, password, role)
# #     return jsonify({'success': True, 'message': 'User created successfully'}), 201
# # @app.route('/login', methods=['POST'])
# # def login():
# #     data = request.json
# #     email = data.get('email')
# #     password = data.get('password')

# #     connection = connect_to_db()
# #     cursor = connection.cursor()
# #     cursor.execute("SELECT id, password FROM users WHERE email = %s", (email,))
# #     result = cursor.fetchone()
# #     cursor.close()
# #     connection.close()

# #     if result and check_password_hash(result[1], password):
# #         token = jwt.encode({
# #             'email': email,
# #             'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
# #         }, SECRET_KEY)
# #         return jsonify({'success': True, 'token': token, 'userId': result[0]})  # Return user ID
# #     return jsonify({'success': False, 'message': 'Invalid email or password'}), 401




# # #admin
# # @app.route('/adminsignup', methods=['POST'])
# # def admin_signup():
# #     data = request.json
# #     first_name = data.get('firstName')
# #     last_name = data.get('lastName')
# #     email = data.get('email')
# #     password = data.get('password')
# #     department = data.get('department')
# #     role = "pending" 

# #     connection = connect_to_db()
# #     cursor = connection.cursor()
# #     hashed_password = generate_password_hash(password)
# #     cursor.execute("INSERT INTO admin (first_name, last_name, email, password, role, department) VALUES (%s, %s, %s, %s, %s, %s)", 
# #                    (first_name, last_name, email, hashed_password, role, department))
# #     connection.commit()
# #     cursor.close()
# #     connection.close()

# #     return jsonify({'success': True, 'message': 'Sign-up request submitted. Awaiting main admin approval.'}), 201

# # # Admin 
# # @app.route('/adminlogin', methods=['POST'])
# # def admin_login():
# #     data = request.json
# #     email = data.get('email')
# #     password = data.get('password')

# #     connection = connect_to_db()
# #     cursor = connection.cursor()
# #     cursor.execute("SELECT password, role FROM admin WHERE email = %s", (email,))
# #     result = cursor.fetchone()
# #     cursor.close()
# #     connection.close()

# #     if result and check_password_hash(result[0], password):
# #         if result[1] == "approved" or result[1] == "main_admin":
# #             token = jwt.encode({
# #                 'email': email,
# #                 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
# #             }, SECRET_KEY)
# #             return jsonify({'success': True, 'token': token})
# #         else:
# #             return jsonify({'success': False, 'message': 'Account pending main admin approval.'}), 403
# #     return jsonify({'success': False, 'message': 'Invalid email or password'}), 401


# # # Approve admin request 
# # @app.route('/approve_admin', methods=['POST'])
# # def approve_admin():
# #     data = request.json
# #     email = data.get('email')
# #     connection = connect_to_db()
# #     cursor = connection.cursor()
# #     cursor.execute("UPDATE admin SET role = 'approved' WHERE email = %s AND role = 'pending'", (email,))
# #     connection.commit()
# #     cursor.close()
# #     connection.close()

# #     return jsonify({'message': 'Admin approved successfully.'})


# # @app.route('/pending_admins', methods=['GET'])
# # def get_pending_admins():
# #     connection = connect_to_db()
# #     cursor = connection.cursor()
# #     cursor.execute("SELECT first_name, last_name, email, department FROM admin WHERE role = 'pending'")
# #     pending_admins = cursor.fetchall()
# #     cursor.close()
# #     connection.close()

# #     return jsonify([{'firstName': admin[0], 'lastName': admin[1], 'email': admin[2], 'department': admin[3]} for admin in pending_admins])

# # @app.route('/decline_admin', methods=['POST'])
# # def decline_admin():
# #     data = request.json
# #     email = data.get('email')
# #     connection = connect_to_db()
# #     cursor = connection.cursor()
# #     cursor.execute("DELETE FROM admin WHERE email = %s AND role = 'pending'", (email,))
# #     connection.commit()
# #     cursor.close()
# #     connection.close()

# #     return jsonify({'message': 'Admin request declined successfully.'})


# # @app.route('/conversation_history', methods=['GET'])
# # def get_conversation_history():
# #     session_id = request.args.get('session_id')
# #     user_id = request.args.get('user_id')

# #     connection = connect_to_db()
# #     cursor = connection.cursor()
# #     cursor.execute(
# #         "SELECT message, sender FROM conversations WHERE session_id = %s AND user_id = %s ORDER BY timestamp",
# #         (session_id, user_id)
# #     )
# #     history = cursor.fetchall()
# #     cursor.close()
# #     connection.close()

# #     return jsonify([
# #         {'sender': row[1], 'message': row[0]}
# #         for row in history
# #     ])


# # if __name__ == '__main__':
# #     app.run(host='0.0.0.0', port=5000)














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










import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, pipeline
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import datetime
import os
import logging
import pickle  # To save and load previously trained data

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def connect_to_db():
    """
    Establish a connection to the MySQL database.
    """
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='root',
            database='chat'
        )
        logging.info("Database connection established.")
        return connection
    except mysql.connector.Error as err:
        logging.error("Database connection error: %s", err)
        return None


def fetch_data(connection, last_trained_at):
    """
    Fetch data updated since the last training timestamp.
    """
    try:
        cursor = connection.cursor()
        query = """
        SELECT category, question, answer, updated_at
        FROM faq_data
        WHERE updated_at > %s AND deleted = FALSE
        LIMIT 200
        """
        cursor.execute(query, (last_trained_at,))
        data = cursor.fetchall()
        cursor.close()
        logging.info("Fetched %d rows of data.", len(data))
        return data
    except Exception as e:
        logging.error("Error fetching data: %s", e)
        return []


def group_by_intent(data):
    """
    Group data by intent (category) while ignoring non-relevant columns.
    """
    grouped_data = {}
    for row in data:
        category, question, answer = row[:3]
        grouped_data.setdefault(category, []).append((question, answer))
    logging.info("Data grouped by %d categories.", len(grouped_data))
    return grouped_data


def augment_data(data):
    """
    Augment data by paraphrasing questions.
    """
    paraphraser_model = "t5-small"
    try:
        paraphraser = pipeline("text2text-generation", model=paraphraser_model)
        logging.info("Paraphraser model loaded successfully.")
    except Exception as e:
        logging.error("Failed to load paraphraser model: %s", e)
        return data

    augmented_data = []
    for category, qa_pairs in data.items():
        for question, answer in qa_pairs:
            augmented_data.append((category, question, answer))
            try:
                paraphrased_questions = paraphraser(f"paraphrase: {question}", num_return_sequences=1)
                for paraphrase in paraphrased_questions:
                    augmented_data.append((category, paraphrase['generated_text'], answer))
            except Exception as e:
                logging.warning("Failed to paraphrase '%s': %s", question, e)
    logging.info("Data augmentation complete with %d entries.", len(augmented_data))
    return augmented_data


def prepare_dataset(data, prev_data_path="prev_data.pkl"):
    """
    Combine new data with previously trained data and split for training and evaluation.
    """
    if os.path.exists(prev_data_path):
        with open(prev_data_path, "rb") as f:
            prev_data = pickle.load(f)
        data = {**prev_data, **data}
        logging.info("Loaded and combined previously trained data.")

    augmented_data = augment_data(data)
    dataset = [{"text": f"Category: {cat}\nUser: {q}\nBot: {a}"} for cat, q, a in augmented_data]
    df = pd.DataFrame(dataset)
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    logging.info("Dataset prepared with %d training and %d evaluation samples.", len(train_df), len(eval_df))

    with open(prev_data_path, "wb") as f:
        pickle.dump(data, f)
    logging.info("Saved updated training data for future use.")

    return train_df, eval_df


def tokenize_function(examples, tokenizer):
    """
    Tokenize the input examples.
    """
    encoding = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=150)
    encoding['labels'] = encoding['input_ids'].copy()
    return encoding


def compute_metrics(p):
    """
    Compute evaluation metrics (accuracy, precision, recall, F1 score).
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=-1)
    labels = p.label_ids.flatten()
    preds = preds.flatten()

    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def fine_tune_model(train_df, eval_df):
    """
    Fine-tune GPT-2 model.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_eval = eval_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir="./fine_tuned_gpt2",
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=50,
        learning_rate=1e-4,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics
    )

    logging.info("Starting model training...")
    trainer.train()
    trainer.save_model("./fine_tuned_gpt2")
    tokenizer.save_pretrained("./fine_tuned_gpt2")
    logging.info("Model fine-tuned and saved successfully.")


def check_and_train():
    """
    Check for new data and train the model incrementally.
    """
    last_trained_at = "1970-01-01 00:00:00"
    if os.path.exists("last_trained_time.txt"):
        with open("last_trained_time.txt", "r") as f:
            last_trained_at = f.read().strip()

    connection = connect_to_db()
    if connection is None:
        return

    while True:
        data = fetch_data(connection, last_trained_at)
        if not data:
            logging.info("No more data to train on.")
            break

        grouped_data = group_by_intent(data)
        train_df, eval_df = prepare_dataset(grouped_data)
        fine_tune_model(train_df, eval_df)

        last_trained_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("last_trained_time.txt", "w") as f:
            f.write(last_trained_at)

    connection.close()


if __name__ == "__main__":
    check_and_train()
