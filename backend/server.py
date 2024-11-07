

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


app = Flask(__name__)
CORS(app)
SECRET_KEY = 'your_secret_key'

def load_gpt_model():
    gpt_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")
    return gpt_model, gpt_tokenizer

gpt_model, gpt_tokenizer = load_gpt_model()

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

def generate_gpt_response(user_input, gpt_model, gpt_tokenizer):
    inputs = gpt_tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
    chat_history_ids = gpt_model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)
    return gpt_tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('user_input')
    corrected_input = correct_spelling(user_input)
    faq_answer = find_best_faq_match(corrected_input, faq_data)
    if faq_answer:
        return jsonify({'response': faq_answer})
    gpt_response = generate_gpt_response(corrected_input, gpt_model, gpt_tokenizer)
    return jsonify({'response': gpt_response if gpt_response else "I'm sorry, I don't have an answer for that."})

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
    if authenticate_user(email, password):
        token = jwt.encode({
            'email': email,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        }, SECRET_KEY)
        return jsonify({'success': True, 'token': token})
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




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
