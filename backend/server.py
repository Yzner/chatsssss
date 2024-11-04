

from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from fuzzywuzzy import fuzz
import subprocess


app = Flask(__name__)
CORS(app)

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
    cursor.execute("SELECT question, answer FROM faq")
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
    cursor.execute("SELECT * FROM faq")
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
    cursor.execute("UPDATE faq SET category=%s, question=%s, answer=%s WHERE id=%s", 
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
    cursor.execute("DELETE FROM faq WHERE id=%s", (id,))
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
    cursor.execute("INSERT INTO faq (category, question, answer) VALUES (%s, %s, %s)", 
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


