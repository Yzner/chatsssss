# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import mysql.connector
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertModel, pipeline
# from fuzzywuzzy import process
# from spellchecker import SpellChecker

# app = Flask(__name__)
# CORS(app)

# spell = SpellChecker()

# # Initialize the models and tokenizers
# def load_models():
#     gpt_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
#     gpt_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")

#     bert_model = BertModel.from_pretrained("bert-base-uncased")
#     bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#     # Load the paraphraser model (T5)
#     paraphraser = pipeline("text2text-generation", model="t5-base")

#     return gpt_model, gpt_tokenizer, bert_model, bert_tokenizer, paraphraser

# gpt_model, gpt_tokenizer, bert_model, bert_tokenizer, paraphraser = load_models()

# def connect_to_db():
#     return mysql.connector.connect(
#         host='localhost',
#         user='root',
#         password='root',
#         database='chat'
#     )

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

# def generate_bert_embeddings(user_input, bert_model, bert_tokenizer):
#     inputs = bert_tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = bert_model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1)

# def generate_gpt_response(user_input, faq_context, gpt_model, gpt_tokenizer):
#     combined_input = f"{faq_context}\nUser: {user_input}\nBot:"
#     inputs = gpt_tokenizer(combined_input, return_tensors='pt', padding=True, truncation=True)
#     chat_history_ids = gpt_model.generate(inputs['input_ids'], max_length=150)
#     return gpt_tokenizer.decode(chat_history_ids[:, inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True)

# # Route for user chat with FAQ matching and GPT-2 response
# @app.route('/chat', methods=['POST'])
# def chat():
#     try:
#         user_input = request.json.get('user_input')
#         conversation_history = request.json.get('conversation_history', [])
#         corrected_input = correct_spelling(user_input)

#         faq_data = fetch_faq_data()
#         if not faq_data:
#             raise Exception("Failed to fetch FAQ data")

#         questions = [q[0] for q in faq_data]
#         answers = {q[0]: q[1] for q in faq_data}

#         best_match, score = process.extractOne(corrected_input, questions)
#         if score >= 70:
#             response = answers[best_match]
#         else:
#             context_embedding = generate_bert_embeddings(corrected_input, bert_model, bert_tokenizer)
#             response = generate_gpt_response(corrected_input, "", gpt_model, gpt_tokenizer)

#         conversation_history.append({"user": corrected_input, "bot": response})

#         return jsonify({
#             'response': response,
#             'conversation_history': conversation_history
#         })
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         return jsonify({'error': str(e)}), 500

# # Paraphrase FAQ questions and add to database
# @app.route('/faqs', methods=['POST'])
# def add_faq():
#     try:
#         data = request.json
#         category = data.get('category')
#         question = data.get('question')
#         answer = data.get('answer')

#         connection = connect_to_db()
#         cursor = connection.cursor()

#         # Insert original question
#         cursor.execute("INSERT INTO faq (category, question, answer) VALUES (%s, %s, %s)", 
#                        (category, question, answer))

#         # Paraphrase the question and insert paraphrased variations
#         paraphrase_prompt = f"paraphrase: {question} </s>"
#         paraphrased_questions = paraphraser(paraphrase_prompt, num_return_sequences=1)
#         for paraphrase in paraphrased_questions:
#             cursor.execute("INSERT INTO faq (category, question, answer) VALUES (%s, %s, %s)", 
#                            (category, paraphrase['generated_text'], answer))

#         connection.commit()
#         cursor.close()
#         connection.close()

#         return jsonify({'message': 'FAQ and its paraphrased versions added successfully'}), 201
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         return jsonify({'error': str(e)}), 500

# # Get all FAQs
# @app.route('/faqs', methods=['GET'])
# def get_faqs():
#     try:
#         connection = connect_to_db()
#         cursor = connection.cursor()
#         cursor.execute("SELECT * FROM faq")
#         faq_data = cursor.fetchall()
#         cursor.close()
#         connection.close()

#         faqs = [{'id': row[0], 'category': row[1], 'question': row[2], 'answer': row[3]} 
#                 for row in faq_data]

#         return jsonify(faqs)
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         return jsonify({'error': str(e)}), 500

# # Edit FAQ
# @app.route('/faqs/<int:id>', methods=['PUT'])
# def edit_faq(id):
#     try:
#         data = request.json
#         category = data.get('category')
#         question = data.get('question')
#         answer = data.get('answer')

#         connection = connect_to_db()
#         cursor = connection.cursor()
#         cursor.execute("UPDATE faq SET category=%s, question=%s, answer=%s WHERE id=%s", 
#                        (category, question, answer, id))
#         connection.commit()
#         cursor.close()
#         connection.close()

#         return jsonify({'message': 'FAQ updated successfully'})
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         return jsonify({'error': str(e)}), 500

# # Delete FAQ
# @app.route('/faqs/<int:id>', methods=['DELETE'])
# def delete_faq(id):
#     try:
#         connection = connect_to_db()
#         cursor = connection.cursor()
#         cursor.execute("DELETE FROM faq WHERE id=%s", (id,))
#         connection.commit()
#         cursor.close()
#         connection.close()

#         return jsonify({'message': 'FAQ deleted successfully'})
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         return jsonify({'error': str(e)}), 500


# if __name__ == '__main__':
#     app.run(port=5000)









from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertModel, pipeline
from fuzzywuzzy import process
from spellchecker import SpellChecker
import logging

app = Flask(__name__)
CORS(app)

spell = SpellChecker()
logging.basicConfig(level=logging.INFO)

# Initialize the models and tokenizers
def load_models():
    gpt_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")

    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load the paraphraser model (T5)
    paraphraser = pipeline("text2text-generation", model="t5-base")

    return gpt_model, gpt_tokenizer, bert_model, bert_tokenizer, paraphraser

gpt_model, gpt_tokenizer, bert_model, bert_tokenizer, paraphraser = load_models()

def connect_to_db():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='root',
        database='chat'
    )

def correct_spelling(user_input):
    corrected_input = []
    for word in user_input.split():
        correction = spell.correction(word)
        # Only correct if confidence is high or the word is not part of the FAQ context
        if correction and correction.lower() != word.lower():
            corrected_input.append(correction)
        else:
            corrected_input.append(word)
    return " ".join(corrected_input)

def fetch_faq_data():
    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute("SELECT question, answer FROM faq")
    faq_data = cursor.fetchall()
    cursor.close()
    connection.close()
    return faq_data

def generate_bert_embeddings(user_input, bert_model, bert_tokenizer):
    inputs = bert_tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def generate_gpt_response(user_input, faq_context, gpt_model, gpt_tokenizer):
    combined_input = f"{faq_context}\nUser: {user_input}\nBot:"
    inputs = gpt_tokenizer(combined_input, return_tensors='pt', padding=True, truncation=True)
    chat_history_ids = gpt_model.generate(inputs['input_ids'], max_length=150)
    return gpt_tokenizer.decode(chat_history_ids[:, inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True)

# Route for user chat with FAQ matching and GPT-2 response
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('user_input')
        conversation_history = request.json.get('conversation_history', [])
        corrected_input = correct_spelling(user_input)
        logging.info(f"Corrected input: {corrected_input}")

        faq_data = fetch_faq_data()
        if not faq_data:
            raise Exception("Failed to fetch FAQ data")

        questions = [q[0] for q in faq_data]
        answers = {q[0]: q[1] for q in faq_data}

        best_match, score = process.extractOne(corrected_input, questions)
        logging.info(f"Best match: {best_match}, Score: {score}")

        if score >= 70:
            response = answers[best_match]
        else:
            context_embedding = generate_bert_embeddings(corrected_input, bert_model, bert_tokenizer)
            response = generate_gpt_response(corrected_input, "", gpt_model, gpt_tokenizer)

        conversation_history.append({"user": corrected_input, "bot": response})

        return jsonify({
            'response': response,
            'conversation_history': conversation_history
        })
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

# Paraphrase FAQ questions and add to database
@app.route('/faqs', methods=['POST'])
def add_faq():
    try:
        data = request.json
        category = data.get('category')
        question = data.get('question')
        answer = data.get('answer')

        connection = connect_to_db()
        cursor = connection.cursor()

        # Insert original question
        cursor.execute("INSERT INTO faq (category, question, answer) VALUES (%s, %s, %s)", 
                       (category, question, answer))

        # Paraphrase the question and insert paraphrased variations
        paraphrase_prompt = f"paraphrase: {question} </s>"
        paraphrased_questions = paraphraser(paraphrase_prompt, num_return_sequences=1)
        for paraphrase in paraphrased_questions:
            cursor.execute("INSERT INTO faq (category, question, answer) VALUES (%s, %s, %s)", 
                           (category, paraphrase['generated_text'], answer))

        connection.commit()
        cursor.close()
        connection.close()

        return jsonify({'message': 'FAQ and its paraphrased versions added successfully'}), 201
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

# Get all FAQs
@app.route('/faqs', methods=['GET'])
def get_faqs():
    try:
        connection = connect_to_db()
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM faq")
        faq_data = cursor.fetchall()
        cursor.close()
        connection.close()

        faqs = [{'id': row[0], 'category': row[1], 'question': row[2], 'answer': row[3]} 
                for row in faq_data]

        return jsonify(faqs)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

# Edit FAQ
@app.route('/faqs/<int:id>', methods=['PUT'])
def edit_faq(id):
    try:
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
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

# Delete FAQ
@app.route('/faqs/<int:id>', methods=['DELETE'])
def delete_faq(id):
    try:
        connection = connect_to_db()
        cursor = connection.cursor()
        cursor.execute("DELETE FROM faq WHERE id=%s", (id,))
        connection.commit()
        cursor.close()
        connection.close()

        return jsonify({'message': 'FAQ deleted successfully'})
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000)
