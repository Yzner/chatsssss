import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import pipeline
import datetime
import os

def augment_data(faq_data):
    paraphraser = pipeline("text2text-generation", model="t5-base")  
    augmented_data = []
    
    for row in faq_data:
        category, question, answer = row[:3]  
        augmented_data.append((category, question, answer))
        paraphrase_prompt = f"paraphrase: {question} </s>"
        paraphrased_questions = paraphraser(paraphrase_prompt, num_return_sequences=1)
        for para in paraphrased_questions:
            augmented_data.append((category, para['generated_text'], answer))
    return augmented_data


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
def fetch_updated_data(connection, last_trained_at):
    cursor = connection.cursor()
    query = """
    SELECT category, question, answer, updated_at
    FROM data
    WHERE updated_at > %s AND deleted = FALSE
    """
    cursor.execute(query, (last_trained_at,))
    faq_data = cursor.fetchall()
    cursor.close()
    return faq_data

def prepare_dataset(faq_data):
    augmented_data = augment_data(faq_data)
    data = [{'text': f"Category: {category}\nUser: {q}\nBot: {a}"} for category, q, a in augmented_data]
    df = pd.DataFrame(data)
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, eval_df

def tokenize_function(examples, tokenizer):
    encoding = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=150)
    encoding['labels'] = encoding['input_ids'].copy()
    return encoding


def compute_metrics(p):
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
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    tokenized_train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_eval_dataset = eval_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir="./fine_tuned_gpt2",
        eval_strategy="steps",
        eval_steps=100,  
        save_steps=100,  
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,  
        logging_dir='./logs',
        logging_steps=10,
        metric_for_best_model="eval_loss"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("./fine_tuned_gpt2")
    tokenizer.save_pretrained("./fine_tuned_gpt2")

def check_and_train():
    last_trained_at = "1970-01-01 00:00:00" 
    if os.path.exists("last_trained_time.txt"):
        with open("last_trained_time.txt", "r") as f:
            last_trained_at = f.read().strip()

    connection = connect_to_db()
    if connection is None:
        return

    faq_data = fetch_updated_data(connection, last_trained_at)
    connection.close()
    
    if not faq_data:
        print("All your data is updated already.")
        return

    train_df, eval_df = prepare_dataset(faq_data)
    fine_tune_model(train_df, eval_df)

    with open("last_trained_time.txt", "w") as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    check_and_train()


# import mysql.connector
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from transformers import Trainer, TrainingArguments
# from datasets import Dataset
# import numpy as np
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from transformers import pipeline
# import datetime
# import os
# import torch

# # --- Existing functions remain the same ---

# def augment_data(faq_data):
#     paraphraser = pipeline("text2text-generation", model="t5-base")  
#     augmented_data = []
    
#     for row in faq_data:
#         category, question, answer = row[:3]  
#         augmented_data.append((category, question, answer))
#         paraphrase_prompt = f"paraphrase: {question} </s>"
#         paraphrased_questions = paraphraser(paraphrase_prompt, num_return_sequences=1)
#         for para in paraphrased_questions:
#             augmented_data.append((category, para['generated_text'], answer))
#     return augmented_data

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

# def fetch_updated_data(connection, last_trained_at):
#     cursor = connection.cursor()
#     query = """
#     SELECT category, question, answer, updated_at
#     FROM data
#     WHERE updated_at > %s AND deleted = FALSE
#     """
#     cursor.execute(query, (last_trained_at,))
#     faq_data = cursor.fetchall()
#     cursor.close()
#     return faq_data

# def prepare_dataset(faq_data):
#     augmented_data = augment_data(faq_data)
#     data = [{'text': f"Category: {category}\nUser: {q}\nBot: {a}"} for category, q, a in augmented_data]
#     df = pd.DataFrame(data)
#     train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
#     return train_df, eval_df

# def tokenize_function(examples, tokenizer):
#     # Tokenize the input and set attention_mask explicitly
#     encoding = tokenizer(
#         examples['text'], 
#         padding="max_length", 
#         truncation=True, 
#         max_length=150, 
#         return_tensors="pt"
#     )
#     # Explicitly set the attention mask for reliable results
#     encoding['labels'] = encoding['input_ids'].clone()
#     return encoding


# def compute_metrics(p):
#     preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
#     preds = np.argmax(preds, axis=-1)
    
#     labels = p.label_ids.flatten()
#     preds = preds.flatten()

#     mask = labels != -100
#     labels = labels[mask]
#     preds = preds[mask]

#     accuracy = accuracy_score(labels, preds)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

#     return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# def fine_tune_model(train_df, eval_df):
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     tokenizer.pad_token = tokenizer.eos_token  
#     model = GPT2LMHeadModel.from_pretrained("gpt2")

#     train_dataset = Dataset.from_pandas(train_df)
#     eval_dataset = Dataset.from_pandas(eval_df)
#     tokenized_train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
#     tokenized_eval_dataset = eval_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

#     training_args = TrainingArguments(
#         output_dir="./fine_tuned_gpt2",
#         eval_strategy="steps",
#         eval_steps=100,  
#         save_steps=100,  
#         learning_rate=5e-5,
#         per_device_train_batch_size=2,
#         num_train_epochs=3,
#         weight_decay=0.01,
#         save_total_limit=2,
#         load_best_model_at_end=True,  
#         logging_dir='./logs',
#         logging_steps=10,
#         metric_for_best_model="eval_loss"
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_train_dataset,
#         eval_dataset=tokenized_eval_dataset,
#         compute_metrics=compute_metrics
#     )

#     trainer.train()
#     trainer.save_model("./fine_tuned_gpt2")
#     tokenizer.save_pretrained("./fine_tuned_gpt2")

# def check_and_train():
#     last_trained_at = "1970-01-01 00:00:00" 
#     if os.path.exists("last_trained_time.txt"):
#         with open("last_trained_time.txt", "r") as f:
#             last_trained_at = f.read().strip()

#     connection = connect_to_db()
#     if connection is None:
#         return

#     faq_data = fetch_updated_data(connection, last_trained_at)
#     connection.close()
    
#     if not faq_data:
#         print("All your data is updated already.")
#         return

#     train_df, eval_df = prepare_dataset(faq_data)
#     fine_tune_model(train_df, eval_df)

#     with open("last_trained_time.txt", "w") as f:
#         f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# # --- New code for inference and conversational memory ---

# class ChatbotMemory:
#     def __init__(self, memory_limit=3):
#         self.conversation_history = []
#         self.memory_limit = memory_limit

#     def add_to_memory(self, user_input, bot_response):
#         self.conversation_history.append((user_input, bot_response))
#         if len(self.conversation_history) > self.memory_limit:
#             self.conversation_history.pop(0)  # Keep only the last few interactions

#     def get_last_bot_response(self):
#         if self.conversation_history:
#             return self.conversation_history[-1][1]
#         return None

# def generate_response(user_input, model, tokenizer, chatbot_memory):
#     # Check for follow-up questions
#     follow_up_queries = ["what is it again?", "repeat that", "can you say it again?"]
#     if user_input.strip().lower() in follow_up_queries:
#         last_response = chatbot_memory.get_last_bot_response()
#         if last_response:
#             return last_response
#         else:
#             return "I'm not sure what you're referring to."

#     # Include conversation history in the input
#     conversation_context = ''
#     for u_input, b_response in chatbot_memory.conversation_history:
#         conversation_context += f"User: {u_input}\nBot: {b_response}\n"
#     conversation_context += f"User: {user_input}\nBot:"

#     input_ids = tokenizer.encode(conversation_context, return_tensors='pt')

#     # Generate a response
#     output_ids = model.generate(
#         input_ids,
#         max_length=150,
#         num_return_sequences=1,
#         no_repeat_ngram_size=2,
#         pad_token_id=tokenizer.eos_token_id
#     )
#     output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
#     # Extract the bot's response from the output
#     bot_response = output[len(conversation_context):].strip()

#     # Add to memory
#     chatbot_memory.add_to_memory(user_input, bot_response)
#     return bot_response

# def chatbot_interface():
#     # Load the fine-tuned model and tokenizer
#     tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")
#     model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
#     model.eval()

#     # Initialize conversation memory
#     chatbot_memory = ChatbotMemory()

#     print("Chatbot is ready to chat! Type 'exit' to quit.")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == 'exit':
#             print("Chatbot: Goodbye!")
#             break
#         response = generate_response(user_input, model, tokenizer, chatbot_memory)
#         print(f"Chatbot: {response}")

# if __name__ == "__main__":
#     # First, check and train if needed
#     check_and_train()
#     # Then, start the chatbot interface
#     chatbot_interface()
