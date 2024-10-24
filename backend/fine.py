# import mysql.connector
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from transformers import Trainer, TrainingArguments
# from datasets import Dataset
# import numpy as np
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from transformers import pipeline
# import random

# # Augmenting data with paraphrased questions using a text2text-generation model
# def augment_data(faq_data, max_augments=2):
#     paraphraser = pipeline("text2text-generation", model="t5-base")  # T5 model for paraphrasing
#     augmented_data = []
    
#     for category, question, answer in faq_data:
#         # Add the original question
#         augmented_data.append((category, question, answer))
        
#         # Generate variations using the paraphrasing task
#         paraphrase_prompt = f"paraphrase: {question} </s>"
#         paraphrased_questions = paraphraser(paraphrase_prompt, num_return_sequences=max_augments, do_sample=True)  # Enable sampling

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

# def fetch_faq_data(connection):
#     cursor = connection.cursor()
#     cursor.execute("SELECT category, question, answer FROM faq")
#     faq_data = cursor.fetchall()
#     cursor.close()
#     return faq_data

# # Prepare dataset with data augmentation
# def prepare_dataset(faq_data):
#     augmented_data = augment_data(faq_data)
#     data = [{'text': f"Category: {category}\nUser: {q}\nBot: {a}"} for category, q, a in augmented_data]
#     df = pd.DataFrame(data)
#     train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
#     return train_df, eval_df

# # Tokenize dataset
# def tokenize_function(examples, tokenizer):
#     encoding = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=150)
#     encoding['labels'] = encoding['input_ids'].copy()
#     return encoding

# # Compute evaluation metrics
# def compute_metrics(p):
#     preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
#     preds = np.argmax(preds, axis=-1)
    
#     labels = p.label_ids.flatten()
#     preds = preds.flatten()

#     # Remove padding token (-100 in labels)
#     mask = labels != -100
#     labels = labels[mask]
#     preds = preds[mask]

#     accuracy = accuracy_score(labels, preds)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

#     return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# # Fine-tune the GPT-2 model
# def fine_tune_model(train_df, eval_df):
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
#     model = GPT2LMHeadModel.from_pretrained("gpt2")

#     train_dataset = Dataset.from_pandas(train_df)
#     eval_dataset = Dataset.from_pandas(eval_df)
#     tokenized_train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
#     tokenized_eval_dataset = eval_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

#     training_args = TrainingArguments(
#         output_dir="./fine_tuned_gpt2",
#         evaluation_strategy="steps",
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

#     # Train the model
#     trainer.train()
#     trainer.save_model("./fine_tuned_gpt2")
#     tokenizer.save_pretrained("./fine_tuned_gpt2")

# def main():
#     connection = connect_to_db()
#     if connection is None:
#         return
#     faq_data = fetch_faq_data(connection)
#     connection.close()
#     train_df, eval_df = prepare_dataset(faq_data)
#     fine_tune_model(train_df, eval_df)

# if __name__ == "__main__":
#     main()

#good but need more improvements




# import mysql.connector
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from transformers import Trainer, TrainingArguments
# from datasets import Dataset
# import numpy as np
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from transformers import pipeline
# import random
# from sentence_transformers import SentenceTransformer, util

# # Augmenting data with paraphrased questions using a text2text-generation model
# def augment_data(faq_data, max_augments=2):
#     paraphraser = pipeline("text2text-generation", model="t5-base")  # T5 model for paraphrasing
#     augmented_data = []
    
#     for category, question, answer in faq_data:
#         # Add the original question
#         augmented_data.append((category, question, answer))
        
#         # Generate variations using paraphrasing
#         paraphrase_prompt = f"paraphrase: {question} </s>"
#         paraphrased_questions = paraphraser(paraphrase_prompt, num_return_sequences=max_augments, do_sample=True)

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

# def fetch_faq_data(connection):
#     cursor = connection.cursor()
#     cursor.execute("SELECT category, question, answer FROM faq")
#     faq_data = cursor.fetchall()
#     cursor.close()
#     return faq_data

# # Prepare dataset with data augmentation
# def prepare_dataset(faq_data):
#     augmented_data = augment_data(faq_data)
#     data = [{'text': f"Category: {category}\nUser: {q}\nBot: {a}"} for category, q, a in augmented_data]
#     df = pd.DataFrame(data)
#     train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
#     return train_df, eval_df

# # Tokenize dataset
# def tokenize_function(examples, tokenizer):
#     encoding = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=150)
#     encoding['labels'] = encoding['input_ids'].copy()
#     return encoding

# # Compute evaluation metrics
# def compute_metrics(p):
#     preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
#     preds = np.argmax(preds, axis=-1)
    
#     labels = p.label_ids.flatten()
#     preds = preds.flatten()

#     # Remove padding token (-100 in labels)
#     mask = labels != -100
#     labels = labels[mask]
#     preds = preds[mask]

#     accuracy = accuracy_score(labels, preds)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

#     return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# # Fine-tune the GPT-2 model
# def fine_tune_model(train_df, eval_df):
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
#     model = GPT2LMHeadModel.from_pretrained("gpt2")

#     train_dataset = Dataset.from_pandas(train_df)
#     eval_dataset = Dataset.from_pandas(eval_df)
#     tokenized_train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
#     tokenized_eval_dataset = eval_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

#     training_args = TrainingArguments(
#         output_dir="./fine_tuned_gpt2",
#         evaluation_strategy="steps",
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

#     # Train the model
#     trainer.train()
#     trainer.save_model("./fine_tuned_gpt2")
#     tokenizer.save_pretrained("./fine_tuned_gpt2")

# def main():
#     connection = connect_to_db()
#     if connection is None:
#         return
#     faq_data = fetch_faq_data(connection)
#     connection.close()
#     train_df, eval_df = prepare_dataset(faq_data)
#     fine_tune_model(train_df, eval_df)

# if __name__ == "__main__":
#     main()

#good but need more improvements





# import mysql.connector
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
# from datasets import Dataset
# from sentence_transformers import SentenceTransformer, util
# import nltk
# from nltk.corpus import wordnet
# nltk.download('wordnet')
# from evaluate import load

# # Database connection function
# def connect_to_db():
#     try:
#         connection = mysql.connector.connect(
#             host='localhost',
#             user='root',  # replace with your username
#             password='root',  # replace with your password
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

# # Augment the dataset by paraphrasing questions
# def paraphrase_question(question):
#     words = question.split()
#     new_sentence = []
#     for word in words:
#         synonyms = wordnet.synsets(word)
#         if synonyms:
#             new_sentence.append(synonyms[0].lemmas()[0].name())
#         else:
#             new_sentence.append(word)
#     return ' '.join(new_sentence)

# def augment_dataset(faq_data):
#     augmented_data = []
#     for q, a in faq_data:
#         augmented_data.append({'text': f"User: {q}\nBot: {a}"})
#         augmented_data.append({'text': f"User: {paraphrase_question(q)}\nBot: {a}"})  # Paraphrased
#     return augmented_data

# # Prepare dataset
# def prepare_dataset(faq_data):
#     data = augment_dataset(faq_data)
#     df = pd.DataFrame(data)
#     train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)  # 80% train, 20% eval
#     return train_df, eval_df

# # Fine-tune the GPT-2 model
# def fine_tune_model(train_df, eval_df):
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     tokenizer.pad_token = tokenizer.eos_token  # Set pad token
#     model = GPT2LMHeadModel.from_pretrained("gpt2")

#     # Tokenize the inputs and prepare labels
#     def tokenize_function(examples):
#         encoding = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=150)
#         encoding['labels'] = encoding['input_ids'].copy()
#         return encoding

#     # Create training and evaluation datasets
#     train_dataset = Dataset.from_pandas(train_df)
#     eval_dataset = Dataset.from_pandas(eval_df)
    
#     tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
#     tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

#     # Custom metrics (example: ROUGE score)
#     rouge = load("rouge")
    
#     def compute_metrics(eval_pred):
#         predictions, labels = eval_pred

#         # If the predictions are logits, we need to take the argmax to get the predicted token IDs
#         if isinstance(predictions, tuple):
#             predictions = predictions[0]
        
#         # Get the predicted token IDs (argmax of logits)
#         predictions = predictions.argmax(-1)
        
#         # Convert the predicted token IDs and labels back into text
#         decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#         decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
#         # Replace empty labels with a space to avoid rouge computation issues
#         decoded_preds = [" ".join(pred.strip().split()) for pred in decoded_preds]
#         decoded_labels = [" ".join(label.strip().split()) for label in decoded_labels]
        
#         # Compute ROUGE scores
#         result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
#         return result


#     # Define training arguments
#     training_args = TrainingArguments(
#         output_dir="./fine_tuned_gpt2",
#         evaluation_strategy="epoch",
#         learning_rate=5e-5,
#         per_device_train_batch_size=4,  # Increased batch size for faster training
#         per_device_eval_batch_size=4,
#         num_train_epochs=5,  # Increased number of epochs
#         weight_decay=0.01,
#         save_steps=100,  # Save every 100 steps
#         save_total_limit=2,
#         warmup_steps=500,  # Use warmup for learning rate
#     )

#     # Create Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_train_dataset,
#         eval_dataset=tokenized_eval_dataset,
#         compute_metrics=compute_metrics  # Add custom metrics
#     )

#     # Train the model
#     trainer.train()
#     trainer.save_model("./fine_tuned_gpt2")
#     tokenizer.save_pretrained("./fine_tuned_gpt2")

# # Find most similar question using SentenceTransformer embeddings
# def find_most_similar_question(user_query, faq_data, model):
#     query_embedding = model.encode(user_query, convert_to_tensor=True)
#     faq_embeddings = model.encode([faq[0] for faq in faq_data], convert_to_tensor=True)
    
#     similarities = util.pytorch_cos_sim(query_embedding, faq_embeddings)
#     best_match_idx = similarities.argmax()
#     return faq_data[best_match_idx]

# # Main function
# def main():
#     # Load sentence-transformer model for semantic similarity
#     similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

#     # Connect to the database
#     connection = connect_to_db()
#     if connection is None:
#         return

#     # Fetch FAQ data
#     faq_data = fetch_faq_data(connection)
#     connection.close()

#     # Prepare dataset
#     train_df, eval_df = prepare_dataset(faq_data)

#     # Fine-tune the model
#     fine_tune_model(train_df, eval_df)

#     # Example usage: Matching a user query to the most similar FAQ question
#     user_query = "How can I reset my password?"
#     best_match = find_most_similar_question(user_query, faq_data, similarity_model)
#     print(f"Best matching FAQ question: {best_match[0]}")

# if __name__ == "__main__":
#     main()


#bulok