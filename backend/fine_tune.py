# import mysql.connector
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from transformers import Trainer, TrainingArguments
# from datasets import Dataset

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

# # Prepare dataset
# def prepare_dataset(faq_data):
#     data = [{'text': f"User: {q}\nBot: {a}"} for q, a in faq_data]
#     df = pd.DataFrame(data)
#     # Split into training and evaluation datasets
#     train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)  # 80% train, 20% eval
#     return train_df, eval_df

# # Fine-tune the model
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

#     # Define training arguments
#     training_args = TrainingArguments(
#         output_dir="./fine_tuned_gpt2",
#         evaluation_strategy="epoch",  # Change to "steps" or "no" to disable
#         learning_rate=5e-5,
#         per_device_train_batch_size=2,
#         num_train_epochs=3,
#         weight_decay=0.01,
#         save_steps=10,
#         save_total_limit=2,
#     )

#     # Create Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_train_dataset,
#         eval_dataset=tokenized_eval_dataset  # Add the evaluation dataset
#     )

#     # Train the model
#     trainer.train()
#     trainer.save_model("./fine_tuned_gpt2")
#     tokenizer.save_pretrained("./fine_tuned_gpt2")  # Save the tokenizer

# def main():
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

# if __name__ == "__main__":
#     main()

#not good





# import mysql.connector
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from transformers import Trainer, TrainingArguments
# from datasets import Dataset

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
#     cursor.execute("SELECT question, answer FROM faq")
#     faq_data = cursor.fetchall()
#     cursor.close()
#     return faq_data

# # Prepare dataset
# def prepare_dataset(faq_data):
#     data = [{'text': f"User: {q}\nBot: {a}"} for q, a in faq_data]
#     df = pd.DataFrame(data)
#     train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42) 
#     return train_df, eval_df

# # Fine-tune the GPT-2 model
# def fine_tune_model(train_df, eval_df):
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     tokenizer.pad_token = tokenizer.eos_token  
#     model = GPT2LMHeadModel.from_pretrained("gpt2")

#     def tokenize_function(examples):
#         encoding = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=150)
#         encoding['labels'] = encoding['input_ids'].copy()
#         return encoding

#     # Create training and evaluation datasets
#     train_dataset = Dataset.from_pandas(train_df)
#     eval_dataset = Dataset.from_pandas(eval_df)
#     tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
#     tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

#     training_args = TrainingArguments(
#         output_dir="./fine_tuned_gpt2",
#         evaluation_strategy="epoch",  
#         learning_rate=5e-5,
#         per_device_train_batch_size=2,
#         num_train_epochs=3,
#         weight_decay=0.01,
#         save_steps=10,
#         save_total_limit=2,
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_train_dataset,
#         eval_dataset=tokenized_eval_dataset
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

#not good






# import mysql.connector
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from transformers import Trainer, TrainingArguments
# from datasets import Dataset
# import numpy as np
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

# # Prepare dataset
# def prepare_dataset(faq_data):
#     data = [{'text': f"Category: {category}\nUser: {q}\nBot: {a}"} for category, q, a in faq_data]
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
#     labels = p.label_ids
#     accuracy = accuracy_score(labels, preds)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
#     return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# # Fine-tune the GPT-2 model
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
#         evaluation_strategy="steps",
#         eval_steps=100,  # Evaluate every 100 steps
#         save_steps=100,  # Save every 100 steps to match eval_steps
#         learning_rate=5e-5,
#         per_device_train_batch_size=2,
#         num_train_epochs=3,
#         weight_decay=0.01,
#         save_total_limit=2,
#         load_best_model_at_end=True,  # Ensure you get the best model
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

#medyo okay





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
# def augment_data(faq_data):
#     paraphraser = pipeline("text2text-generation", model="t5-base")  # T5 model for paraphrasing
#     augmented_data = []
    
#     for category, question, answer in faq_data:
#         # Add the original question
#         augmented_data.append((category, question, answer))
        
#         # Generate variations using the paraphrasing task
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
#     tokenizer.pad_token = tokenizer.eos_token  
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
