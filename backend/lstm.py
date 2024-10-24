import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# LSTM Encoder Definition
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        return hn[-1]  # Return last hidden state

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

def fetch_faq_data(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT category, question, answer FROM faq")
    faq_data = cursor.fetchall()
    cursor.close()
    return faq_data

# Prepare dataset
def prepare_dataset(faq_data):
    data = [{'text': f"Category: {category}\nUser: {q}\nBot: {a}"} for category, q, a in faq_data]
    df = pd.DataFrame(data)
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, eval_df

# Tokenize dataset
def tokenize_function(examples, tokenizer):
    encoding = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=150)
    encoding['labels'] = encoding['input_ids'].copy()
    return encoding

# Compute evaluation metrics
def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=-1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Fine-tune the GPT-2 model with LSTM encoding
def fine_tune_model(train_df, eval_df):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Define LSTM parameters
    lstm_input_size = tokenizer.vocab_size
    lstm_hidden_size = 256  # Choose suitable hidden size
    lstm_encoder = LSTMEncoder(input_size=lstm_input_size, hidden_size=lstm_hidden_size)

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    tokenized_train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_eval_dataset = eval_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir="./fine_tuned_gpt2",
        evaluation_strategy="steps",
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

    # Train the model
    trainer.train()
    trainer.save_model("./fine_tuned_gpt2")
    tokenizer.save_pretrained("./fine_tuned_gpt2")

def main():
    connection = connect_to_db()
    if connection is None:
        return
    faq_data = fetch_faq_data(connection)
    connection.close()
    train_df, eval_df = prepare_dataset(faq_data)
    fine_tune_model(train_df, eval_df)

if __name__ == "__main__":
    main()
