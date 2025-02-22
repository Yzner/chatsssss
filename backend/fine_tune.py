# #

# # GALING KAY NIKITA MY LOVE <3


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


# def fetch_data(connection, last_trained_at):
#     """
#     Fetch new data from the database based on the last training timestamp.
#     """
#     cursor = connection.cursor()
#     query = """
#     SELECT category, question, simple_answer, detailed_answer, step_by_step_answer, updated_at
#     FROM faq_data
#     WHERE updated_at > %s AND deleted = FALSE
#     """
#     cursor.execute(query, (last_trained_at,))
#     faq_data = cursor.fetchall()
#     cursor.close()
#     return faq_data

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


def augment_data(data, paraphraser_model="t5-base"):
    """
    Augment data by paraphrasing questions for the same answers.
    """
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
                paraphrased_questions = paraphraser(
                    f"paraphrase: {question} </s>", 
                    num_return_sequences=1
                )
                for paraphrase in paraphrased_questions:
                    augmented_data.append((category, paraphrase['generated_text'], answer))
            except Exception as e:
                logging.warning("Failed to paraphrase '%s': %s", question, e)
    logging.info("Data augmentation complete with %d entries.", len(augmented_data))
    return augmented_data


def prepare_dataset(data):
    """
    Convert data into a training-ready format and split it for training and evaluation.
    """
    augmented_data = augment_data(data)
    dataset = [{"text": f"Category: {cat}\nUser: {q}\nBot: {a}"} for cat, q, a in augmented_data]
    df = pd.DataFrame(dataset)
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    logging.info("Dataset prepared with %d training and %d evaluation samples.", len(train_df), len(eval_df))
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
    Fine-tune GPT-2 model with adaptive learning rate and overfitting prevention.
    """
    try:
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
            lr_scheduler_type="cosine",  
            warmup_steps=100,  
            per_device_train_batch_size=4,  
            per_device_eval_batch_size=4,
<<<<<<< HEAD
            num_train_epochs=5,
            weight_decay=0.1,  # Stronger regularization
=======
            num_train_epochs=3,
            weight_decay=0.1,  
>>>>>>> 04189a6b44bd4f990b08fe6b0dbfa664dec03dac
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=10,
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,
            report_to="all",
            fp16=True,  
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
    except Exception as e:
        logging.error("Model fine-tuning failed: %s", e)


def check_and_train():
    """
    Check for new data and train the model if updates are found.
    """
    last_trained_at = "1970-01-01 00:00:00"
    if os.path.exists("last_trained_time.txt"):
        with open("last_trained_time.txt", "r") as f:
            last_trained_at = f.read().strip()

    connection = connect_to_db()
    if connection is None:
        return

    data = fetch_data(connection, last_trained_at)
    connection.close()

    if not data:
        logging.info("No new data to train on.")
        return

    grouped_data = group_by_intent(data)
    train_df, eval_df = prepare_dataset(grouped_data)

    fine_tune_model(train_df, eval_df)
    with open("last_trained_time.txt", "w") as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    check_and_train()