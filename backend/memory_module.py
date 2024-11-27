


import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import logging


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

class MemoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MemoryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)  # Pass input through LSTM
        out = self.fc(out[:, -1, :])       # Use the last time step
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device),
                torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device))


# Load fine-tuned GPT-2
gpt_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2").to(device)
gpt_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")

# Data Preparation Function
def prepare_lstm_data(queries, labels):
    """
    Prepares data for the LSTM model by tokenizing and embedding the inputs.
    """
    try:
        inputs = gpt_tokenizer(queries, return_tensors='pt', truncation=True, padding=True, max_length=50)
        input_ids = inputs['input_ids'].to(device)

        # Generate embeddings for the padded input
        with torch.no_grad():
            embeddings = gpt_model.transformer.wte(input_ids)

        # Convert labels to tensor
        y = torch.tensor(labels).float().unsqueeze(1).to(device)
        return embeddings, y
    except Exception as e:
        logging.error(f"Error during data preparation: {e}")
        raise

# Example data for training LSTM
queries = ["Can you elaborate?", "Say that again?", "Clarify your answer?", "repeat", "again"]
labels = [1, 1, 1, 1, 1]  # Binary labels (1 = clarification needed)

# Prepare data
X, y = prepare_lstm_data(queries, labels)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = DataLoader(TensorDataset(X_train, y_train), batch_size=4, shuffle=True)
val_data = DataLoader(TensorDataset(X_val, y_val), batch_size=4)

# Model, Criterion, and Optimizer
hidden_size = 128
num_layers = 2
learning_rate = 1e-3
epochs = 7

lstm = MemoryLSTM(input_size=768, hidden_size=hidden_size, num_layers=num_layers, output_size=1).to(device)
criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Training Loop
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        hidden = model.init_hidden(batch_X.size(0))
        outputs, _ = model(batch_X, hidden)
        loss = criterion(outputs.squeeze(), batch_y.squeeze())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

# Evaluation Loop
def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            hidden = model.init_hidden(batch_X.size(0))
            outputs, _ = model(batch_X, hidden)
            loss = criterion(outputs.squeeze(), batch_y.squeeze())
            val_loss += loss.item()
    return val_loss / len(val_loader)

# Training Process
for epoch in range(epochs):
    train_loss = train_epoch(lstm, train_data, criterion, optimizer)
    val_loss = evaluate(lstm, val_data, criterion)
    logging.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Save the LSTM model after training
model_save_path = "lstm_weights.pth"
torch.save({
    'model_state_dict': lstm.state_dict(),
    'config': {
        'input_size': 768,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'output_size': 1
    }
}, model_save_path)

# Load the model weights from the checkpoint
checkpoint = torch.load("lstm_weights.pth")

# Create the LSTM model instance
lstm_loaded = MemoryLSTM(
    input_size=checkpoint['config']['input_size'],
    hidden_size=checkpoint['config']['hidden_size'],
    num_layers=checkpoint['config']['num_layers'],
    output_size=checkpoint['config']['output_size']
).to(device)

# Load the model state dict
lstm_loaded.load_state_dict(checkpoint['model_state_dict'])
logging.info("Model loaded successfully and ready for use.")