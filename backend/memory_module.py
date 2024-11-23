import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LSTM Definition
class MemoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MemoryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Output size should be 1 for binary classification

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get the output from the last time step
        return out  # Output shape should be (batch_size, 1)

# Load fine-tuned GPT-2
gpt_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2").to(device)
gpt_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")

# Prepare Data
def prepare_lstm_data(queries, labels):
    inputs = gpt_tokenizer(queries, return_tensors='pt', truncation=True, padding=True, max_length=50)
    input_ids = inputs['input_ids'].to(device)
    
    # Generate embeddings for the padded input
    with torch.no_grad():  # Disable gradient calculation for efficiency
        embeddings = gpt_model.transformer.wte(input_ids)
    
    # Convert labels to tensor and move to device
    y = torch.tensor(labels).float().unsqueeze(1).to(device)  # Ensure labels are of shape (batch_size, 1)
    
    return embeddings, y

# Example data
queries = ["Can you elaborate?", "Say that again?", "Clarify your answer?", "repeat", "again", "say that again", "can you repeat", "can you say that again", "what did you say"]
labels = [1, 1, 1, 1, 1, 1, 1, 1, 1]  # 1 = Clarification, 0 = New Query

X, y = prepare_lstm_data(queries, labels)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = DataLoader(TensorDataset(X_train, y_train), batch_size=4, shuffle=True)
val_data = DataLoader(TensorDataset(X_val, y_val), batch_size=4)

# Train LSTM
lstm = MemoryLSTM(input_size=768, hidden_size=128, num_layers=2, output_size=1).to(device)
criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-3)

# Training loop with validation
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y.squeeze())  # Make sure both are the same shape
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y.squeeze())  # Ensure both outputs and labels match
            val_loss += loss.item()
    return val_loss / len(val_loader)

# Training loop with epochs and validation loss
for epoch in range(5):  # Epochs
    train_loss = train_epoch(lstm, train_data, criterion, optimizer)
    val_loss = evaluate(lstm, val_data, criterion)
    
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Save the LSTM model
torch.save(lstm.state_dict(), "lstm_weights.pth")
