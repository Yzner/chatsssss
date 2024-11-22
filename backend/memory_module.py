import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# LSTM Definition
class MemoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MemoryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Load fine-tuned GPT-2
gpt_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")

# Prepare Data
def prepare_lstm_data(queries, labels):
    inputs = gpt_tokenizer(queries, return_tensors='pt', truncation=True, padding=True)
    input_ids = inputs['input_ids']
    
    # Generate embeddings for the padded input
    with torch.no_grad():  # Disable gradient calculation for efficiency
        embeddings = gpt_model.transformer.wte(input_ids)
    
    # Convert labels to tensor
    y = torch.tensor(labels).float()
    
    return embeddings, y


# Example data
queries = ["Can you elaborate?", "Say that again?", "Clarify your answer?"]
labels = [1, 0, 1]  # 1 = Clarification, 0 = New Query

X, y = prepare_lstm_data(queries, labels)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = DataLoader(TensorDataset(X_train, y_train), batch_size=4, shuffle=True)
val_data = DataLoader(TensorDataset(X_val, y_val), batch_size=4)

# Train LSTM
lstm = MemoryLSTM(input_size=768, hidden_size=128, num_layers=2, output_size=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-3)

for epoch in range(5):  # Epochs
    lstm.train()
    for batch_X, batch_y in train_data:
        optimizer.zero_grad()
        outputs = lstm(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item()}")

# Save the LSTM model
torch.save(lstm.state_dict(), "lstm_weights.pth")
