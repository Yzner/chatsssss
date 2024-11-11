# memory_lstm.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Define the LSTM model for memory
class MemoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MemoryLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

# Training data and labels for LSTM (example dataset)
training_data = [
    ("Can you repeat that?", "repeat_request"),
    ("What did you say again?", "repeat_request"),
    ("I didn’t understand, can you explain?", "clarify_request")
]

# Dataset class for training the LSTM
class MemoryDataset(Dataset):
    def __init__(self, training_data):
        self.data = training_data
        self.labels = {"repeat_request": 0, "clarify_request": 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoded_text = np.array([ord(char) for char in text])  # Example encoding, use proper tokenizer here
        return torch.tensor(encoded_text, dtype=torch.float32), self.labels[label]

# Train the LSTM
def train_memory_lstm():
    model = MemoryLSTM(input_size=50, hidden_size=64, output_size=2)  # Adjust input size to match input encoding
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_loader = DataLoader(MemoryDataset(training_data), batch_size=1, shuffle=True)

    for epoch in range(10):  # Simple 10-epoch training
        for inputs, labels in train_loader:
            outputs = model(inputs.unsqueeze(0))
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), "memory_lstm.pth")

# Run the training function
if __name__ == "__main__":
    train_memory_lstm()
