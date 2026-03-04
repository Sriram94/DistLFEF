import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import os

class HNetworkPredictor(nn.Module):
    def __init__(self, num_actions):
        super(HNetworkPredictor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

def train_predictor(data_path, env_name, epochs=50, batch_size=64):
    data = np.load(data_path)
    states = torch.FloatTensor(data['states'])
    human_actions = torch.LongTensor(data['actions'])
    
    num_actions = int(human_actions.max() + 1)
    model = HNetworkPredictor(num_actions)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(states.size(0))
        epoch_loss = 0
        
        for i in range(0, states.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = states[indices], human_actions[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(states):.6f}")
        
    torch.save(model.state_dict(), f"{env_name}_predictor_85.pth")
    print(f"Model saved as {env_name}_predictor_85.pth")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pretrain.py <data_path> <env_name>")
        sys.exit(1)
    train_predictor(sys.argv[1], sys.argv[2])
