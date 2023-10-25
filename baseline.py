import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Read the CSV file
data = pd.read_csv('preProcessedData.csv')

X = data.iloc[:, 1:]
Y = data.iloc[:, 0]

# Split into training & validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

class BaselineMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(BaselineMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)
num_classes = len(label_encoder.classes_)
input_size = X_train.shape[1]
num_classes = len(label_encoder.classes_)  # Number of unique classes

model = BaselineMLP(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    inputs = torch.tensor(X_train.values, dtype=torch.float32)
    labels = torch.tensor(Y_train, dtype=torch.long)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item()}')

# Save
torch.save(model.state_dict(), 'baseline_mlp_model.pth')