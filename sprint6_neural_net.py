from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn

# Same data loading and cleaning as before
dataset = load_dataset("mcauley-lab/amazon-reviews-2023", "raw_meta_All_Beauty", split="full")

prices, descriptions = [], []
seen = set()
for item in dataset:
    price_str = item["price"]
    if price_str is not None and price_str != "None":
        try:
            price = float(price_str)
            if price > 0 and item["title"] not in seen:
                seen.add(item["title"])
                prices.append(price)
                descriptions.append(item["title"])
        except (ValueError, TypeError):
            continue

X_train, X_test, y_train, y_test = train_test_split(descriptions, prices, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# Rescale all input numbers to a similar, small range - neural networks train poorly without this
scaler = StandardScaler()
X_train_vec = scaler.fit_transform(X_train_vec)
X_test_vec = scaler.transform(X_test_vec)

# Convert everything into the number format PyTorch actually understands
X_train_tensor = torch.tensor(X_train_vec, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_vec, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define the network: input -> hidden layers -> a single price prediction
class PricePredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

model = PricePredictor(input_size=1000)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# The actual training loop: predict, measure error, adjust, repeat - now 100 times
print("Training...")
for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = loss_fn(predictions.squeeze(), y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.2f}")

# Honest evaluation on data the model never saw during training
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor).squeeze()
    error = torch.mean(torch.abs(test_predictions - y_test_tensor))
    print(f"\nNeural Network: average error = ${error.item():.2f}")