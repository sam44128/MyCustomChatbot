import json
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
import random
import numpy as np
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')

# Load the training data
with open("train_data.txt", "r") as file:
    data = file.readlines()

# Preprocessing
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

# Extract data
all_words = []
tags = []
xy = []

for line in data:
    parts = line.strip().split("|")
    tag = parts[0]
    pattern = parts[1]
    tags.append(tag)
    words = tokenize(pattern)
    all_words.extend(words)
    xy.append((words, tag))

# Stem and remove duplicates
all_words = [stem(w) for w in all_words if w != "?"]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Neural Network Model
class ChatbotNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

# Hyperparameters
input_size = len(all_words)
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

# Model, Loss, Optimizer
model = ChatbotNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    outputs = model(torch.from_numpy(X_train).float())
    loss = criterion(outputs, torch.from_numpy(y_train).long())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save model data
model_data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

# Save model
torch.save(model_data, "chatbot_model.pth")

print("Training complete. Model saved as chatbot_model.pth")
