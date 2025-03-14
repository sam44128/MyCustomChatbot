import json
import torch
import random
import nltk
from nltk.stem.porter import PorterStemmer
from model import NeuralNet
from utils import bag_of_words, tokenize

# Load intents.json
with open("intents.json", 'r') as file:
    intents = json.load(file)

# Load the trained model
model_data = torch.load("chatbot_model.pth")

input_size = model_data["input_size"]
hidden_size = model_data["hidden_size"]
output_size = model_data["output_size"]
all_words = model_data["all_words"]
tags = model_data["tags"]
model_state = model_data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Chatting loop
print("Chatbot is ready to chat! Type 'quit' to exit.")

while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        break

    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    x = torch.from_numpy(x).float().unsqueeze(0)

    output = model(x)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                print(f"{random.choice(intent['responses'])}")
    else:
        print("I don't understand, bro 😅")

# utils.py file
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# model.py file
import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
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
