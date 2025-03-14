from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import torch
import json
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

# Load the model
model_data = torch.load("chatbot_model.pth")

# Extract model data
all_words = model_data["all_words"]
tags = model_data["tags"]
input_size = model_data["input_size"]
hidden_size = model_data["hidden_size"]
output_size = model_data["output_size"]

# Load the model
class ChatbotNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotNN, self).__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

model = ChatbotNN(input_size, hidden_size, output_size)
model.load_state_dict(model_data["model_state"])
model.eval()

# Preprocessing functions
nltk.download('punkt')
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

# Flask app
app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def whatsapp_reply():
    incoming_message = request.form['Body'].strip().lower()

    # Convert message to bag of words
    tokenized = tokenize(incoming_message)
    bow = bag_of_words(tokenized, all_words)
    bow = torch.from_numpy(bow).float().unsqueeze(0)

    # Get the model's prediction
    output = model(bow)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Custom reply
    if tag == "greeting":
        response = "Hey bro! 👋 How can I help you?"
    elif tag == "goodbye":
        response = "Goodbye bro! 👋 See you soon."
    elif tag == "thanks":
        response = "No problem bro! 😊"
    else:
        response = "I'm not sure bro, but I'm learning! 😎"

    # Twilio response
    reply = MessagingResponse()
    reply.message(response)
    return str(reply)

if __name__ == "__main__":
    app.run(debug=True)
