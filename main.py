import tkinter as tk
from tkinter import scrolledtext
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
from chatbot_training import chatbot_data,vectorizer

classifier = joblib.load('classifier.pkl')
label_encoder = joblib.load('label_encoder.pkl')



stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [ps.stem(word) for word in text if word.isalnum() and word not in stop_words and word not in string.punctuation]
    return ' '.join(text) 

def chat():
    user_input_data = input_field.get()
    input_field.delete(0, tk.END)
    user_input = transform_text(user_input_data)
    predicted_tag_id = classifier.predict(vectorizer.transform([user_input]))
    predicted_tag = label_encoder.inverse_transform(predicted_tag_id)
    response_text = chatbot_data.loc[predicted_tag[0], 'Response']
    display_message("You: " + user_input_data)
    display_message("Bot: " + response_text)

# Define display_message function
def display_message(message):
    chat_history.insert(tk.END, message + "\n")
    chat_history.see(tk.END)

# Create Tkinter window
root = tk.Tk()
root.title("Simple Chatbot")

# Chat history window
chat_history = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=40, height=10)
chat_history.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

# User input field
input_field = tk.Entry(root, width=30)
input_field.grid(row=1, column=0, padx=5, pady=5)

# Send button
send_button = tk.Button(root, text="Send", width=10, command=chat)
send_button.grid(row=1, column=1, padx=5, pady=5)

# Run the Tkinter event loop
root.mainloop()
