import tkinter as tk
from tkinter import scrolledtext
import joblib
from chatbot import transform_text,chatbot,vectorizer

# Load the model from file
classifier = joblib.load('classifier.pkl')
label_encoder=joblib.load('label_encoder.pkl')
class ChatApp:
    def __init__(self, master):
        self.master = master
        master.title("Colorful Chat App")

        # Frames
        self.chat_frame = tk.Frame(master, bg="#f0f0f0")
        self.chat_frame.pack(fill=tk.BOTH, expand=True)

        # Chat history
        self.chat_history = scrolledtext.ScrolledText(self.chat_frame, bg="#ffffff", fg="#333333")
        self.chat_history.pack(fill=tk.BOTH, expand=True)

        # Entry for user input
        self.input_field = tk.Entry(master, bg="#ffffff", fg="#333333", relief=tk.FLAT)
        self.input_field.pack(fill=tk.BOTH, expand=True)
        self.input_field.bind("<Return>", self.send_message)

        # Send button
        self.send_button = tk.Button(master, text="Send", command=self.send_message, bg="#0080ff", fg="#ffffff", relief=tk.RAISED)
        self.send_button.pack(pady=5)

    def send_message(self, event=None):
        message = self.input_field.get()
        if message.strip():
            self.input_field.delete(0, tk.END)
            self.display_message("You: " + message)
            message = transform_text(message)
            predicted_tag_id = classifier.predict(vectorizer.transform([message]))
            predicted_tag = label_encoder.inverse_transform(predicted_tag_id)
            self.display_message("Sahil: " + chatbot.loc[predicted_tag[0],'Response'])
    def display_message(self, message):
        self.chat_history.insert(tk.END, message + "\n")
        self.chat_history.see(tk.END)

def main():
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
