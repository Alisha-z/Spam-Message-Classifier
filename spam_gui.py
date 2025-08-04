import tkinter as tk
from tkinter import messagebox
import joblib
import re

# Load trained pipeline (Vectorizer + Model)
vectorizer = joblib.load("tfidf_vectorizer.joblib")
model = joblib.load("spam_classifier_model.joblib")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Prediction function
def predict_spam():
    message = input_text.get("1.0", tk.END).strip()
    if not message:
        messagebox.showwarning("Input Error", "Please enter a message.")
        return

    cleaned = clean_text(message)
    prediction = model.predict([cleaned])[0]

    result_label.config(text=f"Prediction: {'Spam' if prediction == 'spam' else 'Not Spam'}")

# GUI Setup
root = tk.Tk()
root.title("Spam Message Classifier")

root.geometry("400x300")
root.configure(bg="black")

title = tk.Label(root, text="Spam Message Classifier", font=("Arial", 16, "bold"), fg="green", bg="black")
title.pack(pady=10)

input_text = tk.Text(root, height=5, width=40, font=("Arial", 12))
input_text.pack(pady=10)

predict_button = tk.Button(root, text="Predict", command=predict_spam, bg="green", fg="white", font=("Arial", 12))
predict_button.pack(pady=5)

result_label = tk.Label(root, text="", font=("Arial", 14), fg="white", bg="black")
result_label.pack(pady=10)

root.mainloop()
