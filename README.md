# Spam-Message-Classifier

# 📩 Spam Message Classifier (ML Project with GUI)

This project is a simple yet effective **Spam Message Classifier** built using **Python**, **scikit-learn**, and a user-friendly **Tkinter GUI**. It classifies SMS messages as either **Spam** or **Not Spam (Ham)** using machine learning techniques.

## 🚀 Features

- Cleaned and preprocessed real SMS dataset
- Trained a machine learning pipeline (TfidfVectorizer + Multinomial Naive Bayes)
- Model saved using `joblib` for reuse
- Integrated a modern Tkinter GUI for live message prediction

## 🧠 Machine Learning Model

- **Algorithm**: Multinomial Naive Bayes
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Evaluation**: Accuracy, Confusion Matrix, Classification Report

## 📁 Project Structure
spam_message_classifier.py # Main script with model training and prediction logic
├── spam_gui.py # GUI script for real-time prediction
├── spam_classifier_model.joblib # Saved trained model
├── spam.csv # Dataset file


## 🖥️ How to Run

1. Clone the repository
2. Ensure required libraries are installed:
pip install pandas scikit-learn joblib matplotlib seaborn

3. Run the training file (if needed):
   python spam_message_classifier.py
   
4. Run the GUI:
   python spam_gui.py

## 📊 Dataset

- **Source**: SMS Spam Collection Dataset
- **Columns**: 
- `v1`: Label (ham/spam)
- `v2`: Message text

## 💡 Future Improvements

- Add deep learning model support
- Deploy using Flask/Streamlit
- Make it mobile-responsive

## 👩‍💻 Developed by

Alisha — Intern at High Tech Software House, Python & Machine Learning Enthusiast.

   

   


