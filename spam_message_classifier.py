import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import string
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv("spam_data.txt", sep="\t", header=None, names=["label", "message"])

# Encode labels: ham = 0, spam = 1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.2, random_state=42)

# Create pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "spam_classifier_model.joblib")
print("✅ Model saved as spam_classifier_model.joblib")

# --- Function to Plot Top Spam Words ---
def plot_spam_words(df):
    # Filter spam messages
    spam_messages = df[df["label"] == 1]["message"]
    spam_text = " ".join(spam_messages).lower()
    spam_words = spam_text.translate(str.maketrans("", "", string.punctuation)).split()
    filtered_words = [word for word in spam_words if word not in stop_words]
    word_freq = Counter(filtered_words)
    top_words = word_freq.most_common(20)

    if not top_words:
        print("⚠️ No spam words found.")
        return

    words, counts = zip(*top_words)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(counts), y=list(words), hue=list(words), palette="Reds_r", legend=False)
    plt.title("Top 20 Most Common Words in Spam Messages")
    plt.xlabel("Frequency")
    plt.ylabel("Spam Words")
    plt.tight_layout()
    plt.show()

# --- Show Top Spam Words ---
plot_spam_words(df)

# --- Predict User Messages (CLI) ---
print("\n📨 Enter a message to check if it's SPAM or not. Type 'exit' to quit.")
while True:
    user_input = input(">> Your message: ")
    if user_input.lower() == 'exit':
        print("👋 Exiting. Thank you!")
        break
    prediction = model.predict([user_input])[0]
    result = "🚫 SPAM" if prediction == 1 else "✅ HAM (Not Spam)"
    print("Prediction:", result)
