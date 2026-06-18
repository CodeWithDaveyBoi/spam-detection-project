import streamlit as st
import pickle
import re
import string
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# Load model and vectorizer
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Text cleaning setup
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Prediction function
def predict_spam(message):
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vector)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# UI
st.title("📩 Spam Detection System")

st.write("Enter a message to check whether it is Spam or Not Spam.")

st.write("Developed by Dave Fernandes (A-216) SYMCA 4th Sem.") (1430)

user_input = st.text_area("Enter your message here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        result = predict_spam(user_input)
        
        if result == "Spam":
            st.error("🚫 This is a SPAM message")
        else:
            st.success("✅ This is NOT a spam message")


st.write("Model Accuracy: 97.67% (Naive Bayes)")

st.markdown("---")
st.subheader("How it works")
st.write("This system uses Machine Learning (Naive Bayes + TF-IDF) to classify messages as Spam or Not Spam.")
