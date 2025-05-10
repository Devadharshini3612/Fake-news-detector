import joblib

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_news(text):
    # Transform the input text using the vectorizer
    text_vectorized = vectorizer.transform([text])
    # Predict using the model
    prediction = model.predict(text_vectorized)[0]
    return "Real" if prediction == 1 else "Fake"

# Example usage
sample_text = "Breaking news: Scientists have discovered a new planet that supports life."
print(predict_news(sample_text))
