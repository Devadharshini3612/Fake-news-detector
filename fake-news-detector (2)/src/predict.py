import joblib

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def test_news(news_text):
    vector = vectorizer.transform([news_text])
    prediction = model.predict(vector)
    return "Real News" if prediction[0] == 1 else "Fake News"
