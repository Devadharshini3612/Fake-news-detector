# Fake News Detector

A machine learning-based web application that detects whether a news article is *fake* or *real* using natural language processing (NLP) techniques.

## Features

- Input news text and get real-time predictions
- Machine learning model trained on labeled fake and real news data
- Web interface for user-friendly access
- Text preprocessing and vectorization (TF-IDF)
- Logistic Regression / Naive Bayes classifier (customizable)

## Tech Stack

- *Frontend:* HTML, CSS, JavaScript (optional: React)
- *Backend:* Python (Flask / Django)
- *ML Model:* Scikit-learn, Pandas, Numpy
- *Deployment:* Heroku / Render / Localhost

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
### 2.Create Virtual environment
'''Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


3. Install Dependencies

pip install -r requirements.txt

4. Run the App

python app.py

Open your browser and go to http://127.0.0.1:5000

Example

Input:

> "NASA confirms discovery of water on the moon."



Output:

> REAL



Dataset

The model is trained on the Fake and Real News Dataset from Kaggle.


