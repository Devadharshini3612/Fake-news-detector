# Fake News Detector

A simple machine learning project to detect fake news using Logistic Regression and TF-IDF vectorization. This project includes a Streamlit web app for easy use.

## How to Use

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Train model: `python src/train.py`
4. Run app: `streamlit run app/app.py`

## Files

- `src/train.py` - Trains and saves the model
- `src/predict.py` - Loads model and predicts new input
- `app/app.py` - Streamlit frontend
- `model/` - Stores trained models
