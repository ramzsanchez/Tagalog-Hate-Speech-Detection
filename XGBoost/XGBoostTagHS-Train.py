import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
data = pd.read_csv("hatespeech-train-dataset-edited.csv")

# Replace missing values with an empty string
data["text"] = data["text"].fillna("")

# Transform the text data into numerical features using TF-IDF vectorization
vectorizer = TfidfVectorizer(
    max_features=11024
)  # Adjust the max_features parameter as needed
X_tfidf = vectorizer.fit_transform(data["text"])

# Save the vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Split the dataset into input features (X) and labels (y)
X = X_tfidf
y = data["label"]

# Train the XGBoost model
model = xgb.XGBClassifier()
model.fit(X, y)

# Save the trained model with .model extension
model.save_model("XGBoosTagHS.model")

print("Model Trained Successfully!")
