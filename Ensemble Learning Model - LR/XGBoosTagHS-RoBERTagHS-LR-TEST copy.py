import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

# Load the validation dataset
data_val = pd.read_csv("hatespeech-validate-dataset-edited.csv")

# Replace missing values with an empty string
data_val["text"] = data_val["text"].fillna("")

# Extract the input features and labels from the validation dataset
X_val = data_val["text"]
y_val = data_val["label"]

# Load the TF-IDF vectorizer
vectorizer = joblib.load("vectorizer.pkl")

# Transform the data for logistic regression model
X_tfidf_val = vectorizer.transform(X_val)

# Load the trained Logistic Regression model
lr_meta_model = joblib.load("lr_meta_model.pkl")

# Get predictions from logistic regression meta-learner on the validation dataset
ensemble_preds_val = lr_meta_model.predict(X_tfidf_val)

# Compute evaluation metrics
accuracy_val = accuracy_score(y_val, ensemble_preds_val)
precision_val, _, _, _ = precision_recall_fscore_support(y_val, ensemble_preds_val, average="weighted")
recall_val = recall_score(y_val, ensemble_preds_val, average="weighted")
f1_val = f1_score(y_val, ensemble_preds_val, average="weighted")

# Print evaluation metrics
print("Validation Set Metrics:")
print(f"Accuracy: {100 * accuracy_val:.2f}%")
print(f"Precision: {100 * precision_val:.2f}%")
print(f"Recall: {100 * recall_val:.2f}%")
print(f"F1 Score: {100 * f1_val:.2f}%")
