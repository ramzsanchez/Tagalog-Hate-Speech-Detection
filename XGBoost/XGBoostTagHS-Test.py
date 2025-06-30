import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load the dataset
data = pd.read_csv("hatespeech-test-dataset-edited.csv")

# Replace missing values with an empty string
data["text"] = data["text"].fillna("")

# Extract the input features (X)
X = data["text"]

# Load the XGBoost model
loaded_model = xgb.XGBClassifier()
loaded_model.load_model("XGBoosTagHS.model")

# Load the TF-IDF vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Transform the test data using the loaded vectorizer
X_tfidf = vectorizer.transform(X)

# Batch size for testing
batch_size = 1000

# Make predictions on the dataset
predictions = []
for i in range(0, X_tfidf.shape[0], batch_size):
    batch_X = X_tfidf[i : i + batch_size]
    batch_pred = loaded_model.predict(batch_X)
    predictions.extend(batch_pred)

# Convert predictions to numpy array
y_pred = np.array(predictions)

# Convert labels to numpy array
y_true = data["label"].values

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred) * 100
precision = precision_score(y_true, y_pred, average="weighted") * 100
recall = recall_score(y_true, y_pred, average="weighted") * 100
f1 = f1_score(y_true, y_pred, average="weighted") * 100

# Display the evaluation metrics
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1:.2f}%")
