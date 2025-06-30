import pandas as pd
import numpy as np
import torch
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from transformers import RobertaTokenizer, RobertaModel
import joblib
from tqdm import tqdm

# Load the dataset
data = pd.read_csv("hatespeech-test-dataset-edited.csv")

# Replace missing values with an empty string
data["text"] = data["text"].fillna("")

# Extract the input features and labels
X = data["text"]
y = data["label"]

# --- Load XGBoost Model ---
# Load the XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("XGBoosTagHS.model")

# --- Load RoBERTa Model ---
# Initialize the RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("RoBERTagHS_epoch_30")
roberta_model = RobertaModel.from_pretrained("RoBERTagHS_epoch_30")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
roberta_model.to(device)  # Move the model to the appropriate device

# --- Load TF-IDF Vectorizer ---
# Load the TF-IDF vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- Ensemble Prediction ---
# Transform the data for XGBoost model
X_tfidf = vectorizer.transform(X)

# Get predictions from the XGBoost model
xgb_preds = xgb_model.predict(X_tfidf)

# Tokenize and encode the data for RoBERTa model
batch_size = 32  # Set the batch size
num_samples = len(X)
num_batches = int(np.ceil(num_samples / batch_size))

roberta_outputs = []

with tqdm(total=num_batches, desc="RoBERTa Predictions") as pbar:
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_texts = X[start_idx:end_idx].tolist()

        # Tokenize and encode the batch of texts
        batch_encoded = tokenizer.batch_encode_plus(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        batch_inputs = batch_encoded["input_ids"].to(device)
        attention_mask = batch_encoded["attention_mask"].to(device)

        # Perform forward pass through RoBERTa
        with torch.no_grad():
            roberta_outputs_batch = roberta_model(batch_inputs, attention_mask=attention_mask)

        roberta_outputs.append(roberta_outputs_batch.pooler_output.cpu().numpy())

        pbar.update(1)

roberta_outputs = np.concatenate(roberta_outputs, axis=0)

# Concatenate the XGBoost and RoBERTa predictions as features
meta_features = np.concatenate((X_tfidf.toarray(), roberta_outputs), axis=1)

# --- Logistic Regression Training ---
# Initialize the Logistic Regression model
lr_meta_model = LogisticRegression(max_iter=10000)

# Fit the Logistic Regression model
lr_meta_model.fit(meta_features, y)
print("Logistic Regression Training Complete!")

# Save the Logistic Regression model
joblib.dump(lr_meta_model, "lr_meta_model.pkl")

# Save other necessary files
joblib.dump(tokenizer, "tokenizer.pkl")
joblib.dump(roberta_model, "roberta_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# --- Logistic Regression Prediction ---
# Get predictions from logistic regression meta-learner with progress bar
ensemble_preds = []

with tqdm(total=len(X), desc="Logistic Regression Predictions") as pbar:
    for i in range(len(X)):
        pred = lr_meta_model.predict(meta_features[i].reshape(1, -1))
        ensemble_preds.append(pred[0])
        pbar.update(1)

# Compute evaluation metrics
accuracy = accuracy_score(y, ensemble_preds)
precision, _, _, _ = precision_recall_fscore_support(y, ensemble_preds, average="weighted")
recall = recall_score(y, ensemble_preds, average="weighted")
f1 = f1_score(y, ensemble_preds, average="weighted")

print("Ensemble Model Metrics:")
print(f"Accuracy: {100 * accuracy:.2f}%")
print(f"Precision: {100 * precision:.2f}%")
print(f"Recall: {100 * recall:.2f}%")
print(f"F1 Score: {100 * f1:.2f}%")
